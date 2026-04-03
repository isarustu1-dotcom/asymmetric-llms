import argparse
import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
import mmengine
import torch
import wandb
from mmengine.runner.utils import set_random_seed
from transformers import TrainingArguments

from ib_edl import DATASETS, get_model_and_tokenizer, ClassificationMetric, FTTrainer, plot_predictions, save_predictions, setup_logger, optimize_weights, optimize_temperature_scaling


def parse_args():
    parser = ArgumentParser('Fine-tune model.')
    parser.add_argument('--config_base', help='Path to config file for base model.')
    parser.add_argument('--config_sidekick', help='Path to config file for sidekick model.')
    parser.add_argument('--work-dir', '-w', help='Working directory.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID.')
    parser.add_argument('--run-name', '-n', help='Run name of wandb.')
    parser.add_argument('--run-group', '-g', help='Run group of wandb.')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb.')
    parser.add_argument('--skip-ft', '-s', action='store_true', help='Skip fine-tuning.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument(
        '--cfg-options',
        '-o',
        nargs='+',
        action=mmengine.DictAction,
        help='Override the config entry using xxx=yyy format.')

    return parser.parse_args()


def _sanity_check_cfg(cfg: mmengine.Config, args: argparse.Namespace) -> None:
    if args.skip_ft:
        assert cfg.model.peft_path is not None, 'peft_path is required when skip_ft is True.'


def main():
    args = parse_args()
    set_random_seed(args.seed)
    work_dir = args.work_dir
    mmengine.mkdir_or_exist(work_dir)
    device = torch.device(f'cuda:{args.gpu_id}')

    timestamp = datetime.now().strftime('%m%d_%H%M_%S')

    print(f'Training with seed {args.seed} has begun. \n')

    #Getting base and sidekick names
    base_model_name_list = args.config_base.split('/')[1].split('_')[-2:]
    base_model_name = '_'.join(base_model_name_list)

    sidekick_model_name_list = args.config_sidekick.split('/')[1].split('_')[-2:]
    sidekick_model_name = '_'.join(sidekick_model_name_list)

    model_combination_path = f"{base_model_name}_{sidekick_model_name}"
    work_dir = osp.join(work_dir, model_combination_path)

    # Running training pipeline for base and sidekick models separetely
    for model_type in ['base', 'sidekick']:
        assert args.config_base and args.config_sidekick, f'Base or sidekick config file is missing.'

        if model_type == 'base':
            cfg = mmengine.Config.fromfile(args.config_base)
            if args.cfg_options is not None:
                cfg.merge_from_dict(args.cfg_options)
            _sanity_check_cfg(cfg, args)
        else:
            cfg = mmengine.Config.fromfile(args.config_sidekick)
            if args.cfg_options is not None:
                cfg.merge_from_dict(args.cfg_options)
            _sanity_check_cfg(cfg, args)


        if not args.no_wandb:
            run_name = args.run_name if args.run_name is not None else timestamp
            run_group = args.run_group if args.run_group is not None else None
            wandb.init(project=f'ib-edl-{model_type}', dir=work_dir + f'/{model_type}', name=run_name, group=run_group)
            wandb.config.update({f'ib-edl-{model_type}_config': cfg.to_dict()})
  

        logger = setup_logger(
            name='ib-edl',
            filepath=osp.join(work_dir, f'{model_type}/{timestamp}.log'),
        )
        logger.info(f'Using {model_type} config:\n' + '=' * 60 + f'\n{cfg.pretty_text}\n' + '=' * 60)

  
        cfg.dump(osp.join(work_dir, f'{model_type}/{osp.splitext(osp.basename(cfg.filename))[0]}_{timestamp}.yaml'))

        # Import model & tokenizer (Config variables come from configs -> _base_ & configs -> ib_task_model_.yaml)
        model, tokenizer = get_model_and_tokenizer(**cfg.model, device=device)

        # Downloading the dataset and apply the tokenizer on it
        train_set = DATASETS.build(cfg.data['train'], default_args=dict(tokenizer=tokenizer)) 
        val_set = DATASETS.build(cfg.data['val'], default_args=dict(tokenizer=tokenizer))
        test_set = DATASETS.build(cfg.data['test'], default_args=dict(tokenizer=tokenizer))

        # Getting target_ids for tokenized datasets for base and sidekick models
        train_target_ids = train_set.target_ids
        val_target_ids = val_set.target_ids
        test_target_ids = test_set.target_ids

        assert torch.all(train_target_ids == val_target_ids), f'target_ids of train and val sets are different for {model_type} tokenizer.'
 
        if type(train_set) is type(test_set):
            target_ids = train_target_ids
        else:
            assert args.skip_ft, ('Train and test sets are of different types, indicating that the experiment is In-Out '
                                'distribution test, where the model is train on one dataset and tested on another. In '
                                'this case, the model should not be fine-tuned.')
            target_ids = test_target_ids

        training_args = TrainingArguments(
            output_dir=work_dir + f'/{model_type}/',
            logging_dir=work_dir + f'/{model_type}/',
            report_to='wandb' if not args.no_wandb else 'none',
            remove_unused_columns=False,
            seed = args.seed,
            run_name=timestamp if args.run_name is None else args.run_name,
            **cfg.train_cfg)

        trainer = FTTrainer(
            cfg=cfg,
            target_ids=target_ids,
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=val_set,
            compute_metrics=ClassificationMetric(num_classes=target_ids.shape[-1], **cfg.get('metric_cfg', {})),
            data_collator=train_set.get_collate_fn()
        )

        if not args.skip_ft:
            trainer.train()
            logger.info(f'Fine-tuning for {model_type} model finished.')

        # Logging test and validation metrics
        val_metrics = trainer.evaluate(eval_dataset=val_set, metric_key_prefix='val')
        for key, value in val_metrics.items():
            logger.info(f'Validation metrics for {model_type} model: {key}: {value}')
        
        test_metrics = trainer.evaluate(eval_dataset=test_set, metric_key_prefix='test')
        for key, value in test_metrics.items():
            logger.info(f'Test metrics for {model_type} model: {key}: {value}')

        ## Also, thınk the model complexity part & add ıt to the log fıle.

        # Writing test and validation predictions and logits to npz files.
        if cfg.process_preds['npz_file'] is not None or cfg.process_preds['do_plot']:
            ## Getting row ids for test and validation sets
            test_idx = test_set.get_data_indices()
            val_idx = val_set.get_data_indices()
            ## Getting predictions for test and validation sets
            predictions_test = trainer.predict(test_set)
            predictions_val = trainer.predict(val_set)
            logger.info('Start processing predictions.')
        
            if cfg.process_preds['do_plot']:
                plot_predictions(predictions_test, cfg.process_preds['plot_cfg'], work_dir)
            if cfg.process_preds['npz_file'] is not None:
                save_predictions(predictions_val, osp.join(work_dir, f'{model_type}/val_preds', cfg.process_preds['npz_file']), logger=logger, seed=args.seed, data_idx=val_idx, input_text=val_set.get_input_text())
                save_predictions(predictions_test, osp.join(work_dir, f'{model_type}/test_preds', cfg.process_preds['npz_file']), logger=logger, seed=args.seed, data_idx=test_idx, input_text=test_set.get_input_text())

    # Implementing temperature scaling
    optimize_temperature_scaling(work_dir, cfg.process_preds['npz_file'], seed=args.seed)

    # Implementing duo optimization
    optimize_weights(work_dir, cfg.process_preds['npz_file'], seed=args.seed)

if __name__ == '__main__':
    main()
