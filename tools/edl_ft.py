import argparse
import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
import mmengine
import torch
import wandb
from mmengine.runner.utils import set_random_seed
from transformers import TrainingArguments

from ib_edl import DATASETS, get_model_and_tokenizer, ClassificationMetric, plot_predictions, save_predictions, setup_logger, optimize_weights
from ib_edl.train_eval import EvidentialTrainer, UpdateRegWeightCallback


def parse_args():
    parser = ArgumentParser('Fine-tune model with IB-EDL (EvidentialTrainer).')
    parser.add_argument('--config_base', help='Path to IB-EDL config file for base model.')
    parser.add_argument('--config_sidekick', help='Path to IB-EDL config file for sidekick model.')
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
    assert cfg.get('edl_loss_cfg', None) is not None, (
        'edl_loss_cfg is required in config for edl_ft.py. '
        'Use evidential_ft.py for plain LoRA fine-tuning.'
    )


def main():
    args = parse_args()
    set_random_seed(args.seed)
    work_dir = args.work_dir
    mmengine.mkdir_or_exist(work_dir)
    device = torch.device(f'cuda:{args.gpu_id}')

    timestamp = datetime.now().strftime('%m%d_%H%M_%S')

    print(f'IB-EDL training with seed {args.seed} has begun. \n')

    # Getting base and sidekick names
    base_model_name_list = args.config_base.split('/')[1].split('_')[-2:]
    base_model_name = '_'.join(base_model_name_list)

    sidekick_model_name_list = args.config_sidekick.split('/')[1].split('_')[-2:]
    sidekick_model_name = '_'.join(sidekick_model_name_list)

    model_combination_path = f"{base_model_name}_{sidekick_model_name}"
    work_dir = osp.join(work_dir, model_combination_path)

    # Running IB-EDL training for base and sidekick models separately
    for model_type in ['base', 'sidekick']:
        assert args.config_base and args.config_sidekick, 'Base or sidekick config file is missing.'

        if model_type == 'base':
            cfg = mmengine.Config.fromfile(args.config_base)
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

        # Load model & tokenizer
        model, tokenizer = get_model_and_tokenizer(**cfg.model, device=device)

        # Load datasets
        train_set = DATASETS.build(cfg.data['train'], default_args=dict(tokenizer=tokenizer))
        val_set = DATASETS.build(cfg.data['val'], default_args=dict(tokenizer=tokenizer))
        test_set = DATASETS.build(cfg.data['test'], default_args=dict(tokenizer=tokenizer))

        train_target_ids = train_set.target_ids
        val_target_ids = val_set.target_ids
        test_target_ids = test_set.target_ids

        assert torch.all(train_target_ids == val_target_ids), (
            f'target_ids of train and val sets are different for {model_type} tokenizer.'
        )

        if type(train_set) is type(test_set):
            target_ids = train_target_ids
        else:
            assert args.skip_ft, (
                'Train and test sets are of different types, indicating an OOD experiment. '
                'In this case, the model should not be fine-tuned.'
            )
            target_ids = test_target_ids

        training_args = TrainingArguments(
            output_dir=work_dir + f'/{model_type}/',
            logging_dir=work_dir + f'/{model_type}/',
            report_to='wandb' if not args.no_wandb else 'none',
            remove_unused_columns=False,
            seed=args.seed,
            run_name=timestamp if args.run_name is None else args.run_name,
            **cfg.train_cfg)

        # Build the reg-weight annealing callback from config
        reg_weight_cfg = cfg.edl_loss_cfg['reg_weight_cfg']
        reg_callback = UpdateRegWeightCallback(
            start_epoch=reg_weight_cfg['start_epoch'],
            final_reg_weight=reg_weight_cfg['final_reg_weight'],
        )

        trainer = EvidentialTrainer(
            cfg=cfg,
            target_ids=target_ids,
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=val_set,
            compute_metrics=ClassificationMetric(num_classes=target_ids.shape[-1], **cfg.get('metric_cfg', {})),
            data_collator=train_set.get_collate_fn(),
            callbacks=[reg_callback],
        )

        if not args.skip_ft:
            trainer.train()
            logger.info(f'IB-EDL fine-tuning for {model_type} model finished.')

        # Evaluate on val and test sets
        val_metrics = trainer.evaluate(eval_dataset=val_set, metric_key_prefix='val')
        for key, value in val_metrics.items():
            logger.info(f'Validation metrics for {model_type} model: {key}: {value}')

        test_metrics = trainer.evaluate(eval_dataset=test_set, metric_key_prefix='test')
        for key, value in test_metrics.items():
            logger.info(f'Test metrics for {model_type} model: {key}: {value}')

        # Save predictions (logits + labels + indices + input text) to npz
        if cfg.process_preds['npz_file'] is not None or cfg.process_preds['do_plot']:
            test_idx = test_set.get_data_indices()
            val_idx = val_set.get_data_indices()
            predictions_test = trainer.predict(test_set)
            predictions_val = trainer.predict(val_set)
            logger.info('Start processing predictions.')

            if cfg.process_preds['do_plot']:
                plot_predictions(predictions_test, cfg.process_preds['plot_cfg'], work_dir)
            if cfg.process_preds['npz_file'] is not None:
                save_predictions(
                    predictions_val,
                    osp.join(work_dir, f'{model_type}/val_preds', cfg.process_preds['npz_file']),
                    logger=logger, seed=args.seed, data_idx=val_idx, input_text=val_set.get_input_text()
                )
                save_predictions(
                    predictions_test,
                    osp.join(work_dir, f'{model_type}/test_preds', cfg.process_preds['npz_file']),
                    logger=logger, seed=args.seed, data_idx=test_idx, input_text=test_set.get_input_text()
                )

    # Apply duo optimizer on base+sidekick IB-EDL logits
    optimize_weights(work_dir, cfg.process_preds['npz_file'], seed=args.seed)


if __name__ == '__main__':
    main()
