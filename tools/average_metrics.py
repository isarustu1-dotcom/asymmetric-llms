import pandas as pd
import numpy as np
from scipy.stats import t
from itertools import combinations
from sklearn.metrics import accuracy_score, f1_score
from ib_edl.utils import compute_uncertainty_metrics, _softmax_from_log
from argparse import ArgumentParser
import os.path as osp
import os

def parse_args():
    parser = ArgumentParser('Get average metrics across different seeds.')
    parser.add_argument('--config_base', help='Path to config file for base model.')
    parser.add_argument('--config_sidekick', help='Path to config file for sidekick model.')
    parser.add_argument('--work-dir', '-w', help='Working directory.')


    return parser.parse_args()

def _mean_ci(values: np.ndarray, alpha: float = 0.05):
    """Compute a mean and two-sided confidence interval.

    Uses a t-distribution interval over the provided values.

    Args:
        values (np.ndarray): Metric values to summarize.
        alpha (float): Significance level for the two-sided interval.

    Returns:
        tuple[float, float, float, int]: Mean, CI low, CI high, and sample size.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    mean = float(values.mean()) if n else float("nan")
    if n < 2:
        return mean, float("nan"), float("nan"), n
    sem = values.std(ddof=1) / np.sqrt(n)
    tcrit = t.ppf(1 - alpha / 2, df=n - 1)
    margin = tcrit * sem
    return mean, mean - margin, mean + margin, n


def _summarize_with_ci(data: pd.DataFrame, model_type: str, split: str):
    """Summarize metrics for a given model and split with confidence intervals.

    Args:
        data (pd.DataFrame): Metrics summary data with model/seed/split columns.
        model_type (str): Model identifier (e.g., base, sidekick, duo_constr, duo_unconstr).
        split (str): Split name (val or test).

    Returns:
        pd.DataFrame: Long-form summary with mean and CI per metric.
    """
    data = data[(data["model"] == model_type) & (data["split"] == split)]
    metric_columns = [col for col in data.columns if col not in ["model", "seed", "split", "data"]]
    rows = []
    for col in metric_columns:
        mean, low, high, n = _mean_ci(data[col].values)
        rows.append({"model": model_type, "split": split, "metric": col, "mean": mean, "ci_low": low, "ci_high": high, "n": n})
    return pd.DataFrame(rows)


def _format_ci(mean: float, low: float, high: float) -> str:
    """Format a mean and CI bounds as a display string."""
    if np.isnan(mean):
        return "nan"
    if np.isnan(low) or np.isnan(high):
        return f"{mean:.4f}"
    margin = mean - low
    return f"{mean:.4f} ± {margin:.4f}"

def _pick_seed_pairs(seeds, n, rng):
    """Sample n distinct seed pairs without replacement.

    Args:
        seeds (list[int]): Seed values to pair.
        n (int): Number of pairs to sample.
        rng (np.random.Generator): Random generator.

    Returns:
        list[tuple[int, int]]: Sampled seed pairs.
    """
    all_pairs = list(combinations(seeds, 2))
    return list(rng.choice(all_pairs, size=n, replace=False))
    #return list(rng.choice(all_pairs, size=n))


def _deep_ensemble_metrics(data: pd.DataFrame, seed_pairs, suffix: str) -> pd.DataFrame:
    """Compute deep-ensemble metrics by averaging probabilities across seed pairs.

    Args:
        data (pd.DataFrame): Prediction data with per-class probabilities.
        seed_pairs (list[tuple[int, int]]): Seed pairs to ensemble.
        suffix (str): Model suffix for probability columns.

    Returns:
        pd.DataFrame: Per-pair metrics keyed by the seed pair string.
    """
    metrics_by_pair = {}
    for s1, s2 in seed_pairs:
        seed_data_1 = data[data["seed"] == s1].sort_values(by="idx").reset_index(drop=True)
        seed_data_2 = data[data["seed"] == s2].sort_values(by="idx").reset_index(drop=True)

        prob_cols = [col for col in seed_data_1.columns if col.startswith("prob_label")]
        deep_ensemble_probs = (seed_data_1[prob_cols].values + seed_data_2[prob_cols].values) / 2
        predicted_labels_idx = np.argmax(deep_ensemble_probs, axis=1)

        predicted_labels = np.array([i for i in predicted_labels_idx])


        acc = accuracy_score(np.array(seed_data_1["true_label"]), predicted_labels)
        f1_score_macro = f1_score(seed_data_1["true_label"], predicted_labels, average="macro")
        metrics = {"accuracy": acc, "f1_macro": f1_score_macro}

        metrics.update(compute_uncertainty_metrics(deep_ensemble_probs, seed_data_1["true_label"]))
        metrics_by_pair[f"{s1}_{s2}"] = metrics

    return pd.DataFrame.from_dict(metrics_by_pair, orient="index").reset_index().rename(columns={"index": "seed"})

def _create_metrics_summary_df(data: dict, split_list: list, model_type: list, output_dir: str) -> 'pd.DataFrame':
    """Create a summary DataFrame from the given data dictionary.
    Args:
        data (dict): A dictionary containing the metric data.
        split_list (list): A list of splits to consider (e.g., ['val', 'test']).
        model_type (list): A list of model types to consider (e.g., ['base', 'sidekick']).
    Returns:
        pd.DataFrame: A summary DataFrame with the metrics organized.
    """
    # Get the number of unique seeds, splits, and model types to create summary df
    number_of_seed = len(data.keys())
    number_of_split = len(split_list)
    number_of_model_type = len(model_type)

    # Create an empty DataFrame to hold the summary
    seeds_df = list(data.keys()) * number_of_model_type * number_of_split
    split_df = split_list * number_of_model_type * number_of_seed
    model_type_df = model_type * number_of_seed * number_of_split
    metrics = []
    # Extract metric names from the data. Sufficient to check the first seed and duo keys because structure is consistent.
    for key in data[list(data.keys())[0]].item()['val']:
        if key.startswith('duo'):
            metrics.append(key.replace('duo_', ''))


    # Creating the skeleton of the summary DataFrame
    summary_df = pd.DataFrame({}, columns= ['seed', 'data', 'model', 'split'] + metrics)
    summary_df['seed'] = seeds_df
    summary_df['model'] = model_type_df
    summary_df['split'] = split_df

    # Populate the summary DataFrame with metric values
    for seed in data.keys():
        for m_type in model_type:
            for s in split_list:
                for metric in metrics:
                    col_name = m_type + '_' + metric
                    value = data[seed].item()[s][col_name]
                    data_name = data[seed].item()['data'].split('.')[0]
                    summary_df.loc[(summary_df['seed'] == seed) \
                                   & (summary_df['model'] == m_type) & (summary_df['split'] == s), metric] = value
                    summary_df.loc[(summary_df['seed'] == seed) \
                                   & (summary_df['model'] == m_type) & (summary_df['split'] == s), 'data'] = data_name
                    
    # Write the summary pd
    output_dir = output_dir.replace('.npz', '.csv')
    summary_df.to_csv(output_dir)
    return summary_df

def _create_summary_predictions(data, output_dir, model_type = 'base'):
    """Create a summary DataFrame from the given data dictionary.
    Args:
        data (dict): A dictionary containing the metric data.
        model_type (str): A list of model types to consider (e.g., base).
    Returns:
        pd.DataFrame: A summary DataFrame with the predictions organized.
    """
    seeds = list(data.keys())
    number_of_labels = data[seeds[0]].item()['logits'].shape[1]
    number_of_observations = data[seeds[0]].item()['logits'].shape[0]
    summary_df = pd.DataFrame(columns=['seed', 'model_type', 'idx', 'input', 'true_label', 'predicted_label'] \
                                            + [f'prob_label_{i}' for i in range(number_of_labels)])
    

    for seed in seeds:
        summary_sub_df = pd.DataFrame(columns=['seed', 'model_type', 'idx', 'input', 'true_label', 'predicted_label'] \
                                            + [f'prob_label_{i}' for i in range(number_of_labels)])
        seed_df = [seed] * number_of_observations
        model_type_df = [model_type] * number_of_observations
        seed_data = data[seed].item()
        idx_df = seed_data['idx']
        input_df = seed_data['input']
        true_label_df = seed_data['true_labels']
        label_probs = _softmax_from_log(seed_data['logits'])

        summary_sub_df['seed'] = seed_df
        summary_sub_df['model_type'] = model_type_df
        summary_sub_df['idx'] = idx_df
        summary_sub_df['input'] = input_df
        summary_sub_df['true_label'] = true_label_df
        summary_sub_df['predicted_label'] = label_probs.argmax(axis=1)

        for label_number in range(number_of_labels):
            summary_sub_df[f'prob_label_{label_number}'] = label_probs[:, label_number]

        summary_df = pd.concat([summary_df, summary_sub_df], axis=0, ignore_index=True)

    # Converting idx and true_label colums to integer
    summary_df['idx'] = summary_df['idx'].astype(int)
    summary_df['true_label'] = summary_df['true_label'].astype(int)

    # Write the summary pd
    output_dir = output_dir.replace('.npz', '.csv')
    summary_df.to_csv(output_dir)

    return summary_df


## Update the duo_optimizer in the cloud.
def main():
    args = parse_args()
    work_dir = args.work_dir

    saved_file_name = f'{args.config_base.split('/')[1].split('_')[0] + ".npz"}'
    base_model_name_list = args.config_base.split('/')[1].split('_')[1:]
    base_model_name = '_'.join(base_model_name_list)

    sidekick_model_name_list = args.config_sidekick.split('/')[1].split('_')[1:]
    sidekick_model_name = '_'.join(sidekick_model_name_list)

    model_combination_path = f'{base_model_name}_{sidekick_model_name}/'
    work_dir = osp.join(work_dir, model_combination_path)

    # Reading data
    metrics_path = osp.join(work_dir, f'duo/metrics', saved_file_name)
    metrics_data = dict(np.load(metrics_path, allow_pickle = True))

    base_test_path = osp.join(work_dir, f'base/test_preds', saved_file_name)
    data_test_base = dict(np.load(base_test_path, allow_pickle = True))

    sidekick_test_path = osp.join(work_dir, f'sidekick/test_preds', saved_file_name)
    data_test_sidekick = dict(np.load(sidekick_test_path, allow_pickle = True))

    average_metrics_summary_path = osp.join(work_dir, f'overall_metrics_summary/')
    os.makedirs(average_metrics_summary_path, exist_ok=True) # Create the path for the first time

    # Creating summary pd data
    summary_df = _create_metrics_summary_df(metrics_data, ['val', 'test'], ['base', 'sidekick', 'duo'], output_dir=metrics_path)
    summaries = []
    for model_type in ["base", "sidekick", "duo"]:
        for split in ["val", "test"]:
            summaries.append(_summarize_with_ci(summary_df, model_type, split))

    base_test_data = _create_summary_predictions(data_test_base, model_type='base', output_dir=base_test_path)
    sidekick_test_data = _create_summary_predictions(data_test_sidekick, model_type='sidekick', output_dir = sidekick_test_path)

    
    seeds = base_test_data['seed'].unique()
    rng = np.random.default_rng(seed=42)
    seed_pairs = _pick_seed_pairs(seeds, len(seeds), rng)
    print(f"Deep ensemble seed pairs: {seed_pairs}")

    base_de_test = _deep_ensemble_metrics(base_test_data, seed_pairs, suffix="base")
    side_de_test = _deep_ensemble_metrics(sidekick_test_data, seed_pairs, suffix="sidekick")


    de_frames = [
        base_de_test.assign(model="base_de(2)", split="test"),
        side_de_test.assign(model="sidekick_de(2)", split="test"),
    ]

    de_data = pd.concat(de_frames, ignore_index=True)


    for model_type in ["base_de(2)", "sidekick_de(2)"]:
        summaries.append(_summarize_with_ci(de_data, model_type, "test"))

    summary_df = pd.concat(summaries, ignore_index=True)

    summary_df.to_csv(osp.join(average_metrics_summary_path, "metrics_summary_ci_raw.csv"), index=False)

    metric_order = ["accuracy", "f1_macro", "nll", "brier", "ece", "lppd", "mean_uncertainty"]
    model_order = ["base", "sidekick", "duo", "base_de(2)", "sidekick_de(2)"]
    split_order = ["val", "test"]

    pretty = summary_df.copy()
    pretty["value"] = pretty.apply(lambda r: _format_ci(r["mean"], r["ci_low"], r["ci_high"]), axis=1)
    pretty["metric"] = pd.Categorical(pretty["metric"], categories=metric_order, ordered=True)
    pretty["model"] = pd.Categorical(pretty["model"], categories=model_order, ordered=True)
    pretty["split"] = pd.Categorical(pretty["split"], categories=split_order, ordered=True)
    pretty = pretty.sort_values(["model", "split", "metric"])

    pretty_wide = pretty.pivot_table(
        index=["model", "split"],
        columns="metric",
        values="value",
        aggfunc="first",
    )
    pretty_wide = pretty_wide.reindex(columns=metric_order)
    pretty_wide = pretty_wide.reset_index()

    pretty_wide.to_csv(osp.join(average_metrics_summary_path, "metrics_summary_ci.csv"), index=False)

    print("\n=== Metrics Summary (mean ± 95% CI) ===")

    pretty_test = pretty_wide[pretty_wide["split"] == "test"]
    columns = list(pretty_test.columns)
    rows = [list(map(str, row)) for row in pretty_test.itertuples(index=False, name=None)]
    widths = [len(col) for col in columns]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
    separator = "-+-".join("-" * widths[i] for i in range(len(columns)))
    print(header)
    print(separator)
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(columns))))


if __name__ == '__main__':
    main()

