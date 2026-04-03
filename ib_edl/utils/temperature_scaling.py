import os
import os.path as osp
from typing import Dict
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score
from .uncertainty_metrics import compute_uncertainty_metrics
from .optimizer_helpers import __sanity_check_on_probs, _logsumexp, _softmax_from_log, _sort_by_idx, _read_or_create_path, _save_val_test_results, _merge_dictionaries

def _weighted_nll(temperature: float, logits: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood for a calibrated logit-space ensemble.

    Args:
        temperature (float): Non-negative effective scales for the single model
        logits(np.ndarray): Model logits.
        y (np.ndarray): True labels.

    Returns:
        float: Mean negative log-likelihood.
    """
    log_p = 1/temperature * logits
    log_p_norm = log_p - _logsumexp(log_p, axis=1, keepdims=True)
    return -np.mean(log_p_norm[np.arange(len(y)), y])


def _build_metrics(scale: float, logits: np.ndarray, y_true: np.ndarray, ax:int = 1, model_type = 'base'):
    """Compute base/sidekick/duo metrics for a split.

    Args:
        split_name (str): Split label used for logging.
        scale  (float): Effective scale for base model logits.
        logits (np.ndarray): Base model logits.
        y_true (np.ndarray): True label indices.
        model_type (str): Model type.

    Returns:
        tuple: (metrics dict, duo_logits, duo_probs, duo_pred).
    """
    new_logits = scale * logits
    probs = _softmax_from_log(new_logits)
    __sanity_check_on_probs(probs, model_type)
    preds = probs.argmax(axis=ax)

    metrics = {
        f"{model_type}_temperature_accuracy": accuracy_score(y_true, preds),
        f"{model_type}_temperature_f1_macro": f1_score(y_true, preds, average="macro")
    }
    
    metrics.update({f"{model_type}_temperature_{k}": v for k, v in compute_uncertainty_metrics(probs, y_true).items()})

    return metrics, new_logits, probs, preds


def _fit_temperature_weighted_scales(
    logits: np.ndarray,
    y_true: np.ndarray,
    verbose: bool = True,
    optimizer_method: str = 'L-BFGS-B',
    model_type: str = 'base'
) -> Dict[str, float]:
    """Fit a single temperature scalar that minimises NLL on the provided logits.

    Optimises T > 0 such that softmax((1/T) * logits) best matches y_true under
    NLL loss. Returns both the fitted temperature and its reciprocal
    (the effective logit scale).

    Args:
        logits (np.ndarray): Raw model logits of shape (N, C).
        y_true (np.ndarray): True class indices of shape (N,).
        verbose (bool): If True, prints per-iteration scale values and scipy disp.
        optimizer_method (str): Scipy minimise method; default 'L-BFGS-B'.
        model_type (str): Label used in verbose output to identify the model.

    Returns:
        Dict[str, float]: Dictionary with keys:
            - 'scale'         : fitted 1/T (effective logit multiplier)
            - 'temperature'   : fitted T
            - 'optimizer_fun' : final NLL value at the optimum

    Raises:
        RuntimeError: If the optimiser does not converge.
    """
    x0 = np.array([1.0], dtype=np.float64)
    bounds = [(1e-6, None)]

    callback = None
    if verbose:
        def _print_scales(xk):
            print(f"iter scales -> {model_type}: {xk[0]:.4f}")
        callback = _print_scales

    res = minimize(
        _weighted_nll,
        x0=x0,
        args=(logits, y_true),
        method=optimizer_method,
        bounds=bounds,
        callback=callback,
        options={'disp': verbose},
    )

    if not res.success:
        raise RuntimeError(f'Scale optimization failed: {res.message}')

    temperature = res.x.item()
    scale = 1/temperature

    return {
        'scale': float(scale),
        'temperature': float(temperature),
        'optimizer_fun': float(res.fun),
    }

def _get_optimization_params(logits, true_labels, verbose, optimizer_method):
    params = _fit_temperature_weighted_scales(
        logits,
        true_labels,
        verbose=verbose,
        optimizer_method=optimizer_method
    )
    
    scale = params['scale']
    temperature = params['temperature']
    
    return scale, temperature

def optimize_temperature_scaling(result_dir: str, saved_file_name: str, seed:int = 42, verbose: bool = True, optimizer_method: str = "L-BFGS-B"):
    
    # Directory for validation and test results
    base_model_validation_results_path = osp.join(result_dir, f'base/val_preds', saved_file_name)
    sidekick_model_validation_results_path = osp.join(result_dir, f'sidekick/val_preds', saved_file_name)
    base_model_test_results_path = osp.join(result_dir, f'base/test_preds', saved_file_name)
    sidekick_model_test_results_path = osp.join(result_dir, f'sidekick/test_preds', saved_file_name)

    # Load validation and test results
    base_val_results = dict(np.load(base_model_validation_results_path, allow_pickle = True))
    sidekick_val_results = dict(np.load(sidekick_model_validation_results_path, allow_pickle = True))
    base_test_results = dict(np.load(base_model_test_results_path, allow_pickle = True))
    sidekick_test_results = dict(np.load(sidekick_model_test_results_path, allow_pickle = True))

    # Filter data wrt seed value
    base_val_results = base_val_results[str(seed)].item()
    sidekick_val_results = sidekick_val_results[str(seed)].item()
    base_test_results = base_test_results[str(seed)].item()
    sidekick_test_results = sidekick_test_results[str(seed)].item()


    # Ensure the indices match and sort accordingly
    base_idx_order_val = np.argsort(base_val_results['idx'])
    sidekick_idx_order_val = np.argsort(sidekick_val_results['idx'])
    base_idx_order_test = np.argsort(base_test_results['idx'])
    sidekick_idx_order_test = np.argsort(sidekick_test_results['idx'])

    # Sort logits based on indices
    base_logits_val_sorted = _sort_by_idx(base_val_results['logits'], base_idx_order_val)
    sidekick_logits_val_sorted = _sort_by_idx(sidekick_val_results['logits'], sidekick_idx_order_val)
    base_logits_test_sorted = _sort_by_idx(base_test_results['logits'], base_idx_order_test)
    sidekick_logits_test_sorted = _sort_by_idx(sidekick_test_results['logits'], sidekick_idx_order_test)

    true_labels_val = _sort_by_idx(base_val_results['true_labels'], base_idx_order_val)
    true_labels_test = _sort_by_idx(base_test_results['true_labels'], base_idx_order_test)

    # Get optimization parameters
    scale_base, temperature_base = _get_optimization_params(
        base_logits_val_sorted,
        true_labels_val,
        verbose=verbose,
        optimizer_method=optimizer_method
    )

    scale_sidekick, temperature_sidekick = _get_optimization_params(
        sidekick_logits_val_sorted,
        true_labels_val,
        verbose=verbose,
        optimizer_method=optimizer_method
    )

    base_val_metrics, base_val_logits, base_val_probs, base_val_preds = _build_metrics(scale_base, base_logits_val_sorted, true_labels_val, model_type='base')
    base_test_metrics, base_test_logits, base_test_probs, base_test_preds = _build_metrics(scale_base, base_logits_test_sorted, true_labels_test, model_type='base')

    sidekick_val_metrics, sidekick_val_logits, sidekick_val_probs, sidekick_val_preds = _build_metrics(scale_sidekick, sidekick_logits_val_sorted, true_labels_val, model_type='sidekick')
    sidekick_test_metrics, sidekick_test_logits, sidekick_test_probs, sidekick_test_preds = _build_metrics(scale_sidekick, sidekick_logits_test_sorted, true_labels_test, model_type='sidekick')


    metrics_path = osp.join(result_dir, 'metrics', saved_file_name)
    validation_results_path_base = osp.join(result_dir, f'temperature_scaling/base/val_preds', saved_file_name)
    validation_results_path_sidekick = osp.join(result_dir, f'temperature_scaling/sidekick/val_preds', saved_file_name)
    test_results_path_base = osp.join(result_dir, f'temperature_scaling/base/test_preds', saved_file_name)
    test_results_path_sidekick = osp.join(result_dir, f'temperature_scaling/sidekick/test_preds', saved_file_name)

    # Read data if it is already there or create the directory
    existing_dict_metrics = _read_or_create_path(metrics_path)
    existing_dict_val_base = _read_or_create_path(validation_results_path_base)
    existing_dict_val_sidekick = _read_or_create_path(validation_results_path_sidekick)
    existing_dict_test_base = _read_or_create_path(test_results_path_base)
    existing_dict_test_sidekick = _read_or_create_path(test_results_path_sidekick)


    # Save base and sidekick model validation results
    _save_val_test_results(
        seed,
        saved_file_name,
        idx=_sort_by_idx(base_val_results['idx'], base_idx_order_val),
        input=_sort_by_idx(base_val_results['input'], base_idx_order_val),
        logits=base_val_logits,
        softmax_probs=base_val_probs,
        preds=base_val_preds,
        true_labels=_sort_by_idx(base_val_results['true_labels'], base_idx_order_val),
        existing_dict=existing_dict_val_base,
        path=validation_results_path_base,
    )

    _save_val_test_results(
        seed,
        saved_file_name,
        idx=_sort_by_idx(sidekick_val_results['idx'], sidekick_idx_order_val),
        input=_sort_by_idx(sidekick_val_results['input'], sidekick_idx_order_val),
        logits=sidekick_val_logits,
        softmax_probs=sidekick_val_probs,
        preds=sidekick_val_preds,
        true_labels=_sort_by_idx(sidekick_val_results['true_labels'], sidekick_idx_order_val),
        existing_dict=existing_dict_val_sidekick,
        path=validation_results_path_sidekick,
    )

    # Save base and sidekick model test results
    _save_val_test_results(
        seed,
        saved_file_name,
        idx=_sort_by_idx(base_test_results['idx'], base_idx_order_test),
        input=_sort_by_idx(base_test_results['input'], base_idx_order_test),
        logits=base_test_logits,
        softmax_probs=base_test_probs,
        preds=base_test_preds,
        true_labels=_sort_by_idx(base_test_results['true_labels'], base_idx_order_test),
        existing_dict=existing_dict_test_base,
        path=test_results_path_base,
    )

    _save_val_test_results(
        seed,
        saved_file_name,
        idx=_sort_by_idx(sidekick_test_results['idx'], sidekick_idx_order_test),
        input=_sort_by_idx(sidekick_test_results['input'], sidekick_idx_order_test),
        logits=sidekick_test_logits,
        softmax_probs=sidekick_test_probs,
        preds=sidekick_test_preds,
        true_labels=_sort_by_idx(sidekick_test_results['true_labels'], sidekick_idx_order_test),
        existing_dict=existing_dict_test_sidekick,
        path=test_results_path_sidekick,
    )

    # Loading metrics summary results 
    metrics = {str(seed):{
        'data' : saved_file_name,
        'base_temperature': {f'base': temperature_base},
        'base_scale': {f'base': scale_base},
        'sidekick_temperature': {f'sidekick': temperature_sidekick},
        'sidekick_scale': {f'sidekick': scale_sidekick},
        'val': base_val_metrics | sidekick_val_metrics,
        'test': base_test_metrics | sidekick_test_metrics,
        }
    }

    existing_dict_metrics = _merge_dictionaries(existing_dict_metrics, metrics)
    np.savez_compressed(metrics_path, **existing_dict_metrics)



