import json
import os
import os.path as osp
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score
from .uncertainty_metrics import compute_uncertainty_metrics


def _logsumexp(arr: np.ndarray, axis: int = 1, keepdims: bool = True) -> np.ndarray:
    """Compute numerically stable log-sum-exp.

    Args:
        arr (np.ndarray): Input array.
        axis (int): Axis over which to reduce.
        keepdims (bool): Whether to keep reduced dimensions.

    Returns:
        np.ndarray: Log-sum-exp result.
    """
    m = np.max(arr, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(arr - m), axis=axis, keepdims=True) + 1e-12)

def _weighted_nll(weights: np.ndarray, p_base: np.ndarray, p_sidekick: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood for a weighted log-probability ensemble.

    Args:
        weights (np.ndarray): Base and sidekick weights.
        p_base (np.ndarray): Base probabilities.
        p_sidekick (np.ndarray): Sidekick probabilities.
        y (np.ndarray): True labels.

    Returns:
        float: Mean negative log-likelihood.
    """
    weight_base, weight_sidekick = weights
    log_p = weight_base * np.log(p_base + 1e-12) + weight_sidekick * np.log(p_sidekick + 1e-12)
    log_p_norm = log_p - _logsumexp(log_p, axis=1, keepdims=True)
    return -np.mean(log_p_norm[np.arange(len(y)), y])

def _softmax_from_log(log_p: np.ndarray, ax: int = 1) -> np.ndarray:
    """Convert log-probabilities to normalized probabilities.

    Args:
        log_p (np.ndarray): Log-probability matrix.

    Returns:
        np.ndarray: Normalized probability matrix.
    """
    log_p_norm = log_p - _logsumexp(log_p, axis=ax, keepdims=True)
    return np.exp(log_p_norm)

def _combine_probs(w: float, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """Combine two probability matrices using log-space weighting.

    Args:
        w (float): Weight for the first probability matrix.
        p_a (np.ndarray): First probability matrix.
        p_b (np.ndarray): Second probability matrix.

    Returns:
        np.ndarray: Combined probability matrix.
    """
    log_p = w * np.log(p_a + 1e-12) + (1.0 - w) * np.log(p_b + 1e-12)

    return _softmax_from_log(log_p)

def _build_metrics(split_name: str, w: float, probs_a: np.ndarray, probs_b: np.ndarray, y_true: np.ndarray, ax:int = 1):
    """Compute base/sidekick/duo metrics for a split.

    Args:
        split_name (str): Split label used for logging.
        w (float): Weight for base probabilities.
        probs_a (np.ndarray): Base probabilities.
        probs_b (np.ndarray): Sidekick probabilities.
        y_true (np.ndarray): True label indices.

    Returns:
        tuple: (metrics dict, duo_probs, duo_pred).
    """
    base_pred = probs_a.argmax(axis=ax)
    sidekick_pred = probs_b.argmax(axis=ax)

    duo_probs = _combine_probs(w, probs_a, probs_b)
    duo_pred = duo_probs.argmax(axis=ax)
    y_true = y_true

    
    metrics = {
        "base_accuracy": accuracy_score(y_true, base_pred),
        "sidekick_accuracy": accuracy_score(y_true, sidekick_pred),
        "duo_accuracy": accuracy_score(y_true, duo_pred),
        "base_f1_macro": f1_score(y_true, base_pred, average="macro"),
        "sidekick_f1_macro": f1_score(y_true, sidekick_pred, average="macro"),
        "duo_f1_macro": f1_score(y_true, duo_pred, average="macro"),
    }
    

    metrics.update({f"base_{k}": v for k, v in compute_uncertainty_metrics(probs_a, y_true).items()})
    metrics.update({f"sidekick_{k}": v for k, v in compute_uncertainty_metrics(probs_b, y_true).items()})
    metrics.update({f"duo_{k}": v for k, v in compute_uncertainty_metrics(duo_probs, y_true).items()})
    return metrics, duo_probs, duo_pred

def _sort_by_idx(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Reorder `arr` using indices `idx`.

    Supports:
      - arr shape: (N,)
      - arr shape: (B, N) or (B, N, ...)
      - idx shape: (N,) or (B, N)

    Returns:
      - reordered array with same shape as `arr`
    """
    arr = np.asarray(arr)
    idx = np.asarray(idx)

    if arr.ndim == 1:
        if idx.shape[0] != arr.shape[0]:
            raise ValueError(f"`idx` shape {idx.shape} incompatible with arr size {arr.shape[0]}")
        return arr[idx]
    elif arr.ndim >= 2:
        B, N = arr.shape[0], arr.shape[1]
        if idx.shape[0] != B:
            raise ValueError(f"`idx` shape {idx.shape} incompatible with arr axis=1 size {N}")
        return arr[idx]

    else:
        raise ValueError(f"`arr` must be at least 1D (got shape {arr.shape})")




def optimize_weights(result_dir: str, saved_file_name: str, seed:int = 42, constrain_weights: bool = True, verbose: bool = True, optimizer_method: str = "SLSQP"):

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


    # Check if indices match
    if not np.array_equal(_sort_by_idx(base_val_results['idx'], base_idx_order_val), _sort_by_idx(sidekick_val_results['idx'], sidekick_idx_order_val)):
        raise ValueError('The indices of the base and sidekick validation results do not match. Optimizer will not be reliable. Check the data')

    if not np.array_equal(_sort_by_idx(base_test_results['idx'], base_idx_order_test), _sort_by_idx(sidekick_test_results['idx'], sidekick_idx_order_test)):
        raise ValueError('The indices of the base and sidekick test results do not match. Optimizer will not be reliable. Check the data')
        
    # Sort logits based on indices
    base_logits_val_sorted = _sort_by_idx(base_val_results['logits'], base_idx_order_val)
    sidekick_logits_val_sorted = _sort_by_idx(sidekick_val_results['logits'], sidekick_idx_order_val)
    base_logits_test_sorted = _sort_by_idx(base_test_results['logits'], base_idx_order_test)
    sidekick_logits_test_sorted = _sort_by_idx(sidekick_test_results['logits'], sidekick_idx_order_test)

    # Get validation and test probs and true labels
    base_logits_val = _softmax_from_log(base_logits_val_sorted)
    sidekick_logits_val = _softmax_from_log(sidekick_logits_val_sorted)
    base_logits_test = _softmax_from_log(base_logits_test_sorted)
    sidekick_logits_test = _softmax_from_log(sidekick_logits_test_sorted)

    true_labels_val = _sort_by_idx(base_val_results['true_labels'], base_idx_order_val)
    true_labels_test = _sort_by_idx(base_test_results['true_labels'], base_idx_order_test)

    # Set optimization parameters and run the optimizer
    cons = {"type": "eq", "fun": lambda w: w[0] + w[1] - 1} if constrain_weights else ()
    bounds = [(0.0, 1.0), (0.0, 1.0)] if constrain_weights else None
    x0 = np.array([0.5, 0.5]) if constrain_weights else np.array([1.0, 1.0])

    callback = None
    if verbose:
        def _print_weights(xk):
            print(f"iter weights -> base: {xk[0]:.4f}, sidekick: {xk[1]:.4f}")
        callback = _print_weights

    res = minimize(
        _weighted_nll,
        x0 = x0,
        args=(base_logits_val, sidekick_logits_val, true_labels_val),
        method = optimizer_method,
        bounds = bounds,
        constraints = cons,
        callback = callback,
        options = {'disp': verbose}

    )

    if not res.success:
        raise RuntimeError(f'Weight optimization failed: {res.message}')
    
    w_base, w_sidekick = res.x

    # Get the duo metrics for validation and test sets
    print(f"Optimized weights -> base: {w_base:.4f}, sidekick: {w_sidekick:.4f}")

    # Get the duo metrics for validation and test sets
    val_metrics, duo_val_probs, duo_val_pred = _build_metrics("val", w_base, base_logits_val, sidekick_logits_val, true_labels_val)
    test_metrics, duo_test_probs, duo_test_pred = _build_metrics("test", w_base, base_logits_test, sidekick_logits_test, true_labels_test)

    # Getting paths to save the duo results & create directories if there is none
    duo_metrics_path = osp.join(result_dir, 'duo/metrics', saved_file_name)
    duo_validation_results_path = osp.join(result_dir, 'duo/val_preds', saved_file_name)
    duo_test_results_path = osp.join(result_dir, 'duo/test_preds', saved_file_name)
    

    # Read data if it was written before and update it
    if osp.exists(duo_validation_results_path):
        existing_dict_val = dict(np.load(duo_validation_results_path, allow_pickle = True))
    else:
        os.makedirs(osp.dirname(duo_validation_results_path), exist_ok=True)
        existing_dict_val = {}
        
    if osp.exists(duo_test_results_path):
        existing_dict_test = dict(np.load(duo_test_results_path, allow_pickle = True))
    else:
        os.makedirs(osp.dirname(duo_test_results_path), exist_ok=True)
        existing_dict_test = {}
        
    if osp.exists(duo_metrics_path):
        existing_dict_metrics= dict(np.load(duo_metrics_path, allow_pickle = True))
    else:
        os.makedirs(osp.dirname(duo_metrics_path), exist_ok=True)
        existing_dict_metrics= {}
    
    save_dict_val = {
        str(seed): {
        'data' : saved_file_name,
        'idx': _sort_by_idx(base_val_results['idx'], base_idx_order_val),
        'input': _sort_by_idx(base_val_results['input'], base_idx_order_val),
        'softmax_probs': duo_val_probs,
        'preds': duo_val_pred,
        'true_labels': _sort_by_idx(base_val_results['true_labels'], base_idx_order_val),
        }
    }
    
    save_dict_test = {
        str(seed): {
        'data' : saved_file_name,
        'idx': _sort_by_idx(base_test_results['idx'], base_idx_order_test),
        'input': _sort_by_idx(base_test_results['input'], base_idx_order_test),
        'softmax_probs': duo_test_probs,
        'preds': duo_test_pred,
        'true_labels': _sort_by_idx(base_test_results['true_labels'], base_idx_order_test),
        }
    }

    existing_dict_val.update(save_dict_val)
    np.savez_compressed(duo_validation_results_path, **existing_dict_val)

    existing_dict_test.update(save_dict_test)
    np.savez_compressed(duo_test_results_path, **existing_dict_test)

    # Loading metrics summary results 
    metrics = {str(seed):{
        'data' : saved_file_name,
        'weights': {'base': w_base, 'sidekick': w_sidekick},
        'val': val_metrics,
        'test': test_metrics,
        }
    }

    existing_dict_metrics.update(metrics)
    np.savez_compressed(duo_metrics_path, **existing_dict_metrics)
   
