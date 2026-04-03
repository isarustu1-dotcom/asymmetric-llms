import numpy as np
import os.path as osp
import os

def __sanity_check_on_probs(probs, which_prob):
    total_observations = probs.shape[0]
    
    print(f'Total number of observation for model {which_prob} is {total_observations}')
    print(f'Sum of probs for model {which_prob} is {probs.sum(axis=1).sum()}')

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

def _softmax_from_log(log_p: np.ndarray, ax: int = 1) -> np.ndarray:
    """Convert log-probabilities to normalized probabilities.

    Args:
        log_p (np.ndarray): Log-probability matrix.

    Returns:
        np.ndarray: Normalized probability matrix.
    """
    log_p_norm = log_p - _logsumexp(log_p, axis=ax, keepdims=True)
    return np.exp(log_p_norm)


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


def _read_or_create_path(path: str):

    # Read data if it was written before and update it
    if osp.exists(path):
        data = dict(np.load(path, allow_pickle = True))
    else:
        os.makedirs(osp.dirname(path), exist_ok=True)
        data = {}

    return data

def _merge_dictionaries(dict1, dict2):
    result = dict1.copy()
    for k, v in dict2.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dictionaries(result[k], v)
        else:
            result[k] = v
            
    return result

def _save_val_test_results(seed, data_name, idx, input, logits, softmax_probs, preds, true_labels, existing_dict, path):

    save_dict_val = {
        str(seed): {
        'data' : data_name,
        'idx': idx,
        'input': input,
        'logits': logits,
        'softmax_probs': softmax_probs,
        'preds': preds,
        'true_labels': true_labels,
        }
    }
    merged_dict = _merge_dictionaries(existing_dict, save_dict_val)
    np.savez_compressed(path, **merged_dict)





