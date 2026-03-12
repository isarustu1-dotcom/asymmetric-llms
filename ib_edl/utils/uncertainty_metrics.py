import numpy as np


def _expected_calibration_error(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) with uniform confidence bins.

    Args:
        probs (np.ndarray): Predicted probabilities with shape (N, C).
        labels (np.ndarray): Integer labels with shape (N,).
        num_bins (int): Number of confidence bins.

    Returns:
        float: Expected calibration error.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(num_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if not np.any(mask):
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def compute_uncertainty_metrics(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> dict:
    """Compute common uncertainty metrics from class probabilities and labels.

    Args:
        probs (np.ndarray): Predicted probabilities with shape (1, N, C).
        labels (np.ndarray): Integer labels with shape (N,).
        num_bins (int): Number of bins for calibration metrics.

    Returns:
        dict: Metric values including NLL, Brier, ECE, LPPD, and mean uncertainty.
    """
    labels = labels.astype(int)
    if probs.ndim == 3:
        probs = probs.squeeze(axis=0)
    true_probs = probs[np.arange(len(labels)), labels]

    nll = -np.mean(np.log(true_probs + 1e-12))
    lppd = np.sum(np.log(true_probs + 1e-12))

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))

    ece = _expected_calibration_error(probs, labels, num_bins=num_bins)

    confidences = probs.max(axis=1)
    mean_uncertainty = float(1.0 - confidences.mean())

    return {
        "nll": float(nll),
        "lppd": float(lppd),
        "brier": float(brier),
        "ece": float(ece),
        "mean_uncertainty": mean_uncertainty,
    }