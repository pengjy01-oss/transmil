"""Pseudo 12-subtype label generation.

12 subtypes mapping (fixed order):
  0: 0/-    1: 0/0    2: 0/1
  3: 1/0    4: 1/1    5: 1/2
  6: 2/1    7: 2/2    8: 2/3
  9: 3/2   10: 3/3   11: 3/+

Each main class is split into 3 subtypes based on intra-class severity score S.
Thresholds are computed ONLY from training set to avoid data leakage.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F

NUM_SUBTYPES = 12
NUM_MAIN_CLASSES = 4

SUBTYPE_NAMES = [
    '0/-', '0/0', '0/1',
    '1/0', '1/1', '1/2',
    '2/1', '2/2', '2/3',
    '3/2', '3/3', '3/+',
]


def compute_severity_score(burden_stats, coverage_stats, weights=None):
    """Compute intra-class severity score S from burden/coverage stats.

    Args:
        burden_stats: dict with keys 'soft_ratio', 'score_mean', 'topk_mean', 'score_std'
        coverage_stats: dict with keys 'z_center', 'z_spread', 'active_bins_soft'
                        (may be None)
        weights: dict with keys matching feature names -> float coefficients.
                 Defaults: {'soft_ratio': 1.0, 'topk_mean': 1.0, 'z_spread': 1.0}

    Returns:
        float: severity score S
    """
    if weights is None:
        weights = {'soft_ratio': 1.0, 'topk_mean': 1.0, 'z_spread': 1.0}

    S = 0.0
    for feat_name, coeff in weights.items():
        val = None
        if burden_stats is not None and feat_name in burden_stats:
            val = burden_stats[feat_name]
        if val is None and coverage_stats is not None and feat_name in coverage_stats:
            val = coverage_stats[feat_name]
        if val is not None:
            S += float(coeff) * float(val)
    return S


def calibrate_thresholds(train_scores_by_class, quantiles=(1.0 / 3.0, 2.0 / 3.0)):
    """Compute per-class split thresholds from training set severity scores.

    Args:
        train_scores_by_class: dict {main_class_int: list of S values}
        quantiles: tuple of two floats (lower, upper)

    Returns:
        dict {main_class_int: (t1, t2)}
    """
    thresholds = {}
    q_lo, q_hi = quantiles
    for cls in range(NUM_MAIN_CLASSES):
        scores = train_scores_by_class.get(cls, [])
        if len(scores) < 3:
            # Not enough samples: use even splits
            if len(scores) == 0:
                thresholds[cls] = (0.0, 0.0)
            else:
                arr = np.array(scores, dtype=np.float64)
                thresholds[cls] = (float(arr.min()), float(arr.max()))
        else:
            arr = np.array(scores, dtype=np.float64)
            t1 = float(np.percentile(arr, q_lo * 100.0))
            t2 = float(np.percentile(arr, q_hi * 100.0))
            thresholds[cls] = (t1, t2)
    return thresholds


def assign_pseudo12_label(main_class, severity_score, thresholds):
    """Assign one of 12 subtype labels based on main class, S, and thresholds.

    Args:
        main_class: int in {0, 1, 2, 3}
        severity_score: float S
        thresholds: dict {main_class: (t1, t2)}

    Returns:
        int: pseudo 12-subtype label in [0, 11]
    """
    t1, t2 = thresholds[int(main_class)]
    base = int(main_class) * 3
    if severity_score < t1:
        return base + 0
    elif severity_score < t2:
        return base + 1
    else:
        return base + 2


def generate_pseudo12_labels(severity_scores, main_labels, thresholds):
    """Batch assign pseudo-12 labels.

    Args:
        severity_scores: list/array of S values (one per sample)
        main_labels: list/array of main class labels (0-3)
        thresholds: dict from calibrate_thresholds

    Returns:
        np.ndarray of int64 pseudo-12 labels
    """
    n = len(severity_scores)
    pseudo_labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        pseudo_labels[i] = assign_pseudo12_label(
            main_labels[i], severity_scores[i], thresholds
        )
    return pseudo_labels


def print_pseudo12_distribution(pseudo_labels, main_labels, split_name='Train'):
    """Print pseudo-12 label distribution per main class."""
    print('\n--- Pseudo-12 subtype distribution ({}) ---'.format(split_name))
    for cls in range(NUM_MAIN_CLASSES):
        mask = np.array(main_labels) == cls
        sub_labels = np.array(pseudo_labels)[mask]
        base = cls * 3
        names = SUBTYPE_NAMES[base:base + 3]
        counts = [int(np.sum(sub_labels == (base + j))) for j in range(3)]
        total = int(mask.sum())
        print('  Stage {}: total={}, {} -> {}'.format(
            cls,
            total,
            dict(zip(names, counts)),
            counts,
        ))
    # Overall
    dist = np.bincount(np.array(pseudo_labels), minlength=NUM_SUBTYPES)
    print('  Overall 12-subtype dist: {}'.format(dist.tolist()))
    print('')


def save_thresholds(thresholds, path):
    """Save thresholds to JSON file."""
    data = {str(k): list(v) for k, v in thresholds.items()}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print('Pseudo-12 thresholds saved to {}'.format(path))


def load_thresholds(path):
    """Load thresholds from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return {int(k): tuple(v) for k, v in data.items()}
