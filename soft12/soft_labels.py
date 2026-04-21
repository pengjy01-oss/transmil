"""Soft 12-subtype target distribution generation (Plan C).

Plan C core logic:
  1. Compute severity score S (same linear combination as Plan B).
  2. Normalize S per main class using ONLY training-set statistics:
       r = clip((S - S_min) / (S_max - S_min + eps), 0, 1)
  3. Generate a 3-dim Gaussian-weighted soft distribution over
       intra-class centers [low=0.0, mid=0.5, high=1.0]:
       w_j = exp(-(r - c_j)^2 / tau),  p_j = w_j / sum(w)
  4. Embed the 3-dim distribution into 12-dim space based on main class:
       class 0 -> dims [0,1,2], class 1 -> [3,4,5], ...

The resulting p12_target is a soft probability distribution over 12 subtypes.
It is used with KL divergence loss: KL(p12_target || softmax(logits12)).
"""

import json
import numpy as np

NUM_SUBTYPES = 12
NUM_MAIN_CLASSES = 4

SUBTYPE_NAMES = [
    '0/-', '0/0', '0/1',
    '1/0', '1/1', '1/2',
    '2/1', '2/2', '2/3',
    '3/2', '3/3', '3/+',
]

# Intra-class center positions for Gaussian soft distribution
_INTRA_CENTERS = np.array([0.0, 0.5, 1.0], dtype=np.float64)


def compute_severity_score(burden_stats, coverage_stats, weights=None):
    """Compute intra-class severity score S from burden/coverage stats.

    This is identical to the pseudo12 version and is reproduced here so that
    soft12 is self-contained.

    Args:
        burden_stats: dict with keys 'soft_ratio', 'score_mean', 'topk_mean', 'score_std'
        coverage_stats: dict with keys 'z_center', 'z_spread', 'active_bins_soft' (may be None)
        weights: dict {feature_name -> float coefficient}.
                 Default: {'soft_ratio': 1.0, 'topk_mean': 1.0, 'z_spread': 1.0}

    Returns:
        float: severity score S (unbounded linear combination)
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


def compute_normalization_stats(train_scores_by_class, eps=1e-6):
    """Compute per-class (S_min, S_max) from training set ONLY.

    Args:
        train_scores_by_class: dict {main_class_int -> list of S values}
        eps: small constant added to denominator to avoid division by zero

    Returns:
        dict {main_class_int -> (S_min, S_max)}
    """
    norm_stats = {}
    for cls in range(NUM_MAIN_CLASSES):
        scores = train_scores_by_class.get(cls, [])
        if len(scores) == 0:
            norm_stats[cls] = (0.0, 1.0)  # fallback: identity mapping
        else:
            arr = np.array(scores, dtype=np.float64)
            norm_stats[cls] = (float(arr.min()), float(arr.max()))
    return norm_stats


def compute_intra_r(S, main_class, norm_stats, eps=1e-6):
    """Normalize severity score S to intra-class position r in [0, 1].

    Uses ONLY training-set statistics to avoid data leakage.

    Args:
        S: float, raw severity score
        main_class: int in {0, 1, 2, 3}
        norm_stats: dict from compute_normalization_stats
        eps: small constant added to denominator

    Returns:
        float: r clamped to [0, 1]
    """
    s_min, s_max = norm_stats[int(main_class)]
    denom = s_max - s_min + eps
    r = (float(S) - s_min) / denom
    return float(np.clip(r, 0.0, 1.0))


def compute_intra_soft_dist(r, tau=0.5):
    """Compute 3-dim Gaussian-weighted soft distribution over [low, mid, high].

    Each intra-class center is at [0.0, 0.5, 1.0].
    Weight formula: w_j = exp(-(r - c_j)^2 / tau)
    Result is normalized so sum(p) = 1.

    Args:
        r: float in [0, 1], intra-class normalized position
        tau: float > 0, bandwidth parameter. Smaller tau => sharper peaks.

    Returns:
        np.array [3]: probability distribution [p_low, p_mid, p_high]
    """
    w = np.exp(-((float(r) - _INTRA_CENTERS) ** 2) / float(tau))
    p = w / w.sum()
    return p.astype(np.float32)


def embed_soft12_target(p_intra, main_class):
    """Embed 3-dim intra-class distribution into 12-dim space.

    Mapping:
      class 0 -> dims [0, 1, 2]
      class 1 -> dims [3, 4, 5]
      class 2 -> dims [6, 7, 8]
      class 3 -> dims [9, 10, 11]

    Non-assigned dims are zero, so result sums to 1 overall.

    Args:
        p_intra: np.array [3], intra-class soft distribution
        main_class: int in {0, 1, 2, 3}

    Returns:
        np.array [12]: soft target distribution over 12 subtypes
    """
    p12 = np.zeros(12, dtype=np.float32)
    base = int(main_class) * 3
    p12[base:base + 3] = p_intra
    return p12


def generate_soft12_targets(severity_scores, main_labels, norm_stats, tau=0.5, eps=1e-6):
    """Batch generate soft 12-type target distributions.

    Args:
        severity_scores: list/array of S values (one per sample)
        main_labels: list/array of int main class labels (0-3)
        norm_stats: dict from compute_normalization_stats (training set only)
        tau: float, Gaussian bandwidth for soft distribution
        eps: float, normalization epsilon

    Returns:
        np.ndarray [N, 12], float32: soft target distributions
    """
    n = len(severity_scores)
    targets = np.zeros((n, 12), dtype=np.float32)
    for i in range(n):
        cls = int(main_labels[i])
        r = compute_intra_r(severity_scores[i], cls, norm_stats, eps=eps)
        p_intra = compute_intra_soft_dist(r, tau=tau)
        targets[i] = embed_soft12_target(p_intra, cls)
    return targets


def print_soft12_diagnostics(
    soft_targets, severity_scores, main_labels, norm_stats,
    tau, split_name='Train', n_examples=5
):
    """Print detailed diagnostics for soft 12-type targets.

    Outputs:
    - Per-class S_min / S_max
    - Per-class r distribution statistics
    - Per-class mean p_intra distribution
    - Example samples showing S, r, p_intra, p12_target
    """
    severity_scores = np.asarray(severity_scores, dtype=np.float64)
    main_labels = np.asarray(main_labels, dtype=np.int64)

    print('\n====== Soft-12 Diagnostics ({}) ======'.format(split_name))
    print('tau = {:.4f}'.format(tau))

    for cls in range(NUM_MAIN_CLASSES):
        mask = main_labels == cls
        if not mask.any():
            continue
        s_min, s_max = norm_stats[cls]
        S_cls = severity_scores[mask]
        p12_cls = soft_targets[mask]  # [n_cls, 12]
        base = cls * 3

        # r values for this class
        r_vals = np.clip((S_cls - s_min) / (s_max - s_min + 1e-6), 0.0, 1.0)
        p_intra_mean = p12_cls[:, base:base + 3].mean(axis=0)

        print('\n  Stage {} (n={}):'.format(cls, int(mask.sum())))
        print('    S_min={:.4f}, S_max={:.4f} (train-set norm stats)'.format(s_min, s_max))
        print('    S stats in this split: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}'.format(
            S_cls.mean(), S_cls.std(), S_cls.min(), S_cls.max()
        ))
        print('    r stats: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}'.format(
            r_vals.mean(), r_vals.std(), r_vals.min(), r_vals.max()
        ))
        names = SUBTYPE_NAMES[base:base + 3]
        print('    Mean p_intra: {} = [{:.3f}, {:.3f}, {:.3f}]'.format(
            names, p_intra_mean[0], p_intra_mean[1], p_intra_mean[2]
        ))

    # Example samples
    print('\n  --- Sample examples ({}) ---'.format(split_name))
    indices = list(range(min(n_examples, len(severity_scores))))
    for i in indices:
        cls = int(main_labels[i])
        s_min, s_max = norm_stats[cls]
        r = float(np.clip((severity_scores[i] - s_min) / (s_max - s_min + 1e-6), 0.0, 1.0))
        base = cls * 3
        p_intra = soft_targets[i, base:base + 3]
        p12 = soft_targets[i]
        print('    sample[{}]: stage={}, S={:.4f}, r={:.4f}, p_intra=[{:.3f},{:.3f},{:.3f}], '
              'p12_nonzero={}'.format(
                  i, cls, severity_scores[i], r,
                  p_intra[0], p_intra[1], p_intra[2],
                  [(j, round(float(p12[j]), 3)) for j in range(12) if p12[j] > 0.001]
              ))
    print('====== Soft-12 Diagnostics Done ======\n')


def save_norm_stats(norm_stats, path):
    """Save per-class normalization stats to JSON.

    Args:
        norm_stats: dict {cls: (S_min, S_max)}
        path: str, file path
    """
    data = {str(k): list(v) for k, v in norm_stats.items()}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print('Soft-12 normalization stats saved to {}'.format(path))


def load_norm_stats(path):
    """Load per-class normalization stats from JSON.

    Args:
        path: str, file path

    Returns:
        dict {int -> (S_min, S_max)}
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return {int(k): tuple(v) for k, v in data.items()}
