"""Collect per-case inference details for explainability analysis.

Runs the model on a dataset split and returns a list of per-case dicts
containing attention, instance scores, CORN probabilities, bag features,
and slab metadata.
"""

from __future__ import print_function

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# ── Region display names ─────────────────────────────────────────────────────
REGION_DISPLAY = {
    'left_upper':   'L-Upper',
    'left_middle':  'L-Mid',
    'left_lower':   'L-Lower',
    'right_upper':  'R-Upper',
    'right_middle': 'R-Mid',
    'right_lower':  'R-Lower',
    'global':       'Global',
}

SIX_REGIONS_ORDER = ['left_upper', 'left_middle', 'left_lower',
                     'right_upper', 'right_middle', 'right_lower']


def _corn_to_probs(logits_np, num_classes):
    """Convert CORN logits → class probability vector.

    P(Y=0)   = 1 - sigmoid(l0)
    P(Y=k)   = prod_{j<k} sigma(lj) - prod_{j<=k} sigma(lj)   for 0 < k < C-1
    P(Y=C-1) = prod sigma(lj)
    """
    logits = torch.tensor(logits_np, dtype=torch.float32)
    s = torch.sigmoid(logits)          # [num_classes - 1]
    cum = torch.cumprod(s, dim=0)      # cumulative product

    probs = torch.zeros(num_classes)
    probs[0] = 1.0 - cum[0]
    for k in range(1, num_classes - 1):
        probs[k] = cum[k - 1] - cum[k]
    probs[num_classes - 1] = cum[-1]
    probs = probs.clamp(min=0.0)
    probs = probs / probs.sum().clamp_min(1e-9)
    return probs.numpy()


def _get_case_id(path):
    """Extract case ID (e.g. CT012345) from a NIfTI file path."""
    base = os.path.basename(path)
    # strip .nii.gz or .nii
    for suffix in ('.nii.gz', '.nii'):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def _extract_bag_features(model, logits_tensor, Z_final_tensor):
    """Return a 1-D numpy bag-level feature for embedding plots.

    We capture Z_final inside the forward pass via a temporary hook.
    """
    return Z_final_tensor.detach().cpu().numpy().squeeze(0)  # [D]


def run_inference(model, dataset, device, num_classes, use_cuda=True,
                  collect_bag_features=True, verbose=True):
    """Run model on every sample in ``dataset`` and return per-case dicts.

    Parameters
    ----------
    model : Attention or GatedAttention
    dataset : CTPneNiiBags (or similar); must store ``._last_metadata``
              and ``.samples`` list of (path, label) tuples.
    device  : torch device
    num_classes : int
    collect_bag_features : bool  whether to collect Z_final for t-SNE

    Returns
    -------
    results : list[dict]  – one dict per case with keys:
        case_id, path, true_label, pred_label, corn_logits, corn_probs,
        attention, instance_scores, slab_importance, pos_z, metadata,
        bag_features, burden_stats, coverage_stats, aux12_probs
    """
    model.eval()
    results = []

    # Optional: capture Z_final via hook
    _z_final_store = [None]

    def _zfinal_hook(module, inp, out):
        _z_final_store[0] = out.detach().cpu()  # [1, D]

    z_hook = None
    if collect_bag_features and hasattr(model, 'classifier'):
        z_hook = model.classifier.register_forward_pre_hook(_zfinal_hook)

    n = len(dataset)
    for idx in range(n):
        if verbose and idx % max(1, n // 10) == 0:
            print('  inference [{}/{}]'.format(idx + 1, n))

        # ── Load sample ─────────────────────────────────────────────────────
        item = dataset[idx]
        bag, label, pos_z = item[0], item[1], item[2]

        path, true_label_int = dataset.samples[idx]
        case_id = _get_case_id(path)
        metadata = list(getattr(dataset, '_last_metadata', []))

        bag_dev = bag.unsqueeze(0).to(device)          # [1, K, C, H, W]
        pos_z_dev = pos_z.unsqueeze(0).to(device)      # [1, K]
        _z_final_store[0] = None

        with torch.no_grad():
            logits, y_hat, _ = model(bag_dev, pos_z_dev)

        # ── CORN probabilities ───────────────────────────────────────────────
        logits_np = logits.detach().cpu().numpy().squeeze(0)   # [C-1]
        corn_probs = _corn_to_probs(logits_np, num_classes)
        pred_label = int(y_hat.detach().cpu().item())
        true_label = int(label.item())

        # ── Attention / instance scores ─────────────────────────────────────
        aux = model.last_forward_aux if hasattr(model, 'last_forward_aux') else {}

        # Attention weights: [1, K] or None
        attn = aux.get('attention_weights', None)
        if attn is not None:
            attention_np = attn.detach().cpu().numpy().squeeze(0)   # [K]
        else:
            # TransMIL: compute explicitly
            attention_np = _get_transmil_attention(model, bag_dev, pos_z_dev)

        # Instance scores: [K] or None
        inst_sc = aux.get('instance_scores', None)
        if inst_sc is not None:
            instance_scores_np = inst_sc.detach().cpu().numpy().squeeze()
        else:
            instance_scores_np = None

        # Slab importance = attention * instance_score (if both available)
        if attention_np is not None and instance_scores_np is not None:
            slab_importance = attention_np * instance_scores_np
            if slab_importance.sum() > 1e-12:
                slab_importance = slab_importance / slab_importance.sum()
        elif attention_np is not None:
            slab_importance = attention_np
        elif instance_scores_np is not None:
            s = instance_scores_np.copy()
            s_sum = s.sum()
            slab_importance = s / s_sum if s_sum > 1e-12 else np.ones_like(s) / len(s)
        else:
            k = bag.shape[0]
            slab_importance = np.ones(k, dtype=np.float32) / max(k, 1)

        # ── Bag features (Z_final) ──────────────────────────────────────────
        bag_features_np = None
        if collect_bag_features and _z_final_store[0] is not None:
            bag_features_np = _z_final_store[0].numpy().squeeze(0)

        # ── Auxiliary heads ─────────────────────────────────────────────────
        aux12_probs = None
        aux12_logits = aux.get('aux12_logits', None)
        if aux12_logits is not None:
            ax = aux12_logits.detach().cpu().float()
            aux12_probs = F.softmax(ax, dim=-1).numpy().squeeze(0)

        # ── Burden / coverage stats ─────────────────────────────────────────
        burden_stats = aux.get('burden_stats', None)
        coverage_stats = aux.get('coverage_stats', None)
        pos_z_np = aux.get('pos_z', pos_z).detach().cpu().numpy() if hasattr(
            aux.get('pos_z', pos_z), 'detach') else pos_z.numpy()

        results.append({
            'case_id': case_id,
            'path': path,
            'true_label': true_label,
            'pred_label': pred_label,
            'corn_logits': logits_np,
            'corn_probs': corn_probs,
            'attention': attention_np,
            'instance_scores': instance_scores_np,
            'slab_importance': slab_importance,
            'pos_z': pos_z_np,
            'metadata': metadata,
            'bag_features': bag_features_np,
            'burden_stats': burden_stats,
            'coverage_stats': coverage_stats,
            'aux12_probs': aux12_probs,
            'bag_tensor': bag,          # kept in CPU for Grad-CAM
        })

    if z_hook is not None:
        z_hook.remove()

    return results


def _get_transmil_attention(model, bag_dev, pos_z_dev):
    """Compute per-instance attention for TransMIL / Nystrom aggregators."""
    try:
        with torch.no_grad():
            attn_tensor = model.compute_attention_weights(bag_dev, pos_z_dev)
        if attn_tensor is not None:
            return attn_tensor.cpu().numpy().squeeze(0)
    except Exception:
        pass
    # Fallback: uniform
    k = bag_dev.size(1)
    return np.ones(k, dtype=np.float32) / max(k, 1)


def compute_region_scores(slab_importance_np, metadata, regions=None):
    """Average slab importance per lung region.

    Parameters
    ----------
    slab_importance_np : [K] array
    metadata : list[dict] with 'region' key
    regions  : list of region names to report (default: SIX_REGIONS_ORDER)

    Returns
    -------
    dict  region_name -> mean_score (0 if no slabs in that region)
    """
    if regions is None:
        regions = SIX_REGIONS_ORDER

    scores = {r: [] for r in regions}
    for i, m in enumerate(metadata):
        r = m.get('region', 'global')
        if r in scores and i < len(slab_importance_np):
            scores[r].append(float(slab_importance_np[i]))

    return {r: (float(np.mean(v)) if v else 0.0) for r, v in scores.items()}
