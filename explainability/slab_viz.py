"""Slab-level MIL visualizations: attention curves, z-heatmap, top-K montage.

Produces per-case figures showing how the model distributes importance
across slabs and lung regions.
"""

from __future__ import print_function

import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

CLASS_NAMES = ['Stage 0', 'Stage I', 'Stage II', 'Stage III']

SIX_REGIONS_ORDER = ['left_upper', 'left_middle', 'left_lower',
                     'right_upper', 'right_middle', 'right_lower']

REGION_DISPLAY = {
    'left_upper':   'L-Upper',
    'left_middle':  'L-Mid',
    'left_lower':   'L-Lower',
    'right_upper':  'R-Upper',
    'right_middle': 'R-Mid',
    'right_lower':  'R-Lower',
    'global':       'Global',
}

REGION_COLORS = {
    'left_upper':  '#3498db',
    'left_middle': '#2980b9',
    'left_lower':  '#1a5276',
    'right_upper': '#e74c3c',
    'right_middle': '#c0392b',
    'right_lower': '#7b241c',
    'global':      '#8e44ad',
}


def _save(fig, path):
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── Colour by region ──────────────────────────────────────────────────────────

def _region_colors_for_slabs(metadata):
    """Return list of colors, one per slab, based on region."""
    return [REGION_COLORS.get(m.get('region', 'global'), '#7f8c8d') for m in metadata]


# ── A: Attention / importance score curve ─────────────────────────────────────

def plot_slab_importance_curve(case_dict, save_path):
    """Plot attention and instance score along z-axis for one case."""
    importance = case_dict['slab_importance']
    attention = case_dict.get('attention')
    inst_sc = case_dict.get('instance_scores')
    pos_z = case_dict['pos_z']
    metadata = case_dict['metadata']
    case_id = case_dict['case_id']
    true_l = case_dict['true_label']
    pred_l = case_dict['pred_label']

    K = len(importance)
    x = np.arange(K)

    # Sort by z position for display
    order = np.argsort(pos_z[:K]) if len(pos_z) >= K else np.arange(K)
    pos_sorted = pos_z[order]

    fig, axes = plt.subplots(2, 1, figsize=(max(10, K // 4), 7), sharex=False)

    # ── Sub-plot 1: importance ordered by z ─────────────────────────────────
    ax = axes[0]
    colors_sorted = [REGION_COLORS.get(
        metadata[i].get('region', 'global') if i < len(metadata) else 'global', '#7f8c8d')
        for i in order]
    ax.bar(np.arange(K), importance[order], color=colors_sorted, alpha=0.85, width=0.8)
    ax.set_xlabel('Slab (sorted by z position)')
    ax.set_ylabel('Slab Importance')
    ax.set_title('Slab Importance along z-axis  |  {} true={} pred={}'.format(
        case_id, true_l, pred_l))
    ax.set_xlim(-0.5, K - 0.5)
    ax.grid(True, axis='y', alpha=0.3)

    # Region legend
    seen_regions = {}
    for i, idx in enumerate(order):
        r = metadata[idx].get('region', 'global') if idx < len(metadata) else 'global'
        if r not in seen_regions:
            seen_regions[r] = REGION_COLORS.get(r, '#7f8c8d')
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=REGION_DISPLAY.get(r, r))
                    for r, c in seen_regions.items()]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=8)

    # ── Sub-plot 2: attention + instance_score overlay ───────────────────────
    ax2 = axes[1]
    xs = pos_sorted  # x = normalized z
    if attention is not None and len(attention) >= K:
        ax2.plot(xs, attention[order], 'b-o', markersize=3, label='Attention', alpha=0.7)
    if inst_sc is not None and len(inst_sc) >= K:
        ax2.plot(xs, inst_sc[order], 'r-s', markersize=3, label='Instance Score', alpha=0.7)
    ax2.set_xlabel('Normalized z position')
    ax2.set_ylabel('Score')
    ax2.set_title('Attention & Instance Score vs z')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    fig.tight_layout()
    _save(fig, save_path)


# ── B: Z-axis heatmap strip ──────────────────────────────────────────────────

def plot_z_heatmap_strip(case_dict, save_path):
    """Plot a coloured 1D strip showing importance at each z position."""
    importance = case_dict['slab_importance']
    pos_z = case_dict['pos_z']
    metadata = case_dict['metadata']
    case_id = case_dict['case_id']
    true_l = case_dict['true_label']
    pred_l = case_dict['pred_label']

    K = len(importance)

    # Build a 1D "z-image": map slab importance to a fine z grid
    resolution = 200
    z_grid = np.zeros(resolution, dtype=np.float32)
    count_grid = np.zeros(resolution, dtype=np.int32)

    for i in range(K):
        if i >= len(pos_z):
            continue
        zi = int(np.clip(pos_z[i] * (resolution - 1), 0, resolution - 1))
        z_grid[zi] += float(importance[i])
        count_grid[zi] += 1

    mask = count_grid > 0
    z_grid[mask] /= count_grid[mask]

    # Smooth slightly
    from scipy.ndimage import gaussian_filter1d
    try:
        z_smooth = gaussian_filter1d(z_grid, sigma=3.0)
    except Exception:
        z_smooth = z_grid

    if z_smooth.max() > 0:
        z_smooth = z_smooth / z_smooth.max()

    fig, axes = plt.subplots(2, 1, figsize=(10, 4),
                             gridspec_kw={'height_ratios': [3, 1]})

    # Top: line plot
    ax = axes[0]
    ax.fill_between(np.linspace(0, 1, resolution), z_smooth, alpha=0.4, color='orangered')
    ax.plot(np.linspace(0, 1, resolution), z_smooth, 'r-', linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('Normalized z (cranial → caudal)')
    ax.set_ylabel('Relative importance')
    ax.set_title('Z-axis Slab Importance  |  {} true={} pred={}'.format(
        case_id, true_l, pred_l))
    ax.grid(True, alpha=0.3)

    # Mark individual slabs
    for i in range(K):
        if i >= len(pos_z):
            continue
        color = REGION_COLORS.get(
            metadata[i].get('region', 'global') if i < len(metadata) else 'global', '#999')
        ax.axvline(x=float(pos_z[i]), color=color, alpha=0.25, linewidth=0.8)

    # Bottom: heatmap strip
    ax2 = axes[1]
    strip = z_smooth.reshape(1, -1)
    ax2.imshow(strip, aspect='auto', cmap='hot', vmin=0, vmax=1,
               extent=[0, 1, 0, 1])
    ax2.set_yticks([])
    ax2.set_xlabel('Normalized z')
    ax2.set_title('Heatmap Strip')

    fig.tight_layout()
    _save(fig, save_path)


# ── C: Six-region bar chart ───────────────────────────────────────────────────

def plot_six_region_bar(case_dict, save_path):
    """Bar chart of average slab importance per lung region."""
    importance = case_dict['slab_importance']
    metadata = case_dict['metadata']
    case_id = case_dict['case_id']
    true_l = case_dict['true_label']
    pred_l = case_dict['pred_label']

    # Accumulate
    region_sums = {r: 0.0 for r in SIX_REGIONS_ORDER}
    region_counts = {r: 0 for r in SIX_REGIONS_ORDER}

    for i, m in enumerate(metadata):
        if i >= len(importance):
            break
        r = m.get('region', 'global')
        if r in region_sums:
            region_sums[r] += float(importance[i])
            region_counts[r] += 1

    # mean per region
    vals = []
    counts = []
    for r in SIX_REGIONS_ORDER:
        c = region_counts[r]
        vals.append(region_sums[r] / c if c > 0 else 0.0)
        counts.append(c)

    labels = [REGION_DISPLAY.get(r, r) for r in SIX_REGIONS_ORDER]
    colors = [REGION_COLORS.get(r, '#999') for r in SIX_REGIONS_ORDER]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='white')
    for bar, val, cnt in zip(bars, vals, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                '{:.3f}\n(n={})'.format(val, cnt),
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Mean Slab Importance')
    ax.set_title('Six Lung Regions  |  {} true={} pred={}'.format(case_id, true_l, pred_l))
    ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 0.1)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)


# ── D: Top-K slab montage ─────────────────────────────────────────────────────

def plot_topk_slab_montage(case_dict, save_path, top_k=6, middle_channel=1):
    """Show the top-K most important slabs as a grid of images."""
    importance = case_dict['slab_importance']
    bag_tensor = case_dict['bag_tensor']     # [K, C, H, W]
    metadata = case_dict['metadata']
    pos_z = case_dict['pos_z']
    case_id = case_dict['case_id']
    true_l = case_dict['true_label']
    pred_l = case_dict['pred_label']

    K = bag_tensor.shape[0]
    k = min(top_k, K)
    top_indices = np.argsort(importance)[::-1][:k]

    ncols = min(k, 3)
    nrows = (k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    if k == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for rank, slab_idx in enumerate(top_indices):
        row, col = rank // ncols, rank % ncols
        ax = axes[row, col]

        slab = bag_tensor[slab_idx]   # [C, H, W]
        ch_idx = min(middle_channel, slab.shape[0] - 1)
        img = slab[ch_idx].numpy()
        img = np.clip(img, 0, 1)

        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        region = metadata[slab_idx].get('region', 'unk') if slab_idx < len(metadata) else 'unk'
        cz = metadata[slab_idx].get('center_z', -1) if slab_idx < len(metadata) else -1
        pz_val = float(pos_z[slab_idx]) if slab_idx < len(pos_z) else 0.0
        ax.set_title('Rank#{} slab{}\n{} z={} ({:.2f})'.format(
            rank + 1, slab_idx, REGION_DISPLAY.get(region, region), cz, pz_val),
            fontsize=8)
        ax.axis('off')

        # Importance badge
        color = REGION_COLORS.get(region, '#999')
        ax.text(0.97, 0.03, 'imp={:.4f}'.format(float(importance[slab_idx])),
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

    # Hide empty axes
    for rank in range(k, nrows * ncols):
        row, col = rank // ncols, rank % ncols
        axes[row, col].axis('off')

    fig.suptitle('Top-{} Slabs  |  {} true={} pred={}'.format(
        k, case_id, true_l, pred_l), fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, save_path)


# ── E: Slab ranking table (CSV) ───────────────────────────────────────────────

def save_slab_ranking_csv(case_dict, save_path):
    """Save slab-level ranking table as CSV."""
    importance = case_dict['slab_importance']
    attention = case_dict.get('attention')
    inst_sc = case_dict.get('instance_scores')
    pos_z = case_dict['pos_z']
    metadata = case_dict['metadata']

    K = len(importance)
    ranks = np.argsort(importance)[::-1]

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'slab_idx', 'region', 'center_z', 'pos_z',
                         'importance', 'attention', 'instance_score'])
        for rank, idx in enumerate(ranks):
            region = metadata[idx].get('region', '') if idx < len(metadata) else ''
            center_z = metadata[idx].get('center_z', '') if idx < len(metadata) else ''
            pz = float(pos_z[idx]) if idx < len(pos_z) else ''
            attn_val = float(attention[idx]) if attention is not None and idx < len(attention) else ''
            sc_val = float(inst_sc[idx]) if inst_sc is not None and idx < len(inst_sc) else ''
            writer.writerow([rank + 1, idx, region, center_z, pz,
                             float(importance[idx]), attn_val, sc_val])


# ── Main entry ────────────────────────────────────────────────────────────────

def generate_slab_visualizations(results, save_dir, top_k=6, verbose=True):
    """Generate all slab-level figures and CSV tables for all cases.

    Parameters
    ----------
    results  : list[dict] from inference_engine.run_inference
    save_dir : str  root directory
    top_k    : int  number of top slabs to show in montage
    """
    os.makedirs(save_dir, exist_ok=True)
    n = len(results)

    for i, case_dict in enumerate(results):
        if verbose and i % max(1, n // 10) == 0:
            print('  slab viz [{}/{}]'.format(i + 1, n))

        cid = case_dict['case_id']
        true_l = case_dict['true_label']
        pred_l = case_dict['pred_label']
        prefix = '{}_t{}_p{}'.format(cid, true_l, pred_l)

        case_dir = os.path.join(save_dir, prefix)
        os.makedirs(case_dir, exist_ok=True)

        plot_slab_importance_curve(
            case_dict, os.path.join(case_dir, 'slab_importance_curve.png'))
        plot_z_heatmap_strip(
            case_dict, os.path.join(case_dir, 'z_heatmap_strip.png'))
        plot_six_region_bar(
            case_dict, os.path.join(case_dir, 'six_region_bar.png'))
        plot_topk_slab_montage(
            case_dict, os.path.join(case_dir, 'topk_slabs.png'), top_k=top_k)
        save_slab_ranking_csv(
            case_dict, os.path.join(case_dir, 'slab_ranking.csv'))
