"""Lung region and z-axis heatmap visualizations (Section E of explainability spec).

Produces:
  1. Z-axis heat distribution across all cases (grouped by class)
  2. Six-region heatmap: average importance per region per class
  3. Case-specific lung region visualizations
"""

from __future__ import print_function

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

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

CLASS_NAMES = ['Stage 0', 'Stage I', 'Stage II', 'Stage III']


def _save(fig, path):
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── Helper: accumulate z-grid ────────────────────────────────────────────────

def _build_z_profile(results, class_filter=None, resolution=100):
    """Average slab importance over all cases (optionally filtered by class).

    Returns a 1-D array of shape [resolution] summed and normalized.
    """
    grid = np.zeros(resolution, dtype=np.float64)
    count = np.zeros(resolution, dtype=np.int64)

    for case_dict in results:
        if class_filter is not None and case_dict['true_label'] != class_filter:
            continue
        importance = case_dict['slab_importance']
        pos_z = case_dict['pos_z']
        K = len(importance)
        for i in range(K):
            if i >= len(pos_z):
                continue
            zi = int(np.clip(pos_z[i] * (resolution - 1), 0, resolution - 1))
            grid[zi] += float(importance[i])
            count[zi] += 1

    mask = count > 0
    grid[mask] /= count[mask]
    if grid.max() > 0:
        grid /= grid.max()
    return grid


def _build_region_matrix(results, num_classes):
    """Build [num_classes, 6] mean importance matrix (true labels × regions)."""
    sums = np.zeros((num_classes, len(SIX_REGIONS_ORDER)), dtype=np.float64)
    counts = np.zeros((num_classes, len(SIX_REGIONS_ORDER)), dtype=np.int64)

    for case_dict in results:
        true_l = case_dict['true_label']
        if true_l < 0 or true_l >= num_classes:
            continue
        importance = case_dict['slab_importance']
        metadata = case_dict['metadata']
        for i, m in enumerate(metadata):
            if i >= len(importance):
                break
            r = m.get('region', 'global')
            if r in SIX_REGIONS_ORDER:
                ri = SIX_REGIONS_ORDER.index(r)
                sums[true_l, ri] += float(importance[i])
                counts[true_l, ri] += 1

    mask = counts > 0
    mat = np.zeros_like(sums)
    mat[mask] = sums[mask] / counts[mask]
    return mat


# ── Plot 1: Z-axis importance profile per class ──────────────────────────────

def plot_z_profiles_by_class(results, save_path, num_classes=4, resolution=100):
    """Line plot of average z-axis importance distribution, one curve per class."""
    colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']

    fig, ax = plt.subplots(figsize=(10, 5))
    z_axis = np.linspace(0, 1, resolution)

    for c in range(num_classes):
        profile = _build_z_profile(results, class_filter=c, resolution=resolution)
        if profile.max() > 0:
            ax.plot(z_axis, profile,
                    color=colors[c % len(colors)],
                    label=CLASS_NAMES[c] if c < len(CLASS_NAMES) else 'Class {}'.format(c),
                    linewidth=2, alpha=0.85)
            ax.fill_between(z_axis, profile, alpha=0.1, color=colors[c % len(colors)])

    ax.set_xlabel('Normalized z (cranial → caudal)')
    ax.set_ylabel('Normalized average importance')
    ax.set_title('Z-axis Slab Importance Distribution by Class')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)


# ── Plot 2: Six-region heatmap (class × region) ───────────────────────────────

def plot_six_region_heatmap(results, save_path, num_classes=4):
    """2-D heatmap: rows = class (true label), cols = lung region."""
    mat = _build_region_matrix(results, num_classes)

    # Normalize each class row to [0, 1] for display
    row_max = mat.max(axis=1, keepdims=True)
    mat_norm = np.where(row_max > 0, mat / row_max, 0.0)

    region_labels = [REGION_DISPLAY.get(r, r) for r in SIX_REGIONS_ORDER]
    class_labels = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else 'C{}'.format(c)
                    for c in range(num_classes)]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(3, num_classes + 1)),
                             gridspec_kw={'wspace': 0.4})

    # Raw values
    im0 = axes[0].imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0)
    axes[0].set_xticks(range(len(SIX_REGIONS_ORDER)))
    axes[0].set_xticklabels(region_labels, rotation=30, ha='right', fontsize=9)
    axes[0].set_yticks(range(num_classes))
    axes[0].set_yticklabels(class_labels)
    axes[0].set_title('Mean Slab Importance\n(raw)')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    for i in range(num_classes):
        for j in range(len(SIX_REGIONS_ORDER)):
            axes[0].text(j, i, '{:.3f}'.format(mat[i, j]),
                         ha='center', va='center', fontsize=8)

    # Row-normalized
    im1 = axes[1].imshow(mat_norm, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    axes[1].set_xticks(range(len(SIX_REGIONS_ORDER)))
    axes[1].set_xticklabels(region_labels, rotation=30, ha='right', fontsize=9)
    axes[1].set_yticks(range(num_classes))
    axes[1].set_yticklabels(class_labels)
    axes[1].set_title('Mean Slab Importance\n(row-normalized per class)')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    for i in range(num_classes):
        for j in range(len(SIX_REGIONS_ORDER)):
            axes[1].text(j, i, '{:.2f}'.format(mat_norm[i, j]),
                         ha='center', va='center', fontsize=8)

    fig.suptitle('Six Lung Region Importance Heatmap', fontsize=13)
    fig.tight_layout()
    _save(fig, save_path)


# ── Plot 3: Six-region anatomy layout ────────────────────────────────────────

def plot_six_region_anatomy(results, save_path, num_classes=4):
    """Display six regions as a schematic lung layout with colour intensity.

    Layout (simplified frontal view):
      Left lung (right in image) | Right lung (left in image)
      upper | lower
      (3 vertical bands per side)
    """
    mat = _build_region_matrix(results, num_classes)

    # Overall average across all classes (weighted by class count)
    class_counts = np.array([sum(1 for r in results if r['true_label'] == c)
                              for c in range(num_classes)], dtype=np.float64)
    total = class_counts.sum()
    weights = class_counts / max(total, 1.0)
    region_avg = (mat * weights[:, None]).sum(axis=0)  # [6]

    # Map to region dict
    region_scores = {r: float(region_avg[i]) for i, r in enumerate(SIX_REGIONS_ORDER)}
    max_score = max(region_scores.values()) if region_scores else 1.0
    if max_score <= 0:
        max_score = 1.0

    fig, ax = plt.subplots(figsize=(8, 7))

    # Draw six rectangles in a schematic lung layout
    # (x0, y0, width, height) in axes coordinates [0,1]
    # Left lung = right side, Right lung = left side (radiological convention)
    layout = {
        'left_upper':   (0.52, 0.62, 0.38, 0.30),
        'left_middle':  (0.52, 0.32, 0.38, 0.28),
        'left_lower':   (0.52, 0.05, 0.38, 0.25),
        'right_upper':  (0.10, 0.62, 0.38, 0.30),
        'right_middle': (0.10, 0.32, 0.38, 0.28),
        'right_lower':  (0.10, 0.05, 0.38, 0.25),
    }

    cmap = plt.cm.YlOrRd
    for region, bbox in layout.items():
        score = region_scores.get(region, 0.0)
        intensity = score / max_score
        color = cmap(intensity)
        x0, y0, w, h = bbox
        rect = plt.Rectangle((x0, y0), w, h, transform=ax.transAxes,
                              facecolor=color, edgecolor='gray', linewidth=2,
                              clip_on=False)
        ax.add_patch(rect)
        display = REGION_DISPLAY.get(region, region)
        ax.text(x0 + w / 2, y0 + h / 2, '{}\n{:.3f}'.format(display, score),
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white' if intensity > 0.5 else 'black')

    # Labels
    ax.text(0.71, 0.97, 'Left Lung', transform=ax.transAxes,
            ha='center', fontsize=11, color='steelblue')
    ax.text(0.29, 0.97, 'Right Lung', transform=ax.transAxes,
            ha='center', fontsize=11, color='firebrick')
    ax.set_title('Lung Region Importance (Weighted Average across Classes)', pad=20)
    ax.axis('off')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=max_score))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, label='Mean Slab Importance')

    fig.tight_layout()
    _save(fig, save_path)


# ── Plot 4: Per-class six-region comparison bar ──────────────────────────────

def plot_region_per_class_bars(results, save_path, num_classes=4):
    """Grouped bar chart: classes grouped by region."""
    mat = _build_region_matrix(results, num_classes)   # [num_classes, 6]

    x = np.arange(len(SIX_REGIONS_ORDER))
    width = 0.18
    class_colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']

    fig, ax = plt.subplots(figsize=(12, 5))
    offsets = np.linspace(-(num_classes - 1) * width / 2,
                          (num_classes - 1) * width / 2, num_classes)
    for c in range(num_classes):
        label = CLASS_NAMES[c] if c < len(CLASS_NAMES) else 'C{}'.format(c)
        ax.bar(x + offsets[c], mat[c], width,
               label=label, color=class_colors[c % len(class_colors)],
               alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([REGION_DISPLAY.get(r, r) for r in SIX_REGIONS_ORDER])
    ax.set_ylabel('Mean Slab Importance')
    ax.set_title('Six Lung Region Importance by Stage')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)


# ── Main entry ────────────────────────────────────────────────────────────────

def generate_lung_region_heatmaps(results, save_dir, num_classes=4, verbose=True):
    """Generate all lung region heatmaps.

    Parameters
    ----------
    results  : list[dict] from inference_engine.run_inference
    save_dir : str
    """
    os.makedirs(save_dir, exist_ok=True)
    if verbose:
        print('  z-axis profile by class...')
    plot_z_profiles_by_class(
        results, os.path.join(save_dir, 'z_profiles_by_class.png'),
        num_classes=num_classes)

    if verbose:
        print('  six-region heatmap (class × region)...')
    plot_six_region_heatmap(
        results, os.path.join(save_dir, 'six_region_heatmap.png'),
        num_classes=num_classes)

    if verbose:
        print('  six-region anatomy layout...')
    plot_six_region_anatomy(
        results, os.path.join(save_dir, 'six_region_anatomy.png'),
        num_classes=num_classes)

    if verbose:
        print('  region per-class bar chart...')
    plot_region_per_class_bars(
        results, os.path.join(save_dir, 'region_per_class_bars.png'),
        num_classes=num_classes)
