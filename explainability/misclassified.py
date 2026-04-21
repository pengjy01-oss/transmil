"""Misclassified case analysis.

Collects cases where the model made errors, with special focus on
ambiguous boundary errors (1→2, 2→1, 2→3), and generates:
  - Summary table (CSV)
  - Per-case slab importance figures
  - Comparison plots (correct vs wrong)
"""

from __future__ import print_function

import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .slab_viz import plot_slab_importance_curve, plot_six_region_bar, plot_topk_slab_montage

CLASS_NAMES = ['Stage 0', 'Stage I', 'Stage II', 'Stage III']
PRIORITY_PAIRS = [(1, 2), (2, 1), (2, 3), (3, 2)]  # (true, pred) priority errors


def _save(fig, path):
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def collect_misclassified(results):
    """Split results into correct and misclassified cases.

    Returns
    -------
    misclassified : list[dict]  sorted by error priority
    correct       : list[dict]
    """
    misclassified = []
    correct = []
    for r in results:
        if r['true_label'] != r['pred_label']:
            misclassified.append(r)
        else:
            correct.append(r)

    # Sort: priority pairs first, then by |true - pred| descending
    def _sort_key(r):
        pair = (r['true_label'], r['pred_label'])
        if pair in PRIORITY_PAIRS:
            return (0, PRIORITY_PAIRS.index(pair))
        return (1, -abs(r['true_label'] - r['pred_label']))

    misclassified.sort(key=_sort_key)
    return misclassified, correct


def save_misclassified_csv(misclassified, save_path, num_classes=4):
    """Save misclassified case summary as CSV."""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['case_id', 'true_label', 'pred_label', 'error_type',
                  'priority', 'num_slabs']
        for c in range(num_classes):
            header.append('prob_c{}'.format(c))
        header += ['soft_ratio', 'score_mean', 'topk_mean', 'path']
        writer.writerow(header)

        for r in misclassified:
            true_l = r['true_label']
            pred_l = r['pred_label']
            pair = (true_l, pred_l)
            priority = PRIORITY_PAIRS.index(pair) + 1 if pair in PRIORITY_PAIRS else 99
            error_type = '{}->{}'.format(true_l, pred_l)
            probs = r['corn_probs']
            burden = r.get('burden_stats') or {}
            row = [
                r['case_id'], true_l, pred_l, error_type, priority,
                len(r.get('metadata', [])),
            ]
            row += ['{:.4f}'.format(float(p)) for p in probs]
            row += [
                '{:.4f}'.format(burden.get('soft_ratio', 0.0)),
                '{:.4f}'.format(burden.get('score_mean', 0.0)),
                '{:.4f}'.format(burden.get('topk_mean', 0.0)),
                r['path'],
            ]
            writer.writerow(row)


def plot_misclassified_comparison(results, save_path, num_classes=4, max_cases=20):
    """Bar chart comparing probability distributions: correct vs misclassified."""
    mis = [r for r in results if r['true_label'] != r['pred_label']]
    cor = [r for r in results if r['true_label'] == r['pred_label']]

    if len(mis) == 0:
        return

    # Mean CORN probs for correct and wrong cases, grouped by true class
    fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4),
                             sharey=False)
    if num_classes == 1:
        axes = [axes]

    for c in range(num_classes):
        ax = axes[c]
        mis_c = [r for r in mis if r['true_label'] == c]
        cor_c = [r for r in cor if r['true_label'] == c]

        x = np.arange(num_classes)
        width = 0.35

        if cor_c:
            mean_cor = np.mean([r['corn_probs'] for r in cor_c], axis=0)
            ax.bar(x - width / 2, mean_cor, width, label='Correct (n={})'.format(len(cor_c)),
                   color='steelblue', alpha=0.8)
        if mis_c:
            mean_mis = np.mean([r['corn_probs'] for r in mis_c], axis=0)
            ax.bar(x + width / 2, mean_mis, width, label='Wrong (n={})'.format(len(mis_c)),
                   color='tomato', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([CLASS_NAMES[i] if i < len(CLASS_NAMES) else 'C{}'.format(i)
                            for i in range(num_classes)], fontsize=7, rotation=25)
        ax.set_title('{}'.format(CLASS_NAMES[c] if c < len(CLASS_NAMES) else 'Class {}'.format(c)),
                     fontsize=10)
        ax.set_ylabel('Mean Probability')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Mean CORN Probs: Correct vs Misclassified (per true class)', fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)


def plot_error_confusion(results, save_path, num_classes=4):
    """Stacked bar showing prediction distribution for each true class."""
    class_colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for r in results:
        t, p = r['true_label'], r['pred_label']
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion[t, p] += 1

    x = np.arange(num_classes)
    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = np.zeros(num_classes)
    for p in range(num_classes):
        vals = confusion[:, p].astype(float)
        label = CLASS_NAMES[p] if p < len(CLASS_NAMES) else 'Pred {}'.format(p)
        ax.bar(x, vals, bottom=bottoms, label=label,
               color=class_colors[p % len(class_colors)], alpha=0.85)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_NAMES[c] if c < len(CLASS_NAMES) else 'C{}'.format(c)
                        for c in range(num_classes)])
    ax.set_xlabel('True label')
    ax.set_ylabel('Count')
    ax.set_title('Prediction distribution per true class (stacked)')
    ax.legend(title='Predicted', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)


def generate_misclassified_reports(results, save_dir, model=None,
                                   num_classes=4, top_k=5, run_gradcam=True,
                                   device=None, verbose=True):
    """Generate full misclassified case analysis.

    Parameters
    ----------
    results      : list[dict] from inference_engine.run_inference
    save_dir     : str
    model        : Attention model (for Grad-CAM), or None to skip
    num_classes  : int
    top_k        : int  top-K slabs to visualize
    run_gradcam  : bool
    """
    os.makedirs(save_dir, exist_ok=True)

    misclassified, correct = collect_misclassified(results)

    if verbose:
        print('  Misclassified: {}/{} cases'.format(len(misclassified), len(results)))

    # ── Summary CSV ─────────────────────────────────────────────────────────
    save_misclassified_csv(
        misclassified, os.path.join(save_dir, 'misclassified_summary.csv'),
        num_classes=num_classes)

    # ── Comparison plots ─────────────────────────────────────────────────────
    plot_misclassified_comparison(
        results, os.path.join(save_dir, 'prob_comparison.png'),
        num_classes=num_classes)

    plot_error_confusion(
        results, os.path.join(save_dir, 'error_confusion.png'),
        num_classes=num_classes)

    # ── Per-case reports ─────────────────────────────────────────────────────
    for i, case_dict in enumerate(misclassified):
        cid = case_dict['case_id']
        true_l = case_dict['true_label']
        pred_l = case_dict['pred_label']
        prefix = '{}_t{}_p{}'.format(cid, true_l, pred_l)

        case_dir = os.path.join(save_dir, prefix)
        os.makedirs(case_dir, exist_ok=True)

        if verbose:
            print('  mis case [{}/{}] {} true={} pred={}'.format(
                i + 1, len(misclassified), cid, true_l, pred_l))

        # Slab importance curve
        try:
            plot_slab_importance_curve(
                case_dict, os.path.join(case_dir, 'slab_importance_curve.png'))
        except Exception as e:
            print('    [slab_curve] {}: {}'.format(cid, e))

        # Six-region bar
        try:
            plot_six_region_bar(
                case_dict, os.path.join(case_dir, 'six_region_bar.png'))
        except Exception as e:
            print('    [region_bar] {}: {}'.format(cid, e))

        # Top-K montage
        try:
            plot_topk_slab_montage(
                case_dict, os.path.join(case_dir, 'topk_slabs.png'), top_k=top_k)
        except Exception as e:
            print('    [montage] {}: {}'.format(cid, e))

        # Grad-CAM (only for misclassified, saves time)
        if run_gradcam and model is not None:
            try:
                from .gradcam import run_gradcam_for_case
                gradcam_dir = os.path.join(case_dir, 'gradcam')
                run_gradcam_for_case(
                    case_dict, model, gradcam_dir, top_k=top_k, device=device)
            except Exception as e:
                print('    [gradcam] {}: {}'.format(cid, e))
