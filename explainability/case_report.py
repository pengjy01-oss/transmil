"""Per-case report: prediction card with probabilities, burden stats, aux12 probs.

Saves a PNG figure for each case summarizing the bag-level prediction.
"""

from __future__ import print_function

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ['Stage 0', 'Stage I', 'Stage II', 'Stage III']
CLASS_COLORS = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']


def _save(fig, path):
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_case_card(case_dict, save_path, num_classes=4):
    """Save a multi-panel prediction card for one case.

    Panels:
      Row 1: CORN probabilities bar | burden/coverage stats text
      Row 2: Aux-12 probs (if available)
    """
    case_id = case_dict['case_id']
    true_label = case_dict['true_label']
    pred_label = case_dict['pred_label']
    corn_probs = case_dict['corn_probs']          # [num_classes]
    burden_stats = case_dict.get('burden_stats') or {}
    coverage_stats = case_dict.get('coverage_stats') or {}
    aux12_probs = case_dict.get('aux12_probs')    # [12] or None

    correct = (true_label == pred_label)
    header_color = '#2ecc71' if correct else '#e74c3c'

    has_aux12 = (aux12_probs is not None and len(aux12_probs) == 12)
    nrows = 2 if has_aux12 else 1
    fig_h = 4.5 * nrows + 1.5

    fig = plt.figure(figsize=(14, fig_h))
    gs = fig.add_gridspec(nrows, 2, hspace=0.5, wspace=0.4,
                          top=0.88, bottom=0.08, left=0.07, right=0.97)

    # ── Header ──────────────────────────────────────────────────────────────
    status = 'CORRECT' if correct else 'WRONG'
    title = '{} | True={} ({}) | Pred={} ({}) | {}'.format(
        case_id,
        true_label, CLASS_NAMES[true_label] if true_label < len(CLASS_NAMES) else '',
        pred_label, CLASS_NAMES[pred_label] if pred_label < len(CLASS_NAMES) else '',
        status,
    )
    fig.suptitle(title, fontsize=12, color=header_color, fontweight='bold', y=0.97)

    # ── Panel A: CORN class probabilities ────────────────────────────────────
    ax_prob = fig.add_subplot(gs[0, 0])
    labels = CLASS_NAMES[:num_classes]
    colors = [header_color if i == true_label else CLASS_COLORS[i] for i in range(num_classes)]
    bars = ax_prob.bar(labels, corn_probs, color=colors, alpha=0.85, edgecolor='white')
    ax_prob.set_ylim(0, 1.05)
    ax_prob.set_ylabel('Probability')
    ax_prob.set_title('CORN Class Probabilities', fontsize=10)
    ax_prob.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    for bar, prob in zip(bars, corn_probs):
        ax_prob.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     '{:.3f}'.format(prob), ha='center', va='bottom', fontsize=9)
    # Mark true class
    ax_prob.get_xticklabels()[true_label].set_color('#27ae60')
    ax_prob.get_xticklabels()[true_label].set_fontweight('bold')

    # ── Panel B: Stats text ──────────────────────────────────────────────────
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_stats.axis('off')

    lines = [
        ('Case ID', case_id),
        ('True label', '{} - {}'.format(true_label,
                                        CLASS_NAMES[true_label] if true_label < len(CLASS_NAMES) else '')),
        ('Predicted', '{} - {}'.format(pred_label,
                                       CLASS_NAMES[pred_label] if pred_label < len(CLASS_NAMES) else '')),
        ('Result', status),
    ]
    if burden_stats:
        lines += [
            ('── Burden ──', ''),
            ('soft_ratio', '{:.4f}'.format(burden_stats.get('soft_ratio', 0.0))),
            ('score_mean', '{:.4f}'.format(burden_stats.get('score_mean', 0.0))),
            ('topk_mean', '{:.4f}'.format(burden_stats.get('topk_mean', 0.0))),
            ('score_std', '{:.4f}'.format(burden_stats.get('score_std', 0.0))),
        ]
    if coverage_stats:
        lines += [
            ('── Coverage ──', ''),
            ('z_center', '{:.4f}'.format(coverage_stats.get('z_center', 0.0))),
            ('z_spread', '{:.4f}'.format(coverage_stats.get('z_spread', 0.0))),
            ('active_bins', '{:.4f}'.format(coverage_stats.get('active_bins_soft', 0.0))),
        ]

    y_pos = 0.97
    for key, val in lines:
        if key.startswith('──'):
            ax_stats.text(0.0, y_pos, key, transform=ax_stats.transAxes,
                          fontsize=9, color='gray', fontstyle='italic')
        elif val == '':
            pass
        else:
            ax_stats.text(0.0, y_pos, '{:15s}: {}'.format(key, val),
                          transform=ax_stats.transAxes, fontsize=9,
                          fontfamily='monospace')
        y_pos -= 0.09

    # ── Panel C: Aux-12 probs (if available) ────────────────────────────────
    if has_aux12:
        ax_aux = fig.add_subplot(gs[1, :])
        x12 = np.arange(12)
        colors12 = plt.cm.tab20(np.linspace(0, 1, 12))
        ax_aux.bar(x12, aux12_probs, color=colors12, alpha=0.85, edgecolor='white')
        ax_aux.set_xticks(x12)
        ax_aux.set_xticklabels(['Sub{}'.format(i) for i in range(12)], fontsize=8)
        ax_aux.set_ylabel('Probability')
        ax_aux.set_title('Auxiliary 12-Subtype Head', fontsize=10)
        ax_aux.set_ylim(0, max(float(aux12_probs.max()) * 1.15, 0.2))
        for i, p in enumerate(aux12_probs):
            ax_aux.text(i, p + 0.005, '{:.3f}'.format(p), ha='center', fontsize=7)

    _save(fig, save_path)


def save_case_json(case_dict, save_path, num_classes=4):
    """Save per-case summary as a JSON file."""
    burden = case_dict.get('burden_stats') or {}
    coverage = case_dict.get('coverage_stats') or {}
    summary = {
        'case_id': case_dict['case_id'],
        'path': case_dict['path'],
        'true_label': int(case_dict['true_label']),
        'pred_label': int(case_dict['pred_label']),
        'correct': bool(case_dict['true_label'] == case_dict['pred_label']),
        'corn_probs': [float(p) for p in case_dict['corn_probs']],
        'class_names': CLASS_NAMES[:num_classes],
        'burden_stats': {k: (float(v) if v is not None else None) for k, v in burden.items()},
        'coverage_stats': {k: (float(v) if v is not None else None) for k, v in coverage.items()},
        'num_slabs': int(len(case_dict.get('metadata', []))),
    }
    if case_dict.get('aux12_probs') is not None:
        summary['aux12_probs'] = [float(p) for p in case_dict['aux12_probs']]
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)


def generate_all_case_reports(results, save_dir, num_classes=4, verbose=True):
    """Generate a PNG card and JSON summary for every case.

    Parameters
    ----------
    results  : list[dict] from inference_engine.run_inference
    save_dir : str
    """
    os.makedirs(save_dir, exist_ok=True)
    n = len(results)
    for i, case_dict in enumerate(results):
        if verbose and i % max(1, n // 10) == 0:
            print('  case report [{}/{}]'.format(i + 1, n))
        cid = case_dict['case_id']
        true_l = case_dict['true_label']
        pred_l = case_dict['pred_label']
        prefix = '{}_t{}_p{}'.format(cid, true_l, pred_l)

        plot_case_card(case_dict, os.path.join(save_dir, prefix + '_card.png'),
                       num_classes=num_classes)
        save_case_json(case_dict, os.path.join(save_dir, prefix + '_summary.json'),
                       num_classes=num_classes)
