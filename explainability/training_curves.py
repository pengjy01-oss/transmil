"""Enhanced training curve visualizations from CSV metrics files.

Reads the training_metrics_*.csv files and produces:
  - Loss curves (train/val)
  - Accuracy curves
  - Burden soft_ratio curves
  - Confusion matrix
  - Per-class precision/recall/F1
  - Class distribution comparison
"""

from __future__ import print_function

import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

CLASS_NAMES = ['Stage 0', 'Stage I', 'Stage II', 'Stage III']


def _save(fig, path):
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def _read_csv(csv_path):
    """Read training metrics CSV, return dict of lists (numeric only)."""
    epochs, rows = [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = row.get('epoch', '')
            if ep in ('', 'test'):
                continue
            try:
                epochs.append(int(ep))
                rows.append(row)
            except ValueError:
                continue
    return epochs, rows


def _col(rows, key):
    vals = []
    for r in rows:
        v = r.get(key, '')
        vals.append(float(v) if v not in ('', 'None', None) else None)
    return vals


def _clean_pairs(xs, ys):
    px, py = [], []
    for x, y in zip(xs, ys):
        if y is not None:
            px.append(x)
            py.append(y)
    return px, py


def plot_loss_curves(csv_path, save_dir):
    """Save loss curve PNG."""
    epochs, rows = _read_csv(csv_path)
    if not epochs:
        return

    train_loss = _col(rows, 'train_loss')
    val_loss = _col(rows, 'val_loss')

    fig, ax = plt.subplots(figsize=(9, 4))
    ex, ey = _clean_pairs(epochs, train_loss)
    if ey:
        ax.plot(ex, ey, 'o-', label='Train Loss', markersize=3, linewidth=1.5)
    ex, ey = _clean_pairs(epochs, val_loss)
    if ey:
        ax.plot(ex, ey, 's--', label='Val Loss', markersize=3, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _save(fig, os.path.join(save_dir, 'loss_curves.png'))


def plot_accuracy_curves(csv_path, save_dir):
    """Save accuracy curve PNG."""
    epochs, rows = _read_csv(csv_path)
    if not epochs:
        return

    train_acc = _col(rows, 'train_acc')
    val_acc = _col(rows, 'val_acc')

    fig, ax = plt.subplots(figsize=(9, 4))
    ex, ey = _clean_pairs(epochs, train_acc)
    if ey:
        ax.plot(ex, ey, 'o-', label='Train Acc', markersize=3, linewidth=1.5)
    ex, ey = _clean_pairs(epochs, val_acc)
    if ey:
        ax.plot(ex, ey, 's--', label='Val Acc', markersize=3, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training & Validation Accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _save(fig, os.path.join(save_dir, 'accuracy_curves.png'))


def plot_burden_curves(csv_path, save_dir):
    """Save burden (soft_ratio, score_mean, topk_mean, score_std) curves."""
    epochs, rows = _read_csv(csv_path)
    if not epochs:
        return

    keys = ['soft_ratio', 'score_mean', 'topk_mean', 'score_std']
    train_vals = {k: _col(rows, 'train_' + k) for k in keys}
    val_vals = {k: _col(rows, 'val_' + k) for k in keys}

    has_any = any(any(v is not None for v in vals) for vals in train_vals.values())
    if not has_any:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = {
        'soft_ratio': 'Soft Ratio (burden)',
        'score_mean': 'Score Mean',
        'topk_mean': 'TopK Mean',
        'score_std': 'Score Std',
    }
    for i, k in enumerate(keys):
        ax = axes[i // 2, i % 2]
        ex, ey = _clean_pairs(epochs, train_vals[k])
        if ey:
            ax.plot(ex, ey, 'o-', label='Train', markersize=3)
        ex, ey = _clean_pairs(epochs, val_vals[k])
        if ey:
            ax.plot(ex, ey, 's--', label='Val', markersize=3)
        ax.set_title(titles.get(k, k))
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Burden Feature Curves', fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'burden_curves.png'))


def plot_confusion_matrix(y_true, y_pred, num_classes, save_path,
                          title='Confusion Matrix', split_name=''):
    """Confusion matrix heatmap with counts and row-normalized %."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm / row_sums * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pct, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label='%')

    labels = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else 'C{}'.format(c)
              for c in range(num_classes)]
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    full_title = '{}{}'.format(title, ' ({})'.format(split_name) if split_name else '')
    ax.set_title(full_title, fontsize=12)

    thresh = cm_pct.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            color = 'white' if cm_pct[i, j] > thresh else 'black'
            ax.text(j, i, '{}\n({:.1f}%)'.format(cm[i, j], cm_pct[i, j]),
                    ha='center', va='center', color=color, fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)


def plot_per_class_metrics(y_true, y_pred, num_classes, save_path,
                           title='Per-Class Metrics'):
    """Bar chart of precision, recall, F1 for each class."""
    precision_arr, recall_arr, f1_arr = [], [], []
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precision_arr.append(p)
        recall_arr.append(r)
        f1_arr.append(f)

    x = np.arange(num_classes)
    width = 0.25
    labels = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else 'C{}'.format(c)
              for c in range(num_classes)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, precision_arr, width, label='Precision', color='#4C72B0')
    ax.bar(x, recall_arr, width, label='Recall', color='#55A868')
    ax.bar(x + width, f1_arr, width, label='F1', color='#C44E52')
    for i in range(num_classes):
        ax.text(i - width, precision_arr[i] + 0.02,
                '{:.2f}'.format(precision_arr[i]), ha='center', fontsize=8)
        ax.text(i, recall_arr[i] + 0.02,
                '{:.2f}'.format(recall_arr[i]), ha='center', fontsize=8)
        ax.text(i + width, f1_arr[i] + 0.02,
                '{:.2f}'.format(f1_arr[i]), ha='center', fontsize=8)
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.18)
    ax.grid(True, axis='y', alpha=0.3)
    # Macro F1 annotation
    macro_f1 = float(np.mean(f1_arr))
    ax.text(0.98, 0.98, 'Macro F1: {:.4f}'.format(macro_f1),
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    fig.tight_layout()
    _save(fig, save_path)


def plot_class_distribution(y_true, y_pred, num_classes, save_path,
                            title='Class Distribution'):
    """Side-by-side bar chart of true vs predicted distributions."""
    labels = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else 'C{}'.format(c)
              for c in range(num_classes)]
    true_counts = np.bincount(y_true, minlength=num_classes)
    pred_counts = np.bincount(y_pred, minlength=num_classes)

    x = np.arange(num_classes)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - width / 2, true_counts, width, label='True', color='#4C72B0', alpha=0.85)
    b2 = ax.bar(x + width / 2, pred_counts, width, label='Predicted', color='#C44E52', alpha=0.85)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(int(bar.get_height())), ha='center', fontsize=9)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)


def generate_training_visualizations(csv_path, y_true_test, y_pred_test,
                                     num_classes, save_dir,
                                     y_true_val=None, y_pred_val=None,
                                     verbose=True):
    """Produce all training-phase visualizations from a CSV + arrays.

    Parameters
    ----------
    csv_path      : str  path to training_metrics_*.csv
    y_true_test   : np.ndarray  true labels for test set
    y_pred_test   : np.ndarray  predicted labels for test set
    num_classes   : int
    save_dir      : str
    y_true_val    : np.ndarray or None
    y_pred_val    : np.ndarray or None
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Curve plots from CSV ─────────────────────────────────────────────────
    if csv_path and os.path.exists(csv_path):
        if verbose:
            print('  loss curves...')
        plot_loss_curves(csv_path, save_dir)
        if verbose:
            print('  accuracy curves...')
        plot_accuracy_curves(csv_path, save_dir)
        if verbose:
            print('  burden curves...')
        plot_burden_curves(csv_path, save_dir)

    # ── Test set metrics ─────────────────────────────────────────────────────
    if y_true_test is not None and len(y_true_test) > 0:
        y_t = np.asarray(y_true_test, dtype=np.int64)
        y_p = np.asarray(y_pred_test, dtype=np.int64)

        if verbose:
            print('  test confusion matrix...')
        plot_confusion_matrix(y_t, y_p, num_classes,
                              os.path.join(save_dir, 'confusion_matrix_test.png'),
                              title='Confusion Matrix', split_name='Test')
        if verbose:
            print('  test per-class metrics...')
        plot_per_class_metrics(y_t, y_p, num_classes,
                               os.path.join(save_dir, 'per_class_metrics_test.png'),
                               title='Per-Class Metrics (Test)')
        if verbose:
            print('  test class distribution...')
        plot_class_distribution(y_t, y_p, num_classes,
                                os.path.join(save_dir, 'class_distribution_test.png'),
                                title='Class Distribution (Test)')

    # ── Validation set metrics ───────────────────────────────────────────────
    if y_true_val is not None and len(y_true_val) > 0:
        y_tv = np.asarray(y_true_val, dtype=np.int64)
        y_pv = np.asarray(y_pred_val, dtype=np.int64)

        if verbose:
            print('  val confusion matrix...')
        plot_confusion_matrix(y_tv, y_pv, num_classes,
                              os.path.join(save_dir, 'confusion_matrix_val.png'),
                              title='Confusion Matrix', split_name='Val')
        plot_per_class_metrics(y_tv, y_pv, num_classes,
                               os.path.join(save_dir, 'per_class_metrics_val.png'),
                               title='Per-Class Metrics (Val)')
