"""Post-training visualization: confusion matrix, training curves, per-class metrics."""

from __future__ import print_function

import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ['Stage 0', 'Stage I', 'Stage II', 'Stage III']


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: {}'.format(path))


def plot_confusion_matrix(y_true, y_pred, num_classes, save_path, title='Confusion Matrix'):
    """Confusion matrix heatmap with counts and percentages."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pct, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label='%')

    labels = CLASS_NAMES[:num_classes]
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=13)

    thresh = cm_pct.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            color = 'white' if cm_pct[i, j] > thresh else 'black'
            ax.text(j, i, '{}\n({:.1f}%)'.format(cm[i, j], cm_pct[i, j]),
                    ha='center', va='center', color=color, fontsize=10)

    _save(fig, save_path)


def plot_training_curves(csv_path, save_dir):
    """Loss / accuracy / burden curves from the metrics CSV."""
    epochs, train_loss, val_loss = [], [], []
    train_acc, val_acc = [], []
    train_sr, val_sr = [], []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = row.get('epoch', '')
            if ep == 'test' or ep == '':
                continue
            epochs.append(int(ep))
            train_loss.append(float(row['train_loss']) if row.get('train_loss') else None)
            val_loss.append(float(row['val_loss']) if row.get('val_loss') else None)
            train_acc.append(float(row['train_acc']) if row.get('train_acc') else None)
            val_acc.append(float(row['val_acc']) if row.get('val_acc') else None)
            train_sr.append(float(row['train_soft_ratio']) if row.get('train_soft_ratio') else None)
            val_sr.append(float(row['val_soft_ratio']) if row.get('val_soft_ratio') else None)

    if not epochs:
        return

    def _clean(arr):
        return [v for v in arr if v is not None]

    def _clean_pairs(xs, ys):
        px, py = [], []
        for x, y in zip(xs, ys):
            if y is not None:
                px.append(x)
                py.append(y)
        return px, py

    # Loss curves
    fig, ax = plt.subplots(figsize=(8, 4))
    ex, ey = _clean_pairs(epochs, train_loss)
    if ey:
        ax.plot(ex, ey, 'o-', label='Train Loss', markersize=3)
    ex, ey = _clean_pairs(epochs, val_loss)
    if ey:
        ax.plot(ex, ey, 's-', label='Val Loss', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(save_dir, 'loss_curves.png'))

    # Accuracy curves
    fig, ax = plt.subplots(figsize=(8, 4))
    ex, ey = _clean_pairs(epochs, train_acc)
    if ey:
        ax.plot(ex, ey, 'o-', label='Train Acc', markersize=3)
    ex, ey = _clean_pairs(epochs, val_acc)
    if ey:
        ax.plot(ex, ey, 's-', label='Val Acc', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training & Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    _save(fig, os.path.join(save_dir, 'accuracy_curves.png'))

    # Burden soft_ratio curves
    if any(v is not None for v in train_sr):
        fig, ax = plt.subplots(figsize=(8, 4))
        ex, ey = _clean_pairs(epochs, train_sr)
        if ey:
            ax.plot(ex, ey, 'o-', label='Train Soft Ratio', markersize=3)
        ex, ey = _clean_pairs(epochs, val_sr)
        if ey:
            ax.plot(ex, ey, 's-', label='Val Soft Ratio', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Soft Ratio')
        ax.set_title('Burden Soft Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, os.path.join(save_dir, 'burden_soft_ratio.png'))


def plot_per_class_metrics(y_true, y_pred, num_classes, save_path, title='Per-Class Metrics'):
    """Bar chart of precision, recall, F1 for each class."""
    labels = CLASS_NAMES[:num_classes]
    precision_arr = []
    recall_arr = []
    f1_arr = []

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
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, precision_arr, width, label='Precision', color='#4C72B0')
    ax.bar(x, recall_arr, width, label='Recall', color='#55A868')
    ax.bar(x + width, f1_arr, width, label='F1', color='#C44E52')

    for i in range(num_classes):
        ax.text(i - width, precision_arr[i] + 0.02, '{:.2f}'.format(precision_arr[i]), ha='center', fontsize=8)
        ax.text(i, recall_arr[i] + 0.02, '{:.2f}'.format(recall_arr[i]), ha='center', fontsize=8)
        ax.text(i + width, f1_arr[i] + 0.02, '{:.2f}'.format(f1_arr[i]), ha='center', fontsize=8)

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3)
    _save(fig, save_path)


def plot_class_distribution(y_true, y_pred, num_classes, save_path, title='Class Distribution'):
    """Side-by-side bar chart of true vs predicted class distributions."""
    labels = CLASS_NAMES[:num_classes]
    true_counts = np.bincount(y_true, minlength=num_classes)
    pred_counts = np.bincount(y_pred, minlength=num_classes)

    x = np.arange(num_classes)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, true_counts, width, label='True', color='#4C72B0')
    bars2 = ax.bar(x + width / 2, pred_counts, width, label='Predicted', color='#C44E52')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(int(bar.get_height())), ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(int(bar.get_height())), ha='center', fontsize=9)

    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    _save(fig, save_path)


def plot_error_analysis(y_true, y_pred, num_classes, save_path):
    """Off-by-1 vs off-by-2+ error breakdown (important for ordinal tasks)."""
    labels = CLASS_NAMES[:num_classes]
    correct = 0
    off1 = 0
    off2plus = 0
    per_class_correct = np.zeros(num_classes)
    per_class_off1 = np.zeros(num_classes)
    per_class_off2 = np.zeros(num_classes)

    for t, p in zip(y_true, y_pred):
        diff = abs(int(t) - int(p))
        if diff == 0:
            correct += 1
            per_class_correct[t] += 1
        elif diff == 1:
            off1 += 1
            per_class_off1[t] += 1
        else:
            off2plus += 1
            per_class_off2[t] += 1

    total = len(y_true)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Overall pie
    sizes = [correct, off1, off2plus]
    pie_labels = [
        'Correct\n{} ({:.1f}%)'.format(correct, 100 * correct / total if total else 0),
        'Off-by-1\n{} ({:.1f}%)'.format(off1, 100 * off1 / total if total else 0),
        'Off-by-2+\n{} ({:.1f}%)'.format(off2plus, 100 * off2plus / total if total else 0),
    ]
    colors = ['#55A868', '#FFD92F', '#C44E52']
    axes[0].pie(sizes, labels=pie_labels, colors=colors, startangle=90)
    axes[0].set_title('Overall Error Breakdown (n={})'.format(total))

    # Per-class stacked bar
    x = np.arange(num_classes)
    width = 0.5
    axes[1].bar(x, per_class_correct, width, label='Correct', color='#55A868')
    axes[1].bar(x, per_class_off1, width, bottom=per_class_correct, label='Off-by-1', color='#FFD92F')
    axes[1].bar(x, per_class_off2, width, bottom=per_class_correct + per_class_off1, label='Off-by-2+', color='#C44E52')
    axes[1].set_xlabel('True Class')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Per-Class Error Breakdown')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    axes[1].grid(True, axis='y', alpha=0.3)

    _save(fig, save_path)


def generate_all_plots(csv_path, y_true_test, y_pred_test, num_classes, save_dir,
                       y_true_val=None, y_pred_val=None):
    """Generate all training result visualizations to save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    print('\nGenerating training result plots -> {}'.format(save_dir))

    # Training curves from CSV
    if csv_path and os.path.exists(csv_path):
        plot_training_curves(csv_path, save_dir)

    # Test set confusion matrix
    if len(y_true_test) > 0:
        plot_confusion_matrix(y_true_test, y_pred_test, num_classes,
                              os.path.join(save_dir, 'test_confusion_matrix.png'),
                              title='Test Confusion Matrix')
        plot_per_class_metrics(y_true_test, y_pred_test, num_classes,
                               os.path.join(save_dir, 'test_per_class_metrics.png'),
                               title='Test Per-Class Metrics')
        plot_class_distribution(y_true_test, y_pred_test, num_classes,
                                os.path.join(save_dir, 'test_class_distribution.png'),
                                title='Test Class Distribution')
        plot_error_analysis(y_true_test, y_pred_test, num_classes,
                            os.path.join(save_dir, 'test_error_analysis.png'))

    # Validation set confusion matrix (if available)
    if y_true_val is not None and len(y_true_val) > 0:
        plot_confusion_matrix(y_true_val, y_pred_val, num_classes,
                              os.path.join(save_dir, 'val_confusion_matrix.png'),
                              title='Validation Confusion Matrix')
        plot_per_class_metrics(y_true_val, y_pred_val, num_classes,
                               os.path.join(save_dir, 'val_per_class_metrics.png'),
                               title='Validation Per-Class Metrics')

    print('All plots saved to {}'.format(save_dir))
