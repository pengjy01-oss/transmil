"""Embedding space visualization: t-SNE, PCA (and UMAP if installed).

Collects bag-level features (Z_final) and visualizes them in 2D,
colored by true label and predicted label.
"""

from __future__ import print_function

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CLASS_NAMES = ['Stage 0', 'Stage I', 'Stage II', 'Stage III']
CLASS_COLORS = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']
CLASS_MARKERS = ['o', 's', '^', 'D']


def _save(fig, path):
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _collect_features(results):
    """Return (features_2d_list, true_labels, pred_labels, case_ids) arrays."""
    feats, trues, preds, case_ids = [], [], [], []
    for r in results:
        f = r.get('bag_features')
        if f is None:
            continue
        if np.any(~np.isfinite(f)):
            continue
        feats.append(f)
        trues.append(r['true_label'])
        preds.append(r['pred_label'])
        case_ids.append(r['case_id'])
    if len(feats) == 0:
        return None, None, None, None
    return np.stack(feats, axis=0), np.array(trues), np.array(preds), case_ids


def _scatter_2d(ax, coords, labels, num_classes, title='', show_legend=True):
    """Scatter 2-D embeddings colored by labels."""
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        color = CLASS_COLORS[c % len(CLASS_COLORS)]
        marker = CLASS_MARKERS[c % len(CLASS_MARKERS)]
        label_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else 'C{}'.format(c)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, marker=marker, s=60, alpha=0.75,
                   edgecolors='white', linewidths=0.4,
                   label='{} (n={})'.format(label_name, mask.sum()))

    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    if show_legend:
        ax.legend(fontsize=8, markerscale=1.2)
    ax.grid(True, alpha=0.25)


def _scatter_correct_wrong(ax, coords, trues, preds, title=''):
    """Scatter, marking correctly vs. wrongly classified."""
    correct = trues == preds
    ax.scatter(coords[correct, 0], coords[correct, 1],
               c='steelblue', s=55, alpha=0.7, marker='o',
               edgecolors='white', linewidths=0.3, label='Correct (n={})'.format(correct.sum()))
    ax.scatter(coords[~correct, 0], coords[~correct, 1],
               c='tomato', s=80, alpha=0.9, marker='X',
               edgecolors='black', linewidths=0.4, label='Wrong (n={})'.format((~correct).sum()))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)


# ── PCA ──────────────────────────────────────────────────────────────────────

def _pca_2d(X):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X), pca


def plot_pca(results, save_dir, num_classes=4, verbose=True):
    """PCA embedding plots."""
    X, trues, preds, case_ids = _collect_features(results)
    if X is None:
        if verbose:
            print('  [PCA] No bag features collected, skipping.')
        return

    if verbose:
        print('  PCA on {} samples with {} dims...'.format(X.shape[0], X.shape[1]))

    try:
        coords, pca = _pca_2d(X)
    except Exception as e:
        print('  [PCA] failed: {}'.format(e))
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _scatter_2d(axes[0], coords, trues, num_classes,
                title='PCA – colored by True Label')
    _scatter_2d(axes[1], coords, preds, num_classes,
                title='PCA – colored by Predicted Label')
    _scatter_correct_wrong(axes[2], coords, trues, preds,
                           title='PCA – Correct vs Wrong')
    fig.suptitle('Bag-Level Feature PCA (Z_final)', fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'pca_embeddings.png'))

    # Per-class pairwise overlay (true vs pred)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    for c in range(num_classes):
        mask = trues == c
        if mask.sum() == 0:
            continue
        color = CLASS_COLORS[c % len(CLASS_COLORS)]
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else 'C{}'.format(c)
        # Draw ellipse or just scatter
        ax2.scatter(coords[mask, 0], coords[mask, 1],
                    c=color, s=80, alpha=0.7, marker=CLASS_MARKERS[c % len(CLASS_MARKERS)],
                    edgecolors='white', linewidths=0.4,
                    label='{} (n={})'.format(name, mask.sum()))
    ax2.set_title('PCA – True Labels', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    _save(fig2, os.path.join(save_dir, 'pca_true_labels.png'))


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def plot_tsne(results, save_dir, num_classes=4, perplexity=None, n_iter=1000,
              verbose=True):
    """t-SNE embedding plots."""
    X, trues, preds, case_ids = _collect_features(results)
    if X is None:
        if verbose:
            print('  [t-SNE] No bag features collected, skipping.')
        return

    n = X.shape[0]
    if n < 4:
        if verbose:
            print('  [t-SNE] Too few samples ({}) for t-SNE.'.format(n))
        return

    if perplexity is None:
        perplexity = min(max(5, n // 5), 50)

    if verbose:
        print('  t-SNE on {} samples (perplexity={})...'.format(n, perplexity))

    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=float(perplexity),
                    n_iter=int(n_iter), random_state=42, init='pca',
                    learning_rate='auto')
        coords = tsne.fit_transform(X)
    except Exception as e:
        print('  [t-SNE] failed: {}'.format(e))
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _scatter_2d(axes[0], coords, trues, num_classes,
                title='t-SNE – colored by True Label')
    _scatter_2d(axes[1], coords, preds, num_classes,
                title='t-SNE – colored by Predicted Label')
    _scatter_correct_wrong(axes[2], coords, trues, preds,
                           title='t-SNE – Correct vs Wrong')
    fig.suptitle('Bag-Level Feature t-SNE (Z_final)', fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'tsne_embeddings.png'))


# ── UMAP ─────────────────────────────────────────────────────────────────────

def plot_umap(results, save_dir, num_classes=4, n_neighbors=15, min_dist=0.1,
              verbose=True):
    """UMAP embedding plots (requires ``umap-learn`` package)."""
    try:
        import umap
    except ImportError:
        if verbose:
            print('  [UMAP] umap-learn not installed, skipping. '
                  'Install with: pip install umap-learn')
        return

    X, trues, preds, case_ids = _collect_features(results)
    if X is None:
        if verbose:
            print('  [UMAP] No bag features collected, skipping.')
        return

    n = X.shape[0]
    if n < 4:
        return

    if verbose:
        print('  UMAP on {} samples...'.format(n))

    try:
        reducer = umap.UMAP(n_components=2, n_neighbors=min(n_neighbors, n - 1),
                            min_dist=float(min_dist), random_state=42)
        coords = reducer.fit_transform(X)
    except Exception as e:
        print('  [UMAP] failed: {}'.format(e))
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _scatter_2d(axes[0], coords, trues, num_classes,
                title='UMAP – colored by True Label')
    _scatter_2d(axes[1], coords, preds, num_classes,
                title='UMAP – colored by Predicted Label')
    _scatter_correct_wrong(axes[2], coords, trues, preds,
                           title='UMAP – Correct vs Wrong')
    fig.suptitle('Bag-Level Feature UMAP (Z_final)', fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'umap_embeddings.png'))


# ── Main entry ────────────────────────────────────────────────────────────────

def generate_embedding_visualizations(results, save_dir, num_classes=4,
                                      run_umap=True, verbose=True):
    """Generate PCA, t-SNE, and optionally UMAP plots.

    Parameters
    ----------
    results  : list[dict] from inference_engine.run_inference
    save_dir : str
    """
    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print('  PCA...')
    plot_pca(results, save_dir, num_classes=num_classes, verbose=verbose)

    if verbose:
        print('  t-SNE...')
    plot_tsne(results, save_dir, num_classes=num_classes, verbose=verbose)

    if run_umap:
        if verbose:
            print('  UMAP...')
        plot_umap(results, save_dir, num_classes=num_classes, verbose=verbose)
