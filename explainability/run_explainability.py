"""Main entry point for MIL explainability and visualization.

Usage examples
--------------
# Full analysis on test split
python -m explainability.run_explainability \\
    --config configs/ct25d_resnet18_128.yaml \\
    --checkpoint checkpoints/best_model_resnet18_128_f1.pt \\
    --split test \\
    --out_dir explainability/results

# All splits
python -m explainability.run_explainability \\
    --config configs/ct25d_resnet18_128.yaml \\
    --checkpoint checkpoints/best_model_resnet18_128_f1.pt \\
    --split all

# With Grad-CAM (slower)
python -m explainability.run_explainability \\
    --config configs/ct25d_resnet18_128.yaml \\
    --checkpoint checkpoints/best_model_resnet18_128_f1.pt \\
    --gradcam \\
    --gradcam_topk 5

# From existing training CSV (training curve plots only, no inference)
python -m explainability.run_explainability \\
    --config configs/ct25d_resnet18_128.yaml \\
    --csv training_metrics_20260421_110907.csv \\
    --no_inference \\
    --out_dir explainability/results_curves
"""

from __future__ import print_function

import argparse
import os
import sys

# ── Allow running as ``python -m explainability.run_explainability`` or
#    ``python explainability/run_explainability.py``
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_this_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

import numpy as np
import torch
from datetime import datetime


# ── Argument parsing ─────────────────────────────────────────────────────────

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='MIL Explainability & Visualization System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Core config
    p.add_argument('--config', type=str, default='configs/ct25d_resnet18_128.yaml',
                   help='Path to training YAML config (same as used for training)')
    p.add_argument('--checkpoint', type=str, default='',
                   help='Path to model checkpoint (.pt). Empty = load from config best_model_path')
    p.add_argument('--out_dir', type=str, default='',
                   help='Root output directory. Default: explainability/results/TIMESTAMP')
    p.add_argument('--split', type=str, default='test',
                   choices=['train', 'val', 'test', 'all'],
                   help='Dataset split(s) to run inference on')

    # What to run
    p.add_argument('--no_inference', action='store_true', default=False,
                   help='Skip model inference; only generate training curve plots from CSV')
    p.add_argument('--csv', type=str, default='',
                   help='Training metrics CSV for training curve plots. '
                        'If empty, uses latest training_metrics_*.csv')
    p.add_argument('--gradcam', action='store_true', default=False,
                   help='Run Grad-CAM for top slabs (slower, ~10s/case)')
    p.add_argument('--gradcam_topk', type=int, default=5,
                   help='Number of top slabs to generate Grad-CAM for')
    p.add_argument('--slab_topk', type=int, default=6,
                   help='Number of top slabs in montage')
    p.add_argument('--no_embeddings', action='store_true', default=False,
                   help='Skip embedding visualization (t-SNE, PCA)')
    p.add_argument('--no_umap', action='store_true', default=False,
                   help='Skip UMAP (in addition to t-SNE/PCA)')
    p.add_argument('--no_case_reports', action='store_true', default=False,
                   help='Skip per-case report cards')
    p.add_argument('--no_slab_viz', action='store_true', default=False,
                   help='Skip per-case slab visualizations')
    p.add_argument('--max_cases', type=int, default=0,
                   help='Limit number of cases for debugging (0 = all)')
    p.add_argument('--no_gradcam_for_misclassified', action='store_true', default=False,
                   help='Disable Grad-CAM even for misclassified cases')
    return p


# ── CSV discovery ─────────────────────────────────────────────────────────────

def _find_latest_csv(search_dir='.'):
    """Find the most recently modified training_metrics_*.csv."""
    import glob
    pattern = os.path.join(search_dir, 'training_metrics_*.csv')
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


# ── Dataset builder (mirrors main.py logic) ───────────────────────────────────

def _build_single_dataset(args, split):
    """Build one CTPneNiiBags split, compatible with args from config."""
    from datasets.ct_pne_dataset import CTPneNiiBags

    if args.slab_depth < 1 or (args.slab_depth % 2) == 0:
        raise ValueError('slab_depth must be a positive odd integer')
    half_depth = args.slab_depth // 2
    channel_offsets = tuple(range(-half_depth, half_depth + 1))

    return CTPneNiiBags(
        root_dir=args.data_root,
        split=split,
        num_classes=args.num_classes,
        middle_ratio=args.middle_ratio,
        fixed_num_slices=args.fixed_num_slices,
        channel_offsets=channel_offsets,
        slab_stride=args.slab_stride,
        num_slabs=args.num_slabs,
        center_sampling_mode='uniform',          # deterministic for analysis
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        scale_to_unit=args.scale_to_unit,
        use_zscore=args.use_zscore,
        lung_hu_low=args.lung_hu_low,
        lung_hu_high=args.lung_hu_high,
        min_lung_area_ratio=args.min_lung_area_ratio,
        max_instances=args.max_instances,
        instance_definition=args.instance_definition,
        lung_mask_root=args.lung_mask_root,
        lung_mask_suffix=args.lung_mask_suffix,
        lung_mask_require=args.lung_mask_require,
        cache_root=args.cache_root,
        pseudo_mask_value_threshold=args.pseudo_mask_value_threshold,
        pseudo_mask_min_component_voxels=args.pseudo_mask_min_component_voxels,
        region_num_instances=args.region_num_instances,
        region_out_size=(args.region_out_h, args.region_out_w),
        region_bbox_margin=args.region_bbox_margin,
        region_bbox_min_size=args.region_bbox_min_size,
        region_abs_area_threshold=args.region_abs_area_threshold,
        region_ratio_area_threshold=args.region_ratio_area_threshold,
        debug_save_instances=False,
        debug_save_lung_split=False,
        debug_save_six_regions=False,
        debug_dir='debug_instances',
        debug_max_cases=0,
    )


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_model(args, checkpoint_path, device):
    """Instantiate model and load checkpoint weights."""
    from models.attention import Attention

    model = Attention(
        in_channels=args.in_channels,
        pretrained_backbone=False,    # weights come from checkpoint
        num_classes=args.num_classes,
        instance_batch_size=getattr(args, 'instance_batch_size', 2),
        freeze_backbone=getattr(args, 'freeze_backbone', True),
        backbone_name=getattr(args, 'backbone_name', 'resnet18'),
        use_burden_features=getattr(args, 'use_burden_features', False),
        use_position_embedding=getattr(args, 'use_position_embedding', False),
        position_embed_dim=getattr(args, 'position_embed_dim', 16),
        use_coverage_features=getattr(args, 'use_coverage_features', False),
        coverage_num_bins=getattr(args, 'coverage_num_bins', 6),
        coverage_tau=getattr(args, 'coverage_tau', 0.5),
        coverage_temperature=getattr(args, 'coverage_temperature', 0.1),
        coverage_eps=getattr(args, 'coverage_eps', 1e-6),
        burden_score_hidden_dim=getattr(args, 'burden_score_hidden_dim', 0),
        burden_score_dropout=getattr(args, 'burden_score_dropout', 0.0),
        burden_tau=getattr(args, 'burden_tau', 0.5),
        burden_temperature=getattr(args, 'burden_temperature', 0.1),
        burden_topk_ratio=getattr(args, 'burden_topk_ratio', 0.1),
        aggregator=getattr(args, 'aggregator', 'abmil'),
        transmil_num_heads=getattr(args, 'transmil_num_heads', 8),
        transmil_num_layers=getattr(args, 'transmil_num_layers', 2),
        transmil_dropout=getattr(args, 'transmil_dropout', 0.1),
        nystrom_num_landmarks=getattr(args, 'nystrom_num_landmarks', 64),
        score_logit_reg_weight=getattr(args, 'score_logit_reg_weight', 0.01),
        corn_balanced=getattr(args, 'corn_balanced', True),
        instance_aux_loss_weight=getattr(args, 'instance_aux_loss_weight', 0.0),
        use_pseudo12_guidance=getattr(args, 'use_pseudo12_guidance', False),
        pseudo12_lambda=getattr(args, 'pseudo12_lambda', 0.1),
        use_soft12_guidance=getattr(args, 'use_soft12_guidance', False),
        soft12_lambda=getattr(args, 'soft12_lambda', 0.1),
    )

    state = torch.load(checkpoint_path, map_location=device)
    # Handle potential "module." prefix from DataParallel
    if any(k.startswith('module.') for k in state.keys()):
        state = {k[len('module.'):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    print('Loaded checkpoint: {}'.format(checkpoint_path))
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = _build_arg_parser()
    cli_args = parser.parse_args()

    # ── Load training args from YAML ─────────────────────────────────────────
    from utils.config import parse_args as parse_training_args
    # Inject config path into sys.argv for parse_args
    orig_argv = sys.argv[:]
    sys.argv = ['explainability', '--config', cli_args.config]
    args = parse_training_args()
    sys.argv = orig_argv

    # ── Output directory ─────────────────────────────────────────────────────
    if cli_args.out_dir:
        out_dir = cli_args.out_dir
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join('explainability', 'results', ts)
    os.makedirs(out_dir, exist_ok=True)
    print('\n=== MIL Explainability System ===')
    print('Output directory: {}'.format(os.path.abspath(out_dir)))

    # ── Sub-directories ───────────────────────────────────────────────────────
    dirs = {
        'training_curves':     os.path.join(out_dir, 'training_curves'),
        'confusion_matrix':    os.path.join(out_dir, 'confusion_matrix'),
        'case_reports':        os.path.join(out_dir, 'case_reports'),
        'slab_visualization':  os.path.join(out_dir, 'slab_visualization'),
        'heatmaps':            os.path.join(out_dir, 'heatmaps'),
        'lung_region_heatmaps': os.path.join(out_dir, 'lung_region_heatmaps'),
        'misclassified_cases': os.path.join(out_dir, 'misclassified_cases'),
        'embeddings':          os.path.join(out_dir, 'embeddings'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # ── A: Training curve visualizations ─────────────────────────────────────
    print('\n[A] Training curve visualizations...')
    from explainability.training_curves import generate_training_visualizations

    csv_path = cli_args.csv
    if not csv_path:
        csv_path = _find_latest_csv('.')
        if csv_path:
            print('  Auto-detected CSV: {}'.format(csv_path))
        else:
            print('  No training CSV found; skipping curve plots.')

    if not cli_args.no_inference:
        # We'll pass y_true/y_pred after inference below; call with None for now
        _csv_for_curves = csv_path
    else:
        generate_training_visualizations(
            csv_path=csv_path,
            y_true_test=None,
            y_pred_test=None,
            num_classes=args.num_classes,
            save_dir=dirs['training_curves'],
            verbose=True,
        )

    # ── Skip inference if requested ───────────────────────────────────────────
    if cli_args.no_inference:
        print('\nInference skipped (--no_inference). Done.')
        return

    # ── Device & model ────────────────────────────────────────────────────────
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('\nDevice: {}'.format(device))

    checkpoint_path = cli_args.checkpoint or getattr(args, 'best_model_path', '')
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            'Checkpoint not found: "{}". '
            'Use --checkpoint to specify the path.'.format(checkpoint_path))

    print('\nLoading model...')
    model = _load_model(args, checkpoint_path, device)

    # ── Dataset splits ────────────────────────────────────────────────────────
    splits = ['train', 'val', 'test'] if cli_args.split == 'all' else [cli_args.split]

    all_results = {}
    for split in splits:
        print('\n[Inference] split={}'.format(split))
        try:
            dataset = _build_single_dataset(args, split)
        except Exception as e:
            print('  Failed to build {} dataset: {}'.format(split, e))
            continue

        if cli_args.max_cases > 0:
            dataset.samples = dataset.samples[:cli_args.max_cases]

        from explainability.inference_engine import run_inference
        results = run_inference(
            model=model,
            dataset=dataset,
            device=device,
            num_classes=args.num_classes,
            use_cuda=use_cuda,
            collect_bag_features=True,
            verbose=True,
        )
        all_results[split] = results
        print('  Collected {} cases'.format(len(results)))

    # Work with primary split (test or first available)
    primary_split = 'test' if 'test' in all_results else splits[0]
    results = all_results.get(primary_split, [])

    if len(results) == 0:
        print('No inference results. Exiting.')
        return

    y_true_test = np.array([r['true_label'] for r in results])
    y_pred_test = np.array([r['pred_label'] for r in results])

    # ── A (continued): Training curves with test metrics ─────────────────────
    generate_training_visualizations(
        csv_path=csv_path if csv_path else '',
        y_true_test=y_true_test,
        y_pred_test=y_pred_test,
        num_classes=args.num_classes,
        save_dir=dirs['training_curves'],
        verbose=True,
    )

    # Also save confusion matrix in its own directory
    from explainability.training_curves import plot_confusion_matrix, plot_per_class_metrics
    plot_confusion_matrix(y_true_test, y_pred_test, args.num_classes,
                          os.path.join(dirs['confusion_matrix'],
                                       'confusion_matrix_{}.png'.format(primary_split)),
                          split_name=primary_split)
    plot_per_class_metrics(y_true_test, y_pred_test, args.num_classes,
                           os.path.join(dirs['confusion_matrix'],
                                        'per_class_metrics_{}.png'.format(primary_split)))

    # ── B: Case-level reports ─────────────────────────────────────────────────
    if not cli_args.no_case_reports:
        print('\n[B] Generating case reports...')
        from explainability.case_report import generate_all_case_reports
        generate_all_case_reports(
            results, dirs['case_reports'],
            num_classes=args.num_classes, verbose=True)

    # ── C: Slab-level visualizations ─────────────────────────────────────────
    if not cli_args.no_slab_viz:
        print('\n[C] Slab-level visualizations...')
        from explainability.slab_viz import generate_slab_visualizations
        generate_slab_visualizations(
            results, dirs['slab_visualization'],
            top_k=cli_args.slab_topk, verbose=True)

    # ── D: Grad-CAM for top slabs ─────────────────────────────────────────────
    if cli_args.gradcam:
        print('\n[D] Grad-CAM for all {} cases...'.format(len(results)))
        from explainability.gradcam import run_gradcam_for_case
        for i, case_dict in enumerate(results):
            cid = case_dict['case_id']
            true_l = case_dict['true_label']
            pred_l = case_dict['pred_label']
            print('  gradcam [{}/{}] {}'.format(i + 1, len(results), cid))
            try:
                case_hm_dir = os.path.join(dirs['heatmaps'],
                                           '{}_t{}_p{}'.format(cid, true_l, pred_l))
                run_gradcam_for_case(
                    case_dict, model, case_hm_dir,
                    top_k=cli_args.gradcam_topk, device=device)
            except Exception as e:
                print('    failed: {}'.format(e))

    # ── E: Lung region heatmaps ───────────────────────────────────────────────
    print('\n[E] Lung region heatmaps...')
    from explainability.lung_region_heatmap import generate_lung_region_heatmaps
    generate_lung_region_heatmaps(
        results, dirs['lung_region_heatmaps'],
        num_classes=args.num_classes, verbose=True)

    # ── F: Misclassified case analysis ───────────────────────────────────────
    print('\n[F] Misclassified case analysis...')
    from explainability.misclassified import generate_misclassified_reports
    run_gc_mis = (not cli_args.no_gradcam_for_misclassified)
    generate_misclassified_reports(
        results, dirs['misclassified_cases'],
        model=model if run_gc_mis else None,
        num_classes=args.num_classes,
        top_k=cli_args.gradcam_topk,
        run_gradcam=run_gc_mis,
        device=device,
        verbose=True,
    )

    # ── G: Embedding visualizations ───────────────────────────────────────────
    if not cli_args.no_embeddings:
        print('\n[G] Embedding visualizations...')
        from explainability.embedding_viz import generate_embedding_visualizations
        generate_embedding_visualizations(
            results, dirs['embeddings'],
            num_classes=args.num_classes,
            run_umap=(not cli_args.no_umap),
            verbose=True,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n=== Explainability analysis complete ===')
    print('Results saved to: {}'.format(os.path.abspath(out_dir)))
    print('\nDirectory structure:')
    for name, path in dirs.items():
        n_files = sum(len(fs) for _, _, fs in os.walk(path))
        print('  {:25s}  {:4d} files  ({})'.format(name, n_files, path))


if __name__ == '__main__':
    main()
