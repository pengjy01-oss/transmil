"""Generate attention heatmaps overlaid on CT slices for interpretability.

Usage:
    conda run -n lung_new python scripts/generate_heatmaps.py \
        --config configs/ct25d_transmil.yaml \
        --checkpoint checkpoints/best_model.pt \
        --num_cases 20 --split test
"""

from __future__ import print_function

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import SimpleITK as sitk
import torch
from torch.autograd import Variable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.ct_pne_dataset import CTPneNiiBags, get_case_id_from_path
from datasets.ct_preprocess.cache_io import load_case_cache, unpack_region_context_cache
from losses import _corn_label_from_logits
from models.attention import Attention
from utils.config import parse_args

REGION_NAMES = ['left_upper', 'left_middle', 'left_lower',
                'right_upper', 'right_middle', 'right_lower']
REGION_COLORS = {
    'left_upper': '#FF4444', 'left_middle': '#FF8800', 'left_lower': '#FFCC00',
    'right_upper': '#4444FF', 'right_middle': '#0088FF', 'right_lower': '#00CCFF',
}
CLASS_NAMES = ['Stage 0', 'Stage I', 'Stage II', 'Stage III']


def load_model(args, checkpoint_path):
    """Load trained model from checkpoint."""
    model = Attention(
        in_channels=args.in_channels,
        pretrained_backbone=args.pretrained_backbone,
        num_classes=args.num_classes,
        instance_batch_size=args.instance_batch_size,
        freeze_backbone=args.freeze_backbone,
        use_burden_features=args.use_burden_features,
        use_position_embedding=getattr(args, 'use_position_embedding', False),
        position_embed_dim=getattr(args, 'position_embed_dim', 16),
        use_coverage_features=getattr(args, 'use_coverage_features', False),
        coverage_num_bins=getattr(args, 'coverage_num_bins', 6),
        coverage_tau=getattr(args, 'coverage_tau', 0.5),
        coverage_temperature=getattr(args, 'coverage_temperature', 0.1),
        coverage_eps=getattr(args, 'coverage_eps', 1e-6),
        burden_score_hidden_dim=args.burden_score_hidden_dim,
        burden_score_dropout=args.burden_score_dropout,
        burden_tau=args.burden_tau,
        burden_temperature=args.burden_temperature,
        burden_topk_ratio=args.burden_topk_ratio,
        aggregator=args.aggregator,
        transmil_num_heads=args.transmil_num_heads,
        transmil_num_layers=args.transmil_num_layers,
        transmil_dropout=args.transmil_dropout,
        score_logit_reg_weight=args.score_logit_reg_weight,
        corn_balanced=args.corn_balanced,
    )
    map_location = torch.device('cuda') if args.cuda else torch.device('cpu')
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state_dict)
    if args.cuda:
        model.cuda()
    model.eval()
    return model


def infer_single(model, data, pos_z, cuda):
    """Run inference on a single bag. Returns (pred_label, attention_weights, instance_scores)."""
    with torch.no_grad():
        if cuda:
            data, pos_z = data.cuda(), pos_z.cuda()
        data = Variable(data)
        logits, Y_hat, A = model.forward(data, pos_z=pos_z)
        aux = model.get_latest_aux_outputs()
        instance_scores = aux.get('instance_scores', None)
        if instance_scores is not None:
            instance_scores = instance_scores.detach().cpu().numpy()
        pred = int(Y_hat.view(-1).cpu().item())
        # Get attention weights (works for both ABMIL and TransMIL)
        A_weights = model.compute_attention_weights(data, pos_z=pos_z)
        if A_weights is not None:
            A_weights = A_weights.numpy().flatten()
        return pred, A_weights, instance_scores


def generate_case_heatmap(case_id, nii_path, label, pred, metadata,
                          attention_weights, instance_scores,
                          cache_root, save_dir, num_classes=4):
    """Generate heatmap visualizations for one case."""
    case_dir = os.path.join(save_dir, case_id)
    os.makedirs(case_dir, exist_ok=True)

    # Load CT volume
    image = sitk.ReadImage(nii_path)
    volume = sitk.GetArrayFromImage(image).astype(np.float32)  # [Z, H, W]
    total_z = volume.shape[0]

    # Load cache for region masks
    cached = load_case_cache(case_id, cache_root)
    region_ctx = None
    if cached is not None:
        region_ctx = unpack_region_context_cache(cached['cache'])

    # Build per-region attention aggregation
    region_attn = {r: [] for r in REGION_NAMES}
    region_z_attn = {r: {} for r in REGION_NAMES}  # {region: {z: [attn_values]}}

    K = len(metadata)
    for i in range(K):
        rname = metadata[i].get('region', 'unknown')
        cz = int(metadata[i].get('center_z', 0))
        attn_val = float(attention_weights[i]) if attention_weights is not None else 0.0
        score_val = float(instance_scores[i]) if instance_scores is not None else 0.0
        # Use combined importance: attention * instance_score for richer signal
        importance = attn_val
        if instance_scores is not None:
            importance = attn_val * score_val

        if rname in region_attn:
            region_attn[rname].append(importance)
            if cz not in region_z_attn[rname]:
                region_z_attn[rname][cz] = []
            region_z_attn[rname][cz].append(importance)

    # --- Figure 1: Region-level attention bar chart ---
    region_mean_attn = {}
    for rname in REGION_NAMES:
        vals = region_attn[rname]
        region_mean_attn[rname] = np.mean(vals) if vals else 0.0

    # Normalize for visualization
    max_attn = max(region_mean_attn.values()) if region_mean_attn else 1.0
    if max_attn <= 0:
        max_attn = 1.0

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(REGION_NAMES))
    colors = [REGION_COLORS[r] for r in REGION_NAMES]
    bars = ax.bar(x, [region_mean_attn[r] for r in REGION_NAMES], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('_', '\n') for r in REGION_NAMES], fontsize=9)
    ax.set_ylabel('Mean Attention')
    ax.set_title('{} | True: {} | Pred: {} | {}'.format(
        case_id, CLASS_NAMES[label], CLASS_NAMES[pred],
        'CORRECT' if label == pred else 'WRONG (off-by-{})'.format(abs(label - pred))
    ))
    for bar, rname in zip(bars, REGION_NAMES):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_attn * 0.02,
                '{:.4f}'.format(region_mean_attn[rname]), ha='center', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    fig.savefig(os.path.join(case_dir, 'region_attention_bar.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Figure 2: Z-position attention profile ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for rname in REGION_NAMES:
        z_dict = region_z_attn[rname]
        if not z_dict:
            continue
        zs = sorted(z_dict.keys())
        vals = [np.mean(z_dict[z]) for z in zs]
        ax.plot(zs, vals, 'o-', color=REGION_COLORS[rname], label=rname, markersize=3, alpha=0.8)
    ax.set_xlabel('Z slice')
    ax.set_ylabel('Attention × Score')
    ax.set_title('{} | Attention along Z-axis'.format(case_id))
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(case_dir, 'z_attention_profile.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Figure 3: Top-K instance attention overlaid on CT slices ---
    if attention_weights is not None and region_ctx is not None:
        # Find top-10 highest attention instances
        sorted_idx = np.argsort(attention_weights)[::-1]
        top_k = min(10, K)
        top_indices = sorted_idx[:top_k]

        # Pick up to 4 unique z-slices from top instances
        top_z_slices = []
        seen_z = set()
        for idx in top_indices:
            cz = int(metadata[idx].get('center_z', 0))
            if cz not in seen_z:
                top_z_slices.append((cz, idx))
                seen_z.add(cz)
            if len(top_z_slices) >= 4:
                break

        if top_z_slices:
            n_slices = len(top_z_slices)
            fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5))
            if n_slices == 1:
                axes = [axes]

            left_mask = region_ctx['left_lung_mask']
            right_mask = region_ctx['right_lung_mask']
            region_masks = region_ctx['region_masks_dict']

            for ax_i, (z_slice, top_idx) in enumerate(top_z_slices):
                z_safe = max(0, min(z_slice, total_z - 1))
                ct_slice = volume[z_safe]

                # Window to lung HU range
                ct_display = np.clip(ct_slice, -1000, 400)
                ct_display = (ct_display - ct_display.min()) / (ct_display.max() - ct_display.min() + 1e-8)

                axes[ax_i].imshow(ct_display, cmap='gray', aspect='equal')

                # Overlay region attention heatmap
                heat = np.zeros(ct_slice.shape, dtype=np.float32)
                for rname in REGION_NAMES:
                    rmask = region_masks.get(rname, np.zeros_like(left_mask))
                    if z_safe < rmask.shape[0]:
                        rmask_z = rmask[z_safe].astype(bool)
                        heat[rmask_z] = region_mean_attn.get(rname, 0.0) / max_attn

                axes[ax_i].imshow(heat, cmap='jet', alpha=0.4, vmin=0, vmax=1, aspect='equal')

                rname_top = metadata[top_idx].get('region', '?')
                attn_top = float(attention_weights[top_idx])
                axes[ax_i].set_title('z={} | {} | attn={:.4f}'.format(z_slice, rname_top, attn_top), fontsize=10)
                axes[ax_i].axis('off')

            fig.suptitle('{} | True: {} | Pred: {} | Top attention slices'.format(
                case_id, CLASS_NAMES[label], CLASS_NAMES[pred]), fontsize=12)
            fig.savefig(os.path.join(case_dir, 'top_attention_slices.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)

    # --- Figure 4: Full 6-region spatial heatmap at best slice ---
    if region_ctx is not None:
        pseudo_mask = region_ctx['pseudo_mask']
        per_z = pseudo_mask.reshape(pseudo_mask.shape[0], -1).sum(axis=1)
        z_best = int(np.argmax(per_z))
        z_safe = max(0, min(z_best, total_z - 1))

        ct_slice = volume[z_safe]
        ct_display = np.clip(ct_slice, -1000, 400)
        ct_display = (ct_display - ct_display.min()) / (ct_display.max() - ct_display.min() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original CT
        axes[0].imshow(ct_display, cmap='gray', aspect='equal')
        axes[0].set_title('CT slice z={}'.format(z_safe))
        axes[0].axis('off')

        # Region mask overlay
        region_masks = region_ctx['region_masks_dict']
        rgb = np.stack([ct_display] * 3, axis=-1)
        color_map = {
            'left_upper': [1, 0, 0], 'left_middle': [1, 0.5, 0], 'left_lower': [1, 1, 0],
            'right_upper': [0, 0, 1], 'right_middle': [0, 0.5, 1], 'right_lower': [0, 1, 1],
        }
        for rname, color in color_map.items():
            rmask = region_masks.get(rname, np.zeros_like(pseudo_mask))
            if z_safe < rmask.shape[0]:
                rmask_z = rmask[z_safe].astype(bool)
                for c in range(3):
                    rgb[:, :, c] = np.where(rmask_z, rgb[:, :, c] * 0.5 + color[c] * 0.5, rgb[:, :, c])
        axes[1].imshow(np.clip(rgb, 0, 1), aspect='equal')
        patches = [mpatches.Patch(color=color_map[r], label=r) for r in REGION_NAMES]
        axes[1].legend(handles=patches, fontsize=7, loc='lower right')
        axes[1].set_title('Six regions')
        axes[1].axis('off')

        # Attention heatmap overlay
        heat = np.zeros(ct_slice.shape, dtype=np.float32)
        for rname in REGION_NAMES:
            rmask = region_masks.get(rname, np.zeros_like(pseudo_mask))
            if z_safe < rmask.shape[0]:
                rmask_z = rmask[z_safe].astype(bool)
                heat[rmask_z] = region_mean_attn.get(rname, 0.0) / max_attn
        axes[2].imshow(ct_display, cmap='gray', aspect='equal')
        im = axes[2].imshow(heat, cmap='jet', alpha=0.5, vmin=0, vmax=1, aspect='equal')
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label='Relative Attention')
        axes[2].set_title('Attention heatmap')
        axes[2].axis('off')

        fig.suptitle('{} | True: {} | Pred: {} | z={}'.format(
            case_id, CLASS_NAMES[label], CLASS_NAMES[pred], z_safe), fontsize=13)
        fig.savefig(os.path.join(case_dir, 'spatial_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print('  {} | True={} Pred={} | {}'.format(
        case_id, CLASS_NAMES[label], CLASS_NAMES[pred],
        'CORRECT' if label == pred else 'WRONG'))


def main():
    parser = argparse.ArgumentParser(description='Generate attention heatmaps')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--num_cases', type=int, default=20)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--save_dir', type=str, default='heatmaps')
    parser.add_argument('--cases', type=str, default='', help='Comma-separated case IDs (overrides num_cases)')
    cli_args = parser.parse_args()

    # Parse config
    sys.argv = ['', '--config', cli_args.config]
    args = parse_args()

    print('Loading model from {}'.format(cli_args.checkpoint))
    model = load_model(args, cli_args.checkpoint)

    # Build dataset
    half_depth = args.slab_depth // 2
    channel_offsets = tuple(range(-half_depth, half_depth + 1))
    common_kwargs = dict(
        root_dir=args.data_root,
        num_classes=args.num_classes,
        middle_ratio=args.middle_ratio,
        fixed_num_slices=args.fixed_num_slices,
        channel_offsets=channel_offsets,
        slab_stride=args.slab_stride,
        num_slabs=args.num_slabs,
        center_sampling_mode='uniform',  # deterministic for heatmaps
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

    dataset = CTPneNiiBags(split=cli_args.split, **common_kwargs)
    print('Dataset [{}] size: {}'.format(cli_args.split, len(dataset)))

    # Select cases
    if cli_args.cases:
        target_ids = set(cli_args.cases.split(','))
        indices = []
        for i, (nii_path, _) in enumerate(dataset.samples):
            cid = get_case_id_from_path(nii_path)
            if cid in target_ids:
                indices.append(i)
    else:
        indices = list(range(min(cli_args.num_cases, len(dataset))))

    from datetime import datetime
    save_dir = os.path.join(cli_args.save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)
    print('Generating heatmaps for {} cases -> {}'.format(len(indices), save_dir))

    correct = 0
    total = 0
    for idx in indices:
        nii_path, label = dataset.samples[idx]
        case_id = get_case_id_from_path(nii_path)

        # Get bag and metadata
        bag, bag_label, pos_z = dataset[idx]
        metadata = getattr(dataset, '_last_metadata', None) or []

        # Inference
        pred, A, instance_scores = infer_single(model, bag, pos_z, args.cuda)
        total += 1
        if pred == label:
            correct += 1

        generate_case_heatmap(
            case_id=case_id,
            nii_path=nii_path,
            label=label,
            pred=pred,
            metadata=metadata,
            attention_weights=A,
            instance_scores=instance_scores,
            cache_root=args.cache_root,
            save_dir=save_dir,
            num_classes=args.num_classes,
        )

    print('\nDone! {} cases, accuracy: {}/{} ({:.1f}%)'.format(
        total, correct, total, 100.0 * correct / max(1, total)))
    print('Heatmaps saved to {}'.format(save_dir))


if __name__ == '__main__':
    main()
