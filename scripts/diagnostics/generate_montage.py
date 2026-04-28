#!/usr/bin/env python
"""Generate per-case montage figures for CT MIL diagnostics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import _build_datasets  # noqa: E402
from scripts.diagnostics.audit_best_checkpoint import (  # noqa: E402
    REGION_ORDER,
    _case_id_from_path,
    _rebuild_case_context,
)


REGION_COLORS = {
    'left_upper': '#ff6b6b',
    'left_middle': '#f06595',
    'left_lower': '#cc5de8',
    'right_upper': '#339af0',
    'right_middle': '#22b8cf',
    'right_lower': '#20c997',
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate montage figures for selected CT cases.')
    parser.add_argument('--config', type=str, required=True, help='Training YAML config')
    parser.add_argument('--selection_csv', type=str, default='', help='Case selection CSV from audit_best_checkpoint.py')
    parser.add_argument('--predictions_csv', type=str, default='', help='Per-case prediction CSV for title metadata lookup')
    parser.add_argument('--case_id', type=str, default='', help='Single case mode: case id')
    parser.add_argument('--split', type=str, default='', help='Single case mode: split name')
    parser.add_argument('--output_dir', type=str, default='outputs/diagnostics/montage', help='Montage output root')
    parser.add_argument('--limit', type=int, default=0, help='Optional limit on number of selected cases')
    return parser.parse_args()


def _load_config_args(config_path: Path) -> SimpleNamespace:
    with config_path.open('r') as f:
        config_dict = yaml.safe_load(f) or {}
    args = SimpleNamespace(**config_dict)
    args.no_cuda = bool(getattr(args, 'no_cuda', False))
    args.cuda = False
    return args


def _window_ct(slice_2d: np.ndarray, hu_min: float = -1000.0, hu_max: float = 400.0) -> np.ndarray:
    x = np.clip(slice_2d.astype(np.float32), hu_min, hu_max)
    x = (x - hu_min) / max(hu_max - hu_min, 1e-6)
    return x


def _draw_overlay(ax, base_gray: np.ndarray, overlay_rgb: np.ndarray, title: str) -> None:
    ax.imshow(base_gray, cmap='gray', vmin=0.0, vmax=1.0)
    ax.imshow(overlay_rgb)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def _representative_region_crops(bag_np: np.ndarray, metadata: list[dict]) -> list[tuple[str, int, np.ndarray]]:
    reps = []
    for region in REGION_ORDER:
        candidates = [(idx, meta) for idx, meta in enumerate(metadata) if str(meta.get('region')) == region]
        if not candidates:
            reps.append((region, -1, np.zeros((bag_np.shape[-2], bag_np.shape[-1]), dtype=np.float32)))
            continue
        candidates = sorted(candidates, key=lambda t: int(t[1].get('center_z', 0)))
        chosen_idx, chosen_meta = candidates[len(candidates) // 2]
        center_channel = bag_np[chosen_idx, bag_np.shape[1] // 2]
        reps.append((region, int(chosen_meta.get('center_z', -1)), center_channel))
    return reps


def _find_index_map(dataset) -> dict:
    return {_case_id_from_path(path): idx for idx, (path, _) in enumerate(dataset.samples)}


def _load_case_rows(selection_csv: Path | None, predictions_csv: Path | None, case_id: str, split: str) -> pd.DataFrame:
    if selection_csv is not None and selection_csv.exists():
        rows = pd.read_csv(selection_csv)
        if predictions_csv is not None and predictions_csv.exists():
            pred = pd.read_csv(predictions_csv)
            join_cols = [c for c in pred.columns if c in ('case_id', 'split', 'true_label', 'pred_label_default',
                                                          'cumprob_y_gt_0', 'cumprob_y_gt_1', 'cumprob_y_gt_2',
                                                          'split_method', 'spacing_x', 'spacing_y', 'spacing_z',
                                                          'institution', 'reconstruction_kernel')]
            rows = rows.merge(pred[join_cols].drop_duplicates(['case_id', 'split']), on=['case_id', 'split'], how='left')
            for col in join_cols:
                if col in ('case_id', 'split'):
                    continue
                left_col = '{}_x'.format(col)
                right_col = '{}_y'.format(col)
                if left_col in rows.columns or right_col in rows.columns:
                    left_series = rows[left_col] if left_col in rows.columns else pd.Series(np.nan, index=rows.index)
                    right_series = rows[right_col] if right_col in rows.columns else pd.Series(np.nan, index=rows.index)
                    rows[col] = left_series.combine_first(right_series)
                    drop_cols = [c for c in (left_col, right_col) if c in rows.columns]
                    if drop_cols:
                        rows = rows.drop(columns=drop_cols)
        return rows

    if case_id and split:
        row = {'case_id': case_id, 'split': split, 'group': 'manual'}
        if predictions_csv is not None and predictions_csv.exists():
            pred = pd.read_csv(predictions_csv)
            hit = pred[(pred['case_id'] == case_id) & (pred['split'] == split)]
            if not hit.empty:
                row.update(hit.iloc[0].to_dict())
        return pd.DataFrame([row])

    raise ValueError('Provide either --selection_csv or (--case_id and --split).')


def _plot_case(case_row: pd.Series, dataset, case_idx: int, output_path: Path) -> None:
    item = dataset[case_idx]
    bag = item[0].numpy()
    metadata = list(getattr(dataset, '_last_metadata', []))
    image, volume_raw, _, region_ctx, _ = _rebuild_case_context(dataset, case_idx)
    spacing = image.GetSpacing()
    spacing_txt = '({:.4f}, {:.4f}, {:.4f})'.format(
        float(spacing[0]) if len(spacing) >= 1 else np.nan,
        float(spacing[1]) if len(spacing) >= 2 else np.nan,
        float(spacing[2]) if len(spacing) >= 3 else np.nan,
    )

    pseudo_mask = region_ctx['pseudo_mask']
    left_mask = region_ctx['left_lung_mask']
    right_mask = region_ctx['right_lung_mask']
    z_global = int(np.argmax(pseudo_mask.reshape(pseudo_mask.shape[0], -1).sum(axis=1)))
    ct_slice = _window_ct(volume_raw[z_global])

    overlay_mask = np.zeros((ct_slice.shape[0], ct_slice.shape[1], 4), dtype=np.float32)
    overlay_mask[..., 1] = pseudo_mask[z_global].astype(np.float32)
    overlay_mask[..., 3] = pseudo_mask[z_global].astype(np.float32) * 0.35

    overlay_split = np.zeros((ct_slice.shape[0], ct_slice.shape[1], 4), dtype=np.float32)
    overlay_split[..., 0] = left_mask[z_global].astype(np.float32)
    overlay_split[..., 1] = right_mask[z_global].astype(np.float32)
    overlay_split[..., 3] = np.maximum(left_mask[z_global], right_mask[z_global]).astype(np.float32) * 0.35

    fig = plt.figure(figsize=(18, 12))
    outer = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25], hspace=0.28, wspace=0.12)

    ax0 = fig.add_subplot(outer[0, 0])
    ax0.imshow(ct_slice, cmap='gray', vmin=0.0, vmax=1.0)
    ax0.set_title('Original CT Center Slice', fontsize=10)
    ax0.axis('off')

    ax1 = fig.add_subplot(outer[0, 1])
    _draw_overlay(ax1, ct_slice, overlay_mask, 'Pseudo Lung Mask Overlay')

    ax2 = fig.add_subplot(outer[0, 2])
    _draw_overlay(ax2, ct_slice, overlay_split, 'Left/Right Split Overlay')

    ax3 = fig.add_subplot(outer[0, 3])
    ax3.imshow(ct_slice, cmap='gray', vmin=0.0, vmax=1.0)
    for region_name in REGION_ORDER:
        y0, y1, x0, x1 = region_ctx['region_bboxes_dict'][region_name]
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.8,
                         edgecolor=REGION_COLORS[region_name], facecolor='none')
        ax3.add_patch(rect)
        ax3.text(x0 + 3, y0 + 12, region_name, color=REGION_COLORS[region_name], fontsize=8,
                 bbox={'facecolor': 'black', 'alpha': 0.25, 'pad': 1})
    ax3.set_title('Six-Region BBoxes', fontsize=10)
    ax3.axis('off')

    crop_grid = outer[1, :].subgridspec(2, 3, hspace=0.20, wspace=0.10)
    reps = _representative_region_crops(bag, metadata)
    for i, (region_name, center_z, crop_img) in enumerate(reps):
        ax = fig.add_subplot(crop_grid[i // 3, i % 3])
        ax.imshow(crop_img, cmap='gray')
        ax.set_title('{} | center_z={}'.format(region_name, center_z), fontsize=9)
        ax.axis('off')

    title_parts = [
        str(case_row.get('case_id')),
        'split={}'.format(case_row.get('split', 'unknown')),
    ]
    if not pd.isna(case_row.get('true_label', np.nan)):
        title_parts.append('true={}'.format(int(case_row['true_label'])))
    if not pd.isna(case_row.get('pred_label_default', np.nan)):
        title_parts.append('pred={}'.format(int(case_row['pred_label_default'])))
    if 'cumprob_y_gt_0' in case_row:
        title_parts.append('P(y>0)={:.3f}'.format(float(case_row['cumprob_y_gt_0'])))
    if 'cumprob_y_gt_1' in case_row:
        title_parts.append('P(y>1)={:.3f}'.format(float(case_row['cumprob_y_gt_1'])))
    if 'cumprob_y_gt_2' in case_row:
        title_parts.append('P(y>2)={:.3f}'.format(float(case_row['cumprob_y_gt_2'])))
    title_parts.append('split_method={}'.format(case_row.get('split_method', 'NA')))
    title_parts.append('spacing={}'.format(spacing_txt))
    if not pd.isna(case_row.get('institution', np.nan)):
        title_parts.append('institution={}'.format(case_row['institution']))
    if not pd.isna(case_row.get('reconstruction_kernel', np.nan)):
        title_parts.append('kernel={}'.format(case_row['reconstruction_kernel']))
    fig.suptitle(' | '.join(title_parts), fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    cli_args = _parse_args()
    config_path = Path(cli_args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    args = _load_config_args(config_path)

    train_dataset, val_dataset, test_dataset = _build_datasets(args)
    if hasattr(train_dataset, 'set_epoch'):
        train_dataset.set_epoch(0)
    if hasattr(val_dataset, 'set_epoch') and val_dataset is not None:
        val_dataset.set_epoch(0)
    if hasattr(test_dataset, 'set_epoch') and test_dataset is not None:
        test_dataset.set_epoch(0)
    datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    index_maps = {k: _find_index_map(v) for k, v in datasets.items() if v is not None}

    selection_csv = None
    if cli_args.selection_csv:
        selection_csv = Path(cli_args.selection_csv)
        if not selection_csv.is_absolute():
            selection_csv = (REPO_ROOT / selection_csv).resolve()
    predictions_csv = None
    if cli_args.predictions_csv:
        predictions_csv = Path(cli_args.predictions_csv)
        if not predictions_csv.is_absolute():
            predictions_csv = (REPO_ROOT / predictions_csv).resolve()
    rows = _load_case_rows(selection_csv, predictions_csv, cli_args.case_id, cli_args.split)
    if cli_args.limit > 0:
        rows = rows.head(cli_args.limit)

    output_root = Path(cli_args.output_dir)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    saved_rows = []
    for _, row in rows.iterrows():
        split = str(row['split'])
        case_id = str(row['case_id'])
        dataset = datasets.get(split)
        if dataset is None or case_id not in index_maps.get(split, {}):
            continue
        case_idx = index_maps[split][case_id]
        group_name = str(row.get('group', 'manual')).replace('/', '_')
        out_path = output_root / group_name / '{}_{}.png'.format(split, case_id)
        _plot_case(row, dataset, case_idx, out_path)
        saved_rows.append({'case_id': case_id, 'split': split, 'group': group_name, 'output_png': str(out_path)})

    pd.DataFrame(saved_rows).to_csv(output_root / 'montage_index.csv', index=False)
    print('Saved {} montage figures to {}'.format(len(saved_rows), output_root))


if __name__ == '__main__':
    main()