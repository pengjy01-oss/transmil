#!/usr/bin/env python
"""Checkpoint audit for CT pneumoconiosis MIL experiments.

Exports per-case predictions, split metrics, CORN threshold diagnostics,
split-method analysis, burden/coverage summaries, spacing summaries,
and case lists for visualization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:  # pragma: no cover
    pearsonr = None
    spearmanr = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import _build_datasets, _build_model  # noqa: E402
from datasets.ct_preprocess.lung_regions import get_region_bbox  # noqa: E402


DEFAULT_SPLITS = ('train', 'val', 'test')
REGION_ORDER = (
    'left_upper',
    'left_middle',
    'left_lower',
    'right_upper',
    'right_middle',
    'right_lower',
)
ORDINAL_THRESHOLDS = (0, 1, 2)
NUMERIC_METADATA_FIELDS = (
    'spacing_x',
    'spacing_y',
    'spacing_z',
    'slice_thickness',
)
TEXT_METADATA_FIELDS = (
    'manufacturer',
    'scanner_model',
    'institution',
    'protocol_name',
    'reconstruction_kernel',
)


@dataclass
class SearchResult:
    objective: str
    strategy: str
    thresholds: Tuple[float, float, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Audit CT MIL checkpoint without changing training logic.')
    parser.add_argument('--config', type=str, required=True, help='YAML config path used for training')
    parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint path (default: config best_model_path)')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory for diagnostics')
    parser.add_argument('--splits', type=str, default='train,val,test', help='Comma-separated splits to evaluate')
    parser.add_argument('--max_cases_per_split', type=int, default=0,
                        help='Optional cap per split for quick smoke tests (0 = all cases)')
    parser.add_argument('--train_eval_epoch', type=int, default=0,
                        help='Deterministic epoch index used when materializing train bags')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers used during split evaluation (0 keeps serial dataset[idx] loop)')
    parser.add_argument('--skip_crop_stats', action='store_true',
                        help='Skip per-crop bbox/lung-ratio diagnostics for faster iteration')
    parser.add_argument('--no_extract_dicom_metadata', action='store_true',
                        help='Disable on-demand DICOM header extraction from raw case folders')
    parser.add_argument('--nifti_metadata_csv', type=str,
                        default='dataset_audit_results/nifti_metadata.csv')
    parser.add_argument('--split_assignment_csv', type=str,
                        default='dataset_audit_results/split_assignment.csv')
    parser.add_argument('--region_cache_stats_csv', type=str,
                        default='dataset_audit_results/region_cache_stats.csv')
    return parser.parse_args()


def _load_config(config_path: Path) -> Tuple[dict, SimpleNamespace]:
    with config_path.open('r') as f:
        config_dict = yaml.safe_load(f) or {}
    if not isinstance(config_dict, dict):
        raise ValueError('Config must be a YAML mapping: {}'.format(config_path))
    args = SimpleNamespace(**config_dict)
    args.no_cuda = bool(getattr(args, 'no_cuda', False))
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    args.best_model_path = os.path.expanduser(str(getattr(args, 'best_model_path', '')))
    return config_dict, args


def _resolve_path(base: Path, maybe_rel: str) -> Optional[Path]:
    if not maybe_rel:
        return None
    p = Path(os.path.expanduser(maybe_rel))
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _prepare_args(cli_args: argparse.Namespace) -> Tuple[dict, SimpleNamespace, Path, Path, Path]:
    config_path = _resolve_path(REPO_ROOT, cli_args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError('Config not found: {}'.format(cli_args.config))
    config_dict, args = _load_config(config_path)
    if cli_args.cpu:
        args.cuda = False
        args.no_cuda = True

    checkpoint_path = _resolve_path(REPO_ROOT, cli_args.checkpoint or getattr(args, 'best_model_path', ''))
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError('Checkpoint not found: {}'.format(checkpoint_path))

    if cli_args.output_dir:
        output_dir = _resolve_path(REPO_ROOT, cli_args.output_dir)
    else:
        output_dir = (REPO_ROOT / 'outputs' / 'diagnostics' / checkpoint_path.stem).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    return config_dict, args, config_path, checkpoint_path, output_dir


def _load_csv_if_exists(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _case_id_from_path(path: str) -> str:
    name = os.path.basename(path)
    for suffix in ('.nii.gz', '.nii'):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]


def _safe_sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _decode_with_thresholds(cumprobs: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    pred = np.zeros((cumprobs.shape[0],), dtype=np.int64)
    for k in range(cumprobs.shape[1]):
        passed = cumprobs[:, k] > thresholds[0, k]
        if k == 0:
            pred = pred + passed.astype(np.int64)
        else:
            pred = pred + ((pred == k) & passed).astype(np.int64)
    return pred


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(conf, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)
    return conf


def _class_metrics_from_conf(conf: np.ndarray) -> List[dict]:
    rows = []
    num_classes = conf.shape[0]
    for c in range(num_classes):
        tp = int(conf[c, c])
        fn = int(conf[c, :].sum() - tp)
        fp = int(conf[:, c].sum() - tp)
        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            'class_id': c,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(conf[c, :].sum()),
        })
    return rows


def _quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float('nan')
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    min_rating = int(min(y_true.min(), y_pred.min()))
    max_rating = int(max(y_true.max(), y_pred.max()))
    num_ratings = max_rating - min_rating + 1
    if num_ratings <= 1:
        return 1.0

    conf = np.zeros((num_ratings, num_ratings), dtype=np.float64)
    np.add.at(conf, (y_true - min_rating, y_pred - min_rating), 1.0)
    hist_true = conf.sum(axis=1)
    hist_pred = conf.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / conf.sum().clip(min=1.0)

    grid = np.arange(num_ratings, dtype=np.float64)
    weights = ((grid[:, None] - grid[None, :]) ** 2) / float((num_ratings - 1) ** 2)
    observed_term = float((weights * conf).sum())
    expected_term = float((weights * expected).sum())
    if expected_term <= 1e-12:
        return float('nan')
    return float(1.0 - (observed_term / expected_term))


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    order = np.argsort(y_score, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.shape[0] + 1, dtype=np.float64)

    sorted_scores = y_score[order]
    start = 0
    while start < sorted_scores.shape[0]:
        end = start + 1
        while end < sorted_scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            mean_rank = ranks[order[start:end]].mean()
            ranks[order[start:end]] = mean_rank
        start = end

    rank_sum_pos = float(ranks[pos_mask].sum())
    return float((rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg))


def _compute_metrics(df: pd.DataFrame, pred_col: str, num_classes: int) -> Dict[str, object]:
    if df.empty:
        return {}
    y_true = df['true_label'].to_numpy(dtype=np.int64)
    y_pred = df[pred_col].to_numpy(dtype=np.int64)
    conf = _confusion_matrix(y_true, y_pred, num_classes)
    class_rows = _class_metrics_from_conf(conf)
    acc = float((y_true == y_pred).mean())
    balanced_acc = float(np.mean([row['recall'] for row in class_rows]))
    macro_f1 = float(np.mean([row['f1'] for row in class_rows]))
    mae = float(np.abs(y_true - y_pred).mean())
    adjacent_acc = float((np.abs(y_true - y_pred) <= 1).mean())
    qwk = _quadratic_weighted_kappa(y_true, y_pred)

    true1_mask = (y_true == 1)
    true2_mask = (y_true == 2)
    true1_to_2 = int(np.logical_and(true1_mask, y_pred == 2).sum())
    true2_to_1 = int(np.logical_and(true2_mask, y_pred == 1).sum())

    return {
        'n_cases': int(df.shape[0]),
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'mae': mae,
        'adjacent_accuracy': adjacent_acc,
        'qwk': qwk,
        'confusion_matrix': conf.tolist(),
        'class_metrics': class_rows,
        'class1_recall': class_rows[1]['recall'] if num_classes > 1 else float('nan'),
        'class2_recall': class_rows[2]['recall'] if num_classes > 2 else float('nan'),
        'true1_to_pred2_count': true1_to_2,
        'true1_to_pred2_ratio': (true1_to_2 / float(true1_mask.sum())) if true1_mask.any() else float('nan'),
        'true2_to_pred1_count': true2_to_1,
        'true2_to_pred1_ratio': (true2_to_1 / float(true2_mask.sum())) if true2_mask.any() else float('nan'),
    }


def _metrics_row(split: str, pred_name: str, metrics: Dict[str, object]) -> dict:
    row = {
        'split': split,
        'prediction': pred_name,
        'n_cases': metrics.get('n_cases'),
        'accuracy': metrics.get('accuracy'),
        'balanced_accuracy': metrics.get('balanced_accuracy'),
        'macro_f1': metrics.get('macro_f1'),
        'mae': metrics.get('mae'),
        'adjacent_accuracy': metrics.get('adjacent_accuracy'),
        'qwk': metrics.get('qwk'),
        'class1_recall': metrics.get('class1_recall'),
        'class2_recall': metrics.get('class2_recall'),
        'true1_to_pred2_count': metrics.get('true1_to_pred2_count'),
        'true1_to_pred2_ratio': metrics.get('true1_to_pred2_ratio'),
        'true2_to_pred1_count': metrics.get('true2_to_pred1_count'),
        'true2_to_pred1_ratio': metrics.get('true2_to_pred1_ratio'),
    }
    return row


def _extract_scalar_stats(series: pd.Series) -> dict:
    clean = pd.to_numeric(series, errors='coerce').dropna()
    if clean.empty:
        return {
            'n': 0,
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'q1': np.nan,
            'median': np.nan,
            'q3': np.nan,
            'max': np.nan,
        }
    return {
        'n': int(clean.shape[0]),
        'mean': float(clean.mean()),
        'std': float(clean.std(ddof=0)),
        'min': float(clean.min()),
        'q1': float(clean.quantile(0.25)),
        'median': float(clean.median()),
        'q3': float(clean.quantile(0.75)),
        'max': float(clean.max()),
    }


def _try_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    x_num = pd.to_numeric(x, errors='coerce')
    y_num = pd.to_numeric(y, errors='coerce')
    mask = x_num.notna() & y_num.notna()
    x_arr = x_num[mask].to_numpy(dtype=np.float64)
    y_arr = y_num[mask].to_numpy(dtype=np.float64)
    if x_arr.size < 2 or np.unique(x_arr).size < 2 or np.unique(y_arr).size < 2:
        return float('nan')
    if method == 'pearson' and pearsonr is not None:
        return float(pearsonr(x_arr, y_arr)[0])
    if method == 'spearman' and spearmanr is not None:
        return float(spearmanr(x_arr, y_arr)[0])
    return float(pd.Series(x_arr).corr(pd.Series(y_arr), method=method))


def _record_case_level_crop_summary(case_id: str, split: str, crop_df: pd.DataFrame) -> dict:
    row = {'case_id': case_id, 'split': split}
    for col in (
        'bbox_h',
        'bbox_w',
        'bbox_phys_h_mm',
        'bbox_phys_w_mm',
        'resize_scale_y',
        'resize_scale_x',
        'mask_area',
        'lung_pixel_ratio',
    ):
        stats = _extract_scalar_stats(crop_df[col])
        row['{}_mean'.format(col)] = stats['mean']
        row['{}_std'.format(col)] = stats['std']
        row['{}_median'.format(col)] = stats['median']
        row['{}_q1'.format(col)] = stats['q1']
        row['{}_q3'.format(col)] = stats['q3']
    return row


def _grid_decode_metrics(df: pd.DataFrame, thresholds: Sequence[float], num_classes: int) -> Dict[str, float]:
    cum = df[['cumprob_y_gt_0', 'cumprob_y_gt_1', 'cumprob_y_gt_2']].to_numpy(dtype=np.float32)
    pred = _decode_with_thresholds(cum, thresholds)
    tmp = df[['true_label']].copy()
    tmp['pred'] = pred
    metrics = _compute_metrics(tmp.rename(columns={'pred': 'pred_label'}), 'pred_label', num_classes)
    return {
        'accuracy': float(metrics['accuracy']),
        'balanced_accuracy': float(metrics['balanced_accuracy']),
        'macro_f1': float(metrics['macro_f1']),
        'mae': float(metrics['mae']),
        'qwk': float(metrics['qwk']),
        'class1_recall': float(metrics['class1_recall']),
        'class2_recall': float(metrics['class2_recall']),
        'true1_to_pred2_count': int(metrics['true1_to_pred2_count']),
        'true2_to_pred1_count': int(metrics['true2_to_pred1_count']),
    }


def _objective_value(metrics: Dict[str, float], objective: str) -> float:
    if objective == 'mae':
        return -float(metrics['mae'])
    return float(metrics[objective])


def _search_single_threshold(val_df: pd.DataFrame, test_df: pd.DataFrame, objective: str, num_classes: int) -> SearchResult:
    coarse = np.round(np.arange(0.05, 0.951, 0.05), 4)
    best_t = 0.5
    best_score = -float('inf')
    for t in coarse:
        metrics = _grid_decode_metrics(val_df, (t, t, t), num_classes)
        score = _objective_value(metrics, objective)
        if score > best_score:
            best_score = score
            best_t = float(t)

    lo = max(0.01, best_t - 0.05)
    hi = min(0.99, best_t + 0.05)
    refine = np.round(np.arange(lo, hi + 1e-9, 0.01), 4)
    for t in refine:
        metrics = _grid_decode_metrics(val_df, (t, t, t), num_classes)
        score = _objective_value(metrics, objective)
        if score > best_score:
            best_score = score
            best_t = float(t)

    thresholds = (best_t, best_t, best_t)
    return SearchResult(
        objective=objective,
        strategy='single_threshold',
        thresholds=thresholds,
        val_metrics=_grid_decode_metrics(val_df, thresholds, num_classes),
        test_metrics=_grid_decode_metrics(test_df, thresholds, num_classes),
    )


def _search_per_threshold(val_df: pd.DataFrame, test_df: pd.DataFrame, objective: str, num_classes: int) -> SearchResult:
    coarse = np.round(np.arange(0.05, 0.951, 0.05), 4)
    best_thr = (0.5, 0.5, 0.5)
    best_score = -float('inf')
    for t0 in coarse:
        for t1 in coarse:
            for t2 in coarse:
                thr = (float(t0), float(t1), float(t2))
                metrics = _grid_decode_metrics(val_df, thr, num_classes)
                score = _objective_value(metrics, objective)
                if score > best_score:
                    best_score = score
                    best_thr = thr

    refine_axes = []
    for t in best_thr:
        lo = max(0.01, t - 0.05)
        hi = min(0.99, t + 0.05)
        refine_axes.append(np.round(np.arange(lo, hi + 1e-9, 0.01), 4))

    for t0 in refine_axes[0]:
        for t1 in refine_axes[1]:
            for t2 in refine_axes[2]:
                thr = (float(t0), float(t1), float(t2))
                metrics = _grid_decode_metrics(val_df, thr, num_classes)
                score = _objective_value(metrics, objective)
                if score > best_score:
                    best_score = score
                    best_thr = thr

    return SearchResult(
        objective=objective,
        strategy='per_threshold',
        thresholds=best_thr,
        val_metrics=_grid_decode_metrics(val_df, best_thr, num_classes),
        test_metrics=_grid_decode_metrics(test_df, best_thr, num_classes),
    )


def _threshold_binary_stats(df: pd.DataFrame, split: str) -> List[dict]:
    rows = []
    for thr_idx in ORDINAL_THRESHOLDS:
        y_true = (df['true_label'].to_numpy(dtype=np.int64) > thr_idx).astype(np.int64)
        score = df['cumprob_y_gt_{}'.format(thr_idx)].to_numpy(dtype=np.float64)
        pred = (score > 0.5).astype(np.int64)
        pos = int(y_true.sum())
        neg = int(y_true.shape[0] - pos)
        if pos > 0 and neg > 0:
            auc = _binary_roc_auc(y_true, score)
        else:
            auc = float('nan')
        tp = int(np.logical_and(y_true == 1, pred == 1).sum())
        tn = int(np.logical_and(y_true == 0, pred == 0).sum())
        fp = int(np.logical_and(y_true == 0, pred == 1).sum())
        fn = int(np.logical_and(y_true == 1, pred == 0).sum())
        sens = tp / float(tp + fn) if (tp + fn) > 0 else float('nan')
        spec = tn / float(tn + fp) if (tn + fp) > 0 else float('nan')
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2.0 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        bacc = np.nanmean([sens, spec])
        rows.append({
            'split': split,
            'threshold': thr_idx,
            'binary_target': 'y>{}'.format(thr_idx),
            'pos': pos,
            'neg': neg,
            'auc': auc,
            'balanced_accuracy_at_0.5': float(bacc),
            'sensitivity': float(sens),
            'specificity': float(spec),
            'f1': float(f1),
        })
    return rows


def _per_class_cumprob_stats(df: pd.DataFrame, split: str) -> List[dict]:
    rows = []
    for true_class in sorted(df['true_label'].unique().tolist()):
        sub = df[df['true_label'] == true_class]
        row = {'split': split, 'true_class': int(true_class), 'n': int(sub.shape[0])}
        for k in ORDINAL_THRESHOLDS:
            col = 'cumprob_y_gt_{}'.format(k)
            row['mean_P_y_gt_{}'.format(k)] = float(sub[col].mean())
            row['std_P_y_gt_{}'.format(k)] = float(sub[col].std(ddof=0))
        rows.append(row)
    return rows


def _extract_distribution_row(df: pd.DataFrame, split: str, label_name: str, mask: pd.Series, col: str) -> dict:
    stats = _extract_scalar_stats(df.loc[mask, col])
    return {
        'split': split,
        'group': label_name,
        'column': col,
        **stats,
    }


def _join_external_metadata(pred_df: pd.DataFrame, nifti_df: pd.DataFrame,
                            split_df: pd.DataFrame, region_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()

    if not split_df.empty:
        split_cols = [c for c in ('case_id', 'class', 'split', 'path') if c in split_df.columns]
        split_meta = split_df[split_cols].drop_duplicates('case_id')
        split_meta = split_meta.rename(columns={
            'class': 'split_assignment_class',
            'split': 'split_from_assignment_csv',
            'path': 'source_path_from_split_csv',
        })
        out = out.merge(split_meta, on='case_id', how='left')

    if not nifti_df.empty:
        keep_cols = [c for c in ('case_id', 'class', 'spacing_x', 'spacing_y', 'spacing_z') if c in nifti_df.columns]
        nifti_meta = nifti_df[keep_cols].drop_duplicates('case_id')
        nifti_meta = nifti_meta.rename(columns={
            'class': 'nifti_class',
            'spacing_x': 'nifti_spacing_x',
            'spacing_y': 'nifti_spacing_y',
            'spacing_z': 'nifti_spacing_z',
        })
        out = out.merge(nifti_meta, on='case_id', how='left')

    if not region_df.empty:
        keep_cols = [c for c in region_df.columns if c in (
            'case_id', 'meta_split_method', 'meta_z_spacing_mm', 'total_valid_centers',
            'left_upper_n_centers', 'left_middle_n_centers', 'left_lower_n_centers',
            'right_upper_n_centers', 'right_middle_n_centers', 'right_lower_n_centers'
        )]
        region_meta = region_df[keep_cols].drop_duplicates('case_id')
        out = out.merge(region_meta, on='case_id', how='left')

    if 'split_method' in out.columns:
        out['split_method'] = out['split_method'].fillna(out.get('meta_split_method'))
    elif 'meta_split_method' in out.columns:
        out['split_method'] = out['meta_split_method']
    else:
        out['split_method'] = np.nan

    for field in TEXT_METADATA_FIELDS:
        if field not in out.columns:
            out[field] = np.nan
    if 'slice_thickness' not in out.columns:
        out['slice_thickness'] = np.nan

    return out


def _read_case_dicom_metadata(raw_case_dir: Path) -> dict:
    fields = {
        'manufacturer': np.nan,
        'scanner_model': np.nan,
        'institution': np.nan,
        'protocol_name': np.nan,
        'reconstruction_kernel': np.nan,
        'slice_thickness': np.nan,
    }
    if pydicom is None or not raw_case_dir.exists() or not raw_case_dir.is_dir():
        return fields

    dicom_files = sorted([p for p in raw_case_dir.iterdir() if p.is_file() and p.suffix.lower() == '.dcm'])
    if not dicom_files:
        dicom_files = sorted([p for p in raw_case_dir.iterdir() if p.is_file()])
    if not dicom_files:
        return fields

    try:
        ds = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)
    except Exception:
        return fields

    mapping = {
        'manufacturer': 'Manufacturer',
        'scanner_model': 'ManufacturerModelName',
        'institution': 'InstitutionName',
        'protocol_name': 'ProtocolName',
        'reconstruction_kernel': 'ConvolutionKernel',
        'slice_thickness': 'SliceThickness',
    }
    for out_key, tag_name in mapping.items():
        value = getattr(ds, tag_name, None)
        if value is None or value == '':
            continue
        try:
            fields[out_key] = float(value) if out_key == 'slice_thickness' else str(value)
        except Exception:
            fields[out_key] = str(value)
    return fields


def _build_case_row(split: str, idx: int, dataset, logits_np: np.ndarray, pred_label: int, metadata: List[dict],
                    burden_stats: dict, coverage_stats: dict, spacing_xyz: Optional[Tuple[float, float, float]] = None) -> dict:
    path, _ = dataset.samples[idx]
    case_id = _case_id_from_path(path)
    sigmoids = _safe_sigmoid(logits_np)
    cumprobs = np.cumprod(sigmoids, axis=0)
    region_counts = Counter(str(m.get('region', 'unknown')) for m in metadata)
    split_method = next((str(m.get('split_method')) for m in metadata if m.get('split_method') is not None), np.nan)
    mask_source = next((str(m.get('mask_source')) for m in metadata if m.get('mask_source') is not None), np.nan)

    row = {
        'case_id': case_id,
        'path': path,
        'split': split,
        'true_label': int(dataset.samples[idx][1]),
        'pred_label_default': int(pred_label),
        'logit_0': float(logits_np[0]),
        'logit_1': float(logits_np[1]),
        'logit_2': float(logits_np[2]),
        'sigmoid_0': float(sigmoids[0]),
        'sigmoid_1': float(sigmoids[1]),
        'sigmoid_2': float(sigmoids[2]),
        'cumprob_y_gt_0': float(cumprobs[0]),
        'cumprob_y_gt_1': float(cumprobs[1]),
        'cumprob_y_gt_2': float(cumprobs[2]),
        'split_method': split_method,
        'mask_source': mask_source,
        'instance_count': int(len(metadata)),
        'region_instance_counts_json': json.dumps({k: int(region_counts.get(k, 0)) for k in REGION_ORDER}, sort_keys=True),
        'burden_soft_ratio': burden_stats.get('soft_ratio') if burden_stats else np.nan,
        'burden_score_mean': burden_stats.get('score_mean') if burden_stats else np.nan,
        'burden_topk_mean': burden_stats.get('topk_mean') if burden_stats else np.nan,
        'burden_score_std': burden_stats.get('score_std') if burden_stats else np.nan,
        'coverage_z_center': coverage_stats.get('z_center') if coverage_stats else np.nan,
        'coverage_z_spread': coverage_stats.get('z_spread') if coverage_stats else np.nan,
        'coverage_active_bins_soft': coverage_stats.get('active_bins_soft') if coverage_stats else np.nan,
        'coverage_instance_score_mean': coverage_stats.get('instance_score_mean') if coverage_stats else np.nan,
        'spacing_x': spacing_xyz[0] if spacing_xyz is not None else np.nan,
        'spacing_y': spacing_xyz[1] if spacing_xyz is not None else np.nan,
        'spacing_z': spacing_xyz[2] if spacing_xyz is not None else np.nan,
    }
    for region_name in REGION_ORDER:
        row['region_count_{}'.format(region_name)] = int(region_counts.get(region_name, 0))
    return row


def _rebuild_case_context(dataset, idx: int):
    nii_path, _ = dataset.samples[idx]
    image = sitk.ReadImage(nii_path)
    volume_zyx_raw = sitk.GetArrayFromImage(image).astype(np.float32)
    num_slices = int(volume_zyx_raw.shape[0])
    if dataset.fixed_num_slices > 0:
        selected_idx = dataset._select_lung_aligned_indices(volume_zyx_raw, num_slices)
    else:
        selected_start, selected_end = dataset._select_middle_slice_range(num_slices)
        selected_idx = np.arange(selected_start, selected_end, dtype=np.int64)
        if selected_idx.size < 3:
            selected_idx = np.arange(num_slices, dtype=np.int64)

    if dataset.instance_definition == 'lung_region_thin_slab' and dataset.cache_root:
        full_region_ctx, full_mask_source = dataset._get_or_build_full_region_skeleton(nii_path, volume_zyx_raw)
    else:
        full_region_ctx, full_mask_source = None, None

    selected_volume_raw = volume_zyx_raw[selected_idx]
    if dataset.instance_definition != 'lung_region_thin_slab':
        return image, selected_volume_raw, selected_idx, None, 'global_slab'

    if full_region_ctx is not None:
        region_ctx = dataset._slice_full_region_skeleton(
            full_region_ctx,
            selected_idx,
            num_slices=len(dataset.channel_offsets),
        )
        region_ctx['mask_source'] = full_mask_source
    else:
        region_ctx = dataset._load_region_context_for_case(nii_path, selected_idx, selected_volume_raw)
    return image, selected_volume_raw, selected_idx, region_ctx, region_ctx.get('mask_source', 'unknown')


def _compute_case_crop_rows(dataset, idx: int, split: str, metadata: List[dict]) -> Tuple[List[dict], dict]:
    image, volume_raw, _, region_ctx, mask_source = _rebuild_case_context(dataset, idx)
    spacing = image.GetSpacing()
    spacing_x = float(spacing[0]) if len(spacing) >= 1 else np.nan
    spacing_y = float(spacing[1]) if len(spacing) >= 2 else np.nan
    spacing_z = float(spacing[2]) if len(spacing) >= 3 else np.nan
    if region_ctx is None:
        return [], {}

    case_id = _case_id_from_path(dataset.samples[idx][0])
    crop_rows: List[dict] = []
    offsets = np.arange(-(len(dataset.channel_offsets) // 2), (len(dataset.channel_offsets) // 2) + 1, dtype=np.int64)
    z_total = int(volume_raw.shape[0])
    for inst_idx, meta in enumerate(metadata):
        region_name = str(meta.get('region', 'unknown'))
        center_z = int(meta.get('center_z', 0))
        region_mask = region_ctx['region_masks_dict'][region_name]
        z_ids = np.clip(center_z + offsets, 0, z_total - 1)
        slab_mask = region_mask[z_ids]
        mask_union = np.any(slab_mask, axis=0)
        if np.any(mask_union):
            y0, y1, x0, x1 = get_region_bbox(
                mask_union,
                margin=dataset.region_bbox_margin,
                min_size=dataset.region_bbox_min_size,
                image_shape=volume_raw.shape[1:],
            )
        else:
            y0, y1, x0, x1 = region_ctx['region_bboxes_dict'][region_name]
        bbox_h = int(y1 - y0)
        bbox_w = int(x1 - x0)
        bbox_area = float(max(bbox_h, 0) * max(bbox_w, 0))
        union_crop = mask_union[y0:y1, x0:x1] if np.any(mask_union) else region_mask[center_z, y0:y1, x0:x1]
        mask_area = float(np.asarray(union_crop, dtype=np.uint8).sum())
        crop_rows.append({
            'case_id': case_id,
            'split': split,
            'instance_idx': inst_idx,
            'region': region_name,
            'center_z': center_z,
            'split_method': region_ctx.get('split_method', np.nan),
            'mask_source': mask_source,
            'bbox_y0': int(y0),
            'bbox_y1': int(y1),
            'bbox_x0': int(x0),
            'bbox_x1': int(x1),
            'bbox_h': bbox_h,
            'bbox_w': bbox_w,
            'bbox_phys_h_mm': float(bbox_h * spacing_y) if not math.isnan(spacing_y) else np.nan,
            'bbox_phys_w_mm': float(bbox_w * spacing_x) if not math.isnan(spacing_x) else np.nan,
            'resize_scale_y': float(dataset.region_out_size[0] / max(bbox_h, 1)),
            'resize_scale_x': float(dataset.region_out_size[1] / max(bbox_w, 1)),
            'mask_area': mask_area,
            'lung_pixel_ratio': float(mask_area / bbox_area) if bbox_area > 0 else np.nan,
            'spacing_x': spacing_x,
            'spacing_y': spacing_y,
            'spacing_z': spacing_z,
        })

    crop_df = pd.DataFrame(crop_rows)
    return crop_rows, _record_case_level_crop_summary(case_id, split, crop_df) if not crop_df.empty else {}


def _collect_split_predictions(split: str, dataset, model, device: torch.device,
                               max_cases: int, compute_crop_stats: bool,
                               raw_data_root: Optional[Path], extract_dicom_metadata: bool,
                               num_workers: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    rows: List[dict] = []
    crop_rows: List[dict] = []
    limit = len(dataset) if max_cases <= 0 else min(len(dataset), max_cases)
    dicom_cache: Dict[Tuple[int, str], dict] = {}
    spacing_cache: Dict[str, Tuple[float, float, float]] = {}

    class _AuditEvalDataset(Dataset):
        def __init__(self, base_dataset, dataset_limit: int):
            self.base_dataset = base_dataset
            self.dataset_limit = dataset_limit

        def __len__(self) -> int:
            return self.dataset_limit

        def __getitem__(self, idx: int) -> dict:
            bag, label, pos_z = self.base_dataset[idx]
            return {
                'idx': idx,
                'bag': bag,
                'label': label,
                'pos_z': pos_z,
                'metadata': list(getattr(self.base_dataset, '_last_metadata', [])),
            }

    def _iter_serial() -> Iterable[dict]:
        for idx in range(limit):
            item = dataset[idx]
            yield {
                'idx': idx,
                'bag': item[0],
                'label': item[1],
                'pos_z': item[2],
                'metadata': list(getattr(dataset, '_last_metadata', [])),
            }

    iterator: Iterable[dict]
    if num_workers > 0:
        iterator = DataLoader(
            _AuditEvalDataset(dataset, limit),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
            collate_fn=lambda batch: batch[0],
            persistent_workers=True,
        )
    else:
        iterator = _iter_serial()

    with torch.no_grad():
        for payload in iterator:
            idx = int(payload['idx'])
            bag = payload['bag']
            label = payload['label']
            pos_z = payload['pos_z']
            metadata = list(payload.get('metadata', []))
            bag_dev = bag.unsqueeze(0).to(device, non_blocking=(device.type == 'cuda'))
            pos_z_dev = pos_z.unsqueeze(0).to(device, non_blocking=(device.type == 'cuda'))
            logits, y_hat, _ = model(bag_dev, pos_z_dev)
            logits_np = logits.detach().cpu().numpy().reshape(-1)
            pred_label = int(y_hat.detach().cpu().item())
            aux = model.last_forward_aux if hasattr(model, 'last_forward_aux') else {}
            nii_path = dataset.samples[idx][0]
            if nii_path not in spacing_cache:
                spacing_obj = sitk.ReadImage(nii_path).GetSpacing()
                spacing_cache[nii_path] = (
                    float(spacing_obj[0]) if len(spacing_obj) >= 1 else np.nan,
                    float(spacing_obj[1]) if len(spacing_obj) >= 2 else np.nan,
                    float(spacing_obj[2]) if len(spacing_obj) >= 3 else np.nan,
                )

            row = _build_case_row(
                split=split,
                idx=idx,
                dataset=dataset,
                logits_np=logits_np,
                pred_label=pred_label,
                metadata=metadata,
                burden_stats=aux.get('burden_stats') or {},
                coverage_stats=aux.get('coverage_stats') or {},
                spacing_xyz=spacing_cache[nii_path],
            )
            if extract_dicom_metadata and raw_data_root is not None:
                case_key = (int(row['true_label']), str(row['case_id']))
                if case_key not in dicom_cache:
                    dicom_cache[case_key] = _read_case_dicom_metadata(raw_data_root / str(row['true_label']) / str(row['case_id']))
                row.update(dicom_cache[case_key])
            if compute_crop_stats and metadata:
                case_crop_rows, crop_summary = _compute_case_crop_rows(dataset, idx, split, metadata)
                crop_rows.extend(case_crop_rows)
                row.update(crop_summary)
            rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(crop_rows)


def _save_confusion_tables(output_dir: Path, split_metrics: Dict[Tuple[str, str], Dict[str, object]]) -> None:
    rows = []
    for (split, pred_name), metrics in split_metrics.items():
        conf = np.asarray(metrics.get('confusion_matrix', []), dtype=np.int64)
        if conf.size == 0:
            continue
        for true_cls in range(conf.shape[0]):
            rows.append({
                'split': split,
                'prediction': pred_name,
                'true_label': true_cls,
                'pred0': int(conf[true_cls, 0]),
                'pred1': int(conf[true_cls, 1]),
                'pred2': int(conf[true_cls, 2]),
                'pred3': int(conf[true_cls, 3]),
            })
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / 'confusion_matrices.csv', index=False)


def _case_selection(df_test: pd.DataFrame) -> pd.DataFrame:
    selections = []

    def pick_group(group_name: str, mask: pd.Series, sort_key: str, ascending: bool, limit: int,
                   secondary_mode: Optional[str] = None) -> None:
        sub = df_test.loc[mask].copy()
        if sub.empty:
            return
        if secondary_mode == 'boundary':
            sub['priority'] = np.abs(sub['cumprob_y_gt_1'] - 0.5)
            sub = sub.sort_values(['priority', 'case_id'], ascending=[True, True])
        else:
            sub = sub.sort_values([sort_key, 'case_id'], ascending=[ascending, True])
        for rank, (_, row) in enumerate(sub.head(limit).iterrows(), start=1):
            rec = row.to_dict()
            rec['group'] = group_name
            rec['priority_rank'] = rank
            selections.append(rec)

    false21 = (df_test['true_label'] == 2) & (df_test['pred_label_default'] == 1)
    pick_group('A_false2_to_1_low_py_gt_1', false21, 'cumprob_y_gt_1', True, 5)
    remaining_false21 = false21 & ~df_test['case_id'].isin([r['case_id'] for r in selections])
    pick_group('A_false2_to_1_boundary_py_gt_1', remaining_false21, 'cumprob_y_gt_1', True, 5, secondary_mode='boundary')
    pick_group('B_true2_to_2_high_py_gt_1', (df_test['true_label'] == 2) & (df_test['pred_label_default'] == 2),
               'cumprob_y_gt_1', False, 10)
    pick_group('C_true1_to_1_low_py_gt_1', (df_test['true_label'] == 1) & (df_test['pred_label_default'] == 1),
               'cumprob_y_gt_1', True, 10)
    pick_group('D_true1_to_2_high_py_gt_1', (df_test['true_label'] == 1) & (df_test['pred_label_default'] == 2),
               'cumprob_y_gt_1', False, 10)
    return pd.DataFrame(selections)


def main() -> None:
    cli_args = _parse_args()
    config_dict, args, config_path, checkpoint_path, output_dir = _prepare_args(cli_args)
    requested_splits = tuple(s.strip() for s in cli_args.splits.split(',') if s.strip())
    if not requested_splits:
        raise ValueError('No splits requested.')

    split_assignment_path = _resolve_path(REPO_ROOT, cli_args.split_assignment_csv)
    nifti_metadata_path = _resolve_path(REPO_ROOT, cli_args.nifti_metadata_csv)
    region_cache_stats_path = _resolve_path(REPO_ROOT, cli_args.region_cache_stats_csv)
    split_df = _load_csv_if_exists(split_assignment_path)
    nifti_df = _load_csv_if_exists(nifti_metadata_path)
    region_df = _load_csv_if_exists(region_cache_stats_path)
    raw_data_root = Path(str(getattr(args, 'data_root', ''))).resolve() if getattr(args, 'data_root', '') else None

    train_dataset, val_dataset, test_dataset = _build_datasets(args)
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
    }
    if hasattr(train_dataset, 'set_epoch'):
        train_dataset.set_epoch(cli_args.train_eval_epoch)
    if hasattr(val_dataset, 'set_epoch') and val_dataset is not None:
        val_dataset.set_epoch(0)
    if hasattr(test_dataset, 'set_epoch') and test_dataset is not None:
        test_dataset.set_epoch(0)

    model = _build_model(args)
    device = torch.device('cuda' if args.cuda else 'cpu')
    model.to(device)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    split_frames: Dict[str, pd.DataFrame] = {}
    crop_frames: Dict[str, pd.DataFrame] = {}
    for split in requested_splits:
        dataset = datasets.get(split)
        if dataset is None:
            continue
        pred_df, crop_df = _collect_split_predictions(
            split=split,
            dataset=dataset,
            model=model,
            device=device,
            max_cases=cli_args.max_cases_per_split,
            compute_crop_stats=not cli_args.skip_crop_stats,
            raw_data_root=raw_data_root,
            extract_dicom_metadata=(not cli_args.no_extract_dicom_metadata),
            num_workers=max(0, int(getattr(cli_args, 'num_workers', 0))),
        )
        pred_df = _join_external_metadata(pred_df, nifti_df, split_df, region_df)
        pred_df['abs_error_default'] = (pred_df['pred_label_default'] - pred_df['true_label']).abs()
        pred_df['error_signed_default'] = pred_df['pred_label_default'] - pred_df['true_label']
        split_frames[split] = pred_df
        crop_frames[split] = crop_df
        pred_df.to_csv(output_dir / 'per_case_predictions_{}.csv'.format(split), index=False)
        if not crop_df.empty:
            crop_df.to_csv(output_dir / 'crop_stats_{}.csv'.format(split), index=False)

    combined_df = pd.concat([split_frames[s] for s in split_frames], ignore_index=True) if split_frames else pd.DataFrame()
    if combined_df.empty:
        raise RuntimeError('No predictions were generated.')
    combined_df.to_csv(output_dir / 'per_case_predictions_all.csv', index=False)

    split_metrics: Dict[Tuple[str, str], Dict[str, object]] = {}
    metric_rows = []
    for split, df in split_frames.items():
        metrics = _compute_metrics(df, 'pred_label_default', int(args.num_classes))
        split_metrics[(split, 'default')] = metrics
        metric_rows.append(_metrics_row(split, 'default', metrics))
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(output_dir / 'core_metrics.csv', index=False)
    _save_confusion_tables(output_dir, split_metrics)

    threshold_rows = []
    class_cumprob_rows = []
    py_gt1_rows = []
    false21_rows = []
    for split, df in split_frames.items():
        threshold_rows.extend(_threshold_binary_stats(df, split))
        class_cumprob_rows.extend(_per_class_cumprob_stats(df, split))
        for true_cls in (1, 2):
            sub = df[df['true_label'] == true_cls]
            stats = _extract_scalar_stats(sub['cumprob_y_gt_1'])
            py_gt1_rows.append({'split': split, 'true_class': true_cls, **stats})
        false21 = df[(df['true_label'] == 2) & (df['pred_label_default'] == 1)]
        stats = _extract_scalar_stats(false21['cumprob_y_gt_1'])
        false21_rows.append({
            'split': split,
            **stats,
            'count_between_0.45_0.55': int(false21['cumprob_y_gt_1'].between(0.45, 0.55, inclusive='both').sum()),
            'count_below_0.3': int((false21['cumprob_y_gt_1'] < 0.3).sum()),
            'count_below_0.2': int((false21['cumprob_y_gt_1'] < 0.2).sum()),
        })
    pd.DataFrame(threshold_rows).to_csv(output_dir / 'ordinal_threshold_binary_stats.csv', index=False)
    pd.DataFrame(class_cumprob_rows).to_csv(output_dir / 'ordinal_cumprob_by_true_class.csv', index=False)
    pd.DataFrame(py_gt1_rows).to_csv(output_dir / 'p_y_gt_1_distribution_true1_true2.csv', index=False)
    pd.DataFrame(false21_rows).to_csv(output_dir / 'false2_to_1_p_y_gt_1_distribution.csv', index=False)

    calibration_rows = []
    if 'val' in split_frames and 'test' in split_frames and not split_frames['val'].empty and not split_frames['test'].empty:
        default_val = _grid_decode_metrics(split_frames['val'], (0.5, 0.5, 0.5), int(args.num_classes))
        default_test = _grid_decode_metrics(split_frames['test'], (0.5, 0.5, 0.5), int(args.num_classes))
        for split_name, metrics in (('val', default_val), ('test', default_test)):
            calibration_rows.append({
                'objective': 'default',
                'strategy': 'default_0.5',
                'split': split_name,
                'thresholds': json.dumps([0.5, 0.5, 0.5]),
                **metrics,
            })
        for objective in ('macro_f1', 'balanced_accuracy', 'qwk', 'mae'):
            single_res = _search_single_threshold(split_frames['val'], split_frames['test'], objective, int(args.num_classes))
            per_res = _search_per_threshold(split_frames['val'], split_frames['test'], objective, int(args.num_classes))
            for res in (single_res, per_res):
                calibration_rows.append({
                    'objective': res.objective,
                    'strategy': res.strategy,
                    'split': 'val',
                    'thresholds': json.dumps(list(res.thresholds)),
                    **res.val_metrics,
                })
                calibration_rows.append({
                    'objective': res.objective,
                    'strategy': res.strategy,
                    'split': 'test',
                    'thresholds': json.dumps(list(res.thresholds)),
                    **res.test_metrics,
                })
                if 'test' in split_frames:
                    cum = split_frames['test'][['cumprob_y_gt_0', 'cumprob_y_gt_1', 'cumprob_y_gt_2']].to_numpy(dtype=np.float32)
                    split_frames['test']['pred_{}_{}'.format(res.strategy, res.objective)] = _decode_with_thresholds(cum, res.thresholds)
        pd.DataFrame(calibration_rows).to_csv(output_dir / 'threshold_calibration_results.csv', index=False)

    split_method_rows = []
    for split_name, df in [('all', combined_df)] + list(split_frames.items()):
        sub = df.copy()
        if sub.empty:
            continue
        counts = sub['split_method'].fillna('missing').value_counts(dropna=False)
        total = float(sub.shape[0])
        for method, count in counts.items():
            split_method_rows.append({
                'split': split_name,
                'split_method': method,
                'count': int(count),
                'percentage': float(count / total),
            })
    pd.DataFrame(split_method_rows).to_csv(output_dir / 'split_method_distribution.csv', index=False)

    split_method_crosstab = pd.crosstab(combined_df['true_label'], combined_df['split_method'].fillna('missing'))
    split_method_crosstab.to_csv(output_dir / 'label_by_split_method_crosstab.csv')

    test_split_method_rows = []
    if 'test' in split_frames:
        test_df = split_frames['test']
        for method, sub in test_df.groupby(test_df['split_method'].fillna('missing')):
            metrics = _compute_metrics(sub, 'pred_label_default', int(args.num_classes))
            row = _metrics_row('test', 'default', metrics)
            row['split_method'] = method
            test_split_method_rows.append(row)
    pd.DataFrame(test_split_method_rows).to_csv(output_dir / 'test_metrics_by_split_method.csv', index=False)

    feature_rows = []
    feature_cols = [
        'burden_soft_ratio', 'burden_score_mean', 'burden_topk_mean', 'burden_score_std',
        'coverage_z_center', 'coverage_z_spread', 'coverage_active_bins_soft', 'coverage_instance_score_mean',
    ]
    available_feature_cols = [c for c in feature_cols if c in combined_df.columns]
    for feature in available_feature_cols:
        for label_type in ('true_label', 'pred_label_default'):
            for label_value, sub in combined_df.groupby(label_type):
                feature_rows.append({
                    'feature': feature,
                    'group_by': label_type,
                    'label': int(label_value),
                    'mean': float(pd.to_numeric(sub[feature], errors='coerce').mean()),
                    'std': float(pd.to_numeric(sub[feature], errors='coerce').std(ddof=0)),
                    'n': int(sub.shape[0]),
                })
    pd.DataFrame(feature_rows).to_csv(output_dir / 'burden_coverage_group_stats.csv', index=False)

    corr_rows = []
    for feature in available_feature_cols:
        corr_rows.append({
            'feature': feature,
            'target': 'true_label',
            'pearson': _try_corr(combined_df[feature], combined_df['true_label'], 'pearson'),
            'spearman': _try_corr(combined_df[feature], combined_df['true_label'], 'spearman'),
        })
        corr_rows.append({
            'feature': feature,
            'target': 'pred_label_default',
            'pearson': _try_corr(combined_df[feature], combined_df['pred_label_default'], 'pearson'),
            'spearman': _try_corr(combined_df[feature], combined_df['pred_label_default'], 'spearman'),
        })
        corr_rows.append({
            'feature': feature,
            'target': 'abs_error_default',
            'pearson': _try_corr(combined_df[feature], combined_df['abs_error_default'], 'pearson'),
            'spearman': _try_corr(combined_df[feature], combined_df['abs_error_default'], 'spearman'),
        })
    pd.DataFrame(corr_rows).to_csv(output_dir / 'burden_coverage_correlations.csv', index=False)

    spacing_rows = []
    for split_name, df in [('all', combined_df)] + list(split_frames.items()):
        for label_value in sorted(df['true_label'].unique().tolist()):
            sub = df[df['true_label'] == label_value]
            for col in ('spacing_x', 'spacing_y', 'spacing_z', 'slice_thickness'):
                if col not in sub.columns:
                    continue
                stats = _extract_scalar_stats(sub[col])
                spacing_rows.append({
                    'split': split_name,
                    'true_label': int(label_value),
                    'column': col,
                    **stats,
                })
    pd.DataFrame(spacing_rows).to_csv(output_dir / 'spacing_stats_by_true_label.csv', index=False)

    false21_vs_true22_rows = []
    if 'test' in split_frames:
        test_df = split_frames['test']
        comparison_groups = {
            'false2_to_1': test_df[(test_df['true_label'] == 2) & (test_df['pred_label_default'] == 1)],
            'true2_to_2': test_df[(test_df['true_label'] == 2) & (test_df['pred_label_default'] == 2)],
        }
        for group_name, group_df in comparison_groups.items():
            for col in ('spacing_x', 'spacing_y', 'spacing_z'):
                stats = _extract_scalar_stats(group_df[col]) if col in group_df.columns else _extract_scalar_stats(pd.Series([], dtype=np.float64))
                false21_vs_true22_rows.append({
                    'group': group_name,
                    'column': col,
                    **stats,
                })
    pd.DataFrame(false21_vs_true22_rows).to_csv(output_dir / 'false2_to_1_vs_true2_to_2_spacing.csv', index=False)

    domain_missing = {
        'missing_numeric_fields': [f for f in NUMERIC_METADATA_FIELDS if f not in combined_df.columns or combined_df[f].isna().all()],
        'missing_text_fields': [
            f for f in TEXT_METADATA_FIELDS
            if f not in combined_df.columns or combined_df[f].fillna('').astype(str).str.strip().eq('').all()
        ],
    }
    with (output_dir / 'missing_metadata_report.json').open('w') as f:
        json.dump(domain_missing, f, indent=2)

    if 'test' in split_frames:
        case_selection_df = _case_selection(split_frames['test'])
        case_selection_df.to_csv(output_dir / 'test_case_selection_for_montage.csv', index=False)

    summary = {
        'config_path': str(config_path),
        'checkpoint_path': str(checkpoint_path),
        'output_dir': str(output_dir),
        'evaluated_splits': list(split_frames.keys()),
        'max_cases_per_split': int(cli_args.max_cases_per_split),
        'train_eval_epoch': int(cli_args.train_eval_epoch),
        'skip_crop_stats': bool(cli_args.skip_crop_stats),
        'core_metrics_path': str(output_dir / 'core_metrics.csv'),
        'combined_predictions_path': str(output_dir / 'per_case_predictions_all.csv'),
        'missing_metadata_report_path': str(output_dir / 'missing_metadata_report.json'),
    }
    with (output_dir / 'run_summary.json').open('w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()