from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import random


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import SimpleITK as sitk

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from dataloader import (
    CACHE_VERSION,
    build_case_preprocess_cache,
    iter_case_paths,
    load_case_cache,
    save_case_cache,
    save_region_context_debug,
    select_case_slice_indices,
)


def _iter_with_progress(items, desc):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, leave=False)


def build_one_case(ct_path, case_id, class_id, args):
    existing = load_case_cache(case_id, args.cache_root)
    if existing is not None and not args.overwrite:
        return 'skipped', case_id, None

    image = sitk.ReadImage(ct_path)
    volume_raw = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = image.GetSpacing()
    z_spacing_mm = float(spacing_xyz[2]) if len(spacing_xyz) >= 3 else 1.0

    selected_idx = select_case_slice_indices(
        num_slices=volume_raw.shape[0],
        z_spacing_mm=z_spacing_mm,
        fixed_num_slices=args.fixed_num_slices,
        middle_ratio=args.middle_ratio,
    )
    selected_volume = volume_raw[selected_idx]

    cache_dict, meta_dict, region_ctx = build_case_preprocess_cache(
        masked_ct=selected_volume,
        case_id=case_id,
        source_ct_path=ct_path,
        selected_idx=selected_idx,
        abs_threshold=args.abs_threshold,
        ratio_threshold=args.ratio_threshold,
        num_slices_for_valid_centers=args.num_slices_for_valid_centers,
        cache_version=CACHE_VERSION,
        pseudo_mask_value_threshold=args.pseudo_mask_value_threshold,
        pseudo_mask_min_component_voxels=args.pseudo_mask_min_component_voxels,
    )
    meta_dict['class_id'] = int(class_id)
    meta_dict['fixed_num_slices'] = int(args.fixed_num_slices)
    meta_dict['middle_ratio'] = float(args.middle_ratio)
    meta_dict['z_spacing_mm'] = float(z_spacing_mm)
    meta_dict['selected_count'] = int(len(selected_idx))
    meta_dict['selected_idx_min'] = int(np.min(selected_idx)) if len(selected_idx) > 0 else -1
    meta_dict['selected_idx_max'] = int(np.max(selected_idx)) if len(selected_idx) > 0 else -1
    meta_dict['selected_idx'] = np.asarray(selected_idx, dtype=np.int64).tolist()

    save_case_cache(case_id, args.cache_root, cache_dict, meta_dict)

    if hasattr(args, 'debug_cases_set') and args.debug_cases_set and case_id in args.debug_cases_set:
        save_region_context_debug(case_id, args.debug_root, region_ctx['pseudo_mask'], region_ctx)
        debug_saved = True
    else:
        debug_saved = False

    return 'ok', case_id, debug_saved


def main():
    parser = argparse.ArgumentParser(description='Build offline lung-region cache from masked CT volumes')
    parser.add_argument('--data_root', type=str, required=True, help='root directory containing *_seg_nii folders')
    parser.add_argument('--cache_root', type=str, required=True, help='output cache root directory')
    parser.add_argument('--num_classes', type=int, default=4, help='number of class folders')
    parser.add_argument('--fixed_num_slices', type=int, default=256, help='same slice selection setting as training')
    parser.add_argument('--middle_ratio', type=float, default=0.98, help='same middle slice ratio as training')
    parser.add_argument('--num_slices_for_valid_centers', type=int, default=3, help='thin-slab depth used to compute valid centers')
    parser.add_argument('--abs_threshold', type=float, default=50.0, help='absolute area threshold for valid center selection')
    parser.add_argument('--ratio_threshold', type=float, default=0.05, help='relative area threshold for valid center selection')
    parser.add_argument('--pseudo_mask_value_threshold', type=float, default=1e-6, help='threshold for pseudo mask from masked CT')
    parser.add_argument('--pseudo_mask_min_component_voxels', type=int, default=512, help='minimum component voxels kept in pseudo mask')
    parser.add_argument('--overwrite', action='store_true', default=False, help='rebuild cache even if an existing cache is present')
    parser.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 1))), help='number of worker processes used to build cache')
    parser.add_argument('--seed', type=int, default=1, help='random seed for choosing debug cases')
    parser.add_argument('--debug_cases', type=int, default=0, help='number of randomly selected cases to export debug visualizations')
    parser.add_argument('--debug_root', type=str, default='', help='directory for debug visualizations; defaults to cache_root/_debug')
    args = parser.parse_args()

    os.makedirs(args.cache_root, exist_ok=True)
    debug_root = args.debug_root.strip() or os.path.join(args.cache_root, '_debug')
    os.makedirs(debug_root, exist_ok=True)

    rng = random.Random(args.seed)
    case_entries = list(iter_case_paths(args.data_root, args.num_classes))
    if len(case_entries) == 0:
        raise RuntimeError('No cases found under {}'.format(args.data_root))

    debug_cases_set = set()
    if args.debug_cases > 0:
        debug_cases_set = set(case_id for case_id, _, _ in rng.sample(case_entries, k=min(args.debug_cases, len(case_entries))))
    args.debug_cases_set = debug_cases_set

    total = len(case_entries)
    ok_count = 0
    skip_count = 0
    fail_count = 0

    if args.workers <= 1:
        for case_id, ct_path, class_id in _iter_with_progress(case_entries, desc='building cache'):
            try:
                status, out_case_id, debug_saved = build_one_case(ct_path, case_id, class_id, args)
                if status == 'skipped':
                    skip_count += 1
                    print('[skip] {} exists and cache_version matches'.format(out_case_id))
                else:
                    ok_count += 1
                    print('[ok] {} -> {}'.format(out_case_id, os.path.join(args.cache_root, out_case_id)))
                    if debug_saved:
                        print('[debug] saved visualization for {}'.format(out_case_id))
            except Exception as exc:
                fail_count += 1
                print('[fail] {} -> {}'.format(case_id, exc))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(build_one_case, ct_path, case_id, class_id, args): case_id
                for case_id, ct_path, class_id in case_entries
            }
            for future in _iter_with_progress(as_completed(future_map), desc='building cache'):
                case_id = future_map[future]
                try:
                    status, out_case_id, debug_saved = future.result()
                    if status == 'skipped':
                        skip_count += 1
                        print('[skip] {} exists and cache_version matches'.format(out_case_id))
                    else:
                        ok_count += 1
                        print('[ok] {} -> {}'.format(out_case_id, os.path.join(args.cache_root, out_case_id)))
                        if debug_saved:
                            print('[debug] saved visualization for {}'.format(out_case_id))
                except Exception as exc:
                    fail_count += 1
                    print('[fail] {} -> {}'.format(case_id, exc))

    summary = {
        'cache_version': CACHE_VERSION,
        'data_root': args.data_root,
        'cache_root': args.cache_root,
        'num_cases': total,
        'ok': ok_count,
        'skipped': skip_count,
        'failed': fail_count,
        'abs_threshold': args.abs_threshold,
        'ratio_threshold': args.ratio_threshold,
        'fixed_num_slices': args.fixed_num_slices,
        'middle_ratio': args.middle_ratio,
        'num_slices_for_valid_centers': args.num_slices_for_valid_centers,
    }
    summary_path = os.path.join(args.cache_root, 'cache_build_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print('summary saved to {}'.format(summary_path))
    print(summary)


if __name__ == '__main__':
    main()
