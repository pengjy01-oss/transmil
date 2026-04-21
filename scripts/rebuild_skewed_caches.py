"""Rebuild cache ONLY for cases where left/right lung split is severely skewed."""
from __future__ import annotations

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed

from dataloader import (
    CACHE_VERSION,
    build_case_preprocess_cache,
    iter_case_paths,
    load_case_cache,
    save_case_cache,
    unpack_region_context_cache,
    select_case_slice_indices,
)


def find_skewed_cases(cache_root, skew_threshold=0.85):
    """Return set of case_ids where left_ratio > skew_threshold."""
    skewed = {}
    cases = sorted(d for d in os.listdir(cache_root) if d.startswith('CT'))
    for case_id in cases:
        cached = load_case_cache(case_id, cache_root)
        if cached is None:
            continue
        ctx = unpack_region_context_cache(cached['cache'])
        left_n = int(np.count_nonzero(ctx['left_lung_mask']))
        right_n = int(np.count_nonzero(ctx['right_lung_mask']))
        total = left_n + right_n
        if total == 0:
            continue
        left_ratio = left_n / total
        if left_ratio > skew_threshold or (1 - left_ratio) > skew_threshold:
            skewed[case_id] = max(left_ratio, 1 - left_ratio)
    return skewed


def rebuild_one(ct_path, case_id, class_id, args):
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

    save_case_cache(case_id, args.cache_root, cache_dict, meta_dict)

    # Verify the new cache
    new_cached = load_case_cache(case_id, args.cache_root)
    new_ctx = unpack_region_context_cache(new_cached['cache'])
    left_n = int(np.count_nonzero(new_ctx['left_lung_mask']))
    right_n = int(np.count_nonzero(new_ctx['right_lung_mask']))
    total = left_n + right_n
    new_ratio = left_n / total if total > 0 else 0.5
    split_method = new_cached['meta'].get('split_method', '?')

    return case_id, new_ratio, split_method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data/scratch/c_pne')
    parser.add_argument('--cache_root', type=str, default='/data/scratch/c_pne_cache/lung_region_cache_v1')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--fixed_num_slices', type=int, default=256)
    parser.add_argument('--middle_ratio', type=float, default=0.98)
    parser.add_argument('--num_slices_for_valid_centers', type=int, default=3)
    parser.add_argument('--abs_threshold', type=float, default=50.0)
    parser.add_argument('--ratio_threshold', type=float, default=0.05)
    parser.add_argument('--pseudo_mask_value_threshold', type=float, default=1e-6)
    parser.add_argument('--pseudo_mask_min_component_voxels', type=int, default=512)
    parser.add_argument('--skew_threshold', type=float, default=0.85,
                        help='left_ratio > this value is considered skewed')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    # Phase 1: Find skewed cases
    print("=" * 60)
    print("Phase 1: Scanning existing caches for skewed left/right splits...")
    print("=" * 60)
    t0 = time.time()
    skewed = find_skewed_cases(args.cache_root, args.skew_threshold)
    scan_time = time.time() - t0
    print(f"Scan complete in {scan_time:.1f}s. Found {len(skewed)} skewed cases (threshold={args.skew_threshold})")
    if not skewed:
        print("No skewed cases found. Nothing to rebuild.")
        return

    # Show top 10
    top = sorted(skewed.items(), key=lambda x: -x[1])[:10]
    for cid, ratio in top:
        print(f"  {cid}: dominant_side_ratio={ratio:.3f}")
    if len(skewed) > 10:
        print(f"  ... and {len(skewed) - 10} more")

    # Phase 2: Build case path lookup
    print(f"\n{'=' * 60}")
    print(f"Phase 2: Rebuilding {len(skewed)} skewed cases with {args.workers} workers...")
    print("=" * 60)
    all_entries = {cid: (ct_path, cls_id) for cid, ct_path, cls_id in iter_case_paths(args.data_root, args.num_classes)}

    to_rebuild = []
    for case_id in skewed:
        if case_id in all_entries:
            ct_path, cls_id = all_entries[case_id]
            to_rebuild.append((ct_path, case_id, cls_id))
        else:
            print(f"  [warn] {case_id} not found in data_root, skipping")

    t1 = time.time()
    ok = 0
    fail = 0
    still_skewed = 0

    if args.workers <= 1:
        for i, (ct_path, case_id, cls_id) in enumerate(to_rebuild, 1):
            try:
                cid, new_ratio, method = rebuild_one(ct_path, case_id, cls_id, args)
                ok += 1
                status = "FIXED" if 0.15 <= new_ratio <= 0.85 else "STILL_SKEWED"
                if status == "STILL_SKEWED":
                    still_skewed += 1
                print(f"  [{i}/{len(to_rebuild)}] {cid}: left={new_ratio:.1%} method={method} -> {status}")
            except Exception as e:
                fail += 1
                print(f"  [{i}/{len(to_rebuild)}] {case_id}: FAILED - {e}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(rebuild_one, ct_path, case_id, cls_id, args): case_id
                for ct_path, case_id, cls_id in to_rebuild
            }
            done_count = 0
            for future in as_completed(future_map):
                done_count += 1
                case_id = future_map[future]
                try:
                    cid, new_ratio, method = future.result()
                    ok += 1
                    status = "FIXED" if 0.15 <= new_ratio <= 0.85 else "STILL_SKEWED"
                    if status == "STILL_SKEWED":
                        still_skewed += 1
                    print(f"  [{done_count}/{len(to_rebuild)}] {cid}: left={new_ratio:.1%} method={method} -> {status}")
                except Exception as e:
                    fail += 1
                    print(f"  [{done_count}/{len(to_rebuild)}] {case_id}: FAILED - {e}")

    elapsed = time.time() - t1
    print(f"\n{'=' * 60}")
    print(f"Rebuild complete in {elapsed:.1f}s")
    print(f"  OK: {ok}  |  Failed: {fail}  |  Still skewed: {still_skewed}")
    print("=" * 60)


if __name__ == '__main__':
    main()
