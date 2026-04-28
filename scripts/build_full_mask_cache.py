"""Build full-volume lung mask and region skeleton cache for all CT cases.

Reads each CT NIfTI, computes the pseudo lung mask on the full volume,
and saves it as `full_lung_mask.npz` under the cache directory.
Also saves the selected_idx-independent region skeleton cache:
left/right lung masks, split_method, and six region masks.
This cache is independent of slice-selection strategy (selected_idx),
so it survives any changes to sampling methods.

Usage:
    conda run -n lung_new python scripts/build_full_mask_cache.py \
        --data_root /data/scratch/c_pne \
        --cache_root /data/scratch/c_pne_cache/lung_region_cache_v1 \
        --num_classes 4

Optional arguments:
    --lung_mask_root   Path to real lung masks (skips pseudo mask computation)
    --lung_mask_suffix Suffix for real mask files (default: .nii.gz)
    --overwrite        Rebuild cache even if full_lung_mask.npz already exists
    --workers          Number of worker processes (default: 4)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import SimpleITK as sitk

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from datasets.ct_preprocess.cache_io import (
    load_full_mask_cache,
    load_full_region_skeleton_cache,
    save_full_mask_cache,
    save_full_region_skeleton_cache,
)
from datasets.ct_preprocess.instance_builder import build_lung_region_skeleton_from_mask
from datasets.ct_preprocess.lung_mask import build_pseudo_lung_mask
from datasets.ct_preprocess.slice_utils import get_case_id_from_path


def iter_case_paths(data_root, num_classes):
    """Yield (ct_path, case_id) for all cases in standard N_seg_nii layout."""
    for cls in range(int(num_classes)):
        folder = os.path.join(data_root, '{}_seg_nii'.format(cls))
        if not os.path.isdir(folder):
            continue
        for ct_path in sorted(glob.glob(os.path.join(folder, '*.nii.gz'))):
            yield ct_path, get_case_id_from_path(ct_path)


def resolve_mask_path(ct_path, data_root, lung_mask_root, lung_mask_suffix):
    """Return real lung mask path if lung_mask_root is set, else None."""
    if not lung_mask_root:
        return None
    rel = os.path.relpath(ct_path, data_root)
    rel_base = os.path.splitext(os.path.splitext(rel)[0])[0]
    return os.path.join(lung_mask_root, rel_base + lung_mask_suffix)


def build_one(args_tuple):
    """Worker function: build full-volume lung mask cache for one case."""
    (ct_path, case_id, cache_root, data_root,
     lung_mask_root, lung_mask_suffix, overwrite,
     pseudo_value_threshold, pseudo_min_voxels) = args_tuple

    try:
        # Skip if cache already exists and overwrite is off
        if not overwrite:
            existing = load_full_region_skeleton_cache(case_id, cache_root)
            if existing is not None:
                return 'skipped', case_id, None

        mask_zyx = load_full_mask_cache(case_id, cache_root) if not overwrite else None
        source = 'full_mask_cache' if mask_zyx is not None else None

        if mask_zyx is None:
            mask_path = resolve_mask_path(ct_path, data_root, lung_mask_root, lung_mask_suffix)

            if mask_path is not None and os.path.exists(mask_path):
                mask_img = sitk.ReadImage(mask_path)
                mask_zyx = sitk.GetArrayFromImage(mask_img).astype(bool)
                source = 'real_mask'
            else:
                # Read full CT volume and compute pseudo mask
                image = sitk.ReadImage(ct_path)
                volume_raw = sitk.GetArrayFromImage(image).astype(np.float32)
                mask_zyx = build_pseudo_lung_mask(
                    volume_raw,
                    value_threshold=pseudo_value_threshold,
                    min_component_voxels=pseudo_min_voxels,
                ).astype(bool)
                source = 'pseudo_mask'

        save_full_mask_cache(case_id, cache_root, mask_zyx)
        save_full_region_skeleton_cache(
            case_id,
            cache_root,
            build_lung_region_skeleton_from_mask(mask_zyx),
        )
        return 'ok', case_id, source

    except Exception as e:
        return 'error', case_id, traceback.format_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Pre-build full-volume lung mask cache for all CT cases'
    )
    parser.add_argument('--data_root', type=str, required=True,
                        help='Data root containing N_seg_nii class folders')
    parser.add_argument('--cache_root', type=str, required=True,
                        help='Cache root (same as training cache_root config)')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of class folders (default: 4)')
    parser.add_argument('--lung_mask_root', type=str, default='',
                        help='Root for real lung masks (optional)')
    parser.add_argument('--lung_mask_suffix', type=str, default='.nii.gz',
                        help='Filename suffix for real lung masks (default: .nii.gz)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Rebuild cache even if full_lung_mask.npz already exists')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel worker processes (default: 4)')
    parser.add_argument('--pseudo_mask_value_threshold', type=float, default=1e-6,
                        help='Value threshold for pseudo mask computation')
    parser.add_argument('--pseudo_mask_min_component_voxels', type=int, default=512,
                        help='Min connected component size for pseudo mask')
    args = parser.parse_args()

    os.makedirs(args.cache_root, exist_ok=True)

    case_list = list(iter_case_paths(args.data_root, args.num_classes))
    if not case_list:
        print('[ERROR] No cases found under {}'.format(args.data_root))
        sys.exit(1)

    print('Found {} cases. cache_root={}'.format(len(case_list), args.cache_root))
    print('workers={}, overwrite={}'.format(args.workers, args.overwrite))

    task_args = [
        (ct_path, case_id,
         args.cache_root, args.data_root,
         args.lung_mask_root, args.lung_mask_suffix,
         args.overwrite,
         args.pseudo_mask_value_threshold,
         args.pseudo_mask_min_component_voxels)
        for ct_path, case_id in case_list
    ]

    n_ok = n_skip = n_err = 0
    errors = []

    if args.workers <= 1:
        iterator = (build_one(t) for t in task_args)
        items = task_args
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        futures = {executor.submit(build_one, t): t[1] for t in task_args}
        iterator = as_completed(futures)
        items = futures

    if HAS_TQDM:
        iterator = tqdm(iterator, total=len(task_args), desc='Building full mask cache')

    for result in iterator:
        if args.workers > 1:
            status, case_id, detail = result.result()
        else:
            status, case_id, detail = result

        if status == 'ok':
            n_ok += 1
        elif status == 'skipped':
            n_skip += 1
        else:
            n_err += 1
            errors.append((case_id, detail))
            print('\n[ERROR] {}: {}'.format(case_id, detail))

    if args.workers > 1:
        executor.shutdown(wait=False)

    print('\n=== Done ===')
    print('  built  : {}'.format(n_ok))
    print('  skipped: {} (already cached)'.format(n_skip))
    print('  errors : {}'.format(n_err))
    if errors:
        print('Failed cases:')
        for cid, _ in errors:
            print('  ', cid)


if __name__ == '__main__':
    main()
