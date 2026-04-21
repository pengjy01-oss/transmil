"""Slice selection and case path utilities for CT preprocessing."""

import glob
import os
import numpy as np


CACHE_VERSION = 'lung_region_cache_v1'


def get_case_id_from_path(file_path):
    base = os.path.basename(file_path)
    if base.endswith('.nii.gz'):
        return base[:-7]
    return os.path.splitext(base)[0]


def iter_case_paths(root_dir, num_classes):
    """Yield (case_id, ct_path, class_id) from the standard *_seg_nii layout."""
    for cls in range(int(num_classes)):
        folder = os.path.join(root_dir, '{}_seg_nii'.format(cls))
        for ct_path in sorted(glob.glob(os.path.join(folder, '*.nii.gz'))):
            yield get_case_id_from_path(ct_path), ct_path, int(cls)


def select_case_slice_indices(num_slices, z_spacing_mm=1.0, fixed_num_slices=256, middle_ratio=0.98):
    """Select the same slice subset used by the dataset before region processing."""
    num_slices = int(num_slices)
    if num_slices <= 0:
        return np.asarray([], dtype=np.int64)

    if fixed_num_slices > 0:
        if z_spacing_mm <= 0.0:
            z_spacing_mm = 1.0

        target = int(fixed_num_slices)
        center_pos = 0.5 * float(num_slices - 1)

        half = target // 2
        if target % 2 == 0:
            relative_mm = np.arange(-half, half, dtype=np.float32)
        else:
            relative_mm = np.arange(-half, half + 1, dtype=np.float32)

        relative_idx = np.rint(relative_mm / float(z_spacing_mm)).astype(np.int64)
        centered_idx = np.rint(center_pos).astype(np.int64) + relative_idx
        return np.clip(centered_idx, 0, num_slices - 1).astype(np.int64)

    if num_slices < 3:
        return np.arange(num_slices, dtype=np.int64)

    drop_each_side = int(num_slices * (1.0 - float(middle_ratio)) / 2.0)
    start = drop_each_side
    end = num_slices - drop_each_side
    if (end - start) < 3:
        start, end = 0, num_slices
    selected_idx = np.arange(start, end, dtype=np.int64)
    if selected_idx.size < 3:
        selected_idx = np.arange(num_slices, dtype=np.int64)
    return selected_idx
