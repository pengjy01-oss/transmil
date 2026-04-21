"""Six lung region partitioning: z-range splitting, region masks, valid centers, bboxes."""

import numpy as np


def _split_z_to_three_ranges(mask_3d):
    z_idx = np.where(mask_3d.reshape(mask_3d.shape[0], -1).any(axis=1))[0]
    if z_idx.size == 0:
        return [(0, 0), (0, 0), (0, 0)]

    z_start = int(z_idx.min())
    z_end = int(z_idx.max()) + 1
    z_all = np.arange(z_start, z_end, dtype=np.int64)
    chunks = np.array_split(z_all, 3)
    ranges = []
    for ch in chunks:
        if ch.size == 0:
            ranges.append((z_start, z_start))
        else:
            ranges.append((int(ch[0]), int(ch[-1]) + 1))
    return ranges


def _make_region_mask(side_mask, z_range):
    z0, z1 = int(z_range[0]), int(z_range[1])
    out = np.zeros_like(side_mask, dtype=bool)
    if z1 > z0:
        out[z0:z1] = side_mask[z0:z1]
    return out


def get_lung_region_ranges(lung_mask_left, lung_mask_right):
    """Return per-region z ranges based on each lung's effective z support."""
    left_ranges = _split_z_to_three_ranges(lung_mask_left)
    right_ranges = _split_z_to_three_ranges(lung_mask_right)
    return {
        'left_upper': left_ranges[0],
        'left_middle': left_ranges[1],
        'left_lower': left_ranges[2],
        'right_upper': right_ranges[0],
        'right_middle': right_ranges[1],
        'right_lower': right_ranges[2],
    }


def get_six_lung_regions(left_lung_mask, right_lung_mask):
    """Return dict of six 3D region masks from left/right lung masks."""
    ranges = get_lung_region_ranges(left_lung_mask, right_lung_mask)
    out = {}
    for name, z_range in ranges.items():
        if name.startswith('left_'):
            side = left_lung_mask
        else:
            side = right_lung_mask
        out[name] = _make_region_mask(side, z_range)
    return out


def get_valid_region_slices(region_mask, abs_threshold=50, ratio_threshold=0.05):
    """Valid z slices based on per-slice area threshold."""
    if region_mask.ndim != 3:
        raise ValueError('region_mask must be 3D')

    areas = region_mask.reshape(region_mask.shape[0], -1).sum(axis=1).astype(np.float32)
    amax = float(areas.max()) if areas.size > 0 else 0.0
    thr = max(float(abs_threshold), float(ratio_threshold) * max(amax, 1.0))
    valid = np.where(areas > thr)[0]
    return valid.astype(np.int64)


def get_valid_region_centers(region_mask, num_slices=3, abs_threshold=50, ratio_threshold=0.05):
    """Valid center z for thin-slab with boundary and union-area checks."""
    if num_slices < 1 or (num_slices % 2) == 0:
        raise ValueError('num_slices must be a positive odd integer')

    z_total = int(region_mask.shape[0])
    half = int(num_slices // 2)
    if z_total <= 2 * half:
        return np.asarray([], dtype=np.int64)

    valid_slices = set(get_valid_region_slices(region_mask, abs_threshold, ratio_threshold).tolist())
    areas = region_mask.reshape(z_total, -1).sum(axis=1).astype(np.float32)
    amax = float(areas.max()) if areas.size > 0 else 0.0
    thr = max(float(abs_threshold), float(ratio_threshold) * max(amax, 1.0))

    centers = []
    for z in range(half, z_total - half):
        if z not in valid_slices:
            continue
        z_ids = np.arange(z - half, z + half + 1, dtype=np.int64)
        union_area = float(np.any(region_mask[z_ids], axis=0).sum())
        if union_area > thr:
            centers.append(z)
    return np.asarray(centers, dtype=np.int64)


def get_region_bbox(mask_3d_or_2d, margin=12, min_size=32, image_shape=None):
    """Compute xy bbox [y0, y1, x0, x1] from 2D/3D region mask."""
    if mask_3d_or_2d.ndim == 3:
        m = np.any(mask_3d_or_2d > 0, axis=0)
    elif mask_3d_or_2d.ndim == 2:
        m = (mask_3d_or_2d > 0)
    else:
        raise ValueError('mask_3d_or_2d must be 2D or 3D')

    if image_shape is None:
        h, w = int(m.shape[0]), int(m.shape[1])
    else:
        h, w = int(image_shape[0]), int(image_shape[1])

    ys, xs = np.where(m)
    if ys.size == 0:
        return (0, h, 0, w)

    y0 = int(max(0, ys.min() - margin))
    y1 = int(min(h, ys.max() + 1 + margin))
    x0 = int(max(0, xs.min() - margin))
    x1 = int(min(w, xs.max() + 1 + margin))

    ch = y1 - y0
    cw = x1 - x0
    if ch < min_size:
        pad = min_size - ch
        y0 = max(0, y0 - pad // 2)
        y1 = min(h, y1 + (pad - pad // 2))
    if cw < min_size:
        pad = min_size - cw
        x0 = max(0, x0 - pad // 2)
        x1 = min(w, x1 + (pad - pad // 2))

    if (y1 - y0) <= 0 or (x1 - x0) <= 0:
        return (0, h, 0, w)

    return (y0, y1, x0, x1)
