"""Pytorch dataset object that loads MNIST dataset as bags."""

import glob
import json
import os
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from PIL import Image
from torchvision import datasets, transforms


CACHE_VERSION = 'lung_region_cache_v1'


def _get_case_id_from_path(file_path):
    base = os.path.basename(file_path)
    if base.endswith('.nii.gz'):
        return base[:-7]
    return os.path.splitext(base)[0]


def iter_case_paths(root_dir, num_classes):
    """Yield (case_id, ct_path, class_id) from the standard *_seg_nii layout."""
    for cls in range(int(num_classes)):
        folder = os.path.join(root_dir, '{}_seg_nii'.format(cls))
        for ct_path in sorted(glob.glob(os.path.join(folder, '*.nii.gz'))):
            yield _get_case_id_from_path(ct_path), ct_path, int(cls)


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


def save_case_cache(case_id, cache_root, cache_dict, meta_dict):
    case_dir = os.path.join(cache_root, str(case_id))
    os.makedirs(case_dir, exist_ok=True)

    cache_path = os.path.join(case_dir, 'preprocess_cache.npz')
    meta_path = os.path.join(case_dir, 'preprocess_meta.json')

    np.savez_compressed(cache_path, **cache_dict)
    with open(meta_path, 'w') as f:
        json.dump(meta_dict, f, indent=2, sort_keys=True)

    return cache_path, meta_path


def load_case_cache(case_id, cache_root):
    case_dir = os.path.join(cache_root, str(case_id))
    cache_path = os.path.join(case_dir, 'preprocess_cache.npz')
    meta_path = os.path.join(case_dir, 'preprocess_meta.json')

    if not os.path.exists(cache_path) or not os.path.exists(meta_path):
        return None

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    if meta.get('cache_version') != CACHE_VERSION:
        return None

    try:
        data = np.load(cache_path, allow_pickle=True)
        cache = {k: data[k] for k in data.files}
    except Exception:
        return None
    return {'cache': cache, 'meta': meta, 'cache_path': cache_path, 'meta_path': meta_path}


def _largest_two_components_from_binary(binary_mask):
    cc = sitk.ConnectedComponent(sitk.GetImageFromArray(binary_mask.astype(np.uint8)))
    cc_np = sitk.GetArrayFromImage(cc).astype(np.int32)
    comp_ids = [int(v) for v in np.unique(cc_np) if v > 0]
    if len(comp_ids) < 2:
        return None

    areas = []
    for cid in comp_ids:
        comp = (cc_np == cid)
        areas.append((int(comp.sum()), cid))
    areas.sort(reverse=True)
    c1 = areas[0][1]
    c2 = areas[1][1]
    comp1 = (cc_np == c1)
    comp2 = (cc_np == c2)
    return comp1, comp2


def _largest_n_components(binary_mask, keep_n=2):
    cc = sitk.ConnectedComponent(sitk.GetImageFromArray(binary_mask.astype(np.uint8)))
    cc_np = sitk.GetArrayFromImage(cc).astype(np.int32)
    comp_ids = [int(v) for v in np.unique(cc_np) if v > 0]
    if len(comp_ids) == 0:
        return np.zeros_like(binary_mask, dtype=bool)

    areas = []
    for cid in comp_ids:
        comp = (cc_np == cid)
        areas.append((int(comp.sum()), cid))
    areas.sort(reverse=True)
    keep_ids = set([cid for _, cid in areas[:max(1, int(keep_n))]])
    out = np.isin(cc_np, list(keep_ids))
    return out.astype(bool)


def build_pseudo_lung_mask(masked_ct, value_threshold=1e-6, min_component_voxels=512):
    """Build pseudo binary lung mask from masked CT [Z,H,W]."""
    if masked_ct.ndim != 3:
        raise ValueError('masked_ct must be 3D [Z,H,W]')

    x = masked_ct.astype(np.float32)
    # Auto-detect background as median value (robust when background > 50% of volume).
    # Works for masked CT (bg=-1000), zero-padded CT (bg=0), or any constant background.
    bg_value = float(np.round(np.median(x.ravel())))
    init = np.abs(x - bg_value) > max(float(value_threshold), 0.5)
    if not np.any(init):
        # Degenerate fallback: non-min voxels as weak candidate.
        init = x > float(np.percentile(x, 5))

    # Light denoise and boundary repair.
    img = sitk.GetImageFromArray(init.astype(np.uint8))
    opened = sitk.BinaryMorphologicalOpening(img, [1, 1, 1], sitk.sitkBall)
    closed = sitk.BinaryMorphologicalClosing(opened, [1, 1, 1], sitk.sitkBall)
    filled = sitk.BinaryFillhole(closed)
    mask = sitk.GetArrayFromImage(filled).astype(np.uint8) > 0

    # Keep only main connected components to suppress small artifacts.
    cc = sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(np.uint8)))
    cc_np = sitk.GetArrayFromImage(cc).astype(np.int32)
    comp_ids = [int(v) for v in np.unique(cc_np) if v > 0]
    if len(comp_ids) > 0:
        keep = np.zeros_like(mask, dtype=bool)
        for cid in comp_ids:
            comp = (cc_np == cid)
            if int(comp.sum()) >= int(min_component_voxels):
                keep |= comp
        if np.any(keep):
            mask = keep
        else:
            mask = _largest_n_components(mask, keep_n=2)

    return mask.astype(bool)


def split_left_right_lung(lung_mask):
    """Split binary/label lung mask into left/right lungs.

    Returns:
        left_mask, right_mask, split_method
        split_method in {'cc3d', 'morph_cc3d', 'midline_fallback'}
    """
    if lung_mask.ndim != 3:
        raise ValueError('lung_mask must be 3D [Z, H, W]')

    m = lung_mask.astype(np.int32)
    positive = m > 0
    if not np.any(positive):
        return np.zeros_like(positive), np.zeros_like(positive), 'midline_fallback'

    labels = [int(v) for v in np.unique(m) if v > 0]
    if len(labels) >= 2:
        # Use two largest positive labels if mask already stores separate lungs.
        areas = []
        for lb in labels:
            comp = (m == lb)
            areas.append((int(comp.sum()), lb))
        areas.sort(reverse=True)
        lb1 = areas[0][1]
        lb2 = areas[1][1]
        comp1 = (m == lb1)
        comp2 = (m == lb2)
        x1 = np.where(comp1)[2].mean() if np.any(comp1) else 0.0
        x2 = np.where(comp2)[2].mean() if np.any(comp2) else float(m.shape[2] - 1)
        if x1 <= x2:
            return comp1, comp2, 'cc3d'
        return comp2, comp1, 'cc3d'

    # Priority 1: direct 3D connected components on binary mask.
    pair = _largest_two_components_from_binary(positive)
    if pair is not None:
        comp1, comp2 = pair
        x1 = np.where(comp1)[2].mean() if np.any(comp1) else 0.0
        x2 = np.where(comp2)[2].mean() if np.any(comp2) else float(m.shape[2] - 1)
        if x1 <= x2:
            return comp1, comp2, 'cc3d'
        return comp2, comp1, 'cc3d'

    # Priority 2: light morphology to break weak bridge then cc3d.
    itk_mask = sitk.GetImageFromArray(positive.astype(np.uint8))
    opened = sitk.BinaryMorphologicalOpening(itk_mask, [1, 1, 1], sitk.sitkBall)
    eroded = sitk.BinaryErode(opened, [1, 1, 1], sitk.sitkBall)
    morph_np = sitk.GetArrayFromImage(eroded).astype(np.uint8) > 0
    pair = _largest_two_components_from_binary(morph_np)
    if pair is not None:
        m1, m2 = pair
        x1 = np.where(m1)[2].mean() if np.any(m1) else 0.0
        x2 = np.where(m2)[2].mean() if np.any(m2) else float(m.shape[2] - 1)
        xmid = int(round(0.5 * (x1 + x2)))
        left = positive.copy()
        left[:, :, xmid + 1:] = False
        right = positive.copy()
        right[:, :, :xmid + 1] = False
        if left.sum() > 0 and right.sum() > 0:
            return left, right, 'morph_cc3d'

    # Fallback: approximate split by body midline.
    coords = np.where(positive)
    if coords[0].size == 0:
        return np.zeros_like(positive), np.zeros_like(positive), 'midline_fallback'
    xmin = int(coords[2].min())
    xmax = int(coords[2].max())
    xmid = (xmin + xmax) // 2
    left = positive.copy()
    left[:, :, xmid + 1:] = False
    right = positive.copy()
    right[:, :, :xmid + 1] = False
    return left, right, 'midline_fallback'


def _split_z_to_three_ranges(mask_3d):
    z_idx = np.where(mask_3d.reshape(mask_3d.shape[0], -1).any(axis=1))[0]
    if z_idx.size == 0:
        return [(0, 0), (0, 0), (0, 0)]

    z_start = int(z_idx.min())
    z_end = int(z_idx.max()) + 1  # exclusive
    z_all = np.arange(z_start, z_end, dtype=np.int64)
    chunks = np.array_split(z_all, 3)
    ranges = []
    for ch in chunks:
        if ch.size == 0:
            ranges.append((z_start, z_start))
        else:
            ranges.append((int(ch[0]), int(ch[-1]) + 1))
    return ranges


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

    # Enforce minimum crop size to avoid degenerate tiny crops.
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


def sample_region_centers(candidate_centers, num_samples, rng):
    """Sample centers from legal candidates, allowing repeats when insufficient."""
    candidate_centers = np.asarray(candidate_centers, dtype=np.int64)
    if num_samples <= 0:
        return np.asarray([], dtype=np.int64)
    if candidate_centers.size == 0:
        return np.asarray([], dtype=np.int64)

    if candidate_centers.size >= num_samples:
        # Uniformly pick across z to keep coverage stable.
        idx = np.rint(np.linspace(0, candidate_centers.size - 1, num_samples)).astype(np.int64)
        return candidate_centers[idx]

    # Not enough legal centers: repeat from legal pool only.
    extra_idx = rng.choice(candidate_centers.size, size=(num_samples - candidate_centers.size), replace=True)
    out = np.concatenate([candidate_centers, candidate_centers[extra_idx]], axis=0)
    rng.shuffle(out)
    return out.astype(np.int64)


def _resize_instance_chw(instance_chw, out_size):
    x = torch.from_numpy(instance_chw).unsqueeze(0).float()  # [1, C, H, W]
    x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
    return x.squeeze(0).numpy().astype(np.float32)


def build_2p5d_region_instance(ct_volume, region_mask, center_z, out_size, region_bbox_fallback,
                               num_slices=3, bbox_margin=12, bbox_min_size=32):
    """Build one [C,H,W] instance from a region around center_z with crop+resize."""
    z_total = int(ct_volume.shape[0])
    half = int(num_slices // 2)
    z_ids = [int(np.clip(center_z + dz, 0, z_total - 1)) for dz in range(-half, half + 1)]

    slab = ct_volume[np.asarray(z_ids, dtype=np.int64)]  # [C, H, W]
    slab_mask = region_mask[np.asarray(z_ids, dtype=np.int64)]

    mask_union = np.any(slab_mask, axis=0)
    if np.any(mask_union):
        y0, y1, x0, x1 = get_region_bbox(mask_union, margin=bbox_margin, min_size=bbox_min_size,
                                         image_shape=slab.shape[1:])
    else:
        y0, y1, x0, x1 = region_bbox_fallback

    crop = slab[:, y0:y1, x0:x1]
    if crop.shape[1] <= 1 or crop.shape[2] <= 1:
        crop = slab

    return _resize_instance_chw(crop, out_size=out_size)


def _make_region_mask(side_mask, z_range):
    z0, z1 = int(z_range[0]), int(z_range[1])
    out = np.zeros_like(side_mask, dtype=bool)
    if z1 > z0:
        out[z0:z1] = side_mask[z0:z1]
    return out


def _compute_legal_centers(region_mask, num_slices, abs_threshold, ratio_threshold):
    return get_valid_region_centers(
        region_mask,
        num_slices=num_slices,
        abs_threshold=abs_threshold,
        ratio_threshold=ratio_threshold,
    )


def generate_lung_region_instances(ct_volume, lung_mask, rng, num_instances=64, num_slices=3,
                                   out_size=(224, 224), bbox_margin=12, bbox_min_size=32,
                                   abs_threshold=100.0, ratio_threshold=0.05,
                                   left_mask=None, right_mask=None, split_method='unknown',
                                   region_ctx=None):
    """Generate [N,C,H,W] region-aware 2.5D instances and metadata.

    Returns:
        instances: np.ndarray [N, C, H, W]
        metadata: list[dict] with region and center_z
    """
    if num_slices < 1 or (num_slices % 2) == 0:
        raise ValueError('num_slices must be a positive odd integer')

    lung_mask = (lung_mask > 0)
    if region_ctx is not None:
        left_mask = region_ctx['left_lung_mask'].astype(bool)
        right_mask = region_ctx['right_lung_mask'].astype(bool)
        split_method = str(region_ctx.get('split_method', split_method))
        region_masks = region_ctx['region_masks_dict']
        valid_centers_ctx = region_ctx.get('valid_region_centers_dict', {})
        region_bboxes_ctx = region_ctx.get('region_bboxes_dict', {})
    else:
        if left_mask is None or right_mask is None:
            left_mask, right_mask, split_method = split_left_right_lung(lung_mask)
        else:
            left_mask = (left_mask > 0)
            right_mask = (right_mask > 0)
        region_masks = get_six_lung_regions(left_mask, right_mask)
        valid_centers_ctx = None
        region_bboxes_ctx = None
    region_ranges = get_lung_region_ranges(left_mask, right_mask)
    region_names = [
        'left_upper', 'left_middle', 'left_lower',
        'right_upper', 'right_middle', 'right_lower',
    ]

    region_bboxes = {}
    legal_centers = {}
    legal_counts = {}
    for region_name in region_names:
        rmask = region_masks[region_name]
        if region_bboxes_ctx is not None and region_name in region_bboxes_ctx:
            region_bbox = tuple(int(v) for v in region_bboxes_ctx[region_name])
        else:
            region_bbox = get_region_bbox(rmask, margin=bbox_margin, min_size=bbox_min_size,
                                          image_shape=ct_volume.shape[1:])
        if valid_centers_ctx is not None and region_name in valid_centers_ctx:
            centers = np.asarray(valid_centers_ctx[region_name], dtype=np.int64)
        else:
            centers = _compute_legal_centers(
                rmask,
                num_slices=num_slices,
                abs_threshold=abs_threshold,
                ratio_threshold=ratio_threshold,
            )
        region_masks[region_name] = rmask
        region_bboxes[region_name] = region_bbox
        legal_centers[region_name] = centers
        legal_counts[region_name] = int(centers.size)

    # 64 instances: each region starts with 10, remaining 4 allocated by legal-center proportion.
    alloc = {k: 10 for k in region_names}
    remain = int(num_instances - 60)
    if remain < 0:
        # For smaller bags, distribute uniformly.
        alloc = {k: 0 for k in region_names}
        for i in range(num_instances):
            alloc[region_names[i % len(region_names)]] += 1
    elif remain > 0:
        counts = np.asarray([legal_counts[k] for k in region_names], dtype=np.float32)
        if counts.sum() <= 0:
            extra_regions = rng.choice(len(region_names), size=remain, replace=True)
            for ridx in extra_regions.tolist():
                alloc[region_names[ridx]] += 1
        else:
            probs = counts / counts.sum()
            extra_regions = rng.choice(len(region_names), size=remain, replace=True, p=probs)
            for ridx in extra_regions.tolist():
                alloc[region_names[ridx]] += 1

    instances = []
    metadata = []
    for region_name in region_names:
        need = int(alloc[region_name])
        candidates = legal_centers[region_name]

        # If legal centers are empty, fallback to any in-range center with clipping-safe z.
        if candidates.size == 0:
            z0, z1 = region_ranges[region_name]
            half = num_slices // 2
            z0_safe = max(int(z0), int(half))
            z1_safe = min(int(z1), int(ct_volume.shape[0] - half))
            if z1_safe <= z0_safe:
                z0_safe = int(half)
                z1_safe = int(max(z0_safe + 1, ct_volume.shape[0] - half))
            candidates = np.arange(z0_safe, z1_safe, dtype=np.int64)
            if candidates.size == 0:
                candidates = np.asarray([int(np.clip(ct_volume.shape[0] // 2, half, max(half, ct_volume.shape[0] - half - 1)))], dtype=np.int64)

        chosen = sample_region_centers(candidates, need, rng)
        for zc in chosen.tolist():
            inst = build_2p5d_region_instance(
                ct_volume=ct_volume,
                region_mask=region_masks[region_name],
                center_z=int(zc),
                out_size=out_size,
                region_bbox_fallback=region_bboxes[region_name],
                num_slices=num_slices,
                bbox_margin=bbox_margin,
                bbox_min_size=bbox_min_size,
            )
            instances.append(inst)
            metadata.append({'region': region_name, 'center_z': int(zc), 'split_method': str(split_method)})

    if len(instances) != num_instances:
        # Keep strict bag size for MIL compatibility.
        if len(instances) > num_instances:
            instances = instances[:num_instances]
            metadata = metadata[:num_instances]
        else:
            while len(instances) < num_instances:
                instances.append(instances[-1].copy())
                metadata.append(dict(metadata[-1]))

    return np.stack(instances, axis=0).astype(np.float32), metadata


def build_lung_region_context_from_mask(lung_mask, num_slices=3, abs_threshold=50, ratio_threshold=0.05):
    left_mask, right_mask, split_method = split_left_right_lung(lung_mask)
    region_masks = get_six_lung_regions(left_mask, right_mask)
    valid_centers = {}
    region_bboxes = {}
    for name, rmask in region_masks.items():
        valid_centers[name] = get_valid_region_centers(
            rmask,
            num_slices=num_slices,
            abs_threshold=abs_threshold,
            ratio_threshold=ratio_threshold,
        )
        region_bboxes[name] = get_region_bbox(rmask)

    return {
        'pseudo_mask': (lung_mask > 0).astype(bool),
        'left_lung_mask': left_mask.astype(bool),
        'right_lung_mask': right_mask.astype(bool),
        'split_method': str(split_method),
        'region_masks_dict': region_masks,
        'valid_region_centers_dict': valid_centers,
        'region_bboxes_dict': region_bboxes,
        'num_slices_for_valid_centers': int(num_slices),
        'abs_threshold': float(abs_threshold),
        'ratio_threshold': float(ratio_threshold),
    }


def pack_region_context_cache(region_ctx):
    cache_dict = {
        'pseudo_mask': region_ctx['pseudo_mask'].astype(np.uint8),
        'left_lung_mask': region_ctx['left_lung_mask'].astype(np.uint8),
        'right_lung_mask': region_ctx['right_lung_mask'].astype(np.uint8),
        'split_method': np.asarray(region_ctx['split_method']),
        'num_slices_for_valid_centers': np.asarray(int(region_ctx['num_slices_for_valid_centers']), dtype=np.int32),
        'abs_threshold': np.asarray(float(region_ctx['abs_threshold']), dtype=np.float32),
        'ratio_threshold': np.asarray(float(region_ctx['ratio_threshold']), dtype=np.float32),
    }
    for region_name, region_mask in region_ctx['region_masks_dict'].items():
        cache_dict['region_mask__{}'.format(region_name)] = region_mask.astype(np.uint8)
    for region_name, centers in region_ctx['valid_region_centers_dict'].items():
        cache_dict['valid_centers__{}'.format(region_name)] = np.asarray(centers, dtype=np.int64)
    for region_name, bbox in region_ctx['region_bboxes_dict'].items():
        cache_dict['region_bbox__{}'.format(region_name)] = np.asarray(bbox, dtype=np.int32)
    return cache_dict


def unpack_region_context_cache(cache):
    region_masks_dict = {}
    valid_region_centers_dict = {}
    region_bboxes_dict = {}
    for key, value in cache.items():
        if key.startswith('region_mask__'):
            region_masks_dict[key[len('region_mask__'):]] = value.astype(bool)
        elif key.startswith('valid_centers__'):
            valid_region_centers_dict[key[len('valid_centers__'):]] = value.astype(np.int64)
        elif key.startswith('region_bbox__'):
            region_bboxes_dict[key[len('region_bbox__'):]] = tuple(int(v) for v in np.asarray(value).tolist())

    return {
        'pseudo_mask': cache['pseudo_mask'].astype(bool),
        'left_lung_mask': cache['left_lung_mask'].astype(bool),
        'right_lung_mask': cache['right_lung_mask'].astype(bool),
        'split_method': str(np.asarray(cache['split_method']).item()) if 'split_method' in cache else 'unknown',
        'region_masks_dict': region_masks_dict,
        'valid_region_centers_dict': valid_region_centers_dict,
        'region_bboxes_dict': region_bboxes_dict,
        'num_slices_for_valid_centers': int(np.asarray(cache['num_slices_for_valid_centers']).item()) if 'num_slices_for_valid_centers' in cache else 3,
        'abs_threshold': float(np.asarray(cache['abs_threshold']).item()) if 'abs_threshold' in cache else 50.0,
        'ratio_threshold': float(np.asarray(cache['ratio_threshold']).item()) if 'ratio_threshold' in cache else 0.05,
    }


def save_region_context_debug(case_id, debug_root, pseudo_mask, region_ctx):
    case_dir = os.path.join(debug_root, str(case_id), 'preprocess_debug')
    os.makedirs(case_dir, exist_ok=True)

    per_z = pseudo_mask.reshape(pseudo_mask.shape[0], -1).sum(axis=1)
    if per_z.size == 0:
        return
    z = int(np.argmax(per_z))

    Image.fromarray((pseudo_mask[z].astype(np.uint8) * 255)).save(
        os.path.join(case_dir, '{}_z{:04d}_pseudo_mask.png'.format(case_id, z))
    )

    left_mask = region_ctx['left_lung_mask']
    right_mask = region_ctx['right_lung_mask']
    split_method = str(region_ctx.get('split_method', 'unknown'))

    left_2d = left_mask[z] > 0
    right_2d = right_mask[z] > 0
    overlay = np.zeros((left_2d.shape[0], left_2d.shape[1], 3), dtype=np.uint8)
    overlay[:, :, 0] = left_2d.astype(np.uint8) * 255
    overlay[:, :, 1] = right_2d.astype(np.uint8) * 255

    Image.fromarray((left_2d.astype(np.uint8) * 255)).save(
        os.path.join(case_dir, '{}_z{:04d}_left_{}.png'.format(case_id, z, split_method))
    )
    Image.fromarray((right_2d.astype(np.uint8) * 255)).save(
        os.path.join(case_dir, '{}_z{:04d}_right_{}.png'.format(case_id, z, split_method))
    )
    Image.fromarray(overlay).save(
        os.path.join(case_dir, '{}_z{:04d}_overlay_{}.png'.format(case_id, z, split_method))
    )

    region_dir = os.path.join(case_dir, 'six_regions')
    os.makedirs(region_dir, exist_ok=True)
    for region_name, region_mask in region_ctx['region_masks_dict'].items():
        Image.fromarray((region_mask[z].astype(np.uint8) * 255)).save(
            os.path.join(region_dir, '{}_z{:04d}_{}.png'.format(case_id, z, region_name))
        )


def build_case_preprocess_cache(masked_ct, case_id, source_ct_path, selected_idx, abs_threshold=50, ratio_threshold=0.05,
                                num_slices_for_valid_centers=3, cache_version=CACHE_VERSION,
                                pseudo_mask_value_threshold=1e-6, pseudo_mask_min_component_voxels=512):
    pseudo_mask = build_pseudo_lung_mask(
        masked_ct,
        value_threshold=pseudo_mask_value_threshold,
        min_component_voxels=pseudo_mask_min_component_voxels,
    )
    region_ctx = build_lung_region_context_from_mask(
        pseudo_mask,
        num_slices=num_slices_for_valid_centers,
        abs_threshold=abs_threshold,
        ratio_threshold=ratio_threshold,
    )
    cache_dict = pack_region_context_cache(region_ctx)
    meta_dict = {
        'cache_version': cache_version,
        'case_id': str(case_id),
        'source_ct_path': str(source_ct_path),
        'selected_idx': np.asarray(selected_idx, dtype=np.int64).tolist(),
        'region_definition': 'six_lung_regions',
        'num_slices_for_valid_centers': int(num_slices_for_valid_centers),
        'abs_threshold': float(abs_threshold),
        'ratio_threshold': float(ratio_threshold),
        'pseudo_mask_value_threshold': float(pseudo_mask_value_threshold),
        'pseudo_mask_min_component_voxels': int(pseudo_mask_min_component_voxels),
        'split_method': str(region_ctx['split_method']),
    }
    return cache_dict, meta_dict, region_ctx


class MnistBags(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            if self.train:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
            else:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label


class CTPneNiiBags(data_utils.Dataset):
    """MIL dataset for pneumoconiosis staging from segmented NIfTI CT volumes.

    One patient volume is one bag. Instances are 2.5D triplets [z-1, z, z+1]
    built from centered slices sampled with 1mm interval on z-axis.
    """

    def __init__(
        self,
        root_dir='/data/scratch/c_pne',
        split='train',
        num_classes=4,
        middle_ratio=0.98,
        fixed_num_slices=256,
        channel_offsets=(-1, 0, 1),
        slab_stride=3,
        num_slabs=64,
        center_sampling_mode='uniform',
        test_ratio=0.2,
        val_ratio=0.1,
        seed=1,
        intensity_clip=(-1000.0, 400.0),
        scale_to_unit=True,
        use_zscore=False,
        lung_hu_low=-950.0,
        lung_hu_high=-300.0,
        min_lung_area_ratio=0.01,
        max_instances=0,
        instance_definition='lung_region_thin_slab',
        lung_mask_root='',
        lung_mask_suffix='.nii.gz',
        lung_mask_require=True,
        cache_root='',
        pseudo_mask_value_threshold=1e-6,
        pseudo_mask_min_component_voxels=512,
        region_num_instances=64,
        region_out_size=(224, 224),
        region_bbox_margin=12,
        region_bbox_min_size=32,
        region_abs_area_threshold=100.0,
        region_ratio_area_threshold=0.05,
        debug_save_instances=False,
        debug_save_lung_split=False,
        debug_save_six_regions=False,
        debug_dir='debug_instances',
        debug_max_cases=0,
    ):
        if split not in ('train', 'val', 'test'):
            raise ValueError("split must be 'train', 'val' or 'test'")
        if num_classes < 2:
            raise ValueError('num_classes must be >= 2')
        if len(channel_offsets) < 1:
            raise ValueError('channel_offsets must have at least one element')

        self.root_dir = root_dir
        self.split = split
        self.num_classes = num_classes
        self.middle_ratio = float(middle_ratio)
        self.fixed_num_slices = int(fixed_num_slices)
        self.channel_offsets = tuple(int(v) for v in channel_offsets)
        self.slab_stride = int(slab_stride)
        self.num_slabs = int(num_slabs)
        self.center_sampling_mode = str(center_sampling_mode)
        self.test_ratio = float(test_ratio)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.intensity_clip = intensity_clip
        self.scale_to_unit = bool(scale_to_unit)
        self.use_zscore = use_zscore
        self.lung_hu_low = float(lung_hu_low)
        self.lung_hu_high = float(lung_hu_high)
        self.min_lung_area_ratio = float(min_lung_area_ratio)
        self.max_instances = int(max_instances)
        self.instance_definition = str(instance_definition)
        self.lung_mask_root = str(lung_mask_root) if lung_mask_root is not None else ''
        self.lung_mask_suffix = str(lung_mask_suffix)
        self.lung_mask_require = bool(lung_mask_require)
        self.cache_root = str(cache_root) if cache_root is not None else ''
        self.pseudo_mask_value_threshold = float(pseudo_mask_value_threshold)
        self.pseudo_mask_min_component_voxels = int(pseudo_mask_min_component_voxels)
        self.region_num_instances = int(region_num_instances)
        self.region_out_size = (int(region_out_size[0]), int(region_out_size[1]))
        self.region_bbox_margin = int(region_bbox_margin)
        self.region_bbox_min_size = int(region_bbox_min_size)
        self.region_abs_area_threshold = float(region_abs_area_threshold)
        self.region_ratio_area_threshold = float(region_ratio_area_threshold)
        self.debug_save_instances = bool(debug_save_instances)
        self.debug_save_lung_split = bool(debug_save_lung_split)
        self.debug_save_six_regions = bool(debug_save_six_regions)
        self.debug_dir = str(debug_dir)
        self.debug_max_cases = int(debug_max_cases)
        self.current_epoch = 0
        self._debug_saved_case_ids = set()
        self._debug_saved_split_case_ids = set()
        self._debug_saved_region_case_ids = set()

        if self.test_ratio < 0.0 or self.val_ratio < 0.0:
            raise ValueError('test_ratio and val_ratio must be >= 0')
        if (self.test_ratio + self.val_ratio) >= 1.0:
            raise ValueError('test_ratio + val_ratio must be < 1.0')
        if self.fixed_num_slices < 0:
            raise ValueError('fixed_num_slices must be >= 0')
        if self.slab_stride < 1:
            raise ValueError('slab_stride must be >= 1')
        if self.num_slabs < 0:
            raise ValueError('num_slabs must be >= 0')
        if self.center_sampling_mode not in ('uniform', 'random', 'all'):
            raise ValueError("center_sampling_mode must be 'uniform', 'random' or 'all'")
        if self.instance_definition not in ('lung_region_thin_slab', 'global_slab'):
            raise ValueError("instance_definition must be 'lung_region_thin_slab' or 'global_slab'")
        if not (0.0 <= self.min_lung_area_ratio < 1.0):
            raise ValueError('min_lung_area_ratio must be in [0, 1)')
        if self.lung_hu_low >= self.lung_hu_high:
            raise ValueError('lung_hu_low must be < lung_hu_high')
        if self.region_num_instances <= 0:
            raise ValueError('region_num_instances must be > 0')
        if self.region_out_size[0] <= 0 or self.region_out_size[1] <= 0:
            raise ValueError('region_out_size must contain positive integers')
        if self.region_bbox_margin < 0:
            raise ValueError('region_bbox_margin must be >= 0')
        if self.region_bbox_min_size <= 0:
            raise ValueError('region_bbox_min_size must be > 0')
        if self.region_abs_area_threshold < 0:
            raise ValueError('region_abs_area_threshold must be >= 0')
        if self.region_ratio_area_threshold < 0:
            raise ValueError('region_ratio_area_threshold must be >= 0')
        if self.pseudo_mask_value_threshold < 0.0:
            raise ValueError('pseudo_mask_value_threshold must be >= 0')
        if self.pseudo_mask_min_component_voxels <= 0:
            raise ValueError('pseudo_mask_min_component_voxels must be > 0')
        if self.cache_root:
            os.makedirs(self.cache_root, exist_ok=True)

        self.samples = self._build_samples()
        if len(self.samples) == 0:
            raise RuntimeError('No samples found. Check root_dir and *_seg_nii folders.')

        if (self.debug_save_instances or self.debug_save_lung_split or self.debug_save_six_regions) and self.debug_max_cases > 0:
            os.makedirs(self.debug_dir, exist_ok=True)

    def _build_samples(self):
        rng = np.random.RandomState(self.seed)
        class_to_files = {}

        for cls in range(self.num_classes):
            folder = os.path.join(self.root_dir, '{}_seg_nii'.format(cls))
            files = sorted(glob.glob(os.path.join(folder, '*.nii.gz')))
            if len(files) == 0:
                raise RuntimeError('No .nii.gz files found in {}'.format(folder))
            class_to_files[cls] = files

        selected = []
        for cls, files in class_to_files.items():
            n = len(files)
            indices = np.arange(len(files))
            rng.shuffle(indices)

            test_count = int(round(n * self.test_ratio))
            val_count = int(round(n * self.val_ratio))

            if n >= 3:
                if self.test_ratio > 0.0 and test_count == 0:
                    test_count = 1
                if self.val_ratio > 0.0 and val_count == 0:
                    val_count = 1

            max_holdout = max(0, n - 1)
            if (test_count + val_count) > max_holdout:
                overflow = test_count + val_count - max_holdout
                reduce_val = min(val_count, overflow)
                val_count -= reduce_val
                overflow -= reduce_val
                if overflow > 0:
                    test_count = max(0, test_count - overflow)

            test_end = test_count
            val_end = test_count + val_count

            if self.split == 'test':
                split_indices = indices[:test_end]
            elif self.split == 'val':
                split_indices = indices[test_end:val_end]
            else:
                split_indices = indices[val_end:]

            for idx in split_indices:
                selected.append((files[idx], cls))

        return selected

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        # Used by the training loop so random center sampling can change every epoch.
        self.current_epoch = int(epoch)

    def _select_middle_slice_range(self, num_slices):
        if num_slices < 3:
            return 0, num_slices

        drop_each_side = int(num_slices * (1.0 - self.middle_ratio) / 2.0)
        start = drop_each_side
        end = num_slices - drop_each_side

        # Keep at least 3 slices so [z-1, z, z+1] can be formed.
        if (end - start) < 3:
            start, end = 0, num_slices

        return start, end

    def _select_fixed_middle_indices(self, num_slices, z_spacing_mm=1.0):
        if num_slices <= 0:
            raise ValueError('Volume has no slices')

        if z_spacing_mm <= 0.0:
            z_spacing_mm = 1.0

        target = self.fixed_num_slices
        center_pos = 0.5 * float(num_slices - 1)

        half = target // 2
        if target % 2 == 0:
            relative_mm = np.arange(-half, half, dtype=np.float32)
        else:
            relative_mm = np.arange(-half, half + 1, dtype=np.float32)

        relative_idx = np.rint(relative_mm / float(z_spacing_mm)).astype(np.int64)
        centered_idx = np.rint(center_pos).astype(np.int64) + relative_idx

        return np.clip(centered_idx, 0, num_slices - 1).astype(np.int64)

    def _normalize_volume(self, volume):
        x = volume.astype(np.float32)

        if self.intensity_clip is not None:
            low = float(self.intensity_clip[0])
            high = float(self.intensity_clip[1])
            x = np.clip(x, low, high)
            if self.scale_to_unit:
                x = (x - low) / max(high - low, 1e-6)

        if self.use_zscore:
            mean = float(x.mean())
            std = float(x.std())
            if std < 1e-6:
                std = 1.0
            x = (x - mean) / std

        return x

    def _select_effective_lung_range(self, volume_zyx_raw):
        z = int(volume_zyx_raw.shape[0])
        if z < 3:
            return 0, z

        # Estimate effective lung region by parenchyma-like HU ratio per slice.
        mask = (volume_zyx_raw >= self.lung_hu_low) & (volume_zyx_raw <= self.lung_hu_high)
        ratio = mask.reshape(z, -1).mean(axis=1)
        valid = ratio >= self.min_lung_area_ratio

        if not np.any(valid):
            return 0, z

        start = int(np.argmax(valid))
        end = int(z - np.argmax(valid[::-1]))
        if (end - start) < 3:
            return 0, z

        return start, end

    def _select_centers(self, centers, target_count, rng):
        n = int(len(centers))
        if n == 0 or target_count <= 0 or self.center_sampling_mode == 'all':
            return centers

        if self.center_sampling_mode == 'random':
            if n >= target_count:
                choice = rng.choice(n, size=target_count, replace=False)
            else:
                choice = rng.choice(n, size=target_count, replace=True)
            choice.sort()
            return centers[choice]

        if n >= target_count:
            sampled = []
            edges = np.linspace(0, n, target_count + 1)
            for i in range(target_count):
                left = int(np.floor(edges[i]))
                right = int(np.floor(edges[i + 1]) - 1)
                if right < left:
                    right = left
                idx = (left + right) // 2
                idx = int(np.clip(idx, 0, n - 1))
                sampled.append(centers[idx])
            return np.asarray(sampled, dtype=np.int64)

        # Not enough centers: deterministic upsampling to keep fixed slab count.
        interp_idx = np.rint(np.linspace(0, n - 1, target_count)).astype(np.int64)
        return centers[interp_idx]

    def _resolve_lung_mask_path(self, ct_path):
        if not self.lung_mask_root:
            return None
        rel = os.path.relpath(ct_path, self.root_dir)
        rel_base = os.path.splitext(os.path.splitext(rel)[0])[0]
        return os.path.join(self.lung_mask_root, rel_base + self.lung_mask_suffix)

    def _load_lung_mask_for_case(self, ct_path, selected_idx, volume_zyx_raw):
        mask_path = self._resolve_lung_mask_path(ct_path)
        if mask_path is not None and os.path.exists(mask_path):
            mask_img = sitk.ReadImage(mask_path)
            mask_zyx = sitk.GetArrayFromImage(mask_img)
            if mask_zyx.shape[0] <= 0:
                raise RuntimeError('Invalid empty lung mask: {}'.format(mask_path))

            selected_idx = np.asarray(selected_idx, dtype=np.int64)
            selected_idx = np.clip(selected_idx, 0, mask_zyx.shape[0] - 1)
            mask_zyx = mask_zyx[selected_idx]
            return mask_zyx, 'real_mask'

        if self.lung_mask_require and self.lung_mask_root:
            raise RuntimeError(
                'Lung mask is required but missing. Expected mask path: {}. '
                'If only masked CT is available, set --no_lung_mask_require to enable pseudo mask.'.format(mask_path)
            )

        pseudo = build_pseudo_lung_mask(
            volume_zyx_raw,
            value_threshold=self.pseudo_mask_value_threshold,
            min_component_voxels=self.pseudo_mask_min_component_voxels,
        )
        return pseudo, 'pseudo_mask'

    def _load_region_context_for_case(self, nii_path, selected_idx, volume_zyx_raw):
        case_id = _get_case_id_from_path(nii_path)
        if self.cache_root:
            cached = load_case_cache(case_id, self.cache_root)
            if cached is not None:
                cached_selected = cached['meta'].get('selected_idx')
                if cached_selected is None or np.array_equal(np.asarray(cached_selected, dtype=np.int64), np.asarray(selected_idx, dtype=np.int64)):
                    region_ctx = unpack_region_context_cache(cached['cache'])
                    region_ctx['cache_meta'] = cached['meta']
                    region_ctx['mask_source'] = 'cache'
                    return region_ctx

        lung_mask, mask_source = self._load_lung_mask_for_case(nii_path, selected_idx, volume_zyx_raw)
        region_ctx = self._build_region_context(lung_mask, num_slices=len(self.channel_offsets))
        region_ctx['mask_source'] = mask_source
        return region_ctx

    def _build_region_context(self, lung_mask, num_slices):
        return build_lung_region_context_from_mask(
            lung_mask=lung_mask,
            num_slices=num_slices,
            abs_threshold=self.region_abs_area_threshold,
            ratio_threshold=self.region_ratio_area_threshold,
        )

    def _maybe_save_debug_instances(self, nii_path, bag_np, metadata):
        if (not self.debug_save_instances) or self.debug_max_cases <= 0:
            return

        case_id = os.path.splitext(os.path.splitext(os.path.basename(nii_path))[0])[0]
        if case_id in self._debug_saved_case_ids:
            return
        if len(self._debug_saved_case_ids) >= self.debug_max_cases:
            return

        case_dir = os.path.join(self.debug_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)

        for i in range(min(bag_np.shape[0], len(metadata))):
            inst = bag_np[i]  # [C,H,W]
            if inst.shape[0] >= 3:
                rgb = np.transpose(inst[:3], (1, 2, 0))
            else:
                rgb = np.repeat(inst[:1].transpose(1, 2, 0), 3, axis=2)

            vmin = float(rgb.min())
            vmax = float(rgb.max())
            if vmax <= vmin:
                img = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8)
            else:
                img = ((rgb - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)

            region = str(metadata[i].get('region', 'unknown'))
            center_z = int(metadata[i].get('center_z', -1))
            out_name = '{:03d}_{}_z{:04d}.png'.format(i, region, center_z)
            Image.fromarray(img).save(os.path.join(case_dir, out_name))

        self._debug_saved_case_ids.add(case_id)

    def _maybe_save_debug_lung_split(self, nii_path, lung_mask, left_mask, right_mask, split_method):
        if (not self.debug_save_lung_split) or self.debug_max_cases <= 0:
            return

        case_id = os.path.splitext(os.path.splitext(os.path.basename(nii_path))[0])[0]
        if case_id in self._debug_saved_split_case_ids:
            return
        if len(self._debug_saved_split_case_ids) >= self.debug_max_cases:
            return

        case_dir = os.path.join(self.debug_dir, case_id, 'lung_split')
        os.makedirs(case_dir, exist_ok=True)

        per_z_area = lung_mask.reshape(lung_mask.shape[0], -1).sum(axis=1)
        if per_z_area.size == 0:
            return
        z = int(np.argmax(per_z_area))

        lung_2d = (lung_mask[z] > 0)
        left_2d = (left_mask[z] > 0)
        right_2d = (right_mask[z] > 0)

        # RGB overlay: left=red, right=green, overlap=yellow.
        overlay = np.zeros((lung_2d.shape[0], lung_2d.shape[1], 3), dtype=np.uint8)
        overlay[:, :, 0] = (left_2d.astype(np.uint8) * 255)
        overlay[:, :, 1] = (right_2d.astype(np.uint8) * 255)

        Image.fromarray((lung_2d.astype(np.uint8) * 255)).save(
            os.path.join(case_dir, 'z{:04d}_lung_mask.png'.format(z))
        )
        Image.fromarray((left_2d.astype(np.uint8) * 255)).save(
            os.path.join(case_dir, 'z{:04d}_left_mask_{}.png'.format(z, split_method))
        )
        Image.fromarray((right_2d.astype(np.uint8) * 255)).save(
            os.path.join(case_dir, 'z{:04d}_right_mask_{}.png'.format(z, split_method))
        )
        Image.fromarray(overlay).save(
            os.path.join(case_dir, 'z{:04d}_overlay_{}.png'.format(z, split_method))
        )

        self._debug_saved_split_case_ids.add(case_id)

    def _maybe_save_debug_six_regions(self, nii_path, pseudo_mask, region_ctx):
        if (not self.debug_save_six_regions) or self.debug_max_cases <= 0:
            return

        case_id = os.path.splitext(os.path.splitext(os.path.basename(nii_path))[0])[0]
        if case_id in self._debug_saved_region_case_ids:
            return
        if len(self._debug_saved_region_case_ids) >= self.debug_max_cases:
            return

        case_dir = os.path.join(self.debug_dir, case_id, 'six_regions')
        os.makedirs(case_dir, exist_ok=True)

        per_z = pseudo_mask.reshape(pseudo_mask.shape[0], -1).sum(axis=1)
        if per_z.size == 0:
            return
        z = int(np.argmax(per_z))

        Image.fromarray((pseudo_mask[z].astype(np.uint8) * 255)).save(
            os.path.join(case_dir, 'z{:04d}_pseudo_mask.png'.format(z))
        )

        for region_name, region_mask in region_ctx['region_masks_dict'].items():
            valid_list = region_ctx['valid_region_centers_dict'].get(region_name, np.asarray([], dtype=np.int64))
            valid_count = int(valid_list.size)
            out_name = 'z{:04d}_{}_valid{:03d}.png'.format(z, region_name, valid_count)
            Image.fromarray((region_mask[z].astype(np.uint8) * 255)).save(
                os.path.join(case_dir, out_name)
            )

        self._debug_saved_region_case_ids.add(case_id)

    def __getitem__(self, index):
        nii_path, label = self.samples[index]
        image = sitk.ReadImage(nii_path)
        volume_zyx_raw = sitk.GetArrayFromImage(image).astype(np.float32)  # [Z, H, W]
        volume_zyx = self._normalize_volume(volume_zyx_raw)

        num_slices = int(volume_zyx.shape[0])
        spacing_xyz = image.GetSpacing()
        z_spacing_mm = float(spacing_xyz[2]) if len(spacing_xyz) >= 3 else 1.0

        if self.fixed_num_slices > 0:
            selected_idx = self._select_fixed_middle_indices(num_slices, z_spacing_mm=z_spacing_mm)
        else:
            selected_start, selected_end = self._select_middle_slice_range(num_slices)
            selected_idx = np.arange(selected_start, selected_end, dtype=np.int64)
            if selected_idx.size < 3:
                selected_idx = np.arange(num_slices, dtype=np.int64)

        volume_zyx_raw = volume_zyx_raw[selected_idx]
        volume_zyx = volume_zyx[selected_idx]  # [selected_slices, H, W]
        sampled_slices = int(volume_zyx.shape[0])

        sampling_seed = int((
            self.seed * 1000003
            + int(index) * 9176
            + self.current_epoch * 1361
        ) % (2 ** 32 - 1))
        sampling_rng = np.random.RandomState(sampling_seed)

        if self.instance_definition == 'lung_region_thin_slab':
            region_ctx = self._load_region_context_for_case(nii_path, selected_idx, volume_zyx_raw)
            left_mask = region_ctx['left_lung_mask']
            right_mask = region_ctx['right_lung_mask']
            split_method = region_ctx['split_method']
            pseudo_mask = region_ctx['pseudo_mask']
            mask_source = str(region_ctx.get('mask_source', 'unknown'))

            self._maybe_save_debug_lung_split(
                nii_path=nii_path,
                lung_mask=pseudo_mask,
                left_mask=left_mask,
                right_mask=right_mask,
                split_method=split_method,
            )
            self._maybe_save_debug_six_regions(
                nii_path=nii_path,
                pseudo_mask=pseudo_mask,
                region_ctx=region_ctx,
            )

            bag_np, metadata = generate_lung_region_instances(
                ct_volume=volume_zyx,
                lung_mask=pseudo_mask,
                rng=sampling_rng,
                num_instances=self.region_num_instances,
                num_slices=len(self.channel_offsets),
                out_size=self.region_out_size,
                bbox_margin=self.region_bbox_margin,
                bbox_min_size=self.region_bbox_min_size,
                abs_threshold=self.region_abs_area_threshold,
                ratio_threshold=self.region_ratio_area_threshold,
                left_mask=left_mask,
                right_mask=right_mask,
                split_method=split_method,
                region_ctx=region_ctx,
            )
            for m in metadata:
                m['mask_source'] = str(mask_source)
            if self.max_instances > 0 and bag_np.shape[0] > self.max_instances:
                keep_idx = np.linspace(0, bag_np.shape[0] - 1, self.max_instances, dtype=np.int64)
                bag_np = bag_np[keep_idx]
                metadata = [metadata[i] for i in keep_idx.tolist()]

            self._maybe_save_debug_instances(nii_path, bag_np, metadata)

            # Relative z position in [0,1] along the selected volume order.
            if sampled_slices > 1:
                pos_z = np.asarray(
                    [float(np.clip(m.get('center_z', 0), 0, sampled_slices - 1)) / float(sampled_slices - 1) for m in metadata],
                    dtype=np.float32
                )
            else:
                pos_z = np.zeros((len(metadata),), dtype=np.float32)
        else:
            eff_start, eff_end = self._select_effective_lung_range(volume_zyx_raw)

            min_offset = int(min(self.channel_offsets))
            max_offset = int(max(self.channel_offsets))
            global_start = max(0, -min_offset)
            global_end = sampled_slices - max(0, max_offset)

            center_start = max(global_start, eff_start)
            center_end = min(global_end, eff_end)
            if center_end <= center_start:
                center_start, center_end = global_start, global_end

            centers = np.arange(center_start, center_end, self.slab_stride, dtype=np.int64)
            if self.num_slabs > 0:
                centers = self._select_centers(centers, self.num_slabs, sampling_rng)

            instances = []
            for z in centers.tolist():
                channels = []
                for offset in self.channel_offsets:
                    z_idx = int(np.clip(z + offset, 0, sampled_slices - 1))
                    channels.append(volume_zyx[z_idx])
                instance = np.stack(channels, axis=0)
                instances.append(instance)

            if len(instances) == 0:
                z_low = center_start
                z_high = max(center_start, center_end - 1)
                z = int(np.clip(sampled_slices // 2, z_low, z_high))
                channels = []
                for offset in self.channel_offsets:
                    z_idx = int(np.clip(z + offset, 0, sampled_slices - 1))
                    channels.append(volume_zyx[z_idx])
                instances = [np.stack(channels, axis=0)]

            if self.max_instances > 0 and len(instances) > self.max_instances:
                keep_idx = np.linspace(0, len(instances) - 1, self.max_instances, dtype=np.int64)
                instances = [instances[i] for i in keep_idx]
                centers = centers[keep_idx]

            bag_np = np.stack(instances, axis=0).astype(np.float32)
            if centers.size > 0 and sampled_slices > 1:
                pos_z = np.clip(centers.astype(np.float32) / float(sampled_slices - 1), 0.0, 1.0)
            else:
                pos_z = np.zeros((bag_np.shape[0],), dtype=np.float32)

        bag = torch.from_numpy(bag_np).float()  # [K, C, H, W]
        bag_label = torch.tensor(label, dtype=torch.long)
        bag_pos_z = torch.from_numpy(pos_z).float()  # [K]

        return bag, bag_label, bag_pos_z


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                   mean_bag_length=10,
                                                   var_bag_length=2,
                                                   num_bag=100,
                                                   seed=1,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)

    test_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                  mean_bag_length=10,
                                                  var_bag_length=2,
                                                  num_bag=100,
                                                  seed=1,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label[0].numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_train, len(train_loader),
        np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label[0].numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_test, len(test_loader),
        np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)))
