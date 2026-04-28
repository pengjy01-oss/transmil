"""2.5D slab building and region-aware instance generation."""

import numpy as np
import torch
import torch.nn.functional as F

from .lung_mask import build_pseudo_lung_mask, split_left_right_lung
from .lung_regions import (
    get_lung_region_ranges,
    get_region_bbox,
    get_six_lung_regions,
    get_valid_region_centers,
)


def sample_region_centers(candidate_centers, num_samples, rng):
    """Sample centers from legal candidates, allowing repeats when insufficient."""
    candidate_centers = np.asarray(candidate_centers, dtype=np.int64)
    if num_samples <= 0:
        return np.asarray([], dtype=np.int64)
    if candidate_centers.size == 0:
        return np.asarray([], dtype=np.int64)

    if candidate_centers.size >= num_samples:
        idx = np.rint(np.linspace(0, candidate_centers.size - 1, num_samples)).astype(np.int64)
        return candidate_centers[idx]

    extra_idx = rng.choice(candidate_centers.size, size=(num_samples - candidate_centers.size), replace=True)
    out = np.concatenate([candidate_centers, candidate_centers[extra_idx]], axis=0)
    rng.shuffle(out)
    return out.astype(np.int64)


def _resize_instance_chw(instance_chw, out_size):
    x = torch.from_numpy(instance_chw).unsqueeze(0).float()
    x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
    return x.squeeze(0).numpy().astype(np.float32)


def build_2p5d_region_instance(ct_volume, region_mask, center_z, out_size, region_bbox_fallback,
                               num_slices=3, bbox_margin=12, bbox_min_size=32):
    """Build one [C,H,W] instance from a region around center_z with crop+resize."""
    z_total = int(ct_volume.shape[0])
    half = int(num_slices // 2)
    z_ids = [int(np.clip(center_z + dz, 0, z_total - 1)) for dz in range(-half, half + 1)]

    slab = ct_volume[np.asarray(z_ids, dtype=np.int64)]
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
                                   region_ctx=None, fixed_base_total=0):
    """Generate [N,C,H,W] region-aware 2.5D instances and metadata."""
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

    dense_mode = (num_instances <= 0)

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

    if dense_mode:
        # Dense mode: use ALL valid centers per region
        alloc = {k: int(legal_counts[k]) for k in region_names}
    else:
        n_regions = len(region_names)
        if fixed_base_total > 0:
            # fixed_base_total 均分6区，剩余按肺区有效中心数比例分配
            # e.g. fixed_base_total=96 -> 96//6=16 per region, remain=128-96=32 proportional
            base_per_region = int(fixed_base_total) // n_regions
        else:
            base_per_region = num_instances // n_regions          # e.g. 128//6 = 21
        base_total = base_per_region * n_regions              # e.g. 16*6 = 96
        alloc = {k: base_per_region for k in region_names}
        remain = int(num_instances - base_total)              # e.g. 128-126 = 2
        if remain < 0:
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

        if candidates.size == 0:
            if dense_mode:
                continue  # skip empty regions in dense mode
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

        if dense_mode:
            chosen = np.sort(candidates)  # all centers, z-ordered
        else:
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

    if not dense_mode:
        if len(instances) != num_instances:
            if len(instances) > num_instances:
                instances = instances[:num_instances]
                metadata = metadata[:num_instances]
            else:
                while len(instances) < num_instances:
                    instances.append(instances[-1].copy())
                    metadata.append(dict(metadata[-1]))

    if len(instances) == 0:
        raise RuntimeError('Dense sampling produced 0 instances; check lung masks.')

    return np.stack(instances, axis=0).astype(np.float32), metadata


def build_lung_region_skeleton_from_mask(lung_mask):
    """Build the selected_idx-independent part of region context from a binary lung mask."""
    left_mask, right_mask, split_method = split_left_right_lung(lung_mask)
    region_masks = get_six_lung_regions(left_mask, right_mask)
    return {
        'pseudo_mask': (lung_mask > 0).astype(bool),
        'left_lung_mask': left_mask.astype(bool),
        'right_lung_mask': right_mask.astype(bool),
        'split_method': str(split_method),
        'region_masks_dict': region_masks,
    }


def build_lung_region_context_from_mask(lung_mask, num_slices=3, abs_threshold=50, ratio_threshold=0.05):
    """Build full region context dict from a binary lung mask."""
    region_ctx = build_lung_region_skeleton_from_mask(lung_mask)
    region_masks = region_ctx['region_masks_dict']
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
        'pseudo_mask': region_ctx['pseudo_mask'],
        'left_lung_mask': region_ctx['left_lung_mask'],
        'right_lung_mask': region_ctx['right_lung_mask'],
        'split_method': region_ctx['split_method'],
        'region_masks_dict': region_masks,
        'valid_region_centers_dict': valid_centers,
        'region_bboxes_dict': region_bboxes,
        'num_slices_for_valid_centers': int(num_slices),
        'abs_threshold': float(abs_threshold),
        'ratio_threshold': float(ratio_threshold),
    }
