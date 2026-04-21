"""Lung mask building and left/right lung splitting."""

import numpy as np
import SimpleITK as sitk


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
    bg_value = float(np.round(np.median(x.ravel())))
    init = np.abs(x - bg_value) > max(float(value_threshold), 0.5)
    if not np.any(init):
        init = x > float(np.percentile(x, 5))

    img = sitk.GetImageFromArray(init.astype(np.uint8))
    opened = sitk.BinaryMorphologicalOpening(img, [1, 1, 1], sitk.sitkBall)
    closed = sitk.BinaryMorphologicalClosing(opened, [1, 1, 1], sitk.sitkBall)
    filled = sitk.BinaryFillhole(closed)
    mask = sitk.GetArrayFromImage(filled).astype(np.uint8) > 0

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

    _MIN_RATIO = 0.35  # smaller component must be >= 35% of the larger one

    m = lung_mask.astype(np.int32)
    positive = m > 0
    if not np.any(positive):
        return np.zeros_like(positive), np.zeros_like(positive), 'midline_fallback'

    labels = [int(v) for v in np.unique(m) if v > 0]
    if len(labels) >= 2:
        areas = []
        for lb in labels:
            comp = (m == lb)
            areas.append((int(comp.sum()), lb))
        areas.sort(reverse=True)
        lb1 = areas[0][1]
        lb2 = areas[1][1]
        comp1 = (m == lb1)
        comp2 = (m == lb2)
        if areas[1][0] >= areas[0][0] * _MIN_RATIO:
            x1 = np.where(comp1)[2].mean() if np.any(comp1) else 0.0
            x2 = np.where(comp2)[2].mean() if np.any(comp2) else float(m.shape[2] - 1)
            if x1 <= x2:
                return comp1, comp2, 'cc3d'
            return comp2, comp1, 'cc3d'

    pair = _largest_two_components_from_binary(positive)
    if pair is not None:
        comp1, comp2 = pair
        s1, s2 = int(comp1.sum()), int(comp2.sum())
        if min(s1, s2) >= max(s1, s2) * _MIN_RATIO:
            x1 = np.where(comp1)[2].mean() if np.any(comp1) else 0.0
            x2 = np.where(comp2)[2].mean() if np.any(comp2) else float(m.shape[2] - 1)
            if x1 <= x2:
                return comp1, comp2, 'cc3d'
            return comp2, comp1, 'cc3d'

    itk_mask = sitk.GetImageFromArray(positive.astype(np.uint8))
    opened = sitk.BinaryMorphologicalOpening(itk_mask, [1, 1, 1], sitk.sitkBall)
    eroded = sitk.BinaryErode(opened, [1, 1, 1], sitk.sitkBall)
    morph_np = sitk.GetArrayFromImage(eroded).astype(np.uint8) > 0
    pair = _largest_two_components_from_binary(morph_np)
    if pair is not None:
        m1, m2 = pair
        s1, s2 = int(m1.sum()), int(m2.sum())
        if min(s1, s2) >= max(s1, s2) * _MIN_RATIO:
            x1 = np.where(m1)[2].mean() if np.any(m1) else 0.0
            x2 = np.where(m2)[2].mean() if np.any(m2) else float(m.shape[2] - 1)
            xmid = int(round(0.5 * (x1 + x2)))
            left = positive.copy()
            left[:, :, xmid + 1:] = False
            right = positive.copy()
            right[:, :, :xmid + 1] = False
            sl, sr = int(left.sum()), int(right.sum())
            if sl > 0 and sr > 0 and min(sl, sr) >= max(sl, sr) * _MIN_RATIO:
                return left, right, 'morph_cc3d'

    # Fallback: split at x-axis centroid of the positive mask
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
