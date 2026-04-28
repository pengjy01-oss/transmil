"""Cache I/O for preprocessed lung region data, and debug visualization helpers."""

import json
import os

import numpy as np
from PIL import Image

from .lung_mask import build_pseudo_lung_mask
from .instance_builder import build_lung_region_context_from_mask
from .slice_utils import CACHE_VERSION


def save_case_cache(case_id, cache_root, cache_dict, meta_dict):
    """Save preprocess cache (npz + json meta) for one case."""
    case_dir = os.path.join(cache_root, str(case_id))
    os.makedirs(case_dir, exist_ok=True)

    cache_path = os.path.join(case_dir, 'preprocess_cache.npz')
    meta_path = os.path.join(case_dir, 'preprocess_meta.json')

    np.savez_compressed(cache_path, **cache_dict)
    with open(meta_path, 'w') as f:
        json.dump(meta_dict, f, indent=2, sort_keys=True)

    return cache_path, meta_path


def load_case_cache(case_id, cache_root):
    """Load preprocess cache for one case, returning None if missing/stale."""
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


def pack_region_context_cache(region_ctx):
    """Pack region context dict into numpy arrays for npz serialization."""
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
    """Unpack npz cache dict back into region context dict."""
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


def get_full_mask_cache_path(case_id, cache_root):
    """Return path to full-volume lung mask cache file."""
    return os.path.join(cache_root, str(case_id), 'full_lung_mask.npz')


def get_full_region_skeleton_cache_path(case_id, cache_root):
    """Return path to full-volume region skeleton cache file."""
    return os.path.join(cache_root, str(case_id), 'full_region_skeleton.npz')


def load_full_mask_cache(case_id, cache_root):
    """Load full-volume lung mask cache. Returns bool ndarray [N,H,W] or None."""
    path = get_full_mask_cache_path(case_id, cache_root)
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        return data['mask'].astype(bool)
    except Exception:
        return None


def save_full_mask_cache(case_id, cache_root, mask):
    """Save full-volume lung mask (bool/uint8 [N,H,W]) to compressed cache."""
    path = get_full_mask_cache_path(case_id, cache_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, mask=np.asarray(mask, dtype=np.uint8))


def load_full_region_skeleton_cache(case_id, cache_root):
    """Load selected_idx-independent full-volume region skeleton cache."""
    path = get_full_region_skeleton_cache_path(case_id, cache_root)
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        region_masks_dict = {}
        for key in data.files:
            if key.startswith('region_mask__'):
                region_masks_dict[key[len('region_mask__'):]] = data[key].astype(bool)

        required = ['mask', 'left_lung_mask', 'right_lung_mask', 'split_method']
        if any(key not in data.files for key in required):
            return None

        return {
            'pseudo_mask': data['mask'].astype(bool),
            'left_lung_mask': data['left_lung_mask'].astype(bool),
            'right_lung_mask': data['right_lung_mask'].astype(bool),
            'split_method': str(np.asarray(data['split_method']).item()),
            'region_masks_dict': region_masks_dict,
        }
    except Exception:
        return None


def save_full_region_skeleton_cache(case_id, cache_root, region_ctx):
    """Save selected_idx-independent full-volume region skeleton cache."""
    path = get_full_region_skeleton_cache_path(case_id, cache_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    cache_dict = {
        'mask': np.asarray(region_ctx['pseudo_mask'], dtype=np.uint8),
        'left_lung_mask': np.asarray(region_ctx['left_lung_mask'], dtype=np.uint8),
        'right_lung_mask': np.asarray(region_ctx['right_lung_mask'], dtype=np.uint8),
        'split_method': np.asarray(region_ctx['split_method']),
    }
    for region_name, region_mask in region_ctx['region_masks_dict'].items():
        cache_dict['region_mask__{}'.format(region_name)] = np.asarray(region_mask, dtype=np.uint8)

    np.savez_compressed(path, **cache_dict)


def save_region_context_debug(case_id, debug_root, pseudo_mask, region_ctx):
    """Save debug visualization images for one case's region context."""
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
                                num_slices_for_valid_centers=3, cache_version=None,
                                pseudo_mask_value_threshold=1e-6, pseudo_mask_min_component_voxels=512):
    """Build full preprocess cache for one case (pseudo mask + region context)."""
    if cache_version is None:
        cache_version = CACHE_VERSION

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
