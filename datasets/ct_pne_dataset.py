"""CTPneNiiBags: MIL dataset for pneumoconiosis staging from NIfTI CT volumes."""

import glob
import os

import numpy as np
import SimpleITK as sitk
import torch
import torch.utils.data as data_utils
from PIL import Image

from .ct_preprocess.cache_io import (
    load_case_cache,
    unpack_region_context_cache,
)
from .ct_preprocess.instance_builder import (
    build_lung_region_context_from_mask,
    generate_lung_region_instances,
)
from .ct_preprocess.lung_mask import build_pseudo_lung_mask
from .ct_preprocess.lung_regions import get_region_bbox
from .ct_preprocess.slice_utils import get_case_id_from_path


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
        if self.region_num_instances < 0:
            raise ValueError('region_num_instances must be >= 0 (0 means dense sampling)')
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
        self.current_epoch = int(epoch)

    def set_pseudo12_labels(self, pseudo12_array):
        """Set hard pseudo-12 subtype labels (Plan B). Array must match len(self)."""
        if pseudo12_array is not None:
            assert len(pseudo12_array) == len(self.samples), (
                'pseudo12_labels length {} != samples length {}'.format(len(pseudo12_array), len(self.samples)))
            self._pseudo12_labels = np.asarray(pseudo12_array, dtype=np.int64)
        else:
            self._pseudo12_labels = None

    def set_soft12_targets(self, soft12_array):
        """Set soft 12-subtype target distributions (Plan C). Shape: [N, 12] float32."""
        if soft12_array is not None:
            assert len(soft12_array) == len(self.samples), (
                'soft12_targets length {} != samples length {}'.format(len(soft12_array), len(self.samples)))
            self._soft12_targets = np.asarray(soft12_array, dtype=np.float32)
        else:
            self._soft12_targets = None

    def _select_middle_slice_range(self, num_slices):
        if num_slices < 3:
            return 0, num_slices

        drop_each_side = int(num_slices * (1.0 - self.middle_ratio) / 2.0)
        start = drop_each_side
        end = num_slices - drop_each_side

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
        case_id = get_case_id_from_path(nii_path)
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
            inst = bag_np[i]
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

        # Global pseudo_mask overview: pick z with max area
        per_z = pseudo_mask.reshape(pseudo_mask.shape[0], -1).sum(axis=1)
        if per_z.size == 0:
            return
        z_global = int(np.argmax(per_z))

        Image.fromarray((pseudo_mask[z_global].astype(np.uint8) * 255)).save(
            os.path.join(case_dir, 'z{:04d}_pseudo_mask.png'.format(z_global))
        )

        # Per-region: pick the z slice with max area *within that region*
        for region_name, region_mask in region_ctx['region_masks_dict'].items():
            valid_list = region_ctx['valid_region_centers_dict'].get(region_name, np.asarray([], dtype=np.int64))
            valid_count = int(valid_list.size)

            region_per_z = region_mask.reshape(region_mask.shape[0], -1).sum(axis=1)
            if region_per_z.max() > 0:
                z_region = int(np.argmax(region_per_z))
            else:
                z_region = z_global

            out_name = 'z{:04d}_{}_valid{:03d}.png'.format(z_region, region_name, valid_count)
            Image.fromarray((region_mask[z_region].astype(np.uint8) * 255)).save(
                os.path.join(case_dir, out_name)
            )

        self._debug_saved_region_case_ids.add(case_id)

    def __getitem__(self, index):
        nii_path, label = self.samples[index]
        image = sitk.ReadImage(nii_path)
        volume_zyx_raw = sitk.GetArrayFromImage(image).astype(np.float32)
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
        volume_zyx = volume_zyx[selected_idx]
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

        bag = torch.from_numpy(bag_np).float()
        bag_label = torch.tensor(label, dtype=torch.long)
        bag_pos_z = torch.from_numpy(pos_z).float()

        # Store metadata for heatmap generation (accessible via dataset._last_metadata)
        if self.instance_definition == 'lung_region_thin_slab':
            self._last_metadata = metadata
        else:
            self._last_metadata = [{'region': 'global', 'center_z': 0}] * bag_np.shape[0]

        # Pseudo-12 / Soft-12 label (4th element if set)
        # Plan C soft target takes priority over Plan B hard label
        if hasattr(self, '_soft12_targets') and self._soft12_targets is not None:
            soft_t = torch.from_numpy(self._soft12_targets[index]).float()  # [12]
            return bag, bag_label, bag_pos_z, soft_t
        if hasattr(self, '_pseudo12_labels') and self._pseudo12_labels is not None:
            p12 = torch.tensor(self._pseudo12_labels[index], dtype=torch.long)
            return bag, bag_label, bag_pos_z, p12

        return bag, bag_label, bag_pos_z
