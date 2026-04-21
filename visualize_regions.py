"""Quick script to regenerate six-region debug visualizations."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils.config import parse_args
from datasets.ct_pne_dataset import CTPneNiiBags

def main():
    # Parse config but override debug flags
    sys.argv = [
        'visualize_regions.py',
        '--config', 'configs/ct25d_transmil.yaml',
        '--debug_save_six_regions',
        '--debug_max_cases', '3',
        '--debug_dir', '.',
    ]
    args = parse_args()

    half_depth = args.slab_depth // 2
    channel_offsets = tuple(range(-half_depth, half_depth + 1))

    ds = CTPneNiiBags(
        split='train',
        root_dir=args.data_root,
        num_classes=args.num_classes,
        middle_ratio=args.middle_ratio,
        fixed_num_slices=args.fixed_num_slices,
        channel_offsets=channel_offsets,
        slab_stride=args.slab_stride,
        num_slabs=args.num_slabs,
        center_sampling_mode=args.center_sampling_mode,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        scale_to_unit=args.scale_to_unit,
        use_zscore=args.use_zscore,
        lung_hu_low=args.lung_hu_low,
        lung_hu_high=args.lung_hu_high,
        min_lung_area_ratio=args.min_lung_area_ratio,
        max_instances=args.max_instances,
        instance_definition=args.instance_definition,
        lung_mask_root=args.lung_mask_root,
        lung_mask_suffix=args.lung_mask_suffix,
        lung_mask_require=args.lung_mask_require,
        cache_root=args.cache_root,
        pseudo_mask_value_threshold=args.pseudo_mask_value_threshold,
        pseudo_mask_min_component_voxels=args.pseudo_mask_min_component_voxels,
        region_num_instances=args.region_num_instances,
        region_out_size=(args.region_out_h, args.region_out_w),
        region_bbox_margin=args.region_bbox_margin,
        region_bbox_min_size=args.region_bbox_min_size,
        region_abs_area_threshold=args.region_abs_area_threshold,
        region_ratio_area_threshold=args.region_ratio_area_threshold,
        debug_save_instances=False,
        debug_save_lung_split=False,
        debug_save_six_regions=True,
        debug_dir='.',
        debug_max_cases=3,
    )

    print(f'Dataset has {len(ds)} samples, triggering __getitem__ for first few...')
    for i in range(min(len(ds), len(ds))):
        _ = ds[i]
        # Check if we've saved enough
        if len(ds._debug_saved_region_case_ids) >= 3:
            print(f'Saved debug for {len(ds._debug_saved_region_case_ids)} cases: {ds._debug_saved_region_case_ids}')
            break

    print('Done. Check per-case six_regions/ folders.')

if __name__ == '__main__':
    main()
