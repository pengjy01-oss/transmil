#!/usr/bin/env python3
"""
全链路值域对比：原始DICOM → 分割DICOM → NIfTI
确认值域差异究竟在哪一步引入
"""
import os, glob, sys
import numpy as np
import SimpleITK as sitk
import pydicom
from collections import defaultdict

DATA_ROOT = '/data/scratch/c_pne'
OUTPUT_DIR = '/home/pjy/transmil/TransMIL/dataset_audit_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 5  # 每类采样数

def read_dicom_series_stats(folder):
    """读取一个case的DICOM序列，返回像素统计"""
    dcm_files = sorted(glob.glob(os.path.join(folder, '*.DCM')) + glob.glob(os.path.join(folder, '*.dcm')))
    if not dcm_files:
        return None
    
    slices = []
    hu_slices = []
    has_rescale = False
    slope = 1.0
    intercept = 0.0
    
    for f in dcm_files[:50]:  # 只读前50个切片加速
        try:
            ds = pydicom.dcmread(f)
            arr = ds.pixel_array.astype(np.float64)
            slices.append(arr)
            
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                has_rescale = True
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                hu_arr = arr * slope + intercept
                hu_slices.append(hu_arr)
            else:
                hu_slices.append(arr)
        except Exception as e:
            continue
    
    if not slices:
        return None
    
    vol = np.stack(slices)
    hu_vol = np.stack(hu_slices)
    
    return {
        'n_slices_read': len(slices),
        'raw_min': float(vol.min()),
        'raw_max': float(vol.max()),
        'raw_mean': float(vol.mean()),
        'raw_std': float(vol.std()),
        'has_rescale': has_rescale,
        'slope': slope,
        'intercept': intercept,
        'hu_min': float(hu_vol.min()),
        'hu_max': float(hu_vol.max()),
        'hu_mean': float(hu_vol.mean()),
        'hu_std': float(hu_vol.std()),
        'hu_p1': float(np.percentile(hu_vol, 1)),
        'hu_p5': float(np.percentile(hu_vol, 5)),
        'hu_p50': float(np.percentile(hu_vol, 50)),
        'hu_p95': float(np.percentile(hu_vol, 95)),
        'hu_p99': float(np.percentile(hu_vol, 99)),
    }

def read_nifti_stats(fpath):
    """读取NIfTI文件统计"""
    img = sitk.ReadImage(fpath)
    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    return {
        'nii_shape': arr.shape,
        'nii_min': float(arr.min()),
        'nii_max': float(arr.max()),
        'nii_mean': float(arr.mean()),
        'nii_std': float(arr.std()),
        'nii_p1': float(np.percentile(arr, 1)),
        'nii_p5': float(np.percentile(arr, 5)),
        'nii_p50': float(np.percentile(arr, 50)),
        'nii_p95': float(np.percentile(arr, 95)),
        'nii_p99': float(np.percentile(arr, 99)),
        'spacing': img.GetSpacing(),
    }

# ============================================================
print("=" * 70)
print("全链路值域对比: 原始DICOM → 分割DICOM → NIfTI")
print("=" * 70)

results = []

for cls in range(4):
    orig_dir = os.path.join(DATA_ROOT, str(cls))
    seg_dir = os.path.join(DATA_ROOT, f'{cls}_seg')
    nii_dir = os.path.join(DATA_ROOT, f'{cls}_seg_nii')
    
    # 找共有的case
    orig_cases = set(os.listdir(orig_dir)) if os.path.exists(orig_dir) else set()
    seg_cases = set(os.listdir(seg_dir)) if os.path.exists(seg_dir) else set()
    nii_files = {f.replace('.nii.gz', ''): f for f in os.listdir(nii_dir)} if os.path.exists(nii_dir) else {}
    
    common = sorted(orig_cases & seg_cases & set(nii_files.keys()))
    print(f"\n类别{cls}: 原始{len(orig_cases)}, 分割{len(seg_cases)}, NIfTI{len(nii_files)}, 三阶段共有{len(common)}")
    
    # 采样
    rng = np.random.RandomState(42)
    sample = rng.choice(common, min(NUM_SAMPLES, len(common)), replace=False) if common else []
    
    for case_id in sample:
        print(f"\n  [{cls}] {case_id}:")
        rec = {'class': cls, 'case_id': case_id}
        
        # 1. 原始DICOM
        orig_path = os.path.join(orig_dir, case_id)
        orig_stats = read_dicom_series_stats(orig_path)
        if orig_stats:
            for k, v in orig_stats.items():
                rec[f'orig_{k}'] = v
            print(f"    原始DICOM: HU [{orig_stats['hu_min']:.0f}, {orig_stats['hu_max']:.0f}], "
                  f"mean={orig_stats['hu_mean']:.1f}, std={orig_stats['hu_std']:.1f}, "
                  f"rescale={orig_stats['has_rescale']} (slope={orig_stats['slope']}, intercept={orig_stats['intercept']})")
        
        # 2. 分割后DICOM
        seg_path = os.path.join(seg_dir, case_id)
        seg_stats = read_dicom_series_stats(seg_path)
        if seg_stats:
            for k, v in seg_stats.items():
                rec[f'seg_{k}'] = v
            print(f"    分割DICOM: HU [{seg_stats['hu_min']:.0f}, {seg_stats['hu_max']:.0f}], "
                  f"mean={seg_stats['hu_mean']:.1f}, std={seg_stats['hu_std']:.1f}")
        
        # 3. NIfTI
        nii_path = os.path.join(nii_dir, nii_files[case_id])
        nii_stats = read_nifti_stats(nii_path)
        if nii_stats:
            for k, v in nii_stats.items():
                rec[f'nii_{k}'] = v if not isinstance(v, tuple) else str(v)
            print(f"    NIfTI:     [{nii_stats['nii_min']:.0f}, {nii_stats['nii_max']:.0f}], "
                  f"mean={nii_stats['nii_mean']:.1f}, std={nii_stats['nii_std']:.1f}, "
                  f"shape={nii_stats['nii_shape']}, spacing={nii_stats['spacing']}")
        
        # 对比差异
        if orig_stats and seg_stats:
            print(f"    >>> 分割导致: max从{orig_stats['hu_max']:.0f}变为{seg_stats['hu_max']:.0f}, "
                  f"min从{orig_stats['hu_min']:.0f}变为{seg_stats['hu_min']:.0f}")
        if seg_stats and nii_stats:
            print(f"    >>> DICOM→NII: min从{seg_stats['hu_min']:.0f}变为{nii_stats['nii_min']:.0f}, "
                  f"max从{seg_stats['hu_max']:.0f}变为{nii_stats['nii_max']:.0f}")
        
        results.append(rec)

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 70)
print("汇总: 各类各阶段值域范围")
print("=" * 70)

for cls in range(4):
    cls_results = [r for r in results if r['class'] == cls]
    if not cls_results:
        continue
    
    print(f"\n--- 类别 {cls} ({len(cls_results)} samples) ---")
    
    # 原始
    if any(f'orig_hu_min' in r for r in cls_results):
        orig_mins = [r['orig_hu_min'] for r in cls_results if 'orig_hu_min' in r]
        orig_maxs = [r['orig_hu_max'] for r in cls_results if 'orig_hu_max' in r]
        orig_means = [r['orig_hu_mean'] for r in cls_results if 'orig_hu_mean' in r]
        orig_stds = [r['orig_hu_std'] for r in cls_results if 'orig_hu_std' in r]
        print(f"  原始DICOM HU: min=[{min(orig_mins):.0f},{max(orig_mins):.0f}], "
              f"max=[{min(orig_maxs):.0f},{max(orig_maxs):.0f}], "
              f"mean={np.mean(orig_means):.1f}±{np.std(orig_means):.1f}, "
              f"std={np.mean(orig_stds):.1f}±{np.std(orig_stds):.1f}")
    
    # 分割
    if any(f'seg_hu_min' in r for r in cls_results):
        seg_mins = [r['seg_hu_min'] for r in cls_results if 'seg_hu_min' in r]
        seg_maxs = [r['seg_hu_max'] for r in cls_results if 'seg_hu_max' in r]
        seg_means = [r['seg_hu_mean'] for r in cls_results if 'seg_hu_mean' in r]
        seg_stds = [r['seg_hu_std'] for r in cls_results if 'seg_hu_std' in r]
        print(f"  分割DICOM HU: min=[{min(seg_mins):.0f},{max(seg_mins):.0f}], "
              f"max=[{min(seg_maxs):.0f},{max(seg_maxs):.0f}], "
              f"mean={np.mean(seg_means):.1f}±{np.std(seg_means):.1f}, "
              f"std={np.mean(seg_stds):.1f}±{np.std(seg_stds):.1f}")
    
    # NIfTI
    if any(f'nii_nii_min' in r for r in cls_results):
        nii_mins = [r['nii_nii_min'] for r in cls_results if 'nii_nii_min' in r]
        nii_maxs = [r['nii_nii_max'] for r in cls_results if 'nii_nii_max' in r]
        nii_means = [r['nii_nii_mean'] for r in cls_results if 'nii_nii_mean' in r]
        nii_stds = [r['nii_nii_std'] for r in cls_results if 'nii_nii_std' in r]
        print(f"  NIfTI:        min=[{min(nii_mins):.0f},{max(nii_mins):.0f}], "
              f"max=[{min(nii_maxs):.0f},{max(nii_maxs):.0f}], "
              f"mean={np.mean(nii_means):.1f}±{np.std(nii_means):.1f}, "
              f"std={np.mean(nii_stds):.1f}±{np.std(nii_stds):.1f}")

# 检查0期原始和123期原始是否一致
print("\n" + "=" * 70)
print("关键问题: 原始DICOM阶段，0期与123期值域是否已经不同?")
print("=" * 70)
cls0_orig = [r for r in results if r['class'] == 0 and 'orig_hu_min' in r]
cls123_orig = [r for r in results if r['class'] != 0 and 'orig_hu_min' in r]

if cls0_orig and cls123_orig:
    print(f"  0期原始HU: min范围=[{min(r['orig_hu_min'] for r in cls0_orig):.0f}, {max(r['orig_hu_min'] for r in cls0_orig):.0f}], "
          f"max范围=[{min(r['orig_hu_max'] for r in cls0_orig):.0f}, {max(r['orig_hu_max'] for r in cls0_orig):.0f}]")
    print(f"  123期原始HU: min范围=[{min(r['orig_hu_min'] for r in cls123_orig):.0f}, {max(r['orig_hu_min'] for r in cls123_orig):.0f}], "
          f"max范围=[{min(r['orig_hu_max'] for r in cls123_orig):.0f}, {max(r['orig_hu_max'] for r in cls123_orig):.0f}]")
    
    # 分割后对比
    cls0_seg = [r for r in results if r['class'] == 0 and 'seg_hu_min' in r]
    cls123_seg = [r for r in results if r['class'] != 0 and 'seg_hu_min' in r]
    if cls0_seg and cls123_seg:
        print(f"\n  0期分割后HU: min范围=[{min(r['seg_hu_min'] for r in cls0_seg):.0f}, {max(r['seg_hu_min'] for r in cls0_seg):.0f}], "
              f"max范围=[{min(r['seg_hu_max'] for r in cls0_seg):.0f}, {max(r['seg_hu_max'] for r in cls0_seg):.0f}]")
        print(f"  123期分割后HU: min范围=[{min(r['seg_hu_min'] for r in cls123_seg):.0f}, {max(r['seg_hu_min'] for r in cls123_seg):.0f}], "
              f"max范围=[{min(r['seg_hu_max'] for r in cls123_seg):.0f}, {max(r['seg_hu_max'] for r in cls123_seg):.0f}]")

# 检查rescale参数差异
print("\n" + "=" * 70)
print("Rescale参数对比 (slope/intercept)")
print("=" * 70)
for cls in range(4):
    cls_r = [r for r in results if r['class'] == cls]
    if cls_r:
        slopes = [r.get('orig_slope', None) for r in cls_r]
        intercepts = [r.get('orig_intercept', None) for r in cls_r]
        has_rescale = [r.get('orig_has_rescale', None) for r in cls_r]
        print(f"  类别{cls}: has_rescale={has_rescale}, slopes={slopes}, intercepts={intercepts}")

# 额外检查：0期的DICOM是否缺少RescaleSlope/Intercept
print("\n" + "=" * 70)
print("额外: 检查0期DICOM头部字段")
print("=" * 70)
sample_0 = os.path.join(DATA_ROOT, '0', os.listdir(os.path.join(DATA_ROOT, '0'))[0])
sample_1 = os.path.join(DATA_ROOT, '1', os.listdir(os.path.join(DATA_ROOT, '1'))[0])

for label, folder in [("0期", sample_0), ("1期", sample_1)]:
    dcm_files = sorted(glob.glob(os.path.join(folder, '*.DCM')) + glob.glob(os.path.join(folder, '*.dcm')))
    if dcm_files:
        ds = pydicom.dcmread(dcm_files[0])
        print(f"\n  {label} ({os.path.basename(folder)}):")
        for tag in ['Manufacturer', 'ManufacturerModelName', 'InstitutionName', 
                     'KVP', 'SliceThickness', 'SpacingBetweenSlices',
                     'RescaleSlope', 'RescaleIntercept', 'RescaleType',
                     'WindowCenter', 'WindowWidth', 'BitsAllocated', 'BitsStored',
                     'PixelRepresentation', 'Rows', 'Columns',
                     'PixelSpacing', 'ImagePositionPatient',
                     'ConvolutionKernel', 'ReconstructionDiameter']:
            val = getattr(ds, tag, 'N/A')
            print(f"    {tag}: {val}")

print("\n完成!")
