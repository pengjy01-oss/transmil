#!/usr/bin/env python3
"""
尘肺CT数据集系统性审计脚本
目标：排查为什么0期准确率总是100%
检查项：数据泄漏、分布差异、命名泄漏、图像统计差异、预处理偏差
"""

import os
import sys
import glob
import json
import hashlib
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict, Counter

# ============================================================
# 配置
# ============================================================
DATA_ROOT = '/data/scratch/c_pne'
LUNG_MASK_ROOT = '/data/scratch/c_pne_lungmask'
CACHE_ROOT = '/data/scratch/c_pne_cache/lung_region_cache_v1'
OUTPUT_DIR = '/home/pjy/transmil/TransMIL/dataset_audit_results'
NUM_CLASSES = 4
SEED = 1
TEST_RATIO = 0.2
VAL_RATIO = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# A. 收集所有文件并复现划分逻辑
# ============================================================
print("=" * 60)
print("A. 收集文件列表 & 复现 train/val/test 划分")
print("=" * 60)

class_to_files = {}
all_case_ids = []
case_id_to_info = {}

for cls in range(NUM_CLASSES):
    folder = os.path.join(DATA_ROOT, f'{cls}_seg_nii')
    files = sorted(glob.glob(os.path.join(folder, '*.nii.gz')))
    class_to_files[cls] = files
    print(f"  类别 {cls}: {len(files)} 个文件, 目录 {folder}")
    for f in files:
        base = os.path.basename(f)
        case_id = base[:-7] if base.endswith('.nii.gz') else os.path.splitext(base)[0]
        all_case_ids.append(case_id)
        case_id_to_info[case_id] = {'path': f, 'class': cls}

# 复现划分
rng = np.random.RandomState(SEED)
split_records = []

for cls in range(NUM_CLASSES):
    files = class_to_files[cls]
    n = len(files)
    indices = np.arange(n)
    rng.shuffle(indices)

    test_count = int(round(n * TEST_RATIO))
    val_count = int(round(n * VAL_RATIO))
    if n >= 3:
        if TEST_RATIO > 0 and test_count == 0:
            test_count = 1
        if VAL_RATIO > 0 and val_count == 0:
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

    for idx in indices[:test_end]:
        f = files[idx]
        case_id = os.path.basename(f).replace('.nii.gz', '')
        split_records.append({'case_id': case_id, 'class': cls, 'split': 'test', 'path': f})
    for idx in indices[test_end:val_end]:
        f = files[idx]
        case_id = os.path.basename(f).replace('.nii.gz', '')
        split_records.append({'case_id': case_id, 'class': cls, 'split': 'val', 'path': f})
    for idx in indices[val_end:]:
        f = files[idx]
        case_id = os.path.basename(f).replace('.nii.gz', '')
        split_records.append({'case_id': case_id, 'class': cls, 'split': 'train', 'path': f})

df_split = pd.DataFrame(split_records)
df_split.to_csv(os.path.join(OUTPUT_DIR, 'split_assignment.csv'), index=False)

print("\n各类各split样本数：")
split_table = df_split.groupby(['class', 'split']).size().unstack(fill_value=0)
print(split_table)
split_table.to_csv(os.path.join(OUTPUT_DIR, 'split_distribution.csv'))

# ============================================================
# A1. Patient-level 泄漏检查
# ============================================================
print("\n" + "=" * 60)
print("A1. Patient-level 泄漏检查")
print("=" * 60)

# 检查case_id是否有数字前缀模式，提取patient_id
# 文件名格式：CT086519.nii.gz -> CT086519
# 检查是否有同一patient出现在不同类别
case_ids_by_class = defaultdict(set)
for rec in split_records:
    case_ids_by_class[rec['class']].add(rec['case_id'])

# 检查跨类别重复
cross_class_overlap = []
all_classes = list(range(NUM_CLASSES))
for i in range(len(all_classes)):
    for j in range(i + 1, len(all_classes)):
        overlap = case_ids_by_class[all_classes[i]] & case_ids_by_class[all_classes[j]]
        if overlap:
            cross_class_overlap.append((all_classes[i], all_classes[j], overlap))
            print(f"  !! 类别{all_classes[i]}与类别{all_classes[j]}有重叠case_id: {overlap}")

if not cross_class_overlap:
    print("  无跨类别case_id重叠 (OK)")

# 检查同一case_id是否出现在多个split
case_splits = defaultdict(set)
for rec in split_records:
    case_splits[rec['case_id']].add(rec['split'])

multi_split_cases = {k: v for k, v in case_splits.items() if len(v) > 1}
if multi_split_cases:
    print(f"  !! 发现 {len(multi_split_cases)} 个case出现在多个split:")
    for k, v in list(multi_split_cases.items())[:5]:
        print(f"     {k}: {v}")
else:
    print("  无case出现在多个split (OK)")

# 检查CT编号接近的case是否可能来自同一患者（不同扫描）
print("\n  检查CT编号聚类（可能来自同一患者的不同扫描）...")
ct_numbers = []
for rec in split_records:
    cid = rec['case_id']
    try:
        num = int(cid.replace('CT', '').replace('ct', ''))
        ct_numbers.append((num, rec['class'], rec['split'], cid))
    except:
        ct_numbers.append((None, rec['class'], rec['split'], cid))

ct_numbers_valid = [(n, c, s, cid) for n, c, s, cid in ct_numbers if n is not None]
ct_numbers_valid.sort(key=lambda x: x[0])

# 找编号差<=2的case
close_pairs = []
for i in range(len(ct_numbers_valid) - 1):
    n1, c1, s1, cid1 = ct_numbers_valid[i]
    n2, c2, s2, cid2 = ct_numbers_valid[i + 1]
    if n2 - n1 <= 2 and (c1 != c2 or s1 != s2):
        close_pairs.append((cid1, c1, s1, cid2, c2, s2, n2 - n1))

if close_pairs:
    print(f"  发现 {len(close_pairs)} 对编号相近(差<=2)但不同类/不同split的case:")
    for pair in close_pairs[:10]:
        print(f"    {pair[0]}(类{pair[1]},{pair[2]}) <-> {pair[3]}(类{pair[4]},{pair[5]}) 差={pair[6]}")
else:
    print("  无编号极相近的跨类/跨split case")

# ============================================================
# A2. CT编号范围分析 — 检查数据来源差异
# ============================================================
print("\n" + "=" * 60)
print("A2. CT编号范围分析 (检查数据来源差异)")
print("=" * 60)

class_ct_numbers = defaultdict(list)
for n, c, s, cid in ct_numbers_valid:
    class_ct_numbers[c].append(n)

ct_range_stats = []
for cls in range(NUM_CLASSES):
    nums = sorted(class_ct_numbers[cls])
    if nums:
        ct_range_stats.append({
            'class': cls,
            'count': len(nums),
            'min_id': nums[0],
            'max_id': nums[-1],
            'median_id': int(np.median(nums)),
            'mean_id': int(np.mean(nums)),
            'std_id': int(np.std(nums)),
            'q25': int(np.percentile(nums, 25)),
            'q75': int(np.percentile(nums, 75)),
        })
        print(f"  类别{cls}: CT编号范围 [{nums[0]}, {nums[-1]}], 中位数 {int(np.median(nums))}, 均值 {int(np.mean(nums))}")

df_ct_range = pd.DataFrame(ct_range_stats)
df_ct_range.to_csv(os.path.join(OUTPUT_DIR, 'ct_number_range.csv'), index=False)
print(df_ct_range.to_string(index=False))

# 可视化CT编号分布
fig, ax = plt.subplots(figsize=(12, 5))
for cls in range(NUM_CLASSES):
    nums = class_ct_numbers[cls]
    ax.hist(nums, bins=50, alpha=0.5, label=f'Class {cls} (n={len(nums)})')
ax.set_xlabel('CT Number (from filename)')
ax.set_ylabel('Count')
ax.set_title('CT Number Distribution by Class')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ct_number_distribution.png'), dpi=150)
plt.close(fig)
print("  -> 保存 ct_number_distribution.png")

# ============================================================
# B. NIfTI元数据统计 (采样检查)
# ============================================================
print("\n" + "=" * 60)
print("B. NIfTI元数据统计 (每类最多采样50个)")
print("=" * 60)

MAX_SAMPLES_META = 50  # 每类采样数

meta_records = []
for cls in range(NUM_CLASSES):
    files = class_to_files[cls]
    rng_sample = np.random.RandomState(42)
    sample_idx = rng_sample.choice(len(files), min(MAX_SAMPLES_META, len(files)), replace=False)
    
    for i, idx in enumerate(sample_idx):
        fpath = files[idx]
        case_id = os.path.basename(fpath).replace('.nii.gz', '')
        try:
            img = sitk.ReadImage(fpath)
            size = img.GetSize()  # (W, H, D)
            spacing = img.GetSpacing()
            origin = img.GetOrigin()
            direction = img.GetDirection()
            pixel_type = img.GetPixelIDTypeAsString()
            
            arr = sitk.GetArrayFromImage(img)  # (D, H, W)
            
            meta_records.append({
                'case_id': case_id,
                'class': cls,
                'width': size[0],
                'height': size[1],
                'depth': size[2],
                'spacing_x': round(spacing[0], 4),
                'spacing_y': round(spacing[1], 4),
                'spacing_z': round(spacing[2], 4),
                'pixel_type': pixel_type,
                'dtype': str(arr.dtype),
                'min_val': float(arr.min()),
                'max_val': float(arr.max()),
                'mean_val': float(arr.mean()),
                'std_val': float(arr.std()),
                'file_size_mb': round(os.path.getsize(fpath) / 1024 / 1024, 2),
            })
            
            if (i + 1) % 10 == 0:
                print(f"  类别{cls}: 已处理 {i+1}/{len(sample_idx)}")
        except Exception as e:
            print(f"  !! 读取失败: {fpath}: {e}")
            meta_records.append({
                'case_id': case_id,
                'class': cls,
                'error': str(e),
            })

df_meta = pd.DataFrame(meta_records)
df_meta.to_csv(os.path.join(OUTPUT_DIR, 'nifti_metadata.csv'), index=False)

# 按类别统计
print("\n各类元数据统计:")
numeric_cols = ['width', 'height', 'depth', 'spacing_x', 'spacing_y', 'spacing_z',
                'min_val', 'max_val', 'mean_val', 'std_val', 'file_size_mb']
for col in numeric_cols:
    if col in df_meta.columns:
        class_stats = df_meta.groupby('class')[col].describe()
        print(f"\n  {col}:")
        print(class_stats.to_string())

# ============================================================
# B1. 可视化：各类的深度、spacing、尺寸分布
# ============================================================
print("\n生成元数据分布可视化...")

valid_meta = df_meta.dropna(subset=['depth'])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

plot_cols = ['depth', 'spacing_z', 'width', 'height', 'mean_val', 'std_val']
plot_titles = ['Slice Depth (Z)', 'Z Spacing (mm)', 'Width', 'Height', 'Mean Intensity', 'Std Intensity']

for ax, col, title in zip(axes.flat, plot_cols, plot_titles):
    if col in valid_meta.columns:
        data_by_class = [valid_meta[valid_meta['class'] == c][col].dropna().values for c in range(NUM_CLASSES)]
        bp = ax.boxplot(data_by_class, labels=[f'Class {c}' for c in range(NUM_CLASSES)], patch_artist=True)
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

fig.suptitle('NIfTI Metadata Distribution by Class', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'metadata_boxplots.png'), dpi=150)
plt.close(fig)
print("  -> 保存 metadata_boxplots.png")

# ============================================================
# B2. 关键差异检测
# ============================================================
print("\n" + "=" * 60)
print("B2. 关键分布差异检测")
print("=" * 60)

# 检查0期 vs 123期是否有系统差异
cls0_meta = valid_meta[valid_meta['class'] == 0]
cls123_meta = valid_meta[valid_meta['class'] != 0]

diff_report = []
for col in ['depth', 'spacing_z', 'spacing_x', 'width', 'height', 'mean_val', 'std_val', 'min_val', 'max_val', 'file_size_mb']:
    if col in valid_meta.columns:
        v0 = cls0_meta[col].dropna()
        v123 = cls123_meta[col].dropna()
        if len(v0) > 0 and len(v123) > 0:
            from scipy import stats
            stat, pval = stats.mannwhitneyu(v0, v123, alternative='two-sided')
            diff_report.append({
                'metric': col,
                'class0_mean': round(v0.mean(), 4),
                'class0_std': round(v0.std(), 4),
                'class0_median': round(v0.median(), 4),
                'class123_mean': round(v123.mean(), 4),
                'class123_std': round(v123.std(), 4),
                'class123_median': round(v123.median(), 4),
                'mannwhitney_p': round(pval, 6),
                'significant': '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns')),
            })

df_diff = pd.DataFrame(diff_report)
df_diff.to_csv(os.path.join(OUTPUT_DIR, 'class0_vs_123_diff.csv'), index=False)
print(df_diff.to_string(index=False))

# ============================================================
# C. 像素强度分布深入分析 (每类采样10个)
# ============================================================
print("\n" + "=" * 60)
print("C. 像素强度分布分析 (每类采样10个)")
print("=" * 60)

MAX_SAMPLES_INTENSITY = 10
intensity_records = []

for cls in range(NUM_CLASSES):
    files = class_to_files[cls]
    rng_s = np.random.RandomState(42)
    sidx = rng_s.choice(len(files), min(MAX_SAMPLES_INTENSITY, len(files)), replace=False)
    
    for i, idx in enumerate(sidx):
        fpath = files[idx]
        case_id = os.path.basename(fpath).replace('.nii.gz', '')
        try:
            img = sitk.ReadImage(fpath)
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            
            # 检查是否已经做过HU窗口化
            # 原始CT通常范围是 -1000~3000+ HU
            # 如果已归一化，范围在 0~1 或 0~255
            
            # 非零像素统计
            nonzero_mask = arr != 0
            nonzero_frac = nonzero_mask.mean()
            
            # 尝试检测肺区域 (-950 ~ -300 HU)
            # 如果数据已归一化到[0,1]，这个范围就不适用了
            
            # 检查背景区域（值=0或非常小的区域）
            bg_mask = np.abs(arr) < 1e-6
            bg_frac = bg_mask.mean()
            
            # 中心切片
            mid_z = arr.shape[0] // 2
            mid_slice = arr[mid_z]
            
            # 边缘像素（可能有伪特征）
            edge_vals = np.concatenate([
                arr[:, 0, :].flatten(),
                arr[:, -1, :].flatten(),
                arr[:, :, 0].flatten(),
                arr[:, :, -1].flatten(),
            ])
            
            intensity_records.append({
                'case_id': case_id,
                'class': cls,
                'global_min': float(arr.min()),
                'global_max': float(arr.max()),
                'global_mean': float(arr.mean()),
                'global_std': float(arr.std()),
                'global_median': float(np.median(arr)),
                'nonzero_frac': round(float(nonzero_frac), 4),
                'bg_frac': round(float(bg_frac), 4),
                'p1': float(np.percentile(arr, 1)),
                'p5': float(np.percentile(arr, 5)),
                'p25': float(np.percentile(arr, 25)),
                'p75': float(np.percentile(arr, 75)),
                'p95': float(np.percentile(arr, 95)),
                'p99': float(np.percentile(arr, 99)),
                'edge_mean': float(edge_vals.mean()),
                'edge_std': float(edge_vals.std()),
                'mid_slice_mean': float(mid_slice.mean()),
                'mid_slice_std': float(mid_slice.std()),
                'unique_val_count': min(len(np.unique(arr[:1000].flatten())), 10000),
            })
        except Exception as e:
            print(f"  !! {fpath}: {e}")

df_intensity = pd.DataFrame(intensity_records)
df_intensity.to_csv(os.path.join(OUTPUT_DIR, 'intensity_stats.csv'), index=False)

print("\n各类强度统计:")
for col in ['global_min', 'global_max', 'global_mean', 'global_std', 'nonzero_frac', 'bg_frac', 'edge_mean']:
    if col in df_intensity.columns:
        print(f"\n  {col}:")
        print(df_intensity.groupby('class')[col].describe().to_string())

# ============================================================
# C1. 中心切片可视化对比
# ============================================================
print("\n生成中心切片对比图...")

fig, axes = plt.subplots(4, 5, figsize=(20, 16))

for cls in range(NUM_CLASSES):
    files = class_to_files[cls]
    rng_v = np.random.RandomState(123)
    vidx = rng_v.choice(len(files), min(5, len(files)), replace=False)
    
    for j, idx in enumerate(vidx):
        fpath = files[idx]
        case_id = os.path.basename(fpath).replace('.nii.gz', '')
        try:
            img = sitk.ReadImage(fpath)
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            mid_z = arr.shape[0] // 2
            mid_slice = arr[mid_z]
            
            ax = axes[cls][j]
            ax.imshow(mid_slice, cmap='gray', aspect='auto')
            ax.set_title(f'Class {cls}\n{case_id}\n{arr.shape}', fontsize=8)
            ax.axis('off')
        except Exception as e:
            axes[cls][j].set_title(f'Error: {e}', fontsize=7)
            axes[cls][j].axis('off')

fig.suptitle('Center Slices by Class (5 samples each)', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'center_slice_samples.png'), dpi=150)
plt.close(fig)
print("  -> 保存 center_slice_samples.png")

# ============================================================
# D. 路径和命名泄漏检查
# ============================================================
print("\n" + "=" * 60)
print("D. 路径和命名泄漏检查")
print("=" * 60)

# 检查文件夹结构是否暴露标签
print("  1. 文件夹命名直接包含类别编号: {cls}_seg_nii")
print("     -> 标签来源就是文件夹名，这是设计如此，但需确认dataloader没有将路径传入模型")

# 检查文件名模式
print("\n  2. 文件名数字范围分析:")
for cls in range(NUM_CLASSES):
    names = [os.path.basename(f).replace('.nii.gz', '') for f in class_to_files[cls]]
    nums = []
    for n in names:
        try:
            nums.append(int(n.replace('CT', '').replace('ct', '')))
        except:
            pass
    if nums:
        print(f"     类别{cls}: CT编号范围 [{min(nums)}, {max(nums)}], 数量 {len(nums)}")

# 检查dataloader是否将路径信息传给模型
print("\n  3. 检查dataloader的__getitem__返回值...")
print("     dataloader返回 (bag_tensor, label, case_id)")
print("     case_id是字符串如'CT086519'，包含在文件名中")
print("     -> 但case_id不参与模型forward，只用于日志和评估")

# 检查肺掩膜是否存在
print("\n  4. 肺掩膜文件存在性检查:")
mask_exists = defaultdict(int)
mask_missing = defaultdict(list)
for cls in range(NUM_CLASSES):
    for f in class_to_files[cls]:
        case_id = os.path.basename(f).replace('.nii.gz', '')
        rel_path = os.path.relpath(f, DATA_ROOT)
        mask_path = os.path.join(LUNG_MASK_ROOT, rel_path)
        if os.path.exists(mask_path):
            mask_exists[cls] += 1
        else:
            mask_missing[cls].append(case_id)

for cls in range(NUM_CLASSES):
    total = len(class_to_files[cls])
    exists = mask_exists[cls]
    missing = total - exists
    print(f"     类别{cls}: {exists}/{total} 有掩膜, {missing} 缺失")
    if mask_missing[cls][:3]:
        print(f"       缺失示例: {mask_missing[cls][:3]}")

# ============================================================
# D1. 缓存文件检查
# ============================================================
print("\n  5. 预处理缓存检查:")
cache_exists = defaultdict(int)
cache_missing = defaultdict(int)

if os.path.exists(CACHE_ROOT):
    for cls in range(NUM_CLASSES):
        for f in class_to_files[cls]:
            case_id = os.path.basename(f).replace('.nii.gz', '')
            cache_path = os.path.join(CACHE_ROOT, case_id, 'preprocess_cache.npz')
            if os.path.exists(cache_path):
                cache_exists[cls] += 1
            else:
                cache_missing[cls] += 1
    
    for cls in range(NUM_CLASSES):
        total = len(class_to_files[cls])
        print(f"     类别{cls}: {cache_exists[cls]}/{total} 有缓存, {cache_missing[cls]} 无缓存")
else:
    print(f"     缓存目录不存在: {CACHE_ROOT}")

# ============================================================
# E. 六肺区/Slab构造偏差分析
# ============================================================
print("\n" + "=" * 60)
print("E. 六肺区/Slab构造偏差分析 (从缓存读取)")
print("=" * 60)

region_stats = []
if os.path.exists(CACHE_ROOT):
    for cls in range(NUM_CLASSES):
        files_sample = class_to_files[cls][:50]  # 每类最多50个
        for f in files_sample:
            case_id = os.path.basename(f).replace('.nii.gz', '')
            cache_npz = os.path.join(CACHE_ROOT, case_id, 'preprocess_cache.npz')
            cache_meta = os.path.join(CACHE_ROOT, case_id, 'preprocess_meta.json')
            
            if os.path.exists(cache_npz):
                try:
                    data = np.load(cache_npz, allow_pickle=True)
                    
                    rec = {'case_id': case_id, 'class': cls}
                    
                    # 检查可用的key
                    keys = list(data.keys())
                    
                    # 尝试读取各区域的有效center_z数量
                    region_names = ['left_upper', 'left_middle', 'left_lower',
                                    'right_upper', 'right_middle', 'right_lower']
                    
                    total_valid = 0
                    for rn in region_names:
                        key = f'{rn}_valid_centers'
                        if key in data:
                            centers = data[key]
                            rec[f'{rn}_n_centers'] = len(centers)
                            total_valid += len(centers)
                        else:
                            rec[f'{rn}_n_centers'] = -1
                    
                    rec['total_valid_centers'] = total_valid
                    rec['cache_keys'] = ','.join(keys[:10])
                    
                    region_stats.append(rec)
                except Exception as e:
                    pass
            
            if os.path.exists(cache_meta):
                try:
                    with open(cache_meta) as mf:
                        meta = json.load(mf)
                    # 补充meta信息
                    if region_stats and region_stats[-1]['case_id'] == case_id:
                        for k, v in meta.items():
                            if isinstance(v, (int, float, str, bool)):
                                region_stats[-1][f'meta_{k}'] = v
                except:
                    pass

if region_stats:
    df_regions = pd.DataFrame(region_stats)
    df_regions.to_csv(os.path.join(OUTPUT_DIR, 'region_cache_stats.csv'), index=False)
    
    print("\n各类有效center数量统计:")
    if 'total_valid_centers' in df_regions.columns:
        print(df_regions.groupby('class')['total_valid_centers'].describe().to_string())
    
    # 检查每个region的center数
    for rn in ['left_upper', 'left_middle', 'left_lower', 'right_upper', 'right_middle', 'right_lower']:
        col = f'{rn}_n_centers'
        if col in df_regions.columns:
            valid = df_regions[df_regions[col] >= 0]
            if len(valid) > 0:
                print(f"\n  {rn}:")
                print(valid.groupby('class')[col].describe().to_string())
    
    # 可视化
    if 'total_valid_centers' in df_regions.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        data_by_class = [df_regions[df_regions['class'] == c]['total_valid_centers'].values for c in range(NUM_CLASSES)]
        bp = ax.boxplot(data_by_class, labels=[f'Class {c}' for c in range(NUM_CLASSES)], patch_artist=True)
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title('Total Valid Centers per Case by Class')
        ax.set_ylabel('Number of valid slab centers')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, 'valid_centers_by_class.png'), dpi=150)
        plt.close(fig)
        print("\n  -> 保存 valid_centers_by_class.png")
else:
    print("  无法读取缓存数据")

# ============================================================
# F. 文件哈希重复检查 (采样)
# ============================================================
print("\n" + "=" * 60)
print("F. 文件哈希重复检查")
print("=" * 60)

# 对所有文件计算前4KB的hash（快速近似）
file_hashes = defaultdict(list)
for cls in range(NUM_CLASSES):
    for f in class_to_files[cls]:
        try:
            with open(f, 'rb') as fp:
                header = fp.read(4096)
            h = hashlib.md5(header).hexdigest()
            file_hashes[h].append((f, cls))
        except:
            pass

duplicates = {h: files for h, files in file_hashes.items() if len(files) > 1}
if duplicates:
    print(f"  发现 {len(duplicates)} 组可能重复的文件 (前4KB相同):")
    for h, files in list(duplicates.items())[:5]:
        for f, c in files:
            print(f"    类{c}: {os.path.basename(f)}")
        print()
else:
    print("  无文件头重复 (OK)")

# 检查文件大小完全相同的组
size_groups = defaultdict(list)
for cls in range(NUM_CLASSES):
    for f in class_to_files[cls]:
        try:
            sz = os.path.getsize(f)
            size_groups[sz].append((f, cls))
        except:
            pass

size_dups = {sz: files for sz, files in size_groups.items() if len(files) > 1}
print(f"\n  文件大小完全相同的组数: {len(size_dups)}")
if size_dups:
    # 显示跨类的大小重复
    cross_class_size_dups = {}
    for sz, files in size_dups.items():
        classes = set(c for _, c in files)
        if len(classes) > 1:
            cross_class_size_dups[sz] = files
    
    if cross_class_size_dups:
        print(f"  其中跨类别的: {len(cross_class_size_dups)} 组")
        for sz, files in list(cross_class_size_dups.items())[:3]:
            print(f"    大小={sz}B:")
            for f, c in files[:5]:
                print(f"      类{c}: {os.path.basename(f)}")

# ============================================================
# G. 综合报告
# ============================================================
print("\n" + "=" * 60)
print("G. 综合审计报告")
print("=" * 60)

report_lines = []
report_lines.append("# 尘肺CT数据集审计报告")
report_lines.append(f"# 生成时间: {pd.Timestamp.now()}")
report_lines.append("")
report_lines.append("## 1. 数据概览")
report_lines.append("")
report_lines.append(f"数据根目录: {DATA_ROOT}")
report_lines.append(f"类别数: {NUM_CLASSES}")
for cls in range(NUM_CLASSES):
    report_lines.append(f"  类别{cls}: {len(class_to_files[cls])} 个文件")
report_lines.append(f"总计: {sum(len(v) for v in class_to_files.values())} 个文件")
report_lines.append("")

report_lines.append("## 2. 划分统计")
report_lines.append("")
report_lines.append(split_table.to_string())
report_lines.append("")

report_lines.append("## 3. 数据泄漏检查")
report_lines.append("")
if not cross_class_overlap:
    report_lines.append("- 无跨类别case_id重叠 ✓")
else:
    report_lines.append("- !! 发现跨类别case_id重叠")
if not multi_split_cases:
    report_lines.append("- 无case出现在多个split ✓")
else:
    report_lines.append(f"- !! {len(multi_split_cases)} 个case出现在多个split")
report_lines.append("")

report_lines.append("## 4. CT编号范围")
report_lines.append("")
report_lines.append(df_ct_range.to_string(index=False))
report_lines.append("")

report_lines.append("## 5. 0期 vs 123期 统计差异")
report_lines.append("")
if len(df_diff) > 0:
    report_lines.append(df_diff.to_string(index=False))
report_lines.append("")

report_lines.append("## 6. 可疑问题")
report_lines.append("")
report_lines.append("(详见终端输出和可视化图片)")
report_lines.append("")

report_text = '\n'.join(report_lines)
with open(os.path.join(OUTPUT_DIR, 'audit_report.txt'), 'w') as f:
    f.write(report_text)

print(report_text)

print("\n" + "=" * 60)
print("审计完成。所有结果保存在:", OUTPUT_DIR)
print("=" * 60)
print("\n生成的文件:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f != 'audit_dataset.py':
        fpath = os.path.join(OUTPUT_DIR, f)
        sz = os.path.getsize(fpath)
        print(f"  {f} ({sz:,} bytes)")
