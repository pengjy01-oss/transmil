#!/usr/bin/env python
"""
训练塌缩诊断脚本 —— 定位"第3轮全预测0期"的根因。

用法:
    conda run -n lung_new python debug/diagnose_collapse.py --config configs/ct25d_transmil.yaml --epochs 5

会输出到 debug/ 目录:
    - epoch_stats.csv          每轮汇总统计
    - per_sample_epoch{N}.csv  每个样本的 raw logits / 预测 / 真实标签
    - grad_norm_epoch{N}.csv   每 step 各参数组梯度范数
    - corn_task_losses.csv     CORN 3 个子任务分别的 loss
"""
from __future__ import print_function
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import csv, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable

try:
    import yaml
except ImportError:
    yaml = None

from dataloader import CTPneNiiBags
from model import Attention, _corn_loss, _corn_label_from_logits, _prepare_targets

DEBUG_DIR = os.path.join(os.path.dirname(__file__))
os.makedirs(DEBUG_DIR, exist_ok=True)

# ─── 解析配置 ──────────────────────────────────────────
DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'ct25d_transmil.yaml')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=DEFAULT_CONFIG)
parser.add_argument('--epochs', type=int, default=5)
# 保持与 main.py 相同的默认值; config 文件会覆盖
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--backbone_lr_ratio', type=float, default=0.1)
parser.add_argument('--reg', type=float, default=0.0001)
parser.add_argument('--grad_accum_steps', type=int, default=2)
parser.add_argument('--warmup_epochs', type=int, default=0)
parser.add_argument('--scheduler_factor', type=float, default=0.5)
parser.add_argument('--scheduler_patience', type=int, default=3)
parser.add_argument('--scheduler_min_lr', type=float, default=1e-6)
parser.add_argument('--lr_scheduler', type=str, default='plateau')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--no_cuda', action='store_true', default=False)

# 用 yaml 覆盖默认值
pre_args, _ = parser.parse_known_args()
if pre_args.config and yaml:
    with open(os.path.expanduser(pre_args.config)) as f:
        cfg = yaml.safe_load(f) or {}
    parser.set_defaults(**cfg)
args = parser.parse_args()
args.cuda = (not args.no_cuda) and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# ─── 构建数据集 ──────────────────────────────────────────
half_depth = args.slab_depth // 2
channel_offsets = tuple(range(-half_depth, half_depth + 1))

common_kw = dict(
    root_dir=args.data_root, num_classes=args.num_classes,
    middle_ratio=args.middle_ratio, fixed_num_slices=args.fixed_num_slices,
    channel_offsets=channel_offsets, slab_stride=args.slab_stride,
    num_slabs=args.num_slabs, center_sampling_mode=args.center_sampling_mode,
    test_ratio=args.test_ratio, val_ratio=args.val_ratio, seed=args.seed,
    scale_to_unit=args.scale_to_unit, use_zscore=args.use_zscore,
    lung_hu_low=args.lung_hu_low, lung_hu_high=args.lung_hu_high,
    min_lung_area_ratio=args.min_lung_area_ratio,
    max_instances=getattr(args, 'max_instances', 0),
    instance_definition=args.instance_definition,
    lung_mask_root=args.lung_mask_root, lung_mask_suffix=args.lung_mask_suffix,
    lung_mask_require=getattr(args, 'lung_mask_require', False),
    cache_root=args.cache_root,
    pseudo_mask_value_threshold=getattr(args, 'pseudo_mask_value_threshold', 1e-6),
    pseudo_mask_min_component_voxels=getattr(args, 'pseudo_mask_min_component_voxels', 512),
    region_num_instances=args.region_num_instances,
    region_out_size=(args.region_out_h, args.region_out_w),
    region_bbox_margin=args.region_bbox_margin,
    region_bbox_min_size=args.region_bbox_min_size,
    region_abs_area_threshold=args.region_abs_area_threshold,
    region_ratio_area_threshold=args.region_ratio_area_threshold,
)

train_ds = CTPneNiiBags(split='train', **common_kw)
val_ds = CTPneNiiBags(split='val', **common_kw)

loader_kw = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = data_utils.DataLoader(train_ds, batch_size=1, shuffle=True, **loader_kw)
val_loader = data_utils.DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kw)

# ─── 打印标签分布 ──────────────────────────────────────
def label_dist(ds):
    counts = np.zeros(args.num_classes, dtype=int)
    for _, y in ds.samples:
        counts[int(y)] += 1
    return counts.tolist()

print('Train labels:', label_dist(train_ds))
print('Val   labels:', label_dist(val_ds))

# ─── 构建模型 ──────────────────────────────────────────
model = Attention(
    in_channels=args.in_channels,
    pretrained_backbone=args.pretrained_backbone,
    num_classes=args.num_classes,
    instance_batch_size=getattr(args, 'instance_batch_size', 2),
    freeze_backbone=args.freeze_backbone,
    use_burden_features=args.use_burden_features,
    use_position_embedding=getattr(args, 'use_position_embedding', True),
    position_embed_dim=getattr(args, 'position_embed_dim', 16),
    use_coverage_features=getattr(args, 'use_coverage_features', True),
    coverage_num_bins=getattr(args, 'coverage_num_bins', 6),
    coverage_tau=getattr(args, 'coverage_tau', 0.5),
    coverage_temperature=getattr(args, 'coverage_temperature', 0.1),
    coverage_eps=getattr(args, 'coverage_eps', 1e-6),
    burden_score_hidden_dim=getattr(args, 'burden_score_hidden_dim', 128),
    burden_score_dropout=getattr(args, 'burden_score_dropout', 0.1),
    burden_tau=args.burden_tau,
    burden_temperature=args.burden_temperature,
    burden_topk_ratio=args.burden_topk_ratio,
    aggregator=args.aggregator,
    transmil_num_heads=args.transmil_num_heads,
    transmil_num_layers=args.transmil_num_layers,
    transmil_dropout=args.transmil_dropout,
)
if args.cuda:
    model.cuda()

# 冻结 BN
if (not args.freeze_backbone) and getattr(args, 'freeze_backbone_bn', True):
    from main import _freeze_backbone_batchnorm
    _freeze_backbone_batchnorm(model)

# ─── 构建 optimizer / scheduler ──────────────────────
backbone_params, head_params = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if name.startswith('feature_extractor.'):
        backbone_params.append(p)
    else:
        head_params.append(p)

backbone_lr = args.lr * args.backbone_lr_ratio
param_groups = []
if backbone_params:
    param_groups.append({'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'})
if head_params:
    param_groups.append({'params': head_params, 'lr': args.lr, 'name': 'head'})
optimizer = optim.Adam(param_groups, betas=(0.9, 0.999), weight_decay=args.reg)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    factor=args.scheduler_factor, patience=args.scheduler_patience,
    min_lr=args.scheduler_min_lr)

print('Head lr:', args.lr, '  Backbone lr:', backbone_lr)

# ─── CORN 子任务分解 ──────────────────────────────────
def corn_task_losses(logits, y, num_classes):
    """返回每个子任务的 loss 值 (list of float)."""
    y = _prepare_targets(y)
    results = []
    for k in range(num_classes - 1):
        if k == 0:
            mask = torch.ones_like(y, dtype=torch.bool)
        else:
            mask = y > (k - 1)
        if mask.any():
            targets = (y[mask] > k).float()
            task_logits = logits[mask, k]
            loss_k = F.binary_cross_entropy_with_logits(task_logits, targets, reduction='mean')
            results.append(float(loss_k.item()))
        else:
            results.append(None)
    return results

# ─── 梯度范数工具 ──────────────────────────────────────
def grad_norms_by_group(model):
    groups = {
        'backbone': [],
        'transmil': [],
        'score_head': [],
        'classifier': [],
        'position': [],
        'other': [],
    }
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        gn = p.grad.data.norm(2).item()
        if name.startswith('feature_extractor'):
            groups['backbone'].append(gn)
        elif 'transmil' in name:
            groups['transmil'].append(gn)
        elif 'instance_score_head' in name:
            groups['score_head'].append(gn)
        elif 'classifier' in name:
            groups['classifier'].append(gn)
        elif 'position' in name:
            groups['position'].append(gn)
        else:
            groups['other'].append(gn)
    out = {}
    for k, v in groups.items():
        if v:
            out[k] = float(np.sqrt(sum(x**2 for x in v)))
        else:
            out[k] = 0.0
    return out

# ─── CSV writers ──────────────────────────────────────
epoch_csv = os.path.join(DEBUG_DIR, 'epoch_stats.csv')
corn_csv = os.path.join(DEBUG_DIR, 'corn_task_losses.csv')

with open(epoch_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow([
        'epoch', 'split', 'loss', 'acc',
        'pred_0', 'pred_1', 'pred_2', 'pred_3',
        'true_0', 'true_1', 'true_2', 'true_3',
        'logit0_mean', 'logit1_mean', 'logit2_mean',
        'logit0_std', 'logit1_std', 'logit2_std',
        'score_mean', 'score_std', 'soft_ratio',
        'lr_backbone', 'lr_head',
        'grad_backbone', 'grad_transmil', 'grad_score_head', 'grad_classifier',
        'corn_task0', 'corn_task1', 'corn_task2',
        'cls_weight_norm', 'cls_bias',
    ])


def write_epoch_row(epoch, split, loss, acc, pred_counts, true_counts,
                    logit_means, logit_stds, score_mean, score_std, soft_ratio,
                    lr_bb, lr_hd, gnorms, corn_tasks, cls_w_norm, cls_bias):
    with open(epoch_csv, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            epoch, split, f'{loss:.6f}', f'{acc:.4f}',
            *[int(x) for x in pred_counts],
            *[int(x) for x in true_counts],
            *[f'{x:.4f}' for x in logit_means],
            *[f'{x:.4f}' for x in logit_stds],
            f'{score_mean:.4f}', f'{score_std:.4f}', f'{soft_ratio:.4f}',
            f'{lr_bb:.2e}', f'{lr_hd:.2e}',
            *[f'{gnorms.get(k, 0):.4f}' for k in ('backbone', 'transmil', 'score_head', 'classifier')],
            *[(f'{x:.6f}' if x is not None else '') for x in corn_tasks],
            f'{cls_w_norm:.4f}', cls_bias,
        ])


def write_per_sample(epoch, split, rows):
    path = os.path.join(DEBUG_DIR, f'per_sample_epoch{epoch}_{split}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'true', 'pred', 'logit0', 'logit1', 'logit2',
                     'prob0', 'prob1', 'prob2', 'cumprod0', 'cumprod1', 'cumprod2',
                     'score_mean', 'score_std'])
        for r in rows:
            w.writerow(r)


# ─── 训练/评估循环 ──────────────────────────────────────
def run_epoch(epoch, loader, split, training=True):
    if training:
        model.train()
        if (not args.freeze_backbone) and getattr(args, 'freeze_backbone_bn', True):
            from main import _freeze_backbone_batchnorm
            _freeze_backbone_batchnorm(model)
        optimizer.zero_grad(set_to_none=True)

        if hasattr(loader.dataset, 'set_epoch'):
            loader.dataset.set_epoch(epoch)
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    pred_counts = np.zeros(args.num_classes, dtype=int)
    true_counts = np.zeros(args.num_classes, dtype=int)
    all_logits = []
    all_scores_mean = []
    all_scores_std = []
    all_soft_ratio = []
    corn_task_sums = [0.0, 0.0, 0.0]
    corn_task_counts = [0, 0, 0]
    sample_rows = []
    grad_norms_accum = {}

    ctx = torch.no_grad() if not training else torch.enable_grad()
    with ctx:
        for batch_idx, batch_data in enumerate(loader):
            data = batch_data[0]
            label = batch_data[1]
            pos_z = batch_data[2] if len(batch_data) > 2 else None
            bag_label = label[0] if isinstance(label, (list, tuple)) else label
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
                if pos_z is not None:
                    pos_z = pos_z.cuda()

            logits, Y_hat, A = model.forward(data, pos_z=pos_z)
            loss = _corn_loss(logits, bag_label, args.num_classes)

            if training:
                (loss / float(args.grad_accum_steps)).backward()
                if ((batch_idx + 1) % args.grad_accum_steps == 0) or ((batch_idx + 1) == len(loader)):
                    gn = grad_norms_by_group(model)
                    for k, v in gn.items():
                        grad_norms_accum[k] = grad_norms_accum.get(k, [])
                        grad_norms_accum[k].append(v)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            # 统计
            total_loss += loss.item()
            y_true = int(bag_label.view(-1).item())
            y_pred = int(Y_hat.view(-1).item())
            correct += int(y_true == y_pred)
            pred_counts[min(y_pred, args.num_classes - 1)] += 1
            true_counts[min(y_true, args.num_classes - 1)] += 1
            lg = logits.detach().cpu().numpy().ravel()
            all_logits.append(lg)

            # instance score stats
            aux = model.get_latest_aux_outputs()
            ist = aux.get('instance_scores', None)
            if ist is not None:
                ist_np = ist.detach().cpu().numpy().ravel()
                all_scores_mean.append(float(ist_np.mean()))
                all_scores_std.append(float(ist_np.std()))
                sr = aux.get('burden_stats', {}).get('soft_ratio', 0.0)
                all_soft_ratio.append(float(sr))
            else:
                all_scores_mean.append(0.0)
                all_scores_std.append(0.0)
                all_soft_ratio.append(0.0)

            # CORN 子任务 loss
            ct = corn_task_losses(logits, bag_label, args.num_classes)
            for k_i in range(3):
                if k_i < len(ct) and ct[k_i] is not None:
                    corn_task_sums[k_i] += ct[k_i]
                    corn_task_counts[k_i] += 1

            # per-sample row
            probas = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            cprods = np.cumprod(probas)
            sample_rows.append([
                batch_idx, y_true, y_pred,
                f'{lg[0]:.4f}', f'{lg[1]:.4f}', f'{lg[2]:.4f}',
                f'{probas[0]:.4f}', f'{probas[1]:.4f}', f'{probas[2]:.4f}',
                f'{cprods[0]:.4f}', f'{cprods[1]:.4f}', f'{cprods[2]:.4f}',
                f'{all_scores_mean[-1]:.4f}', f'{all_scores_std[-1]:.4f}',
            ])

            if (batch_idx + 1) % 20 == 0:
                print(f'  [{split}] {batch_idx+1}/{len(loader)} loss={total_loss/(batch_idx+1):.4f}')

    n = len(loader)
    avg_loss = total_loss / max(n, 1)
    acc = correct / max(n, 1)
    logit_arr = np.array(all_logits)
    logit_means = logit_arr.mean(axis=0).tolist()
    logit_stds = logit_arr.std(axis=0).tolist()
    score_m = float(np.mean(all_scores_mean))
    score_s = float(np.mean(all_scores_std))
    sr = float(np.mean(all_soft_ratio))

    corn_avgs = []
    for k_i in range(3):
        if corn_task_counts[k_i] > 0:
            corn_avgs.append(corn_task_sums[k_i] / corn_task_counts[k_i])
        else:
            corn_avgs.append(None)

    gn_avg = {}
    for k, v in grad_norms_accum.items():
        gn_avg[k] = float(np.mean(v)) if v else 0.0

    lrs = [g['lr'] for g in optimizer.param_groups]
    lr_bb = lrs[0] if len(lrs) > 0 else 0.0
    lr_hd = lrs[1] if len(lrs) > 1 else lrs[0]

    # classifier weight/bias
    cls_w = model.classifier.weight.detach().cpu()
    cls_b = model.classifier.bias.detach().cpu()
    cls_w_norm = float(cls_w.norm(2).item())
    cls_bias_str = ','.join([f'{b:.4f}' for b in cls_b.tolist()])

    write_epoch_row(epoch, split, avg_loss, acc, pred_counts, true_counts,
                    logit_means, logit_stds, score_m, score_s, sr,
                    lr_bb, lr_hd, gn_avg, corn_avgs, cls_w_norm, cls_bias_str)
    write_per_sample(epoch, split, sample_rows)

    print(f'Epoch {epoch} [{split}] loss={avg_loss:.4f} acc={acc:.4f} '
          f'pred={pred_counts.tolist()} true={true_counts.tolist()}')
    print(f'  logit_means={[f"{x:.3f}" for x in logit_means]} '
          f'logit_stds={[f"{x:.3f}" for x in logit_stds]}')
    print(f'  score_mean={score_m:.4f} score_std={score_s:.4f} soft_ratio={sr:.4f}')
    print(f'  corn_tasks={[f"{x:.4f}" if x else "N/A" for x in corn_avgs]}')
    print(f'  cls_bias=[{cls_bias_str}]  cls_w_norm={cls_w_norm:.4f}')
    if gn_avg:
        print(f'  grad_norms={gn_avg}')
    print(f'  lr: backbone={lr_bb:.2e} head={lr_hd:.2e}')

    return avg_loss


# ─── 主循环 ──────────────────────────────────────────
if __name__ == '__main__':
    print(f'\n{"="*60}')
    print(f'训练塌缩诊断开始, epochs={args.epochs}')
    print(f'burden_temperature={args.burden_temperature}, burden_tau={args.burden_tau}')
    print(f'coverage_temperature={getattr(args, "coverage_temperature", "N/A")}')
    print(f'{"="*60}\n')

    for epoch in range(1, args.epochs + 1):
        print(f'\n--- Epoch {epoch}/{args.epochs} ---')
        train_loss = run_epoch(epoch, train_loader, 'train', training=True)
        val_loss = run_epoch(epoch, val_loader, 'val', training=False)
        if scheduler is not None:
            scheduler.step(val_loss)

    print(f'\n诊断完成。结果保存在: {DEBUG_DIR}/')
    print('请查看:')
    print(f'  {epoch_csv}  -- 每轮汇总')
    for ep in range(1, args.epochs + 1):
        print(f'  debug/per_sample_epoch{ep}_train.csv')
        print(f'  debug/per_sample_epoch{ep}_val.csv')
