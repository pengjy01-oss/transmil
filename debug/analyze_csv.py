#!/usr/bin/env python
"""
快速离线分析 training_metrics.csv，不需要加载模型或数据。
用法: python debug/analyze_csv.py
"""
import csv, os, sys

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'training_metrics.csv')

with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("=" * 80)
print("训练塌缩诊断 — 基于 training_metrics.csv 的离线分析")
print("=" * 80)

for r in rows:
    ep = r['epoch']
    if ep == 'test':
        continue
    ep = int(ep)

    tr_loss = float(r['train_loss']) if r['train_loss'] else None
    tr_acc  = float(r['train_acc'])  if r['train_acc']  else None
    vl_loss = float(r['val_loss'])   if r['val_loss']   else None
    vl_acc  = float(r['val_acc'])    if r['val_acc']    else None

    tr_sr = float(r['train_soft_ratio'])  if r['train_soft_ratio']  else None
    tr_sm = float(r['train_score_mean'])  if r['train_score_mean']  else None
    tr_tk = float(r['train_topk_mean'])   if r['train_topk_mean']   else None
    tr_sd = float(r['train_score_std'])   if r['train_score_std']   else None

    vl_sr = float(r['val_soft_ratio'])  if r['val_soft_ratio']  else None
    vl_sm = float(r['val_score_mean'])  if r['val_score_mean']  else None
    vl_tk = float(r['val_topk_mean'])   if r['val_topk_mean']   else None
    vl_sd = float(r['val_score_std'])   if r['val_score_std']   else None

    print(f"\n--- Epoch {ep} ---")
    print(f"  Train: loss={tr_loss:.4f}  acc={tr_acc:.4f}")
    print(f"         soft_ratio={tr_sr:.4f}  score_mean={tr_sm:.4f}  topk={tr_tk:.4f}  std={tr_sd:.4f}")
    if vl_loss:
        print(f"  Val:   loss={vl_loss:.4f}  acc={vl_acc:.4f}")
        print(f"         soft_ratio={vl_sr:.4f}  score_mean={vl_sm:.4f}  topk={vl_tk:.4f}  std={vl_sd:.4f}")

print("\n" + "=" * 80)
print("关键异常标记:")
print("=" * 80)

# 检测 soft_ratio 突变
for i in range(1, len(rows)):
    if rows[i]['epoch'] == 'test':
        continue
    ep = int(rows[i]['epoch'])
    prev_ep = int(rows[i-1]['epoch']) if rows[i-1]['epoch'] != 'test' else None
    if prev_ep is None:
        continue

    vsr = float(rows[i]['val_soft_ratio']) if rows[i]['val_soft_ratio'] else None
    vsr_prev = float(rows[i-1]['val_soft_ratio']) if rows[i-1]['val_soft_ratio'] else None
    if vsr and vsr_prev and vsr > 0.9:
        print(f"  ⚠ Epoch {ep}: val_soft_ratio={vsr:.4f} (从 {vsr_prev:.4f} 飙升) — instance score 饱和!")

    va = float(rows[i]['val_acc']) if rows[i]['val_acc'] else None
    va_prev = float(rows[i-1]['val_acc']) if rows[i-1]['val_acc'] else None
    if va and va_prev and (va_prev - va) > 0.3:
        print(f"  ⚠ Epoch {ep}: val_acc={va:.4f} (从 {va_prev:.4f} 暴跌) — 模型塌缩!")

    vl = float(rows[i]['val_loss']) if rows[i]['val_loss'] else None
    vl_prev = float(rows[i-1]['val_loss']) if rows[i-1]['val_loss'] else None
    if vl and vl_prev and (vl - vl_prev) > 0.3:
        print(f"  ⚠ Epoch {ep}: val_loss={vl:.4f} (从 {vl_prev:.4f} 飙升) — loss 突变!")

print("\n" + "=" * 80)
print("根因分析:")
print("=" * 80)
print("""
1. burden_temperature=0.1 导致 sigmoid 门控极陡
   soft_mask = sigmoid((score - 0.5) / 0.1)
   当 score=0.6 时: sigmoid(1.0) = 0.73
   当 score=0.7 时: sigmoid(2.0) = 0.88
   当 score=0.8 时: sigmoid(3.0) = 0.95
   → instance score 只要超过 0.6, soft_ratio 就迅速逼近 1.0

2. Epoch 2→3: instance score 均值从 ~0.37 升到 ~0.86 (val)
   → 所有 instance 被标记为"异常", burden/coverage 特征变成常数
   → 分类器的 7 个输入维度失去区分能力

3. 分类器退化为近似 bias-only 模型:
   → CORN task0 (P(y>0)) 在类别不平衡下被推向负值
   → sigmoid(negative) < 0.5 → cumprod < 0.5 → 全部预测 0 期

4. Epoch 4 score 回落到 0.23 是过度修正的振荡现象
""")

print("修复建议: 将 burden_temperature 从 0.1 提高到 1.0 或更高")
