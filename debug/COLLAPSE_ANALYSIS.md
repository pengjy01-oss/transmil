# 训练塌缩排查报告

## 一、训练流程和关键代码位置

| 模块 | 文件 | 关键行 |
|------|------|--------|
| 数据集 / DataLoader | `dataloader.py` L793-1400 | CTPneNiiBags, 六肺区 lung_region_thin_slab |
| 标签读取 | `dataloader.py` L915-982 | `self.samples = [(nii_path, cls)]`, cls ∈ {0,1,2,3} |
| 模型 forward | `model.py` L266-400 | ResNet18→位置编码→TransMIL→burden/coverage→CORN分类 |
| CORN loss | `model.py` L140-165 | `_corn_loss()` 3个二分类子任务 |
| CORN decode | `model.py` L168-172 | `_corn_label_from_logits()` cumprod(sigmoid) > 0.5 |
| 训练循环 | `main.py` L725-790 | `train()` 函数 |
| 验证循环 | `main.py` L793-870 | `evaluate()` 函数 |
| optimizer | `main.py` L573-595 | Adam, 双参数组 (backbone lr=2e-5, head lr=2e-4) |
| scheduler | `main.py` L598-616 | ReduceLROnPlateau(patience=3, factor=0.5) |
| instance score head | `model.py` L8-18 | Linear(528,128)→ReLU→Dropout→Linear(128,1) |
| burden features | `model.py` L93-130 | soft_ratio/score_mean/topk_mean/score_std |
| coverage features | `model.py` L55-90 | z_center/z_spread/active_bins_soft |
| 最终分类器 | `model.py` L334 | Linear(535, 3) → 3个CORN logits |

## 二、检查过的模块

- [x] train/eval 模式切换（正确）
- [x] CORN loss 实现（正确）
- [x] CORN decode 实现（正确）
- [x] scheduler 逻辑（patience=3, epoch 3 时尚未触发衰减）
- [x] warmup（warmup_epochs=0, 无影响）
- [x] 标签读取/映射（正确，0-3 整数标签）
- [x] sampler（无特殊 sampler，普通 shuffle）
- [x] EMA / AMP / GradScaler（均未使用）
- [x] gradient clipping（未使用 ← 问题之一）
- [x] checkpoint 保存/恢复（正常逻辑）
- [x] backbone BN 冻结（正确，train 时 freeze BN）

## 三、第 1/2/3/4 轮关键统计对比

### training_metrics.csv 原始数据

| Epoch | train_loss | train_acc | val_loss | val_acc |
|-------|-----------|-----------|----------|---------|
| 1     | 0.3737    | 0.4860    | 0.3231   | 0.5105  |
| 2     | 0.3562    | 0.5080    | 0.2678   | **0.6224** |
| 3     | **0.5383** | 0.4279   | **0.7097** | **0.2238** |
| 4     | 0.6684    | 0.2786    | 0.6481   | 0.2238  |

### Instance Score 统计（关键证据）

| Epoch | train_soft_ratio | train_score_mean | val_soft_ratio | val_score_mean | val_score_std |
|-------|-----------------|------------------|----------------|----------------|---------------|
| 1     | 0.242           | 0.334            | 0.220          | 0.306          | 0.062         |
| 2     | 0.484           | 0.524            | 0.352          | 0.368          | 0.125         |
| 3     | 0.668           | 0.634            | **0.970**      | **0.860**      | **0.023**     |
| 4     | 0.228           | 0.245            | 0.237          | 0.233          | 0.038         |

## 四、问题定位：Instance Score Saturation → Burden/Coverage 特征坍缩 → CORN 全 0 预测

### 完整的塌缩链条

```
Epoch 1-2: instance scores 正常增长 (score_mean: 0.31→0.37)
     ↓
Epoch 2→3: score 突然飙升 (0.37 → 0.86 on val)
     ↓
burden_temperature=0.1 → sigmoid((0.86-0.5)/0.1) = sigmoid(3.6) = 0.974
     ↓
所有 instance 的 soft_mask ≈ 1.0 (val_soft_ratio=0.970)
     ↓
所有患者的 burden/coverage 特征变成几乎相同的常数:
  soft_ratio ≈ 0.97, score_mean ≈ 0.86, topk ≈ 0.91, std ≈ 0.02
  z_center ≈ 0.5, z_spread ≈ ?, active_bins ≈ 6.0
     ↓
分类器 535 维输入中, 7 维 burden/coverage 失去区分能力
     ↓
TransMIL CLS token 也趋于一致 (所有样本输入特征相似)
     ↓
分类器退化为近似常数输出
     ↓
CORN task0 = P(y>0): 如果 class 0 占多数, BCE 梯度推 logit[0] → 负
     ↓
sigmoid(negative) < 0.5 → cumprod 全部 < 0.5
     ↓
所有样本预测为 0 期 ✗
```

## 五、最可疑的前 5 个原因（按优先级排序）

### 1. 🔴 `burden_temperature=0.1` 过小 — 直接根因

**证据**：
- `soft_mask = sigmoid((score - tau) / temperature)`
- 当 temperature=0.1 时，score 只要偏离 tau=0.5 超过 0.2，sigmoid 就接近 0 或 1
- CSV 显示：epoch 3 val_soft_ratio=0.970，val_score_std=0.023（所有 instance score 都高且一致）
- 这直接导致 burden/coverage features 成为常数

**同理**，`coverage_temperature=0.1` 存在完全相同的问题。

### 2. 🟠 Instance score head 无直接监督

**证据**：
- instance_score_head 只通过 CORN loss → classifier → burden_features 反向传播获得梯度
- 无任何 instance-level GT 标签作为直接监督
- 这使得 score head 的优化方向完全取决于间接梯度信号
- 当 temperature=0.1 时，sigmoid 梯度在饱和区消失（sigmoid'(3.6)=0.029），形成正反馈死锁

### 3. 🟡 无梯度裁剪

**证据**：
- 当前训练完全没有 gradient clipping
- Instance score head 的梯度可以在某一步突然变大，将所有 score 推过 tau=0.5
- 一旦越过，temperature=0.1 的 sigmoid 门就锁死

### 4. 🟡 active_bins_soft 特征尺度问题

**证据**：
- active_bins_soft ∈ [0, 6]，而其他 burden 特征 ∈ [0, 1]
- 分类器 `Linear(535, 3)` 初始化时各维度权重量级相同
- 这使得 active_bins_soft 对 logits 的贡献 ~6x 大于其他特征
- 当其饱和到 ~6.0 时，对分类器输出产生不成比例的 bias 偏移

### 5. 🟢 CORN task0 的隐含类别不平衡

**证据**：
- CORN task0 用全部样本训练，target = (y > 0)
- 如果 class 0 占 40%+，target=0 的样本多于 target=1
- 当分类器退化为常数输出后，BCE 梯度自然推 logit[0] 为负
- 这是"全预测 0"的最终执行机制，但不是触发原因

## 六、为什么前两轮正常，第 3 轮才崩？

这不是一个突发 bug，而是一个 **正反馈放大过程**：

1. **Epoch 1**：score 均值 ~0.33，远低于 tau=0.5。soft_mask 接近 0，burden 特征尚小但有梯度。
2. **Epoch 2**：score 均值 ~0.52（train），刚越过 tau。soft_mask 开始非线性放大。burden 特征开始携带信息，分类准确率提升到 62%（val）。
3. **Epoch 2→3 过渡**：score head 继续被梯度推高。一旦 score_mean 超过 ~0.6，temperature=0.1 的 sigmoid 快速饱和。**这个过渡发生在 epoch 2 的后半段到 epoch 3 的前半段**。
4. **Epoch 3**：score 全部 >0.7 → soft_ratio ≈ 1.0 → burden 特征成常数 → 分类器崩溃 → 全预测 0。

关键拐点：**score 均值越过 ~0.6 的瞬间**，temperature=0.1 将其从"有信息"放大为"全饱和"。

## 七、最小修复方案

### 修复 A（最优先，必做）：提高 temperature

```yaml
# configs/ct25d_transmil.yaml
burden_temperature: 1.0    # 从 0.1 → 1.0
coverage_temperature: 1.0  # 从 0.1 → 1.0
```

**理由**：temperature=1.0 时，sigmoid((0.86-0.5)/1.0) = sigmoid(0.36) = 0.589，仍有梯度流动，不会饱和。

### 修复 B（推荐，强烈建议）：添加梯度裁剪

在 `main.py` 的 optimizer.step() 之前加入：
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 修复 C（推荐）：归一化 active_bins_soft

在 `_compute_coverage_features` 中，将 `active_bins_soft` 除以 `num_bins`：
```python
active_bins_soft = torch.stack(bin_means, dim=0).sum() / float(bins)  # 归一化到 [0,1]
```

### 修复 D（可选）：降低 instance score head 学习率

通过将 score head 放入单独的 param_group，使用更低的 lr（如 lr * 0.1）。
