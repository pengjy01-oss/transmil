## CT 尘肺四分期审计报告

### 当前状态

- 已完成：train/val/test no-crop、test crop、静态 domain 分析、montage 抽样核查。

### 1. 现象是否真实存在

- 是。train/val/test 都显示 class 2 recall 明显低于 class 1 recall，且 `true2 -> pred1` 明显多于 `true1 -> pred2`。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_train_nocrop/core_metrics.csv`
  - train：accuracy 0.7174，balanced accuracy 0.7304，macro-F1 0.7345，class1 recall 0.6973，class2 recall 0.5328，`true1->pred2=65`，`true2->pred1=114`。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_valtest/core_metrics.csv`
  - val：accuracy 0.7063，balanced accuracy 0.7193，macro-F1 0.7226，class1 recall 0.6757，class2 recall 0.4872，`true1->pred2=9`，`true2->pred1=19`。
  - test：accuracy 0.6702，balanced accuracy 0.6867，macro-F1 0.6882，class1 recall 0.6622，class2 recall 0.3846，`true1->pred2=22`，`true2->pred1=41`。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_train_nocrop/confusion_matrices.csv`
  - train 真类 2 共 274 例，其中 114 例被判为 1，仅 146 例判对。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_valtest/confusion_matrices.csv`
  - test 真类 2 共 78 例，其中 41 例被判为 1，仅 30 例判对。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_train_nocrop/per_class_metrics.csv`
  - train class 1 F1 = 0.6431，class 2 F1 = 0.5299。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_valtest/per_class_metrics.csv`
  - test class 1 F1 = 0.5904，class 2 F1 = 0.4027。

### 1.1 train 结果说明什么

- 这不是“只在 val/test 才出现”的泛化问题，train 内部已经存在明显的 1/2 混淆。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_train_nocrop/false2_to_1_p_y_gt_1_distribution.csv`
  - train `false2_to_1` 的 `P(y>1)` 均值 0.1599，中位数 0.1204，114 例里 96 例低于 0.2，仅 4 例落在 `[0.45, 0.55]`。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_train_nocrop/p_y_gt_1_distribution_true1_true2.csv`
  - train true class 1 `P(y>1)` 中位数 0.1222，true class 2 中位数 0.7503，虽然总体均值有差异，但仍残留大量 class 2 被明显压到 class 1 一侧。
- 结论：1/2 边界问题在训练集内也未被模型吃干净，更像“表示与判别能力不足”，而不只是验证集阈值或测试域偏移。

### 2. 是否是类别不平衡导致

- 当前未找到支持“严重类别不平衡是主因”的证据。
- 证据：`dataset_audit_results/split_distribution.csv`
  - train：class0/1/2/3 = 220/261/274/243。
  - val：32/37/39/35。
  - test：63/74/78/70。
- class 1 与 class 2 在三份 split 中样本数都相近，无法解释为什么 class 2 recall 明显更差。

### 3. 是否只是阈值问题

- 不是单纯阈值问题。调阈值只能带来很小改动，无法根治 class 1 / class 2 分离差。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_valtest/threshold_calibration_results.csv`
  - test 默认阈值 `[0.5, 0.5, 0.5]`：class2 recall = 0.3846。
  - test 单阈值 `[0.4, 0.4, 0.4]`：class2 recall = 0.4103，macro-F1 仅从 0.6882 到 0.6898。
  - test 分阈值 `[0.05, 0.4, 0.1]`：class2 recall 反而降到 0.3333。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_valtest/class12_binary_proxy_stats.csv`
  - 只看 true class 1 vs 2，使用 `P(y>1)` 的 test AUC 只有 0.6003，说明 class 1 / class 2 在这个关键边界上的分数本身就高度重叠。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_valtest/false2_to_1_p_y_gt_1_distribution.csv`
  - test `false2_to_1` 的 `P(y>1)` 均值 0.1607，中位数 0.1197，41 例里 35 例低于 0.2，仅 1 例落在 `[0.45, 0.55]`。
- 结论：大量错分样本不是“刚好卡在阈值附近”，而是模型把它们明显推向 class 1。

### 4. 是否是 crop / 掩膜质量崩坏导致

- 未找到支持“crop 质量崩坏是主因”的证据。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_test_crop/core_metrics.csv`
  - test crop 与 no-crop 的核心指标完全一致。
- 证据：终端对比输出
  - `pred_label_default_equal True`
  - `cumprob_y_gt_0 max_abs_diff 1.19e-07`
  - `cumprob_y_gt_1 max_abs_diff 3.19e-04`
  - `cumprob_y_gt_2 max_abs_diff 4.39e-05`
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_test_crop/crop_case_group_summary.csv`
  - `false2_to_1` vs `true2_to_2` 的 `lung_pixel_ratio_mean`：0.4668 vs 0.4650。
  - `bbox_phys_h_mm_mean`：165.38 vs 164.00。
  - `bbox_phys_w_mm_mean`：131.09 vs 129.17。
  - `resize_scale_x_mean`：1.1790 vs 1.1829。
  - `mask_area_mean`：22522.30 vs 22577.02。
- 证据：`outputs/diagnostics/montage/`
  - 人工核查 `A_false2_to_1_low_py_gt_1` 与 `B_true2_to_2_high_py_gt_1` 组样本，未见明显 mask collapse、左右肺切分崩坏或 gross bbox 错位。

### 5. 是否是 metadata / domain shift 导致

- 有证据表明 domain 差异会放大误差，但目前证据不支持把它认定为唯一主因。

#### 5.1 institution 有影响

- 证据：`outputs/diagnostics/class12_dicom_text_counts.csv`
  - class 1 与 class 2 的 institution 分布不同：
    - class 1：`GZ NO.12 PEOPLE'S HOSPITAL` 76.61%，`HOSPITAL` 18.55%。
    - class 2：`GZ NO.12 PEOPLE'S HOSPITAL` 81.59%，`HOSPITAL` 12.79%。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_valtest/class12_group_metrics.csv`
  - test class 2 recall by institution：
    - `GZ NO.12 PEOPLE'S HOSPITAL` = 0.3881（26/67）
    - `GZ No.12 people's hospital` = 0.6000（3/5）
    - `HOSPITAL` = 0.1667（1/6）
  - test class 1 recall by institution：
    - `GZ NO.12 PEOPLE'S HOSPITAL` = 0.7647（39/51）
    - `GZ No.12 people's hospital` = 0.2500（1/4）
    - `HOSPITAL` = 0.4737（9/19）

#### 5.2 但 institution 不是唯一解释

- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_valtest/test_false2_to_1_vs_true2_to_2_groups.csv`
  - `false2_to_1` 中 87.80% 来自 `GZ NO.12 PEOPLE'S HOSPITAL`。
  - `true2_to_2` 中 86.67% 也来自同一 institution。
- 这说明即便限定在同一主 institution 内，class 2 仍同时存在大量判对与大量误判，institution 只能解释一部分波动。

#### 5.3 scanner / protocol / kernel / manufacturer

- 未找到证据支持这些字段是主因，因为当前文本字段几乎全缺失，无法形成有效比较。
- 证据：`outputs/diagnostics/class12_dicom_text_counts.csv`
  - class 1/2 的 `scanner_model`、`protocol_name`、`reconstruction_kernel`、`manufacturer` 全部为 `missing`（100%）。

### 6. 是否是 split_method 导致

- 有轻度相关，但证据不足以认定为主因。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_valtest/label_by_split_method_crosstab.csv`
  - 合并 val+test 后，class 2 的 `midline_fallback` 数量高于 class 1（58 vs 42），而两类支持数相近（117 vs 111）。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_valtest/class12_group_metrics.csv`
  - test class 2 recall by split_method：
    - `cc3d` = 0.4063
    - `midline_fallback` = 0.3429
    - `morph_cc3d` = 0.4545
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_valtest/test_false2_to_1_vs_true2_to_2_groups.csv`
  - `false2_to_1` 的 split_method 分布：`midline_fallback` 48.78%，`cc3d` 39.02%，`morph_cc3d` 12.20%。
  - `true2_to_2` 的 split_method 分布：`cc3d` 43.33%，`midline_fallback` 40.00%，`morph_cc3d` 16.67%。
- 结论：`midline_fallback` 对 class 2 确实略差，但错误组与正确组分布仍有大面积重叠，不能单独解释 1/2 混淆。

### 7. 是否是 spacing 导致

- 未找到支持“spacing 是主因”的强证据。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_valtest/spacing_stats_by_true_label.csv`
  - test class 1 `spacing_x` 均值约 0.6649，class 2 约 0.6717，差异较小。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_valtest/false2_to_1_vs_true2_to_2_spacing.csv`
  - `false2_to_1` vs `true2_to_2`：`spacing_x` 均值 0.6764 vs 0.6685。
- 结论：存在轻微差异，但量级不足以单独解释 41 个 `true2 -> pred1` 错误。

### 8. 特征头本身是否有足够的 class 1 / 2 判别力

- 当前证据显示附加的 burden / coverage 汇总特征对 class 1 / 2 的区分很弱。
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_valtest/burden_coverage_group_stats.csv`
  - 合并 val+test，true class 1 vs 2：
    - `burden_soft_ratio` = 0.515417 vs 0.515440
    - `burden_score_mean` = 0.562499 vs 0.562589
    - `coverage_z_center` = 0.507255 vs 0.506861
    - `coverage_z_spread` = 0.305776 vs 0.306012
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_valtest/test_class12_feature_summary.csv`
  - test only 也几乎相同。
- 结论：模型额外拼接进去的 7 维 summary 特征，对 class 1 / class 2 几乎没有天然可分性。

### 8.1 train split_method 补充观察

- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_train_nocrop/train_metrics_by_split_method.csv`
  - train class 2 recall by split_method：
    - `cc3d` = 0.5597
    - `midline_fallback` = 0.5089
    - `morph_cc3d` = 0.5000
  - train class 1 recall by split_method：
    - `cc3d` = 0.6599
    - `midline_fallback` = 0.7976
    - `morph_cc3d` = 0.6000
- 证据：`outputs/diagnostics/best_model_resnet18_128_f1_train_nocrop/label_by_split_method_crosstab.csv`
  - train class 1 / 2 在 `cc3d` 与 `midline_fallback` 上都大量重叠（class1: 147/84，class2: 134/112）。
- 结论：train 内部也不是单一 split_method 把 class 2 拉垮，问题跨多种 split_method 都存在。

### 9. 代码路径给出的结构性解释

- 证据：`configs/ct25d_resnet18_128.yaml`
  - 关键行：36, 37, 52-60, 72, 74-75, 82, 92。
  - `region_num_instances=128`，`region_fixed_base_total=96`，`aggregator=transmil`，`use_position_embedding=true`，`use_burden_features=true`，`use_coverage_features=true`，`corn_balanced=true`。
- 证据：`main.py`
  - `DataLoader(... batch_size=1 ...)`：251, 252, 255。
  - `freeze_backbone_batchnorm(model)`：281-283。
  - best checkpoint 选择与保存：345, 347, 407, 409, 411。
- 证据：`train/evaluator.py`
  - `evaluate(...)` 起点：53。
  - `case_acc`、`balanced_acc`、`macro_f1` 计算：147-149。
- 证据：`losses/__init__.py`
  - `_corn_loss(...)`：11。
  - `_corn_label_from_logits(...)`：45。
- 证据：`datasets/ct_preprocess/instance_builder.py`
  - 6 区顺序固定：101。
  - “固定底座 + 剩余按有效中心数比例分配”：137-145。
  - `metadata.append({'region', 'center_z', 'split_method'})`：198。
- 证据：`datasets/ct_pne_dataset.py`
  - `_last_metadata` 写入：772。
  - 真正送入模型的是 `(bag, bag_label, bag_pos_z)`：785。
- 证据：`models/attention.py`
  - `forward(self, x, pos_z=None)`：161。
  - 分类器额外只拼接 burden 4 维和 coverage 3 维：138, 140。
  - 位置嵌入只由 `pos_z` 构造：194。
- 证据：`models/aggregators.py`
  - `TransMILAggregator` 定义：8。
  - `cls_token` 拼接与编码器输入：35, 46。
  - 只有 `CLS token + Transformer`，没有显式 `region-id / side-id / lobe-id` embedding。
- 结构性含义：
  - fixed six-region order 本身不会自动变成“区域身份”信号，因为 Transformer 没有显式 region index embedding。
  - 当前显式位置只剩 `pos_z`，更擅长表示上下位置，不足以稳定区分左右肺和六区来源。
  - 如果 class 1 / class 2 主要依赖更细的“区域分布型差异”，当前输入表达能力偏弱，模型就容易把 class 2 压回 class 1。

### 10. 当前最稳的结论

- 已证实：class 2 被压成 class 1 是稳定现象，不是偶然波动。
- 已证实：这种 1/2 混淆在 train 内部也存在，因此不能只归因为测试域泛化退化。
- 已证实：这不是“只差一个阈值”能解决的问题。
- 已证实：这不是明显 crop 崩坏或 gross mask 失败导致的问题。
- 已证实：institution / split_method 会影响 recall，但无法单独解释全部 class 2 错分。
- 已证实：当前显式 summary 特征几乎不提供 class 1 / 2 可分信息。
- 高度怀疑：模型缺少显式区域身份编码，只靠 `pos_z` 和弱 summary 特征，导致 1/2 边界表示不足。
- 未找到证据：scanner / protocol / reconstruction kernel / manufacturer 是主因。

### 11. 最终判断

- 最核心的问题不是 class 数量失衡，也不是简单阈值，也不是明显 crop 失败。
- 现有证据更支持下面这个组合解释：
  - 模型对 0 和 3 的边界学得较好，但 1 和 2 的边界表示明显不足。
  - 这种不足在 train 内部已经存在，说明它不是纯测试域问题。
  - institution / split_method 会放大波动，但不是唯一解释。
  - 当前输入到聚合器的显式结构信息过弱：`pos_z` 有，上下位置可用；但 region identity / side identity / lobe identity 没有显式编码，而 burden / coverage 7 维 summary 又几乎区分不了 class 1 / 2。
- 因此，现阶段最可信的根因是：
  - 1/2 分界所需的区域分布型信息，没有被当前输入表示和聚合方式稳定表达出来；
  - 结果就是大量 true class 2 被模型系统性压回 class 1。