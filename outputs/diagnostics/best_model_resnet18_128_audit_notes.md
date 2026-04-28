## Current evidence snapshot

- Config / training path
	- `configs/ct25d_resnet18_128.yaml`: `region_num_instances=128`, `region_fixed_base_total=96`, `backbone_name=resnet18`, `aggregator=transmil`, `use_burden_features=true`, `use_position_embedding=true`, `use_coverage_features=true`, `corn_balanced=true`, `freeze_backbone=false`, `freeze_backbone_bn=true`, `grad_accum_steps=2`, `model_selection_metric=macro_f1`.
	- `main.py`: batch size is 1 for train / val / test loaders; best checkpoint selection uses `model_selection_metric`; backbone BN can be frozen while backbone remains trainable.

- Val / test no-crop core metrics
	- `outputs/diagnostics/best_model_resnet18_128_f1_valtest/core_metrics.csv`
	- val: acc 0.7063, bal_acc 0.7193, macro_f1 0.7226, qwk 0.8263, class1_recall 0.6757, class2_recall 0.4872, true1->2 = 9, true2->1 = 19.
	- test: acc 0.6702, bal_acc 0.6867, macro_f1 0.6882, qwk 0.8333, class1_recall 0.6622, class2_recall 0.3846, true1->2 = 22, true2->1 = 41.
	- `outputs/diagnostics/best_model_resnet18_128_f1_valtest/confusion_matrices.csv`

- Per-class metrics
	- `outputs/diagnostics/best_model_resnet18_128_f1_valtest/per_class_metrics.csv`
	- test: class0 f1 1.0000, class1 f1 0.5904, class2 f1 0.4027, class3 f1 0.7597.

- Threshold / ordinal evidence
	- `outputs/diagnostics/best_model_resnet18_128_f1_valtest/threshold_calibration_results.csv`
	- default test class2_recall = 0.3846.
	- best single-threshold candidate `[0.4, 0.4, 0.4]` only raises test class2_recall to 0.4103 and slightly changes macro_f1 (0.6882 -> 0.6898).
	- best per-threshold candidate `[0.05, 0.4, 0.1]` hurts test class2_recall to 0.3333.
	- `outputs/diagnostics/best_model_resnet18_128_f1_valtest/false2_to_1_p_y_gt_1_distribution.csv`: on test, false2->1 mean 0.1607, median 0.1197, 35/41 cases have `P(y>1) < 0.2`, only 1/41 lies in [0.45, 0.55].
	- `outputs/diagnostics/best_model_resnet18_128_f1_valtest/class12_binary_proxy_stats.csv`: using only true class 1 vs 2 and score `P(y>1)`, val AUC = 0.5433, test AUC = 0.6003.

- Domain / metadata evidence
	- `outputs/diagnostics/class12_dicom_text_counts.csv`: for class 1/2, `scanner_model`, `protocol_name`, `reconstruction_kernel`, `manufacturer` are all missing (100%).
	- `outputs/diagnostics/class12_dicom_text_counts.csv`: institution differs between class 1 and 2 overall (`HOSPITAL`: class1 18.55%, class2 12.79%; uppercase hospital: class1 76.61%, class2 81.59%).
	- `outputs/diagnostics/best_model_resnet18_128_f1_valtest/class12_group_metrics.csv`: on test, class2 recall by institution = 0.3881 (uppercase hospital, 26/67), 0.6000 (lowercase hospital, 3/5), 0.1667 (`HOSPITAL`, 1/6).
	- `outputs/diagnostics/best_model_resnet18_128_f1_valtest/class12_group_metrics.csv`: on test, class1 recall by institution = 0.7647 (uppercase hospital, 39/51), 0.2500 (lowercase hospital, 1/4), 0.4737 (`HOSPITAL`, 9/19).
	- `outputs/diagnostics/best_model_resnet18_128_f1_valtest/spacing_stats_by_true_label.csv`: on test, spacing_x mean = 0.6659 for class1 vs 0.6726 for class2.
	- `outputs/diagnostics/best_model_resnet18_128_f1_valtest/label_by_split_method_crosstab.csv`: combined val+test class2 has more `midline_fallback` than class1 (58 vs 42 with similar supports 117 vs 111).
	- `outputs/diagnostics/best_model_resnet18_128_f1_valtest/class12_group_metrics.csv`: on test, class2 recall by split_method = 0.4063 (`cc3d`), 0.3429 (`midline_fallback`), 0.4545 (`morph_cc3d`).

- Feature evidence
	- `outputs/diagnostics/best_model_resnet18_128_f1_valtest/burden_coverage_group_stats.csv`: combined val+test means for class1 vs class2 are nearly identical for `burden_soft_ratio` (0.515417 vs 0.515440), `burden_score_mean` (0.562499 vs 0.562589), `coverage_z_center` (0.507255 vs 0.506861), `coverage_z_spread` (0.305776 vs 0.306012).
	- `outputs/diagnostics/best_model_resnet18_128_f1_valtest/test_class12_feature_summary.csv` gives the same picture on test only.
	- `outputs/diagnostics/best_model_resnet18_128_f1_valtest/test_false2_to_1_vs_true2_to_2_groups.csv`: institution shifts are stronger than split_method shifts inside true class 2 error analysis.
	- `outputs/diagnostics/best_model_resnet18_128_f1_valtest/false2_to_1_vs_true2_to_2_spacing.csv`: false2->1 spacing_x mean 0.6764 vs true2->2 mean 0.6685; difference exists but is modest.

- Crop evidence
	- `outputs/diagnostics/best_model_resnet18_128_f1_test_crop/core_metrics.csv` matches no-crop test metrics.
	- `outputs/diagnostics/best_model_resnet18_128_f1_test_crop/crop_case_group_summary.csv`: false2->1 vs true2->2 crop summaries are very close (`lung_pixel_ratio_mean` 0.4668 vs 0.4650, `bbox_phys_h_mm_mean` 165.38 vs 164.00, `bbox_phys_w_mm_mean` 131.09 vs 129.17, `resize_scale_x_mean` 1.1790 vs 1.1829).

- Montage evidence
	- `outputs/diagnostics/montage/` contains 40 montage figures across groups A/B/C/D.
	- Manual inspection of `outputs/diagnostics/montage/A_false2_to_1_low_py_gt_1/test_CT035488.png` and `outputs/diagnostics/montage/B_true2_to_2_high_py_gt_1/test_CT079911.png` shows no obvious mask collapse, left/right split failure, or gross six-region bbox misplacement.

- Code path evidence / hypotheses
	- `models/attention.py`: position embedding uses only `pos_z`; classifier appends only burden (4 dims) and coverage (3 dims) summary features.
	- `models/components.py`: coverage features are z-center / z-spread / active bins; burden features are soft ratio / mean / top-k mean / std.
	- `models/aggregators.py`: TransMIL aggregator adds CLS token but no extra explicit region-position encoding.
	- `datasets/ct_preprocess/instance_builder.py`: token order is fixed by six regions, but per-region allocation is dynamic, and only `split_method` plus region name / center_z are stored in metadata.
	- `datasets/ct_pne_dataset.py`: model input returned to forward is only `(bag, bag_label, bag_pos_z)`.

- Pending
	- Train no-crop metrics are still running in `outputs/diagnostics/best_model_resnet18_128_f1_train_nocrop/`.
