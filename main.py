"""Main entry point for MIL training."""

from __future__ import print_function

import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils

from datasets.ct_pne_dataset import CTPneNiiBags
from datasets.mnist_bags import MnistBags
from train.evaluator import evaluate
from train.trainer import train_one_epoch
from models.attention import Attention
from models.gated_attention import GatedAttention
from utils.config import parse_args
from utils.logging import append_metrics_csv, init_metrics_csv
from utils.misc import (
    freeze_backbone_batchnorm,
    set_backbone_trainable,
    setup_seed,
)

_BURDEN_KEYS = ('soft_ratio', 'score_mean', 'topk_mean', 'score_std')


def _build_datasets(args):
    """Build train/val/test datasets according to args.dataset."""
    val_dataset = None

    if args.dataset == 'ct25d':
        if args.slab_depth < 1 or (args.slab_depth % 2) == 0:
            raise ValueError('slab_depth must be a positive odd integer, e.g., 3 or 5')
        half_depth = args.slab_depth // 2
        channel_offsets = tuple(range(-half_depth, half_depth + 1))
        if args.in_channels != len(channel_offsets):
            raise ValueError(
                'in_channels ({}) must equal slab_depth ({}) for ct25d'.format(
                    args.in_channels, len(channel_offsets)
                )
            )

        common_kwargs = dict(
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
            debug_save_instances=args.debug_save_instances,
            debug_save_lung_split=args.debug_save_lung_split,
            debug_save_six_regions=args.debug_save_six_regions,
            debug_dir=args.debug_dir,
            debug_max_cases=args.debug_max_cases,
        )

        train_dataset = CTPneNiiBags(split='train', **common_kwargs)
        if args.val_ratio > 0.0:
            val_dataset = CTPneNiiBags(split='val', **common_kwargs)
        test_dataset = CTPneNiiBags(split='test', **common_kwargs)
    else:
        full_train_dataset = MnistBags(
            target_number=args.target_number,
            mean_bag_length=args.mean_bag_length,
            var_bag_length=args.var_bag_length,
            num_bag=args.num_bags_train,
            seed=args.seed,
            train=True,
        )
        val_count = int(round(len(full_train_dataset) * args.val_ratio))
        if len(full_train_dataset) > 1 and val_count >= len(full_train_dataset):
            val_count = len(full_train_dataset) - 1
        val_count = max(0, val_count)
        train_count = len(full_train_dataset) - val_count
        if train_count == 0:
            train_count = len(full_train_dataset)
            val_count = 0
        split_gen = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = data_utils.random_split(
            full_train_dataset, [train_count, val_count], generator=split_gen
        )
        test_dataset = MnistBags(
            target_number=args.target_number,
            mean_bag_length=args.mean_bag_length,
            var_bag_length=args.var_bag_length,
            num_bag=args.num_bags_test,
            seed=args.seed,
            train=False,
        )

    return train_dataset, val_dataset, test_dataset


def _build_model(args):
    """Instantiate the MIL model."""
    if args.model == 'attention':
        model = Attention(
            in_channels=args.in_channels,
            pretrained_backbone=args.pretrained_backbone,
            num_classes=args.num_classes,
            instance_batch_size=args.instance_batch_size,
            freeze_backbone=args.freeze_backbone,
            backbone_name=getattr(args, 'backbone_name', 'resnet18'),
            use_burden_features=args.use_burden_features,
            use_position_embedding=getattr(args, 'use_position_embedding', False),
            position_embed_dim=getattr(args, 'position_embed_dim', 16),
            use_coverage_features=getattr(args, 'use_coverage_features', False),
            coverage_num_bins=getattr(args, 'coverage_num_bins', 6),
            coverage_tau=getattr(args, 'coverage_tau', 0.5),
            coverage_temperature=getattr(args, 'coverage_temperature', 0.1),
            coverage_eps=getattr(args, 'coverage_eps', 1e-6),
            burden_score_hidden_dim=args.burden_score_hidden_dim,
            burden_score_dropout=args.burden_score_dropout,
            burden_tau=args.burden_tau,
            burden_temperature=args.burden_temperature,
            burden_topk_ratio=args.burden_topk_ratio,
            aggregator=args.aggregator,
            transmil_num_heads=args.transmil_num_heads,
            transmil_num_layers=args.transmil_num_layers,
            transmil_dropout=args.transmil_dropout,
            nystrom_num_landmarks=getattr(args, 'nystrom_num_landmarks', 64),
            score_logit_reg_weight=args.score_logit_reg_weight,
            corn_balanced=args.corn_balanced,
            instance_aux_loss_weight=getattr(args, 'instance_aux_loss_weight', 0.0),
            use_pseudo12_guidance=getattr(args, 'use_pseudo12_guidance', False),
            pseudo12_lambda=getattr(args, 'pseudo12_lambda', 0.1),
            use_soft12_guidance=getattr(args, 'use_soft12_guidance', False),
            soft12_lambda=getattr(args, 'soft12_lambda', 0.1),
        )
    elif args.model == 'gated_attention':
        model = GatedAttention(
            in_channels=args.in_channels,
            pretrained_backbone=args.pretrained_backbone,
            num_classes=args.num_classes,
            instance_batch_size=args.instance_batch_size,
            freeze_backbone=args.freeze_backbone,
            use_burden_features=args.use_burden_features,
            burden_score_hidden_dim=args.burden_score_hidden_dim,
            burden_score_dropout=args.burden_score_dropout,
            burden_tau=args.burden_tau,
            burden_temperature=args.burden_temperature,
            burden_topk_ratio=args.burden_topk_ratio,
            score_logit_reg_weight=args.score_logit_reg_weight,
            corn_balanced=args.corn_balanced,
        )
    else:
        raise ValueError("Unknown model: '{}'".format(args.model))

    return model


def _build_optimizer(model, args, freeze_backbone_stage, stage_name):
    """Build optimizer with optional per-group learning rates."""
    if freeze_backbone_stage:
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
        print('Optimizer [{}] -> single lr: {:.2e}'.format(stage_name, args.lr))
        return opt

    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('feature_extractor.'):
            backbone_params.append(p)
        else:
            head_params.append(p)

    backbone_lr = args.lr * args.backbone_lr_ratio
    param_groups = []
    if len(backbone_params) > 0:
        param_groups.append({'params': backbone_params, 'lr': backbone_lr})
    if len(head_params) > 0:
        param_groups.append({'params': head_params, 'lr': args.lr})
    if len(param_groups) == 0:
        raise RuntimeError('No trainable parameters found for optimizer.')

    opt = optim.Adam(param_groups, betas=(0.9, 0.999), weight_decay=args.reg)
    print('Optimizer [{}] -> head lr: {:.2e}, backbone lr: {:.2e} (ratio={:.3f})'.format(
        stage_name, args.lr, backbone_lr, args.backbone_lr_ratio
    ))
    return opt


def _build_scheduler(optimizer_obj, args):
    """Build learning rate scheduler."""
    if args.lr_scheduler != 'plateau':
        return None

    _sel = getattr(args, 'model_selection_metric', 'val_loss')
    _mode = 'max' if _sel == 'macro_f1' else 'min'
    scheduler_kwargs = {
        'mode': _mode,
        'factor': args.scheduler_factor,
        'patience': args.scheduler_patience,
        'min_lr': args.scheduler_min_lr,
    }
    try:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_obj,
            verbose=True,
            **scheduler_kwargs
        )
    except TypeError:
        print('ReduceLROnPlateau verbose flag is not supported in this torch version.')
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_obj,
            **scheduler_kwargs
        )


def main():
    args = parse_args()

    setup_seed(args.seed, use_cuda=args.cuda)
    if args.cuda:
        print('\nGPU is ON!')

    # ---- Datasets ----
    print('Load Train/Val/Test Set')
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_dataset, val_dataset, test_dataset = _build_datasets(args)

    train_loader = data_utils.DataLoader(train_dataset, batch_size=1, shuffle=True, **loader_kwargs)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)
    val_loader = None
    if val_dataset is not None:
        val_loader = data_utils.DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)
    else:
        if args.early_stop_patience > 0:
            print('Validation is disabled, so early stopping and best-model selection are skipped.')

    val_size = len(val_dataset) if val_dataset is not None else 0
    print('Dataset sizes -> train: {}, val: {}, test: {}'.format(len(train_dataset), val_size, len(test_dataset)))

    if args.dataset == 'ct25d':
        def _class_counts(ds):
            counts = np.zeros(args.num_classes, dtype=np.int64)
            for _, y in ds.samples:
                if 0 <= int(y) < args.num_classes:
                    counts[int(y)] += 1
            return counts.tolist()
        print('Class distribution -> train: {}'.format(_class_counts(train_dataset)))
        if val_dataset is not None:
            print('Class distribution -> val: {}'.format(_class_counts(val_dataset)))
        print('Class distribution -> test: {}'.format(_class_counts(test_dataset)))

    # ---- Model ----
    print('Init Model')
    model = _build_model(args)
    if args.cuda:
        model.cuda()

    if (not args.freeze_backbone) and args.freeze_backbone_bn:
        frozen_bn_count = freeze_backbone_batchnorm(model)
        print('Backbone BN frozen layers: {}'.format(frozen_bn_count))

    warmup_active = (not args.freeze_backbone) and (args.warmup_epochs > 0)
    if warmup_active:
        set_backbone_trainable(model, trainable=False)
        print('Warmup active: backbone frozen for first {} epoch(s).'.format(args.warmup_epochs))
        if args.warmup_epochs >= args.epochs:
            print('Warning: warmup_epochs >= epochs, backbone will stay frozen for all epochs.')

    # ---- Pseudo-12 calibration (Plan B: hard labels) ----
    if getattr(args, 'use_pseudo12_guidance', False):
        from pseudo12.calibration import build_pseudo12_labels_for_all_splits
        train_p12, val_p12, test_p12, p12_thresholds = build_pseudo12_labels_for_all_splits(
            model, train_loader, val_loader, test_loader, args,
        )
        # Set pseudo12 labels on datasets
        if hasattr(train_dataset, 'set_pseudo12_labels'):
            train_dataset.set_pseudo12_labels(train_p12)
        if val_dataset is not None and hasattr(val_dataset, 'set_pseudo12_labels'):
            val_dataset.set_pseudo12_labels(val_p12)
        if hasattr(test_dataset, 'set_pseudo12_labels'):
            test_dataset.set_pseudo12_labels(test_p12)
        print('Pseudo-12 (Plan B) labels assigned to all datasets.')

    # ---- Soft-12 calibration (Plan C: soft KL distributions) ----
    if getattr(args, 'use_soft12_guidance', False):
        from soft12.calibration import build_soft12_targets_for_all_splits
        _soft12_save_path = getattr(args, 'soft12_norm_stats_path', '') or None
        train_s12, val_s12, test_s12, s12_norm_stats = build_soft12_targets_for_all_splits(
            model,
            train_loader,
            val_loader,
            test_loader,
            args,
            tau=getattr(args, 'soft12_tau', 0.5),
            eps=getattr(args, 'soft12_eps', 1e-6),
            save_path=_soft12_save_path,
        )
        if hasattr(train_dataset, 'set_soft12_targets'):
            train_dataset.set_soft12_targets(train_s12)
        if val_dataset is not None and hasattr(val_dataset, 'set_soft12_targets'):
            val_dataset.set_soft12_targets(val_s12)
        if hasattr(test_dataset, 'set_soft12_targets'):
            test_dataset.set_soft12_targets(test_s12)
        print('Soft-12 (Plan C) targets assigned to all datasets.')

    # ---- Optimizer / Scheduler ----
    optimizer = _build_optimizer(
        model, args,
        freeze_backbone_stage=(args.freeze_backbone or warmup_active),
        stage_name='warmup' if warmup_active else 'init'
    )
    scheduler = _build_scheduler(optimizer, args)
    print('Gradient accumulation steps: {}'.format(args.grad_accum_steps))

    # ---- CSV logging ----
    from datetime import datetime as _dt
    _csv_base, _csv_ext = os.path.splitext(args.metrics_csv)
    args.metrics_csv = '{}_{}{}'.format(_csv_base, _dt.now().strftime('%Y%m%d_%H%M%S'), _csv_ext or '.csv')
    init_metrics_csv(args.metrics_csv)

    # Model selection metric setup
    _sel_metric = getattr(args, 'model_selection_metric', 'val_loss')
    _sel_higher_better = (_sel_metric == 'macro_f1')
    best_metric_val = -float('inf') if _sel_higher_better else float('inf')
    best_epoch = None
    epochs_without_improve = 0
    early_stop_active = (val_loader is not None and args.early_stop_patience > 0)
    print('Model selection metric: {} ({})'.format(_sel_metric, 'higher=better' if _sel_higher_better else 'lower=better'))

    best_model_dir = os.path.dirname(args.best_model_path)
    if best_model_dir:
        os.makedirs(best_model_dir, exist_ok=True)

    # ---- Training loop ----
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        if warmup_active and epoch == (args.warmup_epochs + 1):
            set_backbone_trainable(model, trainable=True)
            if (not args.freeze_backbone) and args.freeze_backbone_bn:
                freeze_backbone_batchnorm(model)
            optimizer = _build_optimizer(model, args, freeze_backbone_stage=False, stage_name='post-warmup')
            scheduler = _build_scheduler(optimizer, args)
            print('Warmup finished at epoch {}. Backbone is now unfrozen.'.format(epoch - 1))

        train_loss, train_error, train_burden = train_one_epoch(epoch, model, train_loader, optimizer, args)

        if val_loader is not None:
            val_loss, val_error, val_burden, _, _, val_extra = evaluate(val_loader, model, args, split_name='Val', show_examples=False)
        else:
            val_loss, val_error, val_burden = None, None, {k: None for k in _BURDEN_KEYS}
            val_extra = {'macro_f1': 0.0}

        train_acc = None if train_error is None else (1.0 - train_error)
        val_acc = None if val_error is None else (1.0 - val_error)

        append_metrics_csv(
            args.metrics_csv,
            epoch=epoch,
            train_loss=train_loss,
            train_error=train_error,
            train_acc=train_acc,
            train_soft_ratio=train_burden['soft_ratio'],
            train_score_mean=train_burden['score_mean'],
            train_topk_mean=train_burden['topk_mean'],
            train_score_std=train_burden['score_std'],
            val_loss=val_loss,
            val_error=val_error,
            val_acc=val_acc,
            val_soft_ratio=val_burden['soft_ratio'],
            val_score_mean=val_burden['score_mean'],
            val_topk_mean=val_burden['topk_mean'],
            val_score_std=val_burden['score_std'],
            test_loss=None,
            test_error=None,
            test_acc=None,
        )

        if val_loader is not None and val_loss is not None:
            # Model selection
            current_metric = val_extra.get('macro_f1', 0.0) if _sel_higher_better else val_loss
            if scheduler is not None:
                scheduler.step(current_metric)
            if _sel_higher_better:
                improved = current_metric > (best_metric_val + args.early_stop_min_delta)
            else:
                improved = current_metric < (best_metric_val - args.early_stop_min_delta)
            if improved:
                best_metric_val = current_metric
                best_epoch = epoch
                epochs_without_improve = 0
                torch.save(model.state_dict(), args.best_model_path)
                print('Saved best model at epoch {} with {}={:.4f} -> {}'.format(
                    epoch, _sel_metric, current_metric, args.best_model_path
                ))
            else:
                epochs_without_improve += 1

            if early_stop_active and epochs_without_improve >= args.early_stop_patience:
                print('Early stopping triggered at epoch {} (best epoch: {}).'.format(
                    epoch, best_epoch
                ))
                break

    if val_loader is not None and best_epoch is not None and os.path.exists(args.best_model_path):
        map_location = torch.device('cuda') if args.cuda else torch.device('cpu')
        model.load_state_dict(torch.load(args.best_model_path, map_location=map_location))
        print('Loaded best model from epoch {} for final testing.'.format(best_epoch))

    # ---- Final validation (for visualization) ----
    val_true_final, val_pred_final = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if val_loader is not None:
        _, _, _, val_true_final, val_pred_final, _ = evaluate(val_loader, model, args, split_name='Val-Final', show_examples=False)

    # ---- Final test ----
    print('Start Testing')
    test_loss, test_error, test_burden, test_true, test_pred, _ = evaluate(test_loader, model, args, split_name='Test', show_examples=True)
    test_acc = None if test_error is None else (1.0 - test_error)
    append_metrics_csv(
        args.metrics_csv,
        epoch='test',
        train_loss=None,
        train_error=None,
        train_acc=None,
        val_loss=None,
        val_error=None,
        val_acc=None,
        test_loss=test_loss,
        test_error=test_error,
        test_acc=test_acc,
        test_soft_ratio=test_burden['soft_ratio'],
        test_score_mean=test_burden['score_mean'],
        test_topk_mean=test_burden['topk_mean'],
        test_score_std=test_burden['score_std'],
    )
    print('Metrics CSV saved to {}'.format(args.metrics_csv))

    # ---- Generate training result visualizations ----
    from utils.visualize import generate_all_plots
    from datetime import datetime as _dt2
    viz_dir = os.path.join('training_results', _dt2.now().strftime('%Y%m%d_%H%M%S'))
    generate_all_plots(
        csv_path=args.metrics_csv,
        y_true_test=test_true,
        y_pred_test=test_pred,
        num_classes=args.num_classes,
        save_dir=viz_dir,
        y_true_val=val_true_final if len(val_true_final) > 0 else None,
        y_pred_val=val_pred_final if len(val_pred_final) > 0 else None,
    )


if __name__ == '__main__':
    main()
