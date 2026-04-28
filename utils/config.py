"""Centralized argument parsing, YAML config loading, and parameter validation."""
from __future__ import print_function

import argparse
import os

import torch

try:
    import yaml
except ImportError:
    yaml = None

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'configs', 'ct25d_transmil.yaml'
)


def _build_parser():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                             help='path to YAML config file')

    config_args, _ = base_parser.parse_known_args()

    parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example', parents=[base_parser])
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--backbone_lr_ratio', type=float, default=0.1,
                        help='when backbone is unfrozen, backbone lr = lr * backbone_lr_ratio')
    parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                        help='weight decay')
    parser.add_argument('--target_number', type=int, default=9, metavar='T',
                        help='bags have a positive labels if they contain at least one 9')
    parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                        help='average bag length')
    parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                        help='variance of bag length')
    parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                        help='number of bags in training set')
    parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                        help='number of bags in test set')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')
    parser.add_argument('--aggregator', type=str, default='abmil', choices=['abmil', 'transmil', 'nystrom'],
                        help='aggregation method: abmil (baseline), transmil (TransMIL self-attention), or nystrom (Nystrom efficient attention)')
    parser.add_argument('--transmil_num_heads', type=int, default=8,
                        help='number of attention heads in TransMIL aggregator')
    parser.add_argument('--transmil_num_layers', type=int, default=2,
                        help='number of Transformer encoder layers in TransMIL aggregator')
    parser.add_argument('--transmil_dropout', type=float, default=0.1,
                        help='dropout rate in TransMIL aggregator')
    parser.add_argument('--nystrom_num_landmarks', type=int, default=64,
                        help='number of landmark points for Nystrom attention approximation')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='number of channels per instance image (e.g., 3 for 2.5D)')
    parser.add_argument('--use_burden_features', action='store_true', default=False,
                        help='enable burden-aware MIL features concatenated with ABMIL bag embedding')
    parser.add_argument('--no_use_burden_features', action='store_false', dest='use_burden_features',
                        help='disable burden-aware MIL features and keep baseline ABMIL behavior')
    parser.add_argument('--burden_score_hidden_dim', type=int, default=0,
                        help='hidden dim of instance abnormal score head (0 means single linear layer)')
    parser.add_argument('--burden_score_dropout', type=float, default=0.0,
                        help='dropout used in instance abnormal score head')
    parser.add_argument('--burden_tau', type=float, default=0.5,
                        help='tau used by soft burden mask: sigmoid((s - tau) / temperature)')
    parser.add_argument('--burden_temperature', type=float, default=0.1,
                        help='temperature used by soft burden mask')
    parser.add_argument('--burden_topk_ratio', type=float, default=0.1,
                        help='top-k ratio for burden topk_mean (k=max(1, round(K*ratio)))')
    parser.add_argument('--pretrained_backbone', action='store_true', default=True,
                        help='use ImageNet pretrained weights for backbone')
    parser.add_argument('--no_pretrained_backbone', action='store_false', dest='pretrained_backbone',
                        help='disable ImageNet pretrained weights for backbone')
    parser.add_argument('--backbone_name', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'densenet121'],
                        help='backbone architecture for feature extraction')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='freeze ResNet18 backbone to reduce GPU memory')
    parser.add_argument('--no_freeze_backbone', action='store_false', dest='freeze_backbone',
                        help='train ResNet18 backbone end-to-end (higher memory)')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='number of ordinal classes for CORN (e.g., 4 for pneumoconiosis staging)')
    parser.add_argument('--dataset', type=str, default='ct25d', choices=['ct25d', 'mnist'],
                        help='dataset type: ct25d for NIfTI MIL or mnist for toy example')
    parser.add_argument('--data_root', type=str, default='/data/scratch/c_pne',
                        help='root directory that contains 0_seg_nii ... 3_seg_nii folders')
    parser.add_argument('--middle_ratio', type=float, default=0.98,
                        help='keep middle ratio of slices, dropping top/bottom equally')
    parser.add_argument('--fixed_num_slices', type=int, default=256,
                        help='fixed number of centered slices sampled per patient for ct25d')
    parser.add_argument('--lung_trim_ratio', type=float, default=0.05,
                        help='fraction to trim from both ends of the effective lung range (0=no trim, must be < 0.5)')
    parser.add_argument('--slab_depth', type=int, default=3,
                        help='odd number of slices stacked per 2.5D slab, e.g., 3 or 5')
    parser.add_argument('--slab_stride', type=int, default=3,
                        help='step for 2.5D slab center index; 1 means 012/123/234, 2 means 012/234/456')
    parser.add_argument('--num_slabs', type=int, default=64,
                        help='number of uniformly sampled slabs per bag for ct25d (0 keeps all candidate centers)')
    parser.add_argument('--center_sampling_mode', type=str, default='uniform', choices=['uniform', 'random', 'all'],
                        help='how to choose slab centers when num_slabs > 0')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='patient-level test split ratio for ct25d dataset')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='patient-level validation split ratio for ct25d dataset (set 0 to disable val split)')
    parser.add_argument('--max_instances', type=int, default=0,
                        help='optional cap for instances per bag (0 means keep all)')
    parser.add_argument('--scale_to_unit', action='store_true', default=True,
                        help='linearly map clipped HU values to [0, 1] before optional z-score')
    parser.add_argument('--no_scale_to_unit', action='store_false', dest='scale_to_unit',
                        help='disable linear HU scaling and keep clipped values as-is')
    parser.add_argument('--use_zscore', action='store_true', default=False,
                        help='apply per-volume z-score normalization after HU clipping')
    parser.add_argument('--no_use_zscore', action='store_false', dest='use_zscore',
                        help='disable per-volume z-score normalization and keep absolute HU scale')
    parser.add_argument('--lung_hu_low', type=float, default=-950.0,
                        help='lower HU bound for estimating effective lung slices')
    parser.add_argument('--lung_hu_high', type=float, default=-300.0,
                        help='upper HU bound for estimating effective lung slices')
    parser.add_argument('--min_lung_area_ratio', type=float, default=0.01,
                        help='minimum per-slice lung-like area ratio used for effective z-range')
    parser.add_argument('--instance_definition', type=str, default='lung_region_thin_slab',
                        choices=['lung_region_thin_slab', 'global_slab'],
                        help='instance generation strategy for CT MIL bags')
    parser.add_argument('--lung_mask_root', type=str, default='',
                        help='root directory for lung masks aligned with CT files; required for lung_region_thin_slab')
    parser.add_argument('--lung_mask_suffix', type=str, default='.nii.gz',
                        help='suffix appended to CT relative path base to resolve lung mask file path')
    parser.add_argument('--lung_mask_require', action='store_true', default=True,
                        help='raise error when lung mask is missing for lung_region_thin_slab')
    parser.add_argument('--no_lung_mask_require', action='store_false', dest='lung_mask_require',
                        help='allow missing lung mask (not recommended for lung_region_thin_slab)')
    parser.add_argument('--cache_root', type=str, default='',
                        help='root directory for offline lung-region caches; if empty, caching is disabled')
    parser.add_argument('--pseudo_mask_value_threshold', type=float, default=1e-6,
                        help='value threshold used to recover pseudo lung mask from masked CT when real mask is absent')
    parser.add_argument('--pseudo_mask_min_component_voxels', type=int, default=512,
                        help='minimum connected-component volume kept in pseudo mask post-processing')
    parser.add_argument('--region_num_instances', type=int, default=64,
                        help='number of region thin-slab instances per case (0 for dense sampling: all valid centers)')
    parser.add_argument('--region_out_h', type=int, default=224,
                        help='output height of each cropped region instance')
    parser.add_argument('--region_out_w', type=int, default=224,
                        help='output width of each cropped region instance')
    parser.add_argument('--region_bbox_margin', type=int, default=12,
                        help='xy bbox margin (pixels) around region mask')
    parser.add_argument('--region_bbox_min_size', type=int, default=32,
                        help='minimum bbox size for cropped region instance')
    parser.add_argument('--region_abs_area_threshold', type=float, default=100.0,
                        help='absolute union area threshold for legal 3-slice center selection')
    parser.add_argument('--region_ratio_area_threshold', type=float, default=0.05,
                        help='relative union area threshold ratio for legal center selection')
    parser.add_argument('--debug_save_instances', action='store_true', default=False,
                        help='save sampled instance visualizations for debugging')
    parser.add_argument('--debug_save_lung_split', action='store_true', default=False,
                        help='save left-right lung split visualizations for debugging')
    parser.add_argument('--debug_save_six_regions', action='store_true', default=False,
                        help='save pseudo mask and six-lung-region visualizations for debugging')
    parser.add_argument('--debug_dir', type=str, default='debug_instances',
                        help='directory to save debug instance images')
    parser.add_argument('--debug_max_cases', type=int, default=0,
                        help='maximum number of cases to export debug instance images')
    parser.add_argument('--instance_batch_size', type=int, default=8,
                        help='micro-batch size for instance feature extraction inside each bag')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='number of bags to accumulate gradients before each optimizer step')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='number of initial epochs to freeze backbone before unfreezing')
    parser.add_argument('--freeze_backbone_bn', action='store_true', default=True,
                        help='keep backbone BatchNorm in eval mode when backbone is trainable')
    parser.add_argument('--no_freeze_backbone_bn', action='store_false', dest='freeze_backbone_bn',
                        help='allow backbone BatchNorm running stats to update during training')
    parser.add_argument('--lr_scheduler', type=str, default='plateau', choices=['none', 'plateau'],
                        help='learning rate scheduler to use')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='multiplicative factor for ReduceLROnPlateau')
    parser.add_argument('--scheduler_patience', type=int, default=2,
                        help='patience for ReduceLROnPlateau')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6,
                        help='minimum learning rate allowed by the scheduler')
    parser.add_argument('--metrics_csv', type=str, default='training_metrics.csv',
                        help='path to CSV file for epoch metrics')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='early stopping patience on validation loss (<=0 disables early stopping)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='minimum validation loss improvement to reset early stopping counter')
    parser.add_argument('--best_model_path', type=str, default='checkpoints/best_model.pt',
                        help='path to save the best model selected by validation loss')
    parser.add_argument('--model_selection_metric', type=str, default='val_loss',
                        choices=['val_loss', 'macro_f1'],
                        help='metric for best model selection: val_loss (lower=better) or macro_f1 (higher=better)')
    parser.add_argument('--score_logit_reg_weight', type=float, default=0.01,
                        help='L2 regularization weight on instance score logits to prevent saturation')
    parser.add_argument('--corn_balanced', action='store_true', default=True,
                        help='use per-task pos_weight balancing in CORN loss')
    parser.add_argument('--no_corn_balanced', action='store_false', dest='corn_balanced',
                        help='disable per-task balancing in CORN loss')
    parser.add_argument('--instance_aux_loss_weight', type=float, default=0.0,
                        help='weight for auxiliary instance-level BCE loss to supervise score head (0 disables)')

    # ---- Pseudo-12 subtype guidance (Plan B: hard labels) ----
    parser.add_argument('--use_pseudo12_guidance', action='store_true', default=False,
                        help='Plan B: enable pseudo 12-subtype auxiliary head with hard CE loss')
    parser.add_argument('--no_use_pseudo12_guidance', action='store_false', dest='use_pseudo12_guidance',
                        help='disable Plan B pseudo 12-subtype auxiliary head')
    parser.add_argument('--pseudo12_lambda', type=float, default=0.1,
                        help='Plan B: weight for pseudo-12 CE auxiliary loss')

    # ---- Soft-12 subtype guidance (Plan C: soft KL distributions) ----
    parser.add_argument('--use_soft12_guidance', action='store_true', default=False,
                        help='Plan C: enable soft 12-subtype auxiliary head with KL divergence loss')
    parser.add_argument('--no_use_soft12_guidance', action='store_false', dest='use_soft12_guidance',
                        help='disable Plan C soft 12-subtype auxiliary head')
    parser.add_argument('--soft12_lambda', type=float, default=0.1,
                        help='Plan C: weight for soft-12 KL auxiliary loss')
    parser.add_argument('--soft12_tau', type=float, default=0.5,
                        help='Plan C: Gaussian bandwidth tau for intra-class soft distribution. '
                             'Smaller => sharper. Range (0, +inf), typical: 0.1-1.0')
    parser.add_argument('--soft12_eps', type=float, default=1e-6,
                        help='Plan C: epsilon for S_min/S_max normalization')
    parser.add_argument('--soft12_norm_stats_path', type=str, default='',
                        help='Plan C: path to save/load per-class normalization stats JSON; '
                             'empty string disables saving')

    return parser, config_args


def _validate_args(args):
    """Validate argument ranges and constraints."""
    if args.backbone_lr_ratio <= 0.0:
        raise ValueError('backbone_lr_ratio must be > 0')
    if args.early_stop_patience < 0:
        raise ValueError('early_stop_patience must be >= 0')
    if args.early_stop_min_delta < 0.0:
        raise ValueError('early_stop_min_delta must be >= 0')
    if args.grad_accum_steps < 1:
        raise ValueError('grad_accum_steps must be >= 1')
    if args.warmup_epochs < 0:
        raise ValueError('warmup_epochs must be >= 0')
    if args.scheduler_factor <= 0.0 or args.scheduler_factor >= 1.0:
        raise ValueError('scheduler_factor must be in (0, 1)')
    if args.scheduler_patience < 0:
        raise ValueError('scheduler_patience must be >= 0')
    if args.scheduler_min_lr < 0.0:
        raise ValueError('scheduler_min_lr must be >= 0')
    if args.lung_hu_low >= args.lung_hu_high:
        raise ValueError('lung_hu_low must be < lung_hu_high')
    if not (0.0 <= args.lung_trim_ratio < 0.5):
        raise ValueError('lung_trim_ratio must be in [0, 0.5)')
    if not (0.0 <= args.min_lung_area_ratio < 1.0):
        raise ValueError('min_lung_area_ratio must be in [0, 1)')
    if args.region_num_instances < 0:
        raise ValueError('region_num_instances must be >= 0 (0 means dense sampling of all valid centers)')
    if args.region_out_h <= 0 or args.region_out_w <= 0:
        raise ValueError('region_out_h and region_out_w must be > 0')
    if args.region_bbox_margin < 0:
        raise ValueError('region_bbox_margin must be >= 0')
    if args.region_bbox_min_size <= 0:
        raise ValueError('region_bbox_min_size must be > 0')
    if args.region_abs_area_threshold < 0.0:
        raise ValueError('region_abs_area_threshold must be >= 0')
    if args.region_ratio_area_threshold < 0.0:
        raise ValueError('region_ratio_area_threshold must be >= 0')
    if args.pseudo_mask_value_threshold < 0.0:
        raise ValueError('pseudo_mask_value_threshold must be >= 0')
    if args.pseudo_mask_min_component_voxels <= 0:
        raise ValueError('pseudo_mask_min_component_voxels must be > 0')
    if args.burden_score_hidden_dim < 0:
        raise ValueError('burden_score_hidden_dim must be >= 0')
    if args.burden_score_dropout < 0.0 or args.burden_score_dropout >= 1.0:
        raise ValueError('burden_score_dropout must be in [0, 1)')
    if args.burden_temperature <= 0.0:
        raise ValueError('burden_temperature must be > 0')
    if args.burden_topk_ratio <= 0.0 or args.burden_topk_ratio > 1.0:
        raise ValueError('burden_topk_ratio must be in (0, 1]')


def parse_args(argv=None):
    """Parse command-line arguments with optional YAML config overlay.

    Returns:
        args: argparse.Namespace with all training parameters.
    """
    parser, config_args = _build_parser()

    if config_args.config:
        if yaml is None:
            raise ImportError('PyYAML is required for --config. Please install pyyaml.')
        config_path = os.path.expanduser(config_args.config)
        if not os.path.exists(config_path):
            raise FileNotFoundError('Config file not found: {}'.format(config_path))
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        if not isinstance(config_dict, dict):
            raise ValueError('Config file must contain a YAML mapping (key-value pairs).')
        parser.set_defaults(**config_dict)

    args = parser.parse_args(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.best_model_path = os.path.expanduser(args.best_model_path)
    _validate_args(args)
    return args
