from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable

try:
    import yaml
except ImportError:
    yaml = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from dataloader import CTPneNiiBags, MnistBags
from model import Attention, GatedAttention

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'ct25d_transmil.yaml')


def _iter_with_progress(iterable, total, desc):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def _freeze_backbone_batchnorm(model):
    if not hasattr(model, 'feature_extractor'):
        return 0

    bn_count = 0
    for m in model.feature_extractor.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            bn_count += 1

    return bn_count


def _set_dataset_epoch(dataset, epoch):
    if hasattr(dataset, 'set_epoch'):
        dataset.set_epoch(epoch)
        return True

    base_dataset = getattr(dataset, 'dataset', None)
    if base_dataset is not None and hasattr(base_dataset, 'set_epoch'):
        base_dataset.set_epoch(epoch)
        return True

    return False


def _set_backbone_trainable(model, trainable):
    if not hasattr(model, 'feature_extractor'):
        return
    for p in model.feature_extractor.parameters():
        p.requires_grad = bool(trainable)


# Training settings
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
parser.add_argument('--aggregator', type=str, default='abmil', choices=['abmil', 'transmil'],
                    help='aggregation method: abmil (baseline) or transmil (TransMIL self-attention)')
parser.add_argument('--transmil_num_heads', type=int, default=8,
                    help='number of attention heads in TransMIL aggregator')
parser.add_argument('--transmil_num_layers', type=int, default=2,
                    help='number of Transformer encoder layers in TransMIL aggregator')
parser.add_argument('--transmil_dropout', type=float, default=0.1,
                    help='dropout rate in TransMIL aggregator')
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
                    help='use ImageNet pretrained weights for ResNet18 backbone')
parser.add_argument('--no_pretrained_backbone', action='store_false', dest='pretrained_backbone',
                    help='disable ImageNet pretrained weights for ResNet18 backbone')
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
                    help='number of region thin-slab instances per case')
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
parser.add_argument('--score_logit_reg_weight', type=float, default=0.01,
                    help='L2 regularization weight on instance score logits to prevent saturation')
parser.add_argument('--corn_balanced', action='store_true', default=True,
                    help='use per-task pos_weight balancing in CORN loss')
parser.add_argument('--no_corn_balanced', action='store_false', dest='corn_balanced',
                    help='disable per-task balancing in CORN loss')
parser.add_argument('--instance_aux_loss_weight', type=float, default=0.0,
                    help='weight for auxiliary instance-level BCE loss to supervise score head (0 disables)')

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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
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
if not (0.0 <= args.min_lung_area_ratio < 1.0):
    raise ValueError('min_lung_area_ratio must be in [0, 1)')
if args.region_num_instances <= 0:
    raise ValueError('region_num_instances must be > 0')
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
args.best_model_path = os.path.expanduser(args.best_model_path)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train/Val/Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
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

    train_dataset = CTPneNiiBags(
        root_dir=args.data_root,
        split='train',
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
    if args.val_ratio > 0.0:
        val_dataset = CTPneNiiBags(
            root_dir=args.data_root,
            split='val',
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
    test_dataset = CTPneNiiBags(
        root_dir=args.data_root,
        split='test',
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
else:
    full_train_dataset = MnistBags(target_number=args.target_number,
                                   mean_bag_length=args.mean_bag_length,
                                   var_bag_length=args.var_bag_length,
                                   num_bag=args.num_bags_train,
                                   seed=args.seed,
                                   train=True)
    val_count = int(round(len(full_train_dataset) * args.val_ratio))
    if len(full_train_dataset) > 1 and val_count >= len(full_train_dataset):
        val_count = len(full_train_dataset) - 1
    val_count = max(0, val_count)
    train_count = len(full_train_dataset) - val_count
    if train_count == 0:
        train_count = len(full_train_dataset)
        val_count = 0
    split_gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = data_utils.random_split(full_train_dataset, [train_count, val_count], generator=split_gen)
    test_dataset = MnistBags(target_number=args.target_number,
                             mean_bag_length=args.mean_bag_length,
                             var_bag_length=args.var_bag_length,
                             num_bag=args.num_bags_test,
                             seed=args.seed,
                             train=False)

train_loader = data_utils.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    **loader_kwargs
)

test_loader = data_utils.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    **loader_kwargs
)

if val_dataset is not None:
    val_loader = data_utils.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        **loader_kwargs
    )
else:
    val_loader = None
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

print('Init Model')
if args.model=='attention':
    model = Attention(
        in_channels=args.in_channels,
        pretrained_backbone=args.pretrained_backbone,
        num_classes=args.num_classes,
        instance_batch_size=args.instance_batch_size,
        freeze_backbone=args.freeze_backbone,
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
        score_logit_reg_weight=args.score_logit_reg_weight,
        corn_balanced=args.corn_balanced,
    )
elif args.model=='gated_attention':
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
if args.cuda:
    model.cuda()

if (not args.freeze_backbone) and args.freeze_backbone_bn:
    frozen_bn_count = _freeze_backbone_batchnorm(model)
    print('Backbone BN frozen layers: {}'.format(frozen_bn_count))

warmup_active = (not args.freeze_backbone) and (args.warmup_epochs > 0)
if warmup_active:
    _set_backbone_trainable(model, trainable=False)
    print('Warmup active: backbone frozen for first {} epoch(s).'.format(args.warmup_epochs))
    if args.warmup_epochs >= args.epochs:
        print('Warning: warmup_epochs >= epochs, backbone will stay frozen for all epochs.')


def _build_optimizer(freeze_backbone_stage, stage_name):
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


def _build_scheduler(optimizer_obj):
    if args.lr_scheduler != 'plateau':
        return None

    scheduler_kwargs = {
        'mode': 'min',
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
        # Some torch versions do not support the verbose argument.
        print('ReduceLROnPlateau verbose flag is not supported in this torch version.')
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_obj,
            **scheduler_kwargs
        )


optimizer = _build_optimizer(
    freeze_backbone_stage=(args.freeze_backbone or warmup_active),
    stage_name='warmup' if warmup_active else 'init'
)
scheduler = _build_scheduler(optimizer)
print('Gradient accumulation steps: {}'.format(args.grad_accum_steps))


_BURDEN_KEYS = ('soft_ratio', 'score_mean', 'topk_mean', 'score_std')


def _init_burden_sums():
    return {k: 0.0 for k in _BURDEN_KEYS}


def _merge_burden_stats(sums_dict, stats_dict):
    if stats_dict is None:
        return False
    valid = False
    for k in _BURDEN_KEYS:
        if k in stats_dict and stats_dict[k] is not None:
            sums_dict[k] += float(stats_dict[k])
            valid = True
    return valid


def _finalize_burden_means(sums_dict, count):
    if count <= 0:
        return {k: None for k in _BURDEN_KEYS}
    return {k: (sums_dict[k] / float(count)) for k in _BURDEN_KEYS}


def _format_burden_print(prefix, burden_means):
    if burden_means is None or burden_means.get('soft_ratio') is None:
        print('{} burden stats: N/A (use_burden_features={})'.format(prefix, args.use_burden_features))
        return
    print('{} burden stats -> soft_ratio: {:.4f}, score_mean: {:.4f}, topk_mean: {:.4f}, score_std: {:.4f}'.format(
        prefix,
        burden_means['soft_ratio'],
        burden_means['score_mean'],
        burden_means['topk_mean'],
        burden_means['score_std'],
    ))


def init_metrics_csv(csv_path):
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch',
            'train_loss',
            'train_error',
            'train_acc',
            'train_soft_ratio',
            'train_score_mean',
            'train_topk_mean',
            'train_score_std',
            'val_loss',
            'val_error',
            'val_acc',
            'val_soft_ratio',
            'val_score_mean',
            'val_topk_mean',
            'val_score_std',
            'test_loss',
            'test_error',
            'test_acc',
            'test_soft_ratio',
            'test_score_mean',
            'test_topk_mean',
            'test_score_std',
        ])


def append_metrics_csv(csv_path, epoch, train_loss=None, train_error=None, train_acc=None,
                       train_soft_ratio=None, train_score_mean=None, train_topk_mean=None, train_score_std=None,
                       val_loss=None, val_error=None, val_acc=None,
                       val_soft_ratio=None, val_score_mean=None, val_topk_mean=None, val_score_std=None,
                       test_loss=None, test_error=None, test_acc=None,
                       test_soft_ratio=None, test_score_mean=None, test_topk_mean=None, test_score_std=None):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            train_loss,
            train_error,
            train_acc,
            train_soft_ratio,
            train_score_mean,
            train_topk_mean,
            train_score_std,
            val_loss,
            val_error,
            val_acc,
            val_soft_ratio,
            val_score_mean,
            val_topk_mean,
            val_score_std,
            test_loss,
            test_error,
            test_acc,
            test_soft_ratio,
            test_score_mean,
            test_topk_mean,
            test_score_std,
        ])


def train(epoch):
    _set_dataset_epoch(train_loader.dataset, epoch)
    model.train()
    if (not args.freeze_backbone) and args.freeze_backbone_bn:
        _freeze_backbone_batchnorm(model)

    train_loss = 0.
    train_error = 0.
    burden_sums = _init_burden_sums()
    burden_count = 0
    train_pred_counts = np.zeros(args.num_classes, dtype=np.int64)
    optimizer.zero_grad(set_to_none=True)
    loader_iter = _iter_with_progress(
        enumerate(train_loader),
        total=len(train_loader),
        desc='Epoch {}/{} Train'.format(epoch, args.epochs)
    )
    for batch_idx, batch_data in loader_iter:
        data = batch_data[0]
        label = batch_data[1]
        pos_z = batch_data[2] if len(batch_data) > 2 else None
        bag_label = label[0] if isinstance(label, (list, tuple)) else label
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
            if pos_z is not None:
                pos_z = pos_z.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # calculate loss and metrics
        loss, error, predicted_label, _ = model.calculate_objective_and_classification_error(data, bag_label, pos_z=pos_z)
        aux_stats = model.get_latest_aux_metrics() if hasattr(model, 'get_latest_aux_metrics') else None
        if _merge_burden_stats(burden_sums, aux_stats):
            burden_count += 1
        train_loss += loss.item()
        train_error += error
        pred_idx = int(predicted_label.view(-1).detach().cpu().item())
        if 0 <= pred_idx < args.num_classes:
            train_pred_counts[pred_idx] += 1
        # backward pass
        (loss / float(args.grad_accum_steps)).backward()
        # step
        if ((batch_idx + 1) % args.grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if tqdm is not None and (batch_idx + 1) % 5 == 0:
            loader_iter.set_postfix({
                'loss': '{:.4f}'.format(train_loss / float(batch_idx + 1)),
                'err': '{:.4f}'.format(train_error / float(batch_idx + 1)),
                'soft_ratio': '{:.4f}'.format((burden_sums['soft_ratio'] / float(max(1, burden_count)))) if burden_count > 0 else 'NA',
            })

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    burden_means = _finalize_burden_means(burden_sums, burden_count)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))
    _format_burden_print('Train', burden_means)
    print('Train pred distribution: {}'.format(train_pred_counts.tolist()))
    current_lrs = [group['lr'] for group in optimizer.param_groups]
    print('Current learning rates: {}'.format([float(lr) for lr in current_lrs]))
    return train_loss, train_error, burden_means


def evaluate(data_loader, split_name='Eval', show_examples=False):
    model.eval()
    total_loss = 0.
    total_error = 0.
    burden_sums = _init_burden_sums()
    burden_count = 0
    all_true = []
    all_pred = []
    num_batches = len(data_loader)
    if num_batches == 0:
        print('\n{} Set is empty, skipped.'.format(split_name))
        return None, None, {k: None for k in _BURDEN_KEYS}
    
    with torch.no_grad():
        loader_iter = _iter_with_progress(
            enumerate(data_loader),
            total=num_batches,
            desc='{} Progress'.format(split_name)
        )
        for batch_idx, batch_data in loader_iter:
            data = batch_data[0]
            label = batch_data[1]
            pos_z = batch_data[2] if len(batch_data) > 2 else None
            bag_label = label[0] if isinstance(label, (list, tuple)) else label
            instance_labels = label[1] if isinstance(label, (list, tuple)) and len(label) > 1 else None
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
                if pos_z is not None:
                    pos_z = pos_z.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            loss, error, predicted_label, attention_weights = model.calculate_objective_and_classification_error(data, bag_label, pos_z=pos_z)
            aux_stats = model.get_latest_aux_metrics() if hasattr(model, 'get_latest_aux_metrics') else None
            if _merge_burden_stats(burden_sums, aux_stats):
                burden_count += 1
            total_loss += loss.item()
            total_error += error

            true_label = int(bag_label.view(-1).detach().cpu().item())
            pred_label = int(predicted_label.view(-1).detach().cpu().item())
            all_true.append(true_label)
            all_pred.append(pred_label)

            if tqdm is not None and (batch_idx + 1) % 5 == 0:
                loader_iter.set_postfix({
                    'loss': '{:.4f}'.format(total_loss / float(batch_idx + 1)),
                    'err': '{:.4f}'.format(total_error / float(batch_idx + 1)),
                    'soft_ratio': '{:.4f}'.format((burden_sums['soft_ratio'] / float(max(1, burden_count)))) if burden_count > 0 else 'NA',
                })

            if show_examples and batch_idx < 5:
                bag_level = (true_label, pred_label)

                if instance_labels is not None:
                    instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                        np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))
                    print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                          'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))
                else:
                    print('\nTrue Bag Label, Predicted Bag Label: {}'.format(bag_level))

    total_error /= num_batches
    total_loss /= num_batches
    burden_means = _finalize_burden_means(burden_sums, burden_count)

    if len(all_true) > 0:
        y_true = np.asarray(all_true, dtype=np.int64)
        y_pred = np.asarray(all_pred, dtype=np.int64)
        conf = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
        np.add.at(conf, (y_true, y_pred), 1)

        true_dist = np.bincount(y_true, minlength=args.num_classes)
        pred_dist = np.bincount(y_pred, minlength=args.num_classes)
        balanced_recalls = []
        macro_f1_terms = []
        for c in range(args.num_classes):
            tp = conf[c, c]
            fn = conf[c, :].sum() - tp
            fp = conf[:, c].sum() - tp
            recall = (tp / float(tp + fn)) if (tp + fn) > 0 else 0.0
            precision = (tp / float(tp + fp)) if (tp + fp) > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            balanced_recalls.append(recall)
            macro_f1_terms.append(f1)

        case_acc = float((y_true == y_pred).mean())
        balanced_acc = float(np.mean(balanced_recalls))
        macro_f1 = float(np.mean(macro_f1_terms))
        print('{} true distribution: {}'.format(split_name, true_dist.tolist()))
        print('{} pred distribution: {}'.format(split_name, pred_dist.tolist()))
        print('{} case_acc: {:.4f}, balanced_acc: {:.4f}, macro_f1: {:.4f}'.format(
            split_name, case_acc, balanced_acc, macro_f1
        ))
        print('{} confusion matrix (rows=true, cols=pred):\n{}'.format(split_name, conf))

    print('\n{} Set, Loss: {:.4f}, Error: {:.4f}'.format(split_name, total_loss, total_error))
    _format_burden_print(split_name, burden_means)
    return total_loss, total_error, burden_means


if __name__ == "__main__":
    # 按训练启动时间生成唯一 CSV 文件名，避免覆盖
    from datetime import datetime as _dt
    _csv_base, _csv_ext = os.path.splitext(args.metrics_csv)
    args.metrics_csv = '{}_{}{}'.format(_csv_base, _dt.now().strftime('%Y%m%d_%H%M%S'), _csv_ext or '.csv')
    init_metrics_csv(args.metrics_csv)
    best_val_loss = float('inf')
    best_epoch = None
    epochs_without_improve = 0
    early_stop_active = (val_loader is not None and args.early_stop_patience > 0)

    best_model_dir = os.path.dirname(args.best_model_path)
    if best_model_dir:
        os.makedirs(best_model_dir, exist_ok=True)

    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        if warmup_active and epoch == (args.warmup_epochs + 1):
            _set_backbone_trainable(model, trainable=True)
            if (not args.freeze_backbone) and args.freeze_backbone_bn:
                _freeze_backbone_batchnorm(model)
            optimizer = _build_optimizer(freeze_backbone_stage=False, stage_name='post-warmup')
            scheduler = _build_scheduler(optimizer)
            print('Warmup finished at epoch {}. Backbone is now unfrozen.'.format(epoch - 1))

        train_loss, train_error, train_burden = train(epoch)
        if val_loader is not None:
            val_loss, val_error, val_burden = evaluate(val_loader, split_name='Val', show_examples=False)
        else:
            val_loss, val_error, val_burden = None, None, {k: None for k in _BURDEN_KEYS}
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
            if scheduler is not None:
                scheduler.step(val_loss)
            improved = val_loss < (best_val_loss - args.early_stop_min_delta)
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improve = 0
                torch.save(model.state_dict(), args.best_model_path)
                print('Saved best model at epoch {} with val_loss {:.4f} -> {}'.format(
                    epoch, val_loss, args.best_model_path
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

    print('Start Testing')
    test_loss, test_error, test_burden = evaluate(test_loader, split_name='Test', show_examples=True)
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
