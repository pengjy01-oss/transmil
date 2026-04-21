"""Miscellaneous utilities: seed, device, progress bar, backbone freeze helpers."""
from __future__ import print_function

import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def iter_with_progress(iterable, total, desc):
    """Wrap an iterable with tqdm if available."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def setup_seed(seed, use_cuda=False):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def freeze_backbone_batchnorm(model):
    """Set backbone BatchNorm layers to eval mode and freeze their parameters."""
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


def set_dataset_epoch(dataset, epoch):
    """Propagate epoch number to a dataset that supports set_epoch()."""
    if hasattr(dataset, 'set_epoch'):
        dataset.set_epoch(epoch)
        return True

    base_dataset = getattr(dataset, 'dataset', None)
    if base_dataset is not None and hasattr(base_dataset, 'set_epoch'):
        base_dataset.set_epoch(epoch)
        return True

    return False


def set_backbone_trainable(model, trainable):
    """Toggle backbone parameter requires_grad."""
    if not hasattr(model, 'feature_extractor'):
        return
    for p in model.feature_extractor.parameters():
        p.requires_grad = bool(trainable)
