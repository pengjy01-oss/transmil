"""Training loop for one epoch."""

from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable

from utils.misc import freeze_backbone_batchnorm, iter_with_progress, set_dataset_epoch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

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


def _format_burden_print(prefix, burden_means, use_burden_features=False):
    if burden_means is None or burden_means.get('soft_ratio') is None:
        print('{} burden stats: N/A (use_burden_features={})'.format(prefix, use_burden_features))
        return
    print('{} burden stats -> soft_ratio: {:.4f}, score_mean: {:.4f}, topk_mean: {:.4f}, score_std: {:.4f}'.format(
        prefix,
        burden_means['soft_ratio'],
        burden_means['score_mean'],
        burden_means['topk_mean'],
        burden_means['score_std'],
    ))


def train_one_epoch(epoch, model, train_loader, optimizer, args):
    """Run one training epoch. Returns (train_loss, train_error, burden_means)."""
    set_dataset_epoch(train_loader.dataset, epoch)
    model.train()
    if (not args.freeze_backbone) and args.freeze_backbone_bn:
        freeze_backbone_batchnorm(model)

    train_loss = 0.
    train_error = 0.
    burden_sums = _init_burden_sums()
    burden_count = 0
    train_pred_counts = np.zeros(args.num_classes, dtype=np.int64)
    aux12_loss_sum = 0.0
    optimizer.zero_grad(set_to_none=True)
    loader_iter = iter_with_progress(
        enumerate(train_loader),
        total=len(train_loader),
        desc='Epoch {}/{} Train'.format(epoch, args.epochs)
    )
    for batch_idx, batch_data in loader_iter:
        data = batch_data[0]
        label = batch_data[1]
        pos_z = batch_data[2] if len(batch_data) > 2 else None
        bag_label = label[0] if isinstance(label, (list, tuple)) else label

        # Pseudo-12 label (4th element if present)
        p12_label = None
        if len(batch_data) > 3:
            p12_label = batch_data[3]

        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
            if pos_z is not None:
                pos_z = pos_z.cuda()
            if p12_label is not None:
                p12_label = p12_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        loss, error, predicted_label, _ = model.calculate_objective_and_classification_error(
            data, bag_label, pos_z=pos_z, pseudo12_label=p12_label)
        aux_stats = model.get_latest_aux_metrics() if hasattr(model, 'get_latest_aux_metrics') else None
        if _merge_burden_stats(burden_sums, aux_stats):
            burden_count += 1
        train_loss += loss.item()
        train_error += error
        pred_idx = int(predicted_label.view(-1).detach().cpu().item())
        if 0 <= pred_idx < args.num_classes:
            train_pred_counts[pred_idx] += 1

        # Track aux12 loss separately (Plan B CE or Plan C KL)
        _use_b = hasattr(model, 'use_pseudo12_guidance') and model.use_pseudo12_guidance
        _use_c = hasattr(model, 'use_soft12_guidance') and model.use_soft12_guidance
        if (_use_b or _use_c) and p12_label is not None:
            _a12 = model.last_forward_aux.get('aux12_logits', None)
            if _a12 is not None:
                if _use_c:
                    # Plan C: unweighted KL
                    _tgt = p12_label.view(1, 12).float().to(_a12.device)
                    _tgt = _tgt.clamp(min=0.0)
                    _s = _tgt.sum()
                    if _s > 1e-8:
                        _tgt = _tgt / _s
                    _lp = torch.nn.functional.log_softmax(_a12.view(1, 12), dim=-1)
                    aux12_loss_sum += float(torch.nn.functional.kl_div(_lp, _tgt, reduction='batchmean').detach().cpu().item())
                else:
                    # Plan B: unweighted CE
                    _t = p12_label.view(-1).long().to(_a12.device)
                    aux12_loss_sum += float(torch.nn.functional.cross_entropy(_a12, _t).detach().cpu().item())

        (loss / float(args.grad_accum_steps)).backward()

        if ((batch_idx + 1) % args.grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if tqdm is not None and hasattr(loader_iter, 'set_postfix') and (batch_idx + 1) % 5 == 0:
            loader_iter.set_postfix({
                'loss': '{:.4f}'.format(train_loss / float(batch_idx + 1)),
                'err': '{:.4f}'.format(train_error / float(batch_idx + 1)),
                'soft_ratio': '{:.4f}'.format((burden_sums['soft_ratio'] / float(max(1, burden_count)))) if burden_count > 0 else 'NA',
            })

    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    burden_means = _finalize_burden_means(burden_sums, burden_count)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))
    _format_burden_print('Train', burden_means, use_burden_features=args.use_burden_features)
    if hasattr(model, 'use_pseudo12_guidance') and model.use_pseudo12_guidance:
        avg_aux12 = aux12_loss_sum / max(1, len(train_loader))
        print('Train aux12_loss (CE unweighted, Plan B): {:.4f}'.format(avg_aux12))
    if hasattr(model, 'use_soft12_guidance') and model.use_soft12_guidance:
        avg_aux12 = aux12_loss_sum / max(1, len(train_loader))
        print('Train aux12_loss (KL unweighted, Plan C): {:.4f}'.format(avg_aux12))
    print('Train pred distribution: {}'.format(train_pred_counts.tolist()))
    current_lrs = [group['lr'] for group in optimizer.param_groups]
    print('Current learning rates: {}'.format([float(lr) for lr in current_lrs]))
    return train_loss, train_error, burden_means
