"""Evaluation loop for validation or test sets."""

from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable

from utils.misc import iter_with_progress

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


def evaluate(data_loader, model, args, split_name='Eval', show_examples=False):
    """Evaluate model on a data loader. Returns (total_loss, total_error, burden_means)."""
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
        return None, None, {k: None for k in _BURDEN_KEYS}, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    with torch.no_grad():
        loader_iter = iter_with_progress(
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
            loss, error, predicted_label, attention_weights = model.calculate_objective_and_classification_error(
                data, bag_label, pos_z=pos_z, pseudo12_label=p12_label)
            aux_stats = model.get_latest_aux_metrics() if hasattr(model, 'get_latest_aux_metrics') else None
            if _merge_burden_stats(burden_sums, aux_stats):
                burden_count += 1
            total_loss += loss.item()
            total_error += error

            true_label = int(bag_label.view(-1).detach().cpu().item())
            pred_label = int(predicted_label.view(-1).detach().cpu().item())
            all_true.append(true_label)
            all_pred.append(pred_label)

            if tqdm is not None and hasattr(loader_iter, 'set_postfix') and (batch_idx + 1) % 5 == 0:
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
    _format_burden_print(split_name, burden_means, use_burden_features=args.use_burden_features)

    extra_metrics = {'macro_f1': macro_f1 if len(all_true) > 0 else 0.0}
    return total_loss, total_error, burden_means, np.asarray(all_true, dtype=np.int64), np.asarray(all_pred, dtype=np.int64), extra_metrics
