"""Loss functions for ordinal classification (CORN)."""

import torch
import torch.nn.functional as F


def _prepare_targets(y):
    return y.view(-1).long()


def _corn_loss(logits, y, num_classes, balanced=False):
    """CORN ordinal loss: per-rank binary cross entropy."""
    y = _prepare_targets(y)
    total_loss = logits.new_zeros(1)
    total_count = 0

    for k in range(num_classes - 1):
        if k == 0:
            task_mask = torch.ones_like(y, dtype=torch.bool)
        else:
            task_mask = y > (k - 1)

        if task_mask.any():
            task_targets = (y[task_mask] > k).float()
            task_logits = logits[task_mask, k]
            pw = None
            if balanced:
                n_pos = task_targets.sum().clamp_min(1.0)
                n_neg = (task_targets.numel() - n_pos).clamp_min(1.0)
                pw = (n_neg / n_pos).detach()
            total_loss = total_loss + F.binary_cross_entropy_with_logits(
                task_logits,
                task_targets,
                pos_weight=pw,
                reduction='sum'
            )
            total_count += int(task_targets.numel())

    if total_count == 0:
        return logits.sum() * 0.0

    return total_loss / float(total_count)


def _corn_label_from_logits(logits):
    """Convert CORN logits to predicted ordinal labels."""
    probas = torch.sigmoid(logits)
    cumulative_probas = torch.cumprod(probas, dim=1)
    return torch.sum((cumulative_probas > 0.5).long(), dim=1)
