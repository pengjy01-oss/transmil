"""Shared helper modules and functions for MIL models."""

import torch
import torch.nn as nn


def _build_instance_score_head(in_dim, hidden_dim=0, dropout=0.0):
    hidden_dim = int(hidden_dim)
    dropout = float(dropout)
    if hidden_dim <= 0:
        return nn.Linear(in_dim, 1)

    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
        nn.Linear(hidden_dim, 1),
    )


def _build_position_embed_mlp(out_dim):
    out_dim = int(out_dim)
    if out_dim <= 0:
        raise ValueError('position_embed_dim must be > 0 when use_position_embedding=True')
    hidden = max(8, out_dim)
    return nn.Sequential(
        nn.Linear(1, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


def _compute_attention_stats(attention_weights, eps=1e-8):
    if attention_weights is None or attention_weights.numel() == 0:
        return {
            'attention_max_mean': 0.0,
            'attention_entropy': 0.0,
        }

    attn = attention_weights.clamp_min(float(eps))
    attn_max_mean = attention_weights.max(dim=1).values.mean()
    attn_entropy = (-(attn * torch.log(attn)).sum(dim=1)).mean()
    return {
        'attention_max_mean': float(attn_max_mean.detach().cpu().item()),
        'attention_entropy': float(attn_entropy.detach().cpu().item()),
    }


def _compute_coverage_features(instance_scores, pos_z, tau=0.5, temperature=0.1, num_bins=6, eps=1e-6):
    if instance_scores.ndim != 1:
        instance_scores = instance_scores.view(-1)
    if pos_z.ndim != 1:
        pos_z = pos_z.view(-1)

    k_total = int(instance_scores.numel())
    if k_total <= 0:
        z = instance_scores.new_zeros(1)[0]
        coverage = torch.stack([z, z, z], dim=0).unsqueeze(0)
        stats = {
            'z_center': 0.0,
            'z_spread': 0.0,
            'active_bins_soft': 0.0,
            'instance_score_mean': 0.0,
        }
        return coverage, stats

    safe_temp = max(float(temperature), 1e-6)
    safe_eps = max(float(eps), 1e-12)
    m = torch.sigmoid((instance_scores - float(tau)) / safe_temp)
    weight_sum = m.sum().clamp_min(safe_eps)

    z_center = (m * pos_z).sum() / weight_sum
    z_var = (m * (pos_z - z_center).pow(2)).sum() / weight_sum
    z_spread = torch.sqrt(z_var + safe_eps)

    bins = max(1, int(num_bins))
    bin_ids = torch.clamp((pos_z * bins).long(), min=0, max=bins - 1)
    bin_means = []
    for b in range(bins):
        bmask = (bin_ids == b)
        if bmask.any():
            bin_means.append(m[bmask].mean())
        else:
            bin_means.append(m.new_zeros(1)[0])
    active_bins_soft = torch.stack(bin_means, dim=0).sum() / float(bins)

    coverage = torch.stack([z_center, z_spread, active_bins_soft], dim=0).unsqueeze(0)
    stats = {
        'z_center': float(z_center.detach().cpu().item()),
        'z_spread': float(z_spread.detach().cpu().item()),
        'active_bins_soft': float(active_bins_soft.detach().cpu().item()),
        'instance_score_mean': float(instance_scores.mean().detach().cpu().item()),
    }
    return coverage, stats


def _compute_burden_features(instance_scores, tau=0.5, temperature=0.1, topk_ratio=0.1):
    if instance_scores.ndim != 1:
        instance_scores = instance_scores.view(-1)

    k_total = int(instance_scores.numel())
    if k_total <= 0:
        zero = instance_scores.new_zeros(1)
        b = torch.stack([zero[0], zero[0], zero[0], zero[0]], dim=0).unsqueeze(0)
        stats = {
            'soft_ratio': 0.0,
            'score_mean': 0.0,
            'topk_mean': 0.0,
            'score_std': 0.0,
        }
        return b, stats

    safe_temp = max(float(temperature), 1e-6)
    soft_mask = torch.sigmoid((instance_scores - float(tau)) / safe_temp)
    soft_ratio = soft_mask.mean()
    score_mean = instance_scores.mean()

    k = max(1, int(round(float(topk_ratio) * float(k_total))))
    k = min(k, k_total)
    topk_vals = torch.topk(instance_scores, k=k, largest=True, sorted=False).values
    topk_mean = topk_vals.mean()

    if k_total > 1:
        score_std = instance_scores.std(unbiased=False)
    else:
        score_std = instance_scores.new_zeros(1)[0]

    burden_features = torch.stack([soft_ratio, score_mean, topk_mean, score_std], dim=0).unsqueeze(0)
    stats = {
        'soft_ratio': float(soft_ratio.detach().cpu().item()),
        'score_mean': float(score_mean.detach().cpu().item()),
        'topk_mean': float(topk_mean.detach().cpu().item()),
        'score_std': float(score_std.detach().cpu().item()),
    }
    return burden_features, stats
