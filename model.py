"""Backward-compatible wrapper — re-exports all public symbols from new locations."""

from models.attention import Attention  # noqa: F401
from models.gated_attention import GatedAttention  # noqa: F401
from models.backbone import _build_resnet18_feature_extractor  # noqa: F401
from models.aggregators import TransMILAggregator  # noqa: F401
from models.components import (  # noqa: F401
    _build_instance_score_head,
    _build_position_embed_mlp,
    _compute_attention_stats,
    _compute_burden_features,
    _compute_coverage_features,
)
from losses import _prepare_targets, _corn_loss, _corn_label_from_logits  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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
    # attention_weights: [B, K] (B is attention branches in this model)
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
    # instance_scores: [K], pos_z: [K] in [0,1]
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
    active_bins_soft = torch.stack(bin_means, dim=0).sum() / float(bins)  # normalize to [0,1]

    coverage = torch.stack([z_center, z_spread, active_bins_soft], dim=0).unsqueeze(0)
    stats = {
        'z_center': float(z_center.detach().cpu().item()),
        'z_spread': float(z_spread.detach().cpu().item()),
        'active_bins_soft': float(active_bins_soft.detach().cpu().item()),
        'instance_score_mean': float(instance_scores.mean().detach().cpu().item()),
    }
    return coverage, stats


def _compute_burden_features(instance_scores, tau=0.5, temperature=0.1, topk_ratio=0.1):
    # instance_scores: [K] after sigmoid
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


def _prepare_targets(y):
    return y.view(-1).long()


def _corn_loss(logits, y, num_classes, balanced=False):
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
    # P(y > k) = product_{j<=k} sigmoid(logit_j)
    probas = torch.sigmoid(logits)
    cumulative_probas = torch.cumprod(probas, dim=1)
    return torch.sum((cumulative_probas > 0.5).long(), dim=1)


def _build_resnet18_feature_extractor(in_channels=1, pretrained=False):
    # Support both new and old torchvision APIs.
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
    except AttributeError:
        backbone = models.resnet18(pretrained=pretrained)

    if in_channels != 3:
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        with torch.no_grad():
            if pretrained:
                if in_channels == 1:
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                elif in_channels > 3:
                    new_conv.weight[:, :3].copy_(old_conv.weight)
                    new_conv.weight[:, 3:].copy_(old_conv.weight[:, :1].repeat(1, in_channels - 3, 1, 1))
                else:
                    new_conv.weight.copy_(old_conv.weight[:, :in_channels])
        backbone.conv1 = new_conv

    # Output shape after this module is Kx512x1x1.
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    return feature_extractor


class TransMILAggregator(nn.Module):
    """TransMIL-style aggregator: CLS token + Transformer self-attention.

    Replaces ABMIL attention pooling with self-attention interaction
    among instances, using a learnable [CLS] token to produce
    the bag-level representation.
    """

    def __init__(self, in_dim, num_heads=8, num_layers=2, dropout=0.1):
        super(TransMILAggregator, self).__init__()
        if in_dim % num_heads != 0:
            raise ValueError(
                'TransMILAggregator: in_dim ({}) must be divisible by num_heads ({})'.format(in_dim, num_heads))
        self.in_dim = in_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=in_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        """Aggregate instance features into a bag-level representation.

        Args:
            x: [K, in_dim] instance feature tokens (no batch dimension).
        Returns:
            z: [1, in_dim] bag-level representation (matches ABMIL Z shape).
        """
        x = x.unsqueeze(0)  # [1, K, in_dim]
        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # [1, 1, in_dim]
        x = torch.cat([cls_token, x], dim=1)  # [1, K+1, in_dim]
        x = self.encoder(x)  # [1, K+1, in_dim]
        z = self.norm(x[:, 0, :])  # [1, in_dim]
        return z


class Attention(nn.Module):
    def __init__(self, in_channels=1, pretrained_backbone=False, num_classes=2, instance_batch_size=8,
                 freeze_backbone=True,
                 use_burden_features=False,
                 use_position_embedding=False,
                 position_embed_dim=16,
                 use_coverage_features=False,
                 coverage_num_bins=6,
                 coverage_tau=0.5,
                 coverage_temperature=0.1,
                 coverage_eps=1e-6,
                 burden_score_hidden_dim=0,
                 burden_score_dropout=0.0,
                 burden_tau=0.5,
                 burden_temperature=0.1,
                 burden_topk_ratio=0.1,
                 aggregator='abmil',
                 transmil_num_heads=8,
                 transmil_num_layers=2,
                 transmil_dropout=0.1,
                 score_logit_reg_weight=0.01,
                 corn_balanced=True,
                 instance_aux_loss_weight=0.0):
        super(Attention, self).__init__()
        self.M = 512
        self.L = 128
        self.ATTENTION_BRANCHES = 1
        self.num_classes = num_classes
        self.instance_batch_size = int(instance_batch_size)
        self.score_logit_reg_weight = float(score_logit_reg_weight)
        self.corn_balanced = bool(corn_balanced)
        self.instance_aux_loss_weight = float(instance_aux_loss_weight)
        self.freeze_backbone = bool(freeze_backbone)
        self.use_burden_features = bool(use_burden_features)
        self.use_position_embedding = bool(use_position_embedding)
        self.position_embed_dim = int(position_embed_dim)
        self.use_coverage_features = bool(use_coverage_features)
        self.coverage_num_bins = int(coverage_num_bins)
        self.coverage_tau = float(coverage_tau)
        self.coverage_temperature = float(coverage_temperature)
        self.coverage_eps = float(coverage_eps)
        self.burden_tau = float(burden_tau)
        self.burden_temperature = float(burden_temperature)
        self.burden_topk_ratio = float(burden_topk_ratio)
        self.last_forward_aux = {}
        if self.num_classes < 2:
            raise ValueError('num_classes must be >= 2')
        if self.burden_temperature <= 0.0:
            raise ValueError('burden_temperature must be > 0')
        if self.burden_topk_ratio <= 0.0 or self.burden_topk_ratio > 1.0:
            raise ValueError('burden_topk_ratio must be in (0, 1]')
        if self.coverage_temperature <= 0.0:
            raise ValueError('coverage_temperature must be > 0')
        if self.coverage_num_bins <= 0:
            raise ValueError('coverage_num_bins must be > 0')
        if self.coverage_eps <= 0.0:
            raise ValueError('coverage_eps must be > 0')
        if self.use_position_embedding and self.position_embed_dim <= 0:
            raise ValueError('position_embed_dim must be > 0 when use_position_embedding=True')

        self.feature_extractor = _build_resnet18_feature_extractor(
            in_channels=in_channels,
            pretrained=pretrained_backbone
        )
        if self.freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        if self.use_position_embedding:
            self.position_embed = _build_position_embed_mlp(self.position_embed_dim)
            self.attention_in_dim = self.M + self.position_embed_dim
        else:
            self.position_embed = None
            self.attention_in_dim = self.M

        self.aggregator_type = str(aggregator).lower()
        if self.aggregator_type not in ('abmil', 'transmil'):
            raise ValueError("aggregator must be 'abmil' or 'transmil', got '{}'".format(aggregator))

        if self.aggregator_type == 'transmil':
            self.transmil_agg = TransMILAggregator(
                in_dim=self.attention_in_dim,
                num_heads=int(transmil_num_heads),
                num_layers=int(transmil_num_layers),
                dropout=float(transmil_dropout),
            )
        else:
            self.attention = nn.Sequential(
                nn.Linear(self.attention_in_dim, self.L), # matrix V
                nn.Tanh(),
                nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
            )

        score_head_in_dim = self.attention_in_dim
        if self.use_burden_features or self.use_coverage_features:
            self.instance_score_head = _build_instance_score_head(
                in_dim=score_head_in_dim,
                hidden_dim=burden_score_hidden_dim,
                dropout=burden_score_dropout,
            )
        else:
            self.instance_score_head = None

        classifier_in_dim = self.attention_in_dim * self.ATTENTION_BRANCHES
        if self.use_burden_features:
            classifier_in_dim += 4
        if self.use_coverage_features:
            classifier_in_dim += 3

        self.classifier = nn.Linear(classifier_in_dim, self.num_classes - 1)

    def forward(self, x, pos_z=None):
        x = x.squeeze(0)

        feature_chunks = []
        step = self.instance_batch_size if self.instance_batch_size > 0 else x.size(0)
        for i in range(0, x.size(0), step):
            x_chunk = x[i:i + step]
            if self.freeze_backbone:
                self.feature_extractor.eval()
                with torch.no_grad():
                    h_chunk = self.feature_extractor(x_chunk)
            else:
                h_chunk = self.feature_extractor(x_chunk)
            h_chunk = h_chunk.view(h_chunk.size(0), -1)
            feature_chunks.append(h_chunk)
        H = torch.cat(feature_chunks, dim=0)  # KxM

        k_instances = int(H.size(0))
        if pos_z is None:
            if k_instances > 1:
                pos_z_vec = torch.linspace(0.0, 1.0, steps=k_instances, device=H.device, dtype=H.dtype)
            else:
                pos_z_vec = torch.zeros((k_instances,), device=H.device, dtype=H.dtype)
        else:
            pos_z_vec = pos_z.to(device=H.device, dtype=H.dtype).view(-1)
            if int(pos_z_vec.numel()) != k_instances:
                if k_instances > 1:
                    pos_z_vec = torch.linspace(0.0, 1.0, steps=k_instances, device=H.device, dtype=H.dtype)
                else:
                    pos_z_vec = torch.zeros((k_instances,), device=H.device, dtype=H.dtype)
            pos_z_vec = torch.clamp(pos_z_vec, 0.0, 1.0)

        if self.use_position_embedding:
            pos_emb = self.position_embed(pos_z_vec.unsqueeze(1))
            H_att = torch.cat([H, pos_emb], dim=1)
        else:
            pos_emb = None
            H_att = H

        if self.aggregator_type == 'transmil':
            Z = self.transmil_agg(H_att)  # [1, attention_in_dim]
            A = None
        else:
            A = self.attention(H_att)  # KxATTENTION_BRANCHES
            A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
            A = F.softmax(A, dim=1)  # softmax over K
            Z = torch.mm(A, H_att)  # ATTENTION_BRANCHESx(attention_in_dim)

        burden_features = None
        coverage_features = None
        instance_scores = None
        burden_stats = None
        coverage_stats = None
        score_logits_raw = None
        if self.use_burden_features or self.use_coverage_features:
            score_logits_raw = self.instance_score_head(H_att).view(-1)
            instance_scores = torch.sigmoid(score_logits_raw)

        if self.use_burden_features:
            burden_features, burden_stats = _compute_burden_features(
                instance_scores,
                tau=self.burden_tau,
                temperature=self.burden_temperature,
                topk_ratio=self.burden_topk_ratio,
            )

        if self.use_coverage_features:
            coverage_features, coverage_stats = _compute_coverage_features(
                instance_scores,
                pos_z_vec,
                tau=self.coverage_tau,
                temperature=self.coverage_temperature,
                num_bins=self.coverage_num_bins,
                eps=self.coverage_eps,
            )

        final_parts = [Z]
        if burden_features is not None:
            final_parts.append(burden_features)
        if coverage_features is not None:
            final_parts.append(coverage_features)
        Z_final = torch.cat(final_parts, dim=1)

        logits = self.classifier(Z_final)
        Y_hat = _corn_label_from_logits(logits)

        attn_stats = _compute_attention_stats(A)
        metrics = dict(attn_stats)
        if burden_stats is not None:
            metrics.update(burden_stats)
        if coverage_stats is not None:
            metrics.update(coverage_stats)
        elif instance_scores is not None:
            metrics['instance_score_mean'] = float(instance_scores.mean().detach().cpu().item())

        self.last_forward_aux = {
            'attention_weights': A,
            'instance_scores': instance_scores,
            'score_logits': score_logits_raw,
            'pos_z': pos_z_vec,
            'position_embeddings': pos_emb,
            'burden_features': burden_features,
            'burden_stats': burden_stats,
            'coverage_features': coverage_features,
            'coverage_stats': coverage_stats,
            'metrics': metrics,
            'instance_bag_score': None,
        }

        # Auxiliary: bag-level prediction from instance scores for direct score head supervision.
        if self.instance_aux_loss_weight > 0 and instance_scores is not None:
            k_aux = max(1, int(instance_scores.size(0) * 0.1))
            topk_scores = torch.topk(instance_scores, k=k_aux, largest=True, sorted=False).values
            self.last_forward_aux['instance_bag_score'] = topk_scores.mean()

        return logits, Y_hat, A

    def get_latest_aux_metrics(self):
        return self.last_forward_aux.get('metrics', None)

    def get_latest_aux_outputs(self):
        return self.last_forward_aux

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y, pos_z=None):
        Y = _prepare_targets(Y)
        _, Y_hat, _ = self.forward(X, pos_z=pos_z)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def _score_logit_reg(self):
        sl = self.last_forward_aux.get('score_logits', None)
        if sl is None or self.score_logit_reg_weight <= 0.0:
            return 0.0
        return self.score_logit_reg_weight * sl.pow(2).mean()

    def _instance_aux_loss(self, Y):
        """Auxiliary loss: top-k mean instance score should predict bag positivity (label > 0)."""
        ibs = self.last_forward_aux.get('instance_bag_score', None)
        if ibs is None or self.instance_aux_loss_weight <= 0:
            return 0.0
        bag_positive = (_prepare_targets(Y) > 0).float().squeeze()
        return self.instance_aux_loss_weight * F.binary_cross_entropy(
            ibs.clamp(1e-7, 1 - 1e-7), bag_positive)

    def calculate_objective(self, X, Y, pos_z=None):
        logits, _, A = self.forward(X, pos_z=pos_z)
        neg_log_likelihood = _corn_loss(logits, Y, self.num_classes, balanced=self.corn_balanced)
        neg_log_likelihood = neg_log_likelihood + self._score_logit_reg()
        neg_log_likelihood = neg_log_likelihood + self._instance_aux_loss(Y)

        return neg_log_likelihood, A

    def calculate_objective_and_classification_error(self, X, Y, pos_z=None):
        Y = _prepare_targets(Y)
        logits, Y_hat, A = self.forward(X, pos_z=pos_z)
        neg_log_likelihood = _corn_loss(logits, Y, self.num_classes, balanced=self.corn_balanced)
        neg_log_likelihood = neg_log_likelihood + self._score_logit_reg()
        neg_log_likelihood = neg_log_likelihood + self._instance_aux_loss(Y)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return neg_log_likelihood, error, Y_hat, A

class GatedAttention(nn.Module):
    def __init__(self, in_channels=1, pretrained_backbone=False, num_classes=2, instance_batch_size=8,
                 freeze_backbone=True,
                 use_burden_features=False,
                 use_position_embedding=False,
                 position_embed_dim=16,
                 use_coverage_features=False,
                 coverage_num_bins=6,
                 coverage_tau=0.5,
                 coverage_temperature=0.1,
                 coverage_eps=1e-6,
                 burden_score_hidden_dim=0,
                 burden_score_dropout=0.0,
                 burden_tau=0.5,
                 burden_temperature=0.1,
                 burden_topk_ratio=0.1,
                 score_logit_reg_weight=0.01,
                 corn_balanced=True):
        super(GatedAttention, self).__init__()
        self.M = 512
        self.L = 128
        self.ATTENTION_BRANCHES = 1
        self.num_classes = num_classes
        self.instance_batch_size = int(instance_batch_size)
        self.score_logit_reg_weight = float(score_logit_reg_weight)
        self.corn_balanced = bool(corn_balanced)
        self.freeze_backbone = bool(freeze_backbone)
        self.use_burden_features = bool(use_burden_features)
        self.use_position_embedding = bool(use_position_embedding)
        self.position_embed_dim = int(position_embed_dim)
        self.use_coverage_features = bool(use_coverage_features)
        self.coverage_num_bins = int(coverage_num_bins)
        self.coverage_tau = float(coverage_tau)
        self.coverage_temperature = float(coverage_temperature)
        self.coverage_eps = float(coverage_eps)
        self.burden_tau = float(burden_tau)
        self.burden_temperature = float(burden_temperature)
        self.burden_topk_ratio = float(burden_topk_ratio)
        self.last_forward_aux = {}
        if self.num_classes < 2:
            raise ValueError('num_classes must be >= 2')
        if self.burden_temperature <= 0.0:
            raise ValueError('burden_temperature must be > 0')
        if self.burden_topk_ratio <= 0.0 or self.burden_topk_ratio > 1.0:
            raise ValueError('burden_topk_ratio must be in (0, 1]')
        if self.coverage_temperature <= 0.0:
            raise ValueError('coverage_temperature must be > 0')
        if self.coverage_num_bins <= 0:
            raise ValueError('coverage_num_bins must be > 0')
        if self.coverage_eps <= 0.0:
            raise ValueError('coverage_eps must be > 0')
        if self.use_position_embedding and self.position_embed_dim <= 0:
            raise ValueError('position_embed_dim must be > 0 when use_position_embedding=True')

        self.feature_extractor = _build_resnet18_feature_extractor(
            in_channels=in_channels,
            pretrained=pretrained_backbone
        )
        if self.freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        if self.use_position_embedding:
            self.position_embed = _build_position_embed_mlp(self.position_embed_dim)
            self.attention_in_dim = self.M + self.position_embed_dim
        else:
            self.position_embed = None
            self.attention_in_dim = self.M

        self.attention_V = nn.Sequential(
            nn.Linear(self.attention_in_dim, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.attention_in_dim, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        score_head_in_dim = self.attention_in_dim
        if self.use_burden_features or self.use_coverage_features:
            self.instance_score_head = _build_instance_score_head(
                in_dim=score_head_in_dim,
                hidden_dim=burden_score_hidden_dim,
                dropout=burden_score_dropout,
            )
        else:
            self.instance_score_head = None

        classifier_in_dim = self.attention_in_dim * self.ATTENTION_BRANCHES
        if self.use_burden_features:
            classifier_in_dim += 4
        if self.use_coverage_features:
            classifier_in_dim += 3

        self.classifier = nn.Linear(classifier_in_dim, self.num_classes - 1)

    def forward(self, x, pos_z=None):
        x = x.squeeze(0)

        feature_chunks = []
        step = self.instance_batch_size if self.instance_batch_size > 0 else x.size(0)
        for i in range(0, x.size(0), step):
            x_chunk = x[i:i + step]
            if self.freeze_backbone:
                self.feature_extractor.eval()
                with torch.no_grad():
                    h_chunk = self.feature_extractor(x_chunk)
            else:
                h_chunk = self.feature_extractor(x_chunk)
            h_chunk = h_chunk.view(h_chunk.size(0), -1)
            feature_chunks.append(h_chunk)
        H = torch.cat(feature_chunks, dim=0)  # KxM

        k_instances = int(H.size(0))
        if pos_z is None:
            if k_instances > 1:
                pos_z_vec = torch.linspace(0.0, 1.0, steps=k_instances, device=H.device, dtype=H.dtype)
            else:
                pos_z_vec = torch.zeros((k_instances,), device=H.device, dtype=H.dtype)
        else:
            pos_z_vec = pos_z.to(device=H.device, dtype=H.dtype).view(-1)
            if int(pos_z_vec.numel()) != k_instances:
                if k_instances > 1:
                    pos_z_vec = torch.linspace(0.0, 1.0, steps=k_instances, device=H.device, dtype=H.dtype)
                else:
                    pos_z_vec = torch.zeros((k_instances,), device=H.device, dtype=H.dtype)
            pos_z_vec = torch.clamp(pos_z_vec, 0.0, 1.0)

        if self.use_position_embedding:
            pos_emb = self.position_embed(pos_z_vec.unsqueeze(1))
            H_att = torch.cat([H, pos_emb], dim=1)
        else:
            pos_emb = None
            H_att = H

        A_V = self.attention_V(H_att)  # KxL
        A_U = self.attention_U(H_att)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H_att)  # ATTENTION_BRANCHESx(attention_in_dim)

        burden_features = None
        coverage_features = None
        instance_scores = None
        burden_stats = None
        coverage_stats = None
        score_logits_raw = None
        if self.use_burden_features or self.use_coverage_features:
            score_logits_raw = self.instance_score_head(H_att).view(-1)
            instance_scores = torch.sigmoid(score_logits_raw)

        if self.use_burden_features:
            burden_features, burden_stats = _compute_burden_features(
                instance_scores,
                tau=self.burden_tau,
                temperature=self.burden_temperature,
                topk_ratio=self.burden_topk_ratio,
            )

        if self.use_coverage_features:
            coverage_features, coverage_stats = _compute_coverage_features(
                instance_scores,
                pos_z_vec,
                tau=self.coverage_tau,
                temperature=self.coverage_temperature,
                num_bins=self.coverage_num_bins,
                eps=self.coverage_eps,
            )

        final_parts = [Z]
        if burden_features is not None:
            final_parts.append(burden_features)
        if coverage_features is not None:
            final_parts.append(coverage_features)
        Z_final = torch.cat(final_parts, dim=1)

        logits = self.classifier(Z_final)
        Y_hat = _corn_label_from_logits(logits)

        attn_stats = _compute_attention_stats(A)
        metrics = dict(attn_stats)
        if burden_stats is not None:
            metrics.update(burden_stats)
        if coverage_stats is not None:
            metrics.update(coverage_stats)
        elif instance_scores is not None:
            metrics['instance_score_mean'] = float(instance_scores.mean().detach().cpu().item())

        self.last_forward_aux = {
            'attention_weights': A,
            'instance_scores': instance_scores,
            'score_logits': score_logits_raw,
            'pos_z': pos_z_vec,
            'position_embeddings': pos_emb,
            'burden_features': burden_features,
            'burden_stats': burden_stats,
            'coverage_features': coverage_features,
            'coverage_stats': coverage_stats,
            'metrics': metrics,
        }

        return logits, Y_hat, A

    def get_latest_aux_metrics(self):
        return self.last_forward_aux.get('metrics', None)

    def get_latest_aux_outputs(self):
        return self.last_forward_aux

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y, pos_z=None):
        Y = _prepare_targets(Y)
        _, Y_hat, _ = self.forward(X, pos_z=pos_z)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def _score_logit_reg(self):
        sl = self.last_forward_aux.get('score_logits', None)
        if sl is None or self.score_logit_reg_weight <= 0.0:
            return 0.0
        return self.score_logit_reg_weight * sl.pow(2).mean()

    def calculate_objective(self, X, Y, pos_z=None):
        logits, _, A = self.forward(X, pos_z=pos_z)
        neg_log_likelihood = _corn_loss(logits, Y, self.num_classes, balanced=self.corn_balanced)
        neg_log_likelihood = neg_log_likelihood + self._score_logit_reg()

        return neg_log_likelihood, A

    def calculate_objective_and_classification_error(self, X, Y, pos_z=None):
        Y = _prepare_targets(Y)
        logits, Y_hat, A = self.forward(X, pos_z=pos_z)
        neg_log_likelihood = _corn_loss(logits, Y, self.num_classes, balanced=self.corn_balanced)
        neg_log_likelihood = neg_log_likelihood + self._score_logit_reg()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return neg_log_likelihood, error, Y_hat, A
