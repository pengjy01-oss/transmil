"""Attention-based MIL model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import _corn_loss, _corn_label_from_logits, _prepare_targets
from models.backbone import build_feature_extractor
from models.components import (
    _build_instance_score_head,
    _build_position_embed_mlp,
    _compute_attention_stats,
    _compute_burden_features,
    _compute_coverage_features,
)
from models.aggregators import TransMILAggregator, NystromTransMILAggregator


class Attention(nn.Module):
    def __init__(self, in_channels=1, pretrained_backbone=False, num_classes=2, instance_batch_size=8,
                 freeze_backbone=True,
                 backbone_name='resnet18',
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
                 nystrom_num_landmarks=64,
                 score_logit_reg_weight=0.01,
                 corn_balanced=True,
                 instance_aux_loss_weight=0.0,
                 use_pseudo12_guidance=False,
                 pseudo12_lambda=0.1,
                 use_soft12_guidance=False,
                 soft12_lambda=0.1):
        super(Attention, self).__init__()
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

        self.feature_extractor, self.M = build_feature_extractor(
            backbone_name=backbone_name,
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
        if self.aggregator_type not in ('abmil', 'transmil', 'nystrom'):
            raise ValueError("aggregator must be 'abmil', 'transmil', or 'nystrom', got '{}'".format(aggregator))

        if self.aggregator_type == 'transmil':
            self.transmil_agg = TransMILAggregator(
                in_dim=self.attention_in_dim,
                num_heads=int(transmil_num_heads),
                num_layers=int(transmil_num_layers),
                dropout=float(transmil_dropout),
            )
        elif self.aggregator_type == 'nystrom':
            self.transmil_agg = NystromTransMILAggregator(
                in_dim=self.attention_in_dim,
                num_heads=int(transmil_num_heads),
                num_layers=int(transmil_num_layers),
                num_landmarks=int(nystrom_num_landmarks),
                dropout=float(transmil_dropout),
            )
        else:
            self.attention = nn.Sequential(
                nn.Linear(self.attention_in_dim, self.L),
                nn.Tanh(),
                nn.Linear(self.L, self.ATTENTION_BRANCHES)
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

        # ---- Pseudo-12 / Soft-12 auxiliary head ----
        # Plan B: use_pseudo12_guidance -> hard CE loss with int label
        # Plan C: use_soft12_guidance   -> soft KL loss with float [12] target
        self.use_pseudo12_guidance = bool(use_pseudo12_guidance)
        self.pseudo12_lambda = float(pseudo12_lambda)
        self.use_soft12_guidance = bool(use_soft12_guidance)
        self.soft12_lambda = float(soft12_lambda)
        if self.use_pseudo12_guidance or self.use_soft12_guidance:
            self.classifier_aux12 = nn.Sequential(
                nn.Linear(classifier_in_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(64, 12),
            )
        else:
            self.classifier_aux12 = None

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
        H = torch.cat(feature_chunks, dim=0)

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

        if self.aggregator_type in ('transmil', 'nystrom'):
            Z = self.transmil_agg(H_att)
            A = None  # Computed on demand via compute_transmil_attention()
        else:
            A = self.attention(H_att)
            A = torch.transpose(A, 1, 0)
            A = F.softmax(A, dim=1)
            Z = torch.mm(A, H_att)

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

        # Pseudo-12 auxiliary logits
        aux12_logits = None
        if self.use_pseudo12_guidance and self.classifier_aux12 is not None:
            aux12_logits = self.classifier_aux12(Z_final)

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
            'aux12_logits': aux12_logits,
        }

        if self.instance_aux_loss_weight > 0 and instance_scores is not None:
            k_aux = max(1, int(instance_scores.size(0) * 0.1))
            topk_scores = torch.topk(instance_scores, k=k_aux, largest=True, sorted=False).values
            self.last_forward_aux['instance_bag_score'] = topk_scores.mean()

        return logits, Y_hat, A

    def get_latest_aux_metrics(self):
        return self.last_forward_aux.get('metrics', None)

    def get_latest_aux_outputs(self):
        return self.last_forward_aux

    def compute_attention_weights(self, X, pos_z=None):
        """Compute per-instance attention weights [1, K]. Call after forward()."""
        if self.aggregator_type in ('transmil', 'nystrom'):
            # Need to re-extract H_att and run separate attention pass
            x = X.squeeze(0)
            feature_chunks = []
            step = self.instance_batch_size if self.instance_batch_size > 0 else x.size(0)
            with torch.no_grad():
                for i in range(0, x.size(0), step):
                    h_chunk = self.feature_extractor(x[i:i + step])
                    h_chunk = h_chunk.view(h_chunk.size(0), -1)
                    feature_chunks.append(h_chunk)
            H = torch.cat(feature_chunks, dim=0)
            k_instances = int(H.size(0))
            if pos_z is None:
                pos_z_vec = torch.linspace(0.0, 1.0, steps=k_instances, device=H.device, dtype=H.dtype)
            else:
                pos_z_vec = pos_z.to(device=H.device, dtype=H.dtype).view(-1)
                pos_z_vec = torch.clamp(pos_z_vec, 0.0, 1.0)
            if self.use_position_embedding:
                pos_emb = self.position_embed(pos_z_vec.unsqueeze(1))
                H_att = torch.cat([H, pos_emb], dim=1)
            else:
                H_att = H
            attn = self.transmil_agg.compute_attention_weights(H_att)
            return attn.unsqueeze(0).detach().cpu()  # [1, K]
        else:
            # ABMIL: attention was already computed in forward()
            A = self.last_forward_aux.get('attention_weights', None)
            if A is not None:
                return A.detach().cpu()
            return None

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
        ibs = self.last_forward_aux.get('instance_bag_score', None)
        if ibs is None or self.instance_aux_loss_weight <= 0:
            return 0.0
        bag_positive = (_prepare_targets(Y) > 0).float().squeeze()
        return self.instance_aux_loss_weight * F.binary_cross_entropy(
            ibs.clamp(1e-7, 1 - 1e-7), bag_positive)

    def _pseudo12_aux_loss(self, pseudo12_label):
        """Plan B: CE loss with hard integer 12-subtype label."""
        if not self.use_pseudo12_guidance or pseudo12_label is None:
            return 0.0
        aux12_logits = self.last_forward_aux.get('aux12_logits', None)
        if aux12_logits is None:
            return 0.0
        target = pseudo12_label.view(-1).long()
        if target.device != aux12_logits.device:
            target = target.to(aux12_logits.device)
        return self.pseudo12_lambda * F.cross_entropy(aux12_logits, target)

    def _soft12_aux_loss(self, soft12_target):
        """Plan C: KL divergence loss with soft [12] float target distribution.

        L_aux12 = soft12_lambda * KL(p12_target || softmax(logits12))
        where KL(P||Q) = sum_i P_i * (log P_i - log Q_i).
        F.kl_div(log_Q, P, reduction='batchmean') computes this correctly.
        """
        if not self.use_soft12_guidance or soft12_target is None:
            return 0.0
        aux12_logits = self.last_forward_aux.get('aux12_logits', None)
        if aux12_logits is None:
            return 0.0
        # soft12_target: [12] float tensor or [1,12]
        target = soft12_target.view(1, 12).float()
        if target.device != aux12_logits.device:
            target = target.to(aux12_logits.device)
        # Re-normalize after potential floating-point drift
        target = target.clamp(min=0.0)
        t_sum = target.sum()
        if t_sum > 1e-8:
            target = target / t_sum
        else:
            target = torch.ones_like(target) / 12.0
        log_pred = F.log_softmax(aux12_logits.view(1, 12), dim=-1)
        # F.kl_div(log_input, target) = sum(target * (log(target) - log_input))
        kl = F.kl_div(log_pred, target, reduction='batchmean')
        return self.soft12_lambda * kl

    def calculate_objective(self, X, Y, pos_z=None, pseudo12_label=None):
        logits, _, A = self.forward(X, pos_z=pos_z)
        neg_log_likelihood = _corn_loss(logits, Y, self.num_classes, balanced=self.corn_balanced)
        neg_log_likelihood = neg_log_likelihood + self._score_logit_reg()
        neg_log_likelihood = neg_log_likelihood + self._instance_aux_loss(Y)
        # Plan B hard CE
        neg_log_likelihood = neg_log_likelihood + self._pseudo12_aux_loss(pseudo12_label)
        # Plan C soft KL  (pseudo12_label doubles as soft12_target when Plan C is on)
        if self.use_soft12_guidance:
            neg_log_likelihood = neg_log_likelihood + self._soft12_aux_loss(pseudo12_label)
        return neg_log_likelihood, A

    def calculate_objective_and_classification_error(self, X, Y, pos_z=None, pseudo12_label=None):
        Y = _prepare_targets(Y)
        logits, Y_hat, A = self.forward(X, pos_z=pos_z)
        neg_log_likelihood = _corn_loss(logits, Y, self.num_classes, balanced=self.corn_balanced)
        neg_log_likelihood = neg_log_likelihood + self._score_logit_reg()
        neg_log_likelihood = neg_log_likelihood + self._instance_aux_loss(Y)
        # Plan B hard CE
        neg_log_likelihood = neg_log_likelihood + self._pseudo12_aux_loss(pseudo12_label)
        # Plan C soft KL  (pseudo12_label doubles as soft12_target when Plan C is on)
        if self.use_soft12_guidance:
            neg_log_likelihood = neg_log_likelihood + self._soft12_aux_loss(pseudo12_label)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return neg_log_likelihood, error, Y_hat, A
