"""Grad-CAM implementation for 2.5D MIL slabs.

Computes a spatial activation map for a single slab image, showing
which spatial regions the model considers most important for predicting
that slab as abnormal.

Strategy
--------
For each slab, we run it through the backbone (up to the last conv layer,
layer4 for ResNet18), record spatial feature maps and their gradients
w.r.t. the instance abnormal score (or ABMIL attention score if no
score head is available), then produce a class activation map.

This gives a MIL-native interpretation: the CAM shows which pixels
drive each slab's contribution to the bag-level prediction.
"""

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── Target layer resolution ───────────────────────────────────────────────────

def _get_gradcam_target_layer(model):
    """Return the last convolutional block (before global average pool).

    For ResNet18/34: feature_extractor is Sequential of 9 modules:
      0 conv1, 1 bn1, 2 relu, 3 maxpool, 4 layer1, 5 layer2, 6 layer3,
      7 layer4, 8 avgpool
    → target = feature_extractor[7]

    For DenseNet121: the last dense block is inside features, then
    an ReLU + AdaptiveAvgPool2d. We hook the features module.
    """
    fe = model.feature_extractor
    children = list(fe.children())

    if len(children) >= 9:
        # ResNet-style: layer4 is index 7
        return children[7]
    elif len(children) >= 3:
        # DenseNet-style: features is index 0
        return children[0]
    else:
        # Fallback: second-to-last child
        return children[-2] if len(children) >= 2 else children[-1]


def _build_backbone_up_to_target(model, target_layer):
    """Return (pre_target_net, target_mod, post_target_net) from feature_extractor."""
    children = list(model.feature_extractor.children())
    target_idx = None
    for i, c in enumerate(children):
        if c is target_layer:
            target_idx = i
            break
    if target_idx is None:
        target_idx = len(children) - 2  # fallback

    pre = nn.Sequential(*children[:target_idx])
    tgt = children[target_idx]
    post = nn.Sequential(*children[target_idx + 1:])
    return pre, tgt, post


# ── Core Grad-CAM ─────────────────────────────────────────────────────────────

class SlabGradCAM:
    """Compute Grad-CAM for individual slabs using instance abnormal score.

    The gradient source is:
      1. instance_score_head output (if model has one)  ← preferred
      2. ABMIL attention raw logit                      ← fallback
      3. Feature L2-norm                                ← last resort

    Parameters
    ----------
    model : Attention model (with .feature_extractor, .instance_score_head, etc.)
    """

    def __init__(self, model):
        self.model = model
        self.target_layer = _get_gradcam_target_layer(model)
        self._pre, self._tgt, self._post = _build_backbone_up_to_target(model, self.target_layer)

        self._activations = None
        self._gradients = None

        self._fwd_hook = self._tgt.register_forward_hook(self._save_activations)
        self._bwd_hook = self._tgt.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __del__(self):
        try:
            self._fwd_hook.remove()
            self._bwd_hook.remove()
        except Exception:
            pass

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    @torch.enable_grad()
    def compute(self, slab_tensor, pos_z_scalar=0.5):
        """Compute Grad-CAM for one slab.

        Parameters
        ----------
        slab_tensor : [C, H, W] float tensor (on CPU)
        pos_z_scalar : float  normalized z position [0, 1]

        Returns
        -------
        cam : [H_orig, W_orig] numpy array in [0, 1]
        """
        model = self.model
        device = next(model.parameters()).device

        slab = slab_tensor.unsqueeze(0).to(device).float()  # [1, C, H, W]
        pz = torch.tensor([[pos_z_scalar]], device=device, dtype=torch.float32)

        model.eval()
        model.zero_grad()
        self._activations = None
        self._gradients = None

        # ── Forward through backbone (with grad enabled) ─────────────────────
        x = self._pre(slab)                     # [1, *, h1, w1]
        x = self._tgt(x)                        # [1, C_feat, h, w]   ← hook fires
        feat_spatial = x                        # keep for activation
        feat_pooled = self._post(feat_spatial)  # [1, M, 1, 1] or [1, M]
        if feat_pooled.dim() > 2:
            feat = feat_pooled.view(1, -1)
        else:
            feat = feat_pooled

        # Position embedding
        if model.use_position_embedding and model.position_embed is not None:
            pe = model.position_embed(pz)
            feat_att = torch.cat([feat, pe], dim=1)
        else:
            feat_att = feat

        # ── Compute score to differentiate ──────────────────────────────────
        if model.instance_score_head is not None:
            score = model.instance_score_head(feat_att)
            score = torch.sigmoid(score).squeeze()
            target = score
        elif hasattr(model, 'attention'):
            # ABMIL: use raw attention logit as proxy
            raw = model.attention(feat_att)   # [1, 1]
            target = raw.squeeze()
        else:
            # TransMIL fallback: feature norm
            target = feat_att.norm()

        target.backward()

        acts = self._activations[0]   # [C_feat, h, w]
        grads = self._gradients[0]    # [C_feat, h, w]

        # GAP over spatial dims
        weights = grads.mean(dim=(-2, -1))  # [C_feat]

        # Weighted combination
        cam = (weights[:, None, None] * acts).sum(0)   # [h, w]
        cam = F.relu(cam)

        # Upsample to original slab size
        h_orig, w_orig = slab_tensor.shape[-2], slab_tensor.shape[-1]
        cam_up = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(h_orig, w_orig),
            mode='bilinear',
            align_corners=False
        ).squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        cmin, cmax = cam_up.min(), cam_up.max()
        if cmax > cmin:
            cam_up = (cam_up - cmin) / (cmax - cmin)
        else:
            cam_up = np.zeros_like(cam_up)

        return cam_up


# ── Overlay helpers ───────────────────────────────────────────────────────────

def _slab_to_rgb(slab_tensor, channel_idx=1):
    """Convert a [C, H, W] float slab to an [H, W, 3] uint8 image.

    Uses the middle channel (index channel_idx) as the grayscale image.
    """
    if slab_tensor.shape[0] > channel_idx:
        img = slab_tensor[channel_idx].numpy()
    else:
        img = slab_tensor[0].numpy()
    img = np.clip(img, 0.0, 1.0)
    gray = (img * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def save_gradcam_figure(slab_tensor, cam_np, save_path, slab_title='',
                        alpha=0.5, colormap='jet'):
    """Save a 3-panel figure: original | heatmap | overlay.

    Parameters
    ----------
    slab_tensor : [C, H, W] float tensor
    cam_np      : [H, W] float array in [0, 1]
    save_path   : str
    slab_title  : str
    alpha       : float  blending alpha for overlay
    colormap    : str    matplotlib colormap name
    """
    orig_rgb = _slab_to_rgb(slab_tensor)

    cmap_fn = cm.get_cmap(colormap)
    heat_rgba = cmap_fn(cam_np)
    heat_rgb = (heat_rgba[:, :, :3] * 255).astype(np.uint8)

    # Overlay
    overlay = (orig_rgb.astype(np.float32) * (1 - alpha) +
               heat_rgb.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_rgb, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    im = axes[1].imshow(cam_np, cmap=colormap, vmin=0, vmax=1)
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    if slab_title:
        fig.suptitle(slab_title, fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def run_gradcam_for_case(case_dict, model, save_dir, top_k=5, colormap='jet',
                         device=None):
    """Compute and save Grad-CAM for the top-K slabs of one case.

    Parameters
    ----------
    case_dict : dict from inference_engine.run_inference
    model     : Attention model
    save_dir  : str  directory to save .png files
    top_k     : int  number of top slabs to visualize
    colormap  : str
    device    : torch device (inferred from model if None)

    Returns
    -------
    cam_results : list[dict]  {slab_idx, cam, slab_img, importance}
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = next(model.parameters()).device

    bag_tensor = case_dict['bag_tensor']   # [K, C, H, W]
    importance = case_dict['slab_importance']
    metadata = case_dict['metadata']
    pos_z_np = case_dict['pos_z']
    case_id = case_dict['case_id']
    true_label = case_dict['true_label']
    pred_label = case_dict['pred_label']
    K = bag_tensor.shape[0]

    # Top-K slab indices
    k = min(top_k, K)
    top_indices = np.argsort(importance)[::-1][:k].tolist()

    gradcam = SlabGradCAM(model)
    cam_results = []

    for rank, slab_idx in enumerate(top_indices):
        slab = bag_tensor[slab_idx]           # [C, H, W]
        pz = float(pos_z_np[slab_idx]) if slab_idx < len(pos_z_np) else 0.5

        try:
            cam = gradcam.compute(slab, pos_z_scalar=pz)
        except Exception as e:
            print('    [GradCAM] slab {} failed: {}'.format(slab_idx, e))
            cam = np.zeros((slab.shape[-2], slab.shape[-1]), dtype=np.float32)

        region = metadata[slab_idx].get('region', 'unk') if slab_idx < len(metadata) else 'unk'
        center_z = metadata[slab_idx].get('center_z', -1) if slab_idx < len(metadata) else -1

        title = '{} | true={} pred={} | rank{} slab{} {} z={}'.format(
            case_id, true_label, pred_label, rank + 1, slab_idx, region, center_z)

        fname = 'gradcam_rank{:02d}_slab{:04d}_{}.png'.format(rank + 1, slab_idx, region)
        save_gradcam_figure(slab, cam, os.path.join(save_dir, fname),
                            slab_title=title, colormap=colormap)

        cam_results.append({
            'slab_idx': slab_idx,
            'rank': rank + 1,
            'cam': cam,
            'slab_tensor': slab,
            'importance': float(importance[slab_idx]),
            'region': region,
            'center_z': center_z,
        })

    gradcam.remove_hooks()
    return cam_results
