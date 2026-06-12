"""Pure-PyTorch building blocks for the wireframe VAE.

Trimmed from the original CLR-Wire ``modules.py``: the ``diffusers`` /
``x_transformers`` dependencies (and a large amount of unused MLP-Mixer / U-Net /
permutator / squeeze-excite code) were removed. What remains is exactly what the
wireframe VAE uses:

  * :func:`MLP` -- the small prediction-head MLP.
  * :class:`PointEmbed` (+ :class:`MPConv`) -- magnitude-preserving Fourier point
    embedding for the segment endpoints.
  * :func:`ce_loss` / :class:`FocalLoss` -- classification losses.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ====================================================================
# prediction-head MLP
# ====================================================================
def MLP(in_dim, out_dim=None, expansion_factor=4.0, dropout=0.0, dense=nn.Linear):
    out_dim = in_dim if out_dim is None else out_dim
    inner_dim = int(in_dim * expansion_factor)
    return nn.Sequential(
        dense(in_dim, inner_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        dense(inner_dim, out_dim),
    )


# ====================================================================
# magnitude-preserving Fourier point embedding
# ====================================================================
def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class MPConv(nn.Module):
    """Magnitude-preserving linear (forced weight normalization), kernel=[]."""

    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,))


class PointEmbed(nn.Module):
    """Magnitude-preserving Fourier embedding of ``(N, 3[+other])`` points."""

    def __init__(self, hidden_dim=48, dim=128, other_dim=0):
        super().__init__()
        assert hidden_dim % 6 == 0
        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x (hidden_dim/2)
        self.mlp = MPConv(self.embedding_dim + 3 + other_dim, dim, kernel=[])

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum('nd,de->ne', input, basis)
        return torch.cat([projections.sin(), projections.cos()], dim=1)

    def forward(self, input):
        if input.shape[1] != 3:
            input, others = input[:, :3], input[:, 3:]
        else:
            others = None
        if others is None:
            embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=1))
        else:
            embed = self.mlp(
                torch.cat([self.embed(input, self.basis), input, others], dim=1))
        return embed


# ====================================================================
# classification losses
# ====================================================================
def ce_loss(pred, target, label_smoothing=0.005, weight=None,
            reduction='none', num_classes=None):
    return F.cross_entropy(
        pred, target, reduction=reduction,
        label_smoothing=label_smoothing, weight=weight)


class FocalLoss(nn.Module):
    """Multi-class focal loss (drop-in for :func:`ce_loss`)."""

    def __init__(self, gamma=2, alpha=None, task_type='multi-class', num_classes=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.task_type = task_type
        self.num_classes = num_classes
        if (task_type == 'multi-class' and alpha is not None
                and isinstance(alpha, (list, torch.Tensor))):
            assert num_classes is not None, "num_classes must be specified"
            self.alpha = torch.Tensor(alpha) if isinstance(alpha, list) else alpha

    def forward(self, inputs, targets, num_classes=None, label_smoothing=0.0,
                weight=None, reduction='none'):
        return self.multi_class_focal_loss(
            inputs, targets, num_classes, label_smoothing, weight, reduction)

    def multi_class_focal_loss(self, inputs, targets, num_classes=None,
                               label_smoothing=0.0, weight=None, reduction='none'):
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
        if num_classes is None:
            num_classes = self.num_classes

        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        if targets_one_hot.ndim == 3:
            targets_one_hot = rearrange(targets_one_hot, 'b n c -> b c n')

        ce = F.cross_entropy(
            inputs, targets, reduction='none',
            label_smoothing=label_smoothing, weight=weight)

        p_t = torch.sum(probs * targets_one_hot, dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce = alpha_t.unsqueeze(1) * ce
        loss = focal_weight * ce

        if reduction == 'mean':
            return loss.mean()
        if reduction == 'sum':
            return loss.sum()
        return loss
