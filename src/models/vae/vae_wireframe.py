"""Wireframe VAE (pure PyTorch, no diffusers / x_transformers).

Set-to-set attention VAE derived from CLR-Wire (SIGGRAPH 2025). It autoencodes a
*set of curves* (each = endpoints + differential adjacency + a frozen curve-VAE
latent) into a fixed ``(wireframe_latent_num, latent_channels)`` latent and back.

  * **Encoder** -- per-curve tokens (Fourier endpoints + col/row-diff embeddings +
    curve latent) are cross-attended by ``wireframe_latent_num`` learnable queries
    (padding-masked), producing the Gaussian latent.
  * **Decoder** -- the latent tokens self-attend; ``1 + max_curves`` learnable
    queries then cross-attend to them. A ``cls`` token predicts the curve count;
    the remaining tokens predict per-curve endpoints / col+row-diff logits /
    curve-VAE latent via small MLP heads.

The diffusers ``ConfigMixin``/``ModelMixin`` wrappers and the
``DiagonalGaussianDistribution`` / ``AutoencoderKLOutput`` / ``DecoderOutput``
containers were replaced by a plain ``nn.Module`` and the lightweight
``GaussianLatent`` shared with the curve VAE. Attention uses native
``nn.Transformer`` blocks (see ``modules.SelfAttention`` / ``CrossAttention``).

Public surface used by ``packing.py`` / ``pc2wireframe.py`` / ``module.py``:
``encode(xs, flag_diffs) -> GaussianLatent`` (``.mode()`` is ``(B, C, N)``),
``decode(z) -> dec`` tokens, ``mlp_predict`` / ``linear_predict`` heads,
``loss(...)`` and ``forward(..., return_loss=True) -> (loss, parts)``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Module
from einops import rearrange, repeat, pack

from .modules import MLP, PointEmbed, FocalLoss, ce_loss, SelfAttention, CrossAttention
from .vae_curve import GaussianLatent


class EmbeddingLayerFactory:
    def __init__(
        self,
        point_embed_dim: int,
        max_col_diff: int,
        col_diff_embed_dim: int,
        max_row_diff: int,
        row_diff_embed_dim: int,
    ):
        self.point_embed_dim = point_embed_dim
        self.max_col_diff = max_col_diff
        self.col_diff_embed_dim = col_diff_embed_dim
        self.max_row_diff = max_row_diff
        self.row_diff_embed_dim = row_diff_embed_dim

    def create_embeddings(self):
        point_embed = PointEmbed(dim=self.point_embed_dim)
        col_diff_embed = nn.Embedding(self.max_col_diff, self.col_diff_embed_dim)
        row_diff_embed = nn.Embedding(self.max_row_diff, self.row_diff_embed_dim)
        return point_embed, col_diff_embed, row_diff_embed


class Encoder1D(Module):
    """Per-curve tokens -> ``wireframe_latent_num`` latent moments via cross-attn."""

    def __init__(
        self,
        out_channels=8,
        coor_embed_dim=128,
        max_col_diff=6,
        max_row_diff=32,
        col_diff_embed_dim=16,
        row_diff_embed_dim=32,
        max_curves_num=128,
        curve_latent_channels=12,
        curve_latent_embed_dim=128,
        attn_kwargs: dict = dict(dim=512, depth=4, heads=8),
        double_z=True,
        wireframe_latent_num=64,
    ):
        super().__init__()
        self.max_curves_num = max_curves_num
        self.wireframe_latent_num = wireframe_latent_num

        embedding_factory = EmbeddingLayerFactory(
            point_embed_dim=coor_embed_dim * 3,
            max_col_diff=max_col_diff,
            col_diff_embed_dim=col_diff_embed_dim,
            max_row_diff=max_row_diff,
            row_diff_embed_dim=row_diff_embed_dim,
        )
        (self.point_embed, self.col_diff_embed,
         self.row_diff_embed) = embedding_factory.create_embeddings()

        attn_dim = attn_kwargs['dim']

        self.latent_embed = nn.Linear(curve_latent_channels, curve_latent_embed_dim)
        self.enc_learnable_queries = nn.Parameter(
            torch.randn(wireframe_latent_num, attn_dim))
        self.pos_emb = nn.Parameter(torch.randn(max_curves_num, attn_dim))

        init_dim = (coor_embed_dim * 6 + col_diff_embed_dim
                    + row_diff_embed_dim + curve_latent_embed_dim)
        self.attn_project_in = nn.Linear(init_dim, attn_dim)

        self.cross_attn = CrossAttention(**attn_kwargs)

        out_channels = 2 * out_channels if double_z else out_channels
        self.project_out = nn.Linear(attn_dim, out_channels)

    def forward(self, *, xs, flag_diffs):
        bs = xs.shape[0]

        line_coords = xs[..., :6]  # (b, nl, 6)
        points = rearrange(line_coords, 'b nl (nlv d) -> b nl nlv d', nlv=2)
        points = rearrange(points, 'b nl nlv d -> (b nl nlv) d')

        line_coor_embed = self.point_embed(points)
        line_coor_embed = rearrange(
            line_coor_embed, '(b nl nlv) d -> b nl nlv d', b=bs, nlv=2)
        line_coor_embed = rearrange(line_coor_embed, 'b nl nlv d -> b nl (nlv d)')

        flag = flag_diffs[..., 0].unsqueeze(-1)  # (b, nl, 1)
        diffs = flag_diffs[..., 1:]              # (b, nl, 2)
        col_diff_embed = self.col_diff_embed(diffs[..., 0])
        row_diff_embed = self.row_diff_embed(diffs[..., 1])

        curve_latent = xs[..., 6:]  # (b, nl, curve_latent_channels)
        curve_latent_embed = self.latent_embed(curve_latent)

        wire_embed, _ = pack(
            [line_coor_embed, col_diff_embed, row_diff_embed, curve_latent_embed],
            'b nl *')
        wire_embed = self.attn_project_in(wire_embed)
        wire_embed = self.pos_emb + wire_embed

        enc_query = repeat(self.enc_learnable_queries, 'n d -> b n d', b=bs)
        # valid curves only (True == keep); pad slots are masked out.
        context_keep = rearrange(flag >= 0.5, 'b n c -> b (n c)')
        wireframe_latent_embed = self.cross_attn(
            enc_query, wire_embed, context_mask=context_keep)
        return self.project_out(wireframe_latent_embed)


class Decoder1D(Module):
    """Latent tokens -> ``1 + max_curves`` decoded tokens (self- then cross-attn)."""

    def __init__(
        self,
        *,
        in_channels: int = 8,
        attn_kwargs: dict = dict(dim=512, heads=8, self_depth=6, cross_depth=2),
        max_curves_num=128,
        wireframe_latent_num=64,
        use_latent_pos_emb: bool = False,
    ):
        super().__init__()
        self.max_curves_num = max_curves_num
        self.use_latent_pos_emb = use_latent_pos_emb

        attn_dim = attn_kwargs['dim']
        self.proj_in = nn.Linear(in_channels, attn_dim)
        if self.use_latent_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(wireframe_latent_num, attn_dim))
        self.dec_learnable_query = nn.Parameter(
            torch.randn(1 + max_curves_num, attn_dim))

        self.self_attn = SelfAttention(
            dim=attn_dim, heads=attn_kwargs['heads'], depth=attn_kwargs['self_depth'])
        self.cross_attn = CrossAttention(
            dim=attn_dim, heads=attn_kwargs['heads'], depth=attn_kwargs['cross_depth'])
        self.proj_out = nn.Linear(attn_dim, attn_dim)

    def forward(self, zs):
        bs = zs.shape[0]
        wireframe_latent = self.proj_in(zs)
        if self.use_latent_pos_emb:
            wireframe_latent = self.pos_emb + wireframe_latent

        wireframe_latent = self.self_attn(wireframe_latent)
        query_embed = repeat(self.dec_learnable_query, 'n d -> b n d', b=bs)
        query_embed = self.cross_attn(query_embed, wireframe_latent)
        return self.proj_out(query_embed)


class AutoencoderKLWireframe(nn.Module):
    """Set-to-set wireframe VAE (pure PyTorch)."""

    def __init__(
        self,
        latent_channels: int = 8,
        max_col_diff=6,
        max_row_diff=32,
        attn_encoder_depth: int = 4,
        attn_decoder_self_depth: int = 6,
        attn_decoder_cross_depth: int = 2,
        attn_dim: int = 512,
        num_heads: int = 8,
        max_curves_num: int = 128,
        wireframe_latent_num: int = 64,
        label_smoothing: float = 0.005,
        cls_loss_weight: float = 1.,
        segment_loss_weight: float = 1.,
        col_diff_loss_weight: float = 1.,
        row_diff_loss_weight: float = 1.,
        curve_latent_loss_weight: float = 1.,
        kl_loss_weight: float = 2e-4,
        curve_latent_embed_dim: int = 256,
        use_mlp_predict: bool = False,
        use_focal_loss: bool = False,
        use_latent_pos_emb: bool = False,
        input_is_curve_latent: bool = True,
        **kwargs,  # tolerate legacy keys (e.g. curve_vae_args)
    ):
        super().__init__()

        self.max_col_diff = max_col_diff
        self.max_row_diff = max_row_diff
        self.max_curves_num = max_curves_num
        self.input_is_curve_latent = input_is_curve_latent

        if not self.input_is_curve_latent:
            raise NotImplementedError(
                "input_is_curve_latent=False (encoding raw curves inside the "
                "wireframe VAE) is no longer supported; the staged pipeline "
                "always feeds precomputed curve latents."
            )

        self.encoder = Encoder1D(
            out_channels=latent_channels,
            attn_kwargs=dict(dim=attn_dim, heads=num_heads, depth=attn_encoder_depth),
            max_curves_num=max_curves_num,
            wireframe_latent_num=wireframe_latent_num,
            curve_latent_embed_dim=curve_latent_embed_dim,
        )
        self.decoder = Decoder1D(
            in_channels=latent_channels,
            attn_kwargs=dict(
                dim=attn_dim, heads=num_heads,
                self_depth=attn_decoder_self_depth,
                cross_depth=attn_decoder_cross_depth),
            max_curves_num=max_curves_num,
            wireframe_latent_num=wireframe_latent_num,
            use_latent_pos_emb=use_latent_pos_emb,
        )

        self.use_mlp_predict = use_mlp_predict
        if use_mlp_predict:
            dim = attn_dim
            self.predict_cls = MLP(in_dim=dim, out_dim=max_curves_num, expansion_factor=1.0, dropout=0.1)
            self.predict_diffs = MLP(in_dim=dim, out_dim=max_col_diff + max_row_diff, expansion_factor=1.0, dropout=0.1)
            self.predict_segments = MLP(in_dim=dim, out_dim=6, expansion_factor=1.0)
            self.predict_curve_latent = MLP(in_dim=dim, out_dim=12, expansion_factor=1.0)
        else:
            out_dim = 6 + 6 + 32 + 12  # segments + col_diffs + row_diffs + curve_latent
            self.predict_cls = nn.Linear(attn_dim, max_curves_num)
            self.predict_features = nn.Linear(attn_dim, out_dim)

        self.quant_proj = nn.Linear(2 * latent_channels, 2 * latent_channels)
        self.post_quant_proj = nn.Linear(latent_channels, latent_channels)

        # loss functions / weights
        self.mse_loss_fn = nn.MSELoss(reduction='none')
        self.ce_loss = FocalLoss(gamma=2) if use_focal_loss else ce_loss
        self.cls_loss_weight = cls_loss_weight
        self.segment_loss_weight = segment_loss_weight
        self.col_diff_loss_weight = col_diff_loss_weight
        self.row_diff_loss_weight = row_diff_loss_weight
        self.curve_latent_loss_weight = curve_latent_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.pad_id = -1
        self.label_smoothing = label_smoothing

        col_diff_class_weights = torch.exp(torch.linspace(-1, 1, self.max_col_diff))
        row_diff_class_weights = torch.exp(torch.linspace(-1, 1, self.max_row_diff))
        t = torch.linspace(0, 2, self.max_curves_num)
        col_weights = 1.2 - 0.2 * torch.log(t + 1.7183)
        self.register_buffer('col_diff_class_weights', col_diff_class_weights)
        self.register_buffer('row_diff_class_weights', row_diff_class_weights)
        self.register_buffer('col_weights', col_weights)

    # ------------------------------------------------------------------
    def encode(self, *, xs, flag_diffs) -> GaussianLatent:
        """GT wireframe ``(xs, flag_diffs)`` -> Gaussian posterior over ``(B, C, N)``."""
        h = self.encoder(xs=xs, flag_diffs=flag_diffs)  # (b, n, 2C)
        moments = self.quant_proj(h)
        moments = rearrange(moments, 'b n d -> b d n')  # (b, 2C, n)
        return GaussianLatent(moments)

    def decode(self, *, z) -> torch.Tensor:
        """Latent ``(B, C, N)`` -> decoded tokens ``(B, 1 + max_curves, attn_dim)``."""
        zs = rearrange(z, 'b d n -> b n d')
        zs = self.post_quant_proj(zs)
        return self.decoder(zs)

    # ------------------------------------------------------------------
    def linear_predict(self, dec):
        num_segments = 6
        num_diffs = num_segments + self.max_col_diff + self.max_row_diff
        pred_cls_logits = self.predict_cls(dec[:, 0])
        pred_features_logits = self.predict_features(dec[:, 1:])
        pred_segments = pred_features_logits[..., :num_segments]
        pred_diffs_logits = pred_features_logits[..., num_segments:num_diffs]
        pred_curve_latent = pred_features_logits[..., num_diffs:]
        return pred_cls_logits, pred_segments, pred_diffs_logits, pred_curve_latent

    def mlp_predict(self, dec):
        cls_token = dec[:, 0]
        features_tokens = dec[:, 1:]
        pred_cls_logits = self.predict_cls(cls_token)
        pred_segments = self.predict_segments(features_tokens)
        pred_diffs_logits = self.predict_diffs(features_tokens)
        pred_curve_latent = self.predict_curve_latent(features_tokens)
        return pred_cls_logits, pred_segments, pred_diffs_logits, pred_curve_latent

    # ------------------------------------------------------------------
    def loss(self, *, gt_segment_coords, gt_flag_diffs, gt_curve_latent, xs_mask, preds):
        pred_cls_logits = preds['cls']
        pred_segments = preds['segments']
        pred_diffs_logits = preds['diffs']
        pred_curve_latent_mu = preds['curve_latent']

        bs = pred_cls_logits.shape[0]
        cls = gt_flag_diffs[..., 0].sum(dim=-1) - 1
        diffs = gt_flag_diffs[..., 1:]

        # curve-count classification
        cls_ce_loss = self.ce_loss(
            pred_cls_logits, cls, reduction='mean',
            label_smoothing=self.label_smoothing, num_classes=self.max_curves_num)

        # endpoint regression (valid curves only)
        segment_mse_loss = self.mse_loss_fn(pred_segments, gt_segment_coords)
        line_mask = repeat(xs_mask, 'b nl -> b nl r', r=6)
        segment_mse_loss = segment_mse_loss[line_mask].mean()

        # col / row differential-adjacency classification
        rearranged_logits = rearrange(pred_diffs_logits, 'b ... c -> b c (...)')
        pred_col_diff_logits, pred_row_diff_logits = rearranged_logits.split(
            [self.max_col_diff, rearranged_logits.shape[1] - self.max_col_diff], dim=1)
        col_diff_ce_loss = self.ce_loss(
            pred_col_diff_logits, diffs[..., 0], num_classes=self.max_col_diff,
            label_smoothing=self.label_smoothing,
            weight=self.col_diff_class_weights, reduction='none')
        row_diff_ce_loss = self.ce_loss(
            pred_row_diff_logits, diffs[..., 1], num_classes=self.max_row_diff,
            label_smoothing=self.label_smoothing,
            weight=self.row_diff_class_weights, reduction='none')
        col_weights = repeat(self.col_weights, 'n -> b n', b=bs)
        col_diff_ce_loss = (col_diff_ce_loss * col_weights)[xs_mask].mean()
        row_diff_ce_loss = row_diff_ce_loss[xs_mask].mean()

        # per-curve latent regression (down-weight by GT posterior std)
        gt_curve_latent_std = torch.clamp(gt_curve_latent[..., 12:], 0., 1.)
        mu_weights = 1.2 - 0.5 * torch.log(gt_curve_latent_std + 1.7183)
        curve_latent_mask = repeat(xs_mask, 'b nl -> b nl r', r=12)
        curve_latent_loss = (
            mu_weights * self.mse_loss_fn(pred_curve_latent_mu, gt_curve_latent[..., :12])
        )[curve_latent_mask].mean()

        return cls_ce_loss, segment_mse_loss, col_diff_ce_loss, row_diff_ce_loss, curve_latent_loss

    # ------------------------------------------------------------------
    def forward(
        self,
        xs,                      # (b, nl, 6 + 2*curve_latent_dim)
        flag_diffs,              # (b, nl, 1 + 2)
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
        return_loss: bool = False,
        **kwargs,
    ):
        xs_mask = flag_diffs[..., 0] > 0.5
        # only [endpoints | curve mu] (6 + curve_latent_dim) feed the encoder.
        posterior = self.encode(xs=xs[..., :18], flag_diffs=flag_diffs)
        segment_coords = xs[..., :6]

        z = posterior.sample(generator) if sample_posterior else posterior.mode()
        dec = self.decode(z=z)

        if self.use_mlp_predict:
            cls, seg, diffs, curve_latent = self.mlp_predict(dec)
        else:
            cls, seg, diffs, curve_latent = self.linear_predict(dec)
        preds = {'cls': cls, 'segments': seg, 'diffs': diffs, 'curve_latent': curve_latent}

        if not return_loss:
            return preds

        kl_loss = posterior.kl().mean()
        if not sample_posterior:
            kl_loss = 0.0 * kl_loss

        (cls_ce_loss, segment_mse_loss, col_diff_ce_loss,
         row_diff_ce_loss, curve_latent_loss) = self.loss(
            gt_segment_coords=segment_coords,
            gt_flag_diffs=flag_diffs,
            gt_curve_latent=xs[..., 6:],
            xs_mask=xs_mask,
            preds=preds,
        )

        loss = (self.cls_loss_weight * cls_ce_loss
                + self.segment_loss_weight * segment_mse_loss
                + self.col_diff_loss_weight * col_diff_ce_loss
                + self.row_diff_loss_weight * row_diff_ce_loss
                + self.curve_latent_loss_weight * curve_latent_loss
                + self.kl_loss_weight * kl_loss)

        all_losses = dict(
            cls_ce_loss=cls_ce_loss,
            segment_mse_loss=segment_mse_loss,
            col_diff_ce_loss=col_diff_ce_loss,
            row_diff_ce_loss=row_diff_ce_loss,
            curve_latent_loss=curve_latent_loss,
            kl_loss=kl_loss,
            mu=posterior.mean.abs().mean().detach(),
            std=posterior.std.mean().detach(),
        )
        return loss, all_losses
