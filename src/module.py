"""LightningModules for the staged PC2Wireframe training.

The pipeline is trained in three independent stages, each with its own config
(``configs/{curve_vae,wireframe_vae,pc2wireframe}.yaml``) and its own module:

  1. :class:`CurveVAEModule` -- train the per-curve neural parametric VAE
     (``AutoencoderKL1D``) on canonicalised GT curves.
  2. :class:`WireframeVAEModule` -- train the wireframe VAE
     (``AutoencoderKLWireframe``) as an autoencoder over packed wireframes,
     with the **curve VAE frozen** (loaded from stage 1) to produce the
     per-curve latent targets.
  3. :class:`PC2WireframeModule` -- train the point-cloud encoder
     (PTv3 + latent compressor) to predict the latent, with the **wireframe
     VAE frozen** (loaded from stage 2) and the **curve VAE frozen** (stage 1),
     via latent regression toward the frozen teacher posterior + decode-through
     supervision of the (frozen) decoder heads.

All three share the ``AdamW`` + linear-warmup/cosine-decay schedule in
:class:`_BaseModule` and are driven through ``LightningCLI`` (see
``src/main.py``); the model class is selected per-stage via ``class_path``.
"""
from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .metrics import WireframeScore
from .models.packing import normalized_curves_from_batch
from .models.pc2wireframe import ClrWireframeBase, PC2WireframeModel


# ----------------------------------------------------------------------
# default sub-config builders (overridable from YAML)
# ----------------------------------------------------------------------
def _default_pc_encoder() -> dict[str, Any]:
    return dict(
        in_channels=3,
        grid_size=0.01,
        cls_mode=False,
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        stride=(2, 2, 2, 2),
        enable_flash=True,
        latent_num=64,
        latent_dim=64,
        compressor_heads=8,
        variational=True,
    )


def _default_wireframe_vae() -> dict[str, Any]:
    # latent_channels * wireframe_latent_num must equal the PC-encoder latent
    # (and stay <= 4096 floats).
    return dict(
        latent_channels=64,
        wireframe_latent_num=64,
        max_col_diff=6,
        max_row_diff=32,
        max_curves_num=512,
        attn_encoder_depth=4,
        attn_decoder_self_depth=12,
        attn_decoder_cross_depth=2,
        attn_dim=768,
        num_heads=12,
        curve_latent_embed_dim=256,
        use_mlp_predict=True,
        use_latent_pos_emb=True,
        input_is_curve_latent=True,
    )


def _default_curve_vae() -> dict[str, Any]:
    return dict(
        in_channels=3,
        out_channels=3,
        latent_channels=3,
        down_block_types=("DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D"),
        block_out_channels=(128, 256),
        layers_per_block=2,
        act_fn="silu",
        norm_num_groups=32,
        sample_points_num=16,
    )


# ----------------------------------------------------------------------
# checkpoint / freezing helpers
# ----------------------------------------------------------------------
def _load_submodule(
    dest: torch.nn.Module, ckpt_path: str, candidate_prefixes: list[str]
) -> None:
    """Load one submodule's weights out of a (Lightning) checkpoint.

    A previous-stage Lightning checkpoint stores everything under a module
    path (e.g. ``model.wireframe_vae.*`` or ``curve_vae.*``). We strip the
    first ``candidate_prefix`` that actually matches keys and load the rest
    into ``dest`` (non-strict, so extra/missing keys are reported not fatal).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    best: tuple[str, dict[str, torch.Tensor]] | None = None
    for pref in candidate_prefixes:
        p = (pref + ".") if pref else ""
        sub = {
            (k[len(p):] if p else k): v
            for k, v in sd.items()
            if k.startswith(p)
        }
        if sub and (best is None or len(sub) > len(best[1])):
            best = (pref, sub)
    if best is None:
        raise RuntimeError(
            f"No keys matching prefixes {candidate_prefixes!r} in {ckpt_path!r}"
        )
    missing, unexpected = dest.load_state_dict(best[1], strict=False)
    print(
        f"[load] {ckpt_path} prefix={best[0]!r} -> {dest.__class__.__name__} "
        f"(loaded={len(best[1])}, missing={len(missing)}, "
        f"unexpected={len(unexpected)})"
    )


def _freeze(module: torch.nn.Module) -> None:
    """Put ``module`` in eval mode and disable grads for all its params."""
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def _gt_wireframes(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-sample GT wireframes (numpy) for metric computation."""
    from .data.dataset import unbatch_wireframe_graphs

    graphs = unbatch_wireframe_graphs(batch)
    out: list[dict[str, Any]] = []
    for g in graphs:
        out.append({
            "vertices": g["vertices"].detach().cpu().numpy(),
            "edge_index": g["edge_index"].detach().cpu().numpy(),
            "edge_points": g["edge_points"].detach().cpu().numpy(),
        })
    return out


# ----------------------------------------------------------------------
# shared base: optimizer + frozen-module bookkeeping
# ----------------------------------------------------------------------
class _BaseModule(pl.LightningModule):
    """AdamW + linear-warmup/cosine-decay, optimising only trainable params."""

    def frozen_modules(self) -> list[torch.nn.Module]:
        """Submodules that must stay in eval mode even during ``.train()``."""
        return []

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        # Keep frozen sub-VAEs deterministic (no dropout / running-stat drift).
        for m in self.frozen_modules():
            m.eval()
        return self

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=tuple(self.hparams.betas),
        )

        warmup = max(1, int(self.hparams.warmup_steps))
        try:
            est = int(self.trainer.estimated_stepping_batches)
        except (RuntimeError, AttributeError):
            est = 0
        total = max(warmup + 1, est if est > 0 else int(self.hparams.max_steps))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        """Clip grads by global norm using the module's ``grad_clip`` hparam.

        This is the single source of truth for clipping; the trainer configs
        therefore leave ``gradient_clip_val`` unset. A ``grad_clip <= 0``
        disables clipping.
        """
        clip = float(getattr(self.hparams, "grad_clip", 0.0) or 0.0)
        if clip > 0.0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip,
                gradient_clip_algorithm="norm",
            )


# ----------------------------------------------------------------------
# stage 1: curve VAE
# ----------------------------------------------------------------------
class CurveVAEModule(_BaseModule):
    """Stage 1 -- train the per-curve VAE (``AutoencoderKL1D``) alone."""

    def __init__(
        self,
        curve_vae: dict[str, Any] | None = None,
        # ----- optimization -----
        lr: float = 1.0e-4,
        weight_decay: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 500,
        max_steps: int = 100_000,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        from .models.vae import AutoencoderKL1D

        self.curve_vae = AutoencoderKL1D(**(curve_vae or _default_curve_vae()))

    def _step(self, batch: dict[str, Any], stage: str):
        curves = normalized_curves_from_batch(batch)  # (Esum, U, 3) or None
        if curves is None or curves.shape[0] == 0:
            return None
        loss, parts = self.curve_vae(
            curves,
            sample_posterior=(stage == "train"),
            return_loss=True,
        )
        bs = int(curves.shape[0])
        self.log(f"{stage}/loss", loss, batch_size=bs, prog_bar=True, sync_dist=True)
        for key, value in parts.items():
            self.log(f"{stage}/{key}", value, batch_size=bs, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")



# ----------------------------------------------------------------------
# stage 2: wireframe VAE (curve VAE frozen)
# ----------------------------------------------------------------------
class WireframeVAEModule(_BaseModule):
    """Stage 2 -- train the wireframe VAE; the curve VAE is frozen."""

    def __init__(
        self,
        wireframe_vae: dict[str, Any] | None = None,
        curve_vae: dict[str, Any] | None = None,
        # ----- frozen curve VAE warm-start (stage-1 ckpt) -----
        curve_vae_ckpt: str | None = None,
        # ----- eval metric (CCD / TA / VPE -> weighted final score) -----
        eval_w_ccd: float = 0.3,
        eval_w_ta: float = 0.4,
        eval_w_vpe: float = 0.3,
        eval_ccd_tau: float = 0.1,
        eval_vpe_tau: float = 0.1,
        eval_match_thresh: float = 0.1,
        eval_num_per_edge: int = 32,
        # ----- optimization -----
        lr: float = 1.0e-4,
        weight_decay: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 500,
        max_steps: int = 100_000,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = ClrWireframeBase(
            wireframe_vae=wireframe_vae or _default_wireframe_vae(),
            curve_vae=curve_vae or _default_curve_vae(),
        )
        if curve_vae_ckpt:
            _load_submodule(
                self.model.curve_vae, curve_vae_ckpt,
                ["curve_vae", "model.curve_vae"],
            )
        _freeze(self.model.curve_vae)

        self.val_metrics = WireframeScore(
            w_ccd=eval_w_ccd, w_ta=eval_w_ta, w_vpe=eval_w_vpe,
            ccd_tau=eval_ccd_tau, vpe_tau=eval_vpe_tau,
            match_thresh=eval_match_thresh, num_per_edge=eval_num_per_edge,
        )

    def frozen_modules(self):
        return [self.model.curve_vae]

    def _vae_step(self, batch, stage):
        clr = self.model.graph_to_clr_inputs(batch)
        loss, parts = self.model.wireframe_vae(
            xs=clr["xs"],
            flag_diffs=clr["flag_diffs"],
            sample_posterior=(stage == "train"),
            return_loss=True,
        )
        bs = int(batch["num_graphs"])
        self.log(f"{stage}/loss", loss, batch_size=bs, prog_bar=True, sync_dist=True)
        for key, value in parts.items():
            self.log(f"{stage}/{key}", value, batch_size=bs, sync_dist=True)
        return loss, clr

    def training_step(self, batch, batch_idx):
        loss, _ = self._vae_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        from einops import rearrange

        _, clr = self._vae_step(batch, "val")
        # Reconstruction quality of the VAE itself: encode GT -> decode -> score.
        posterior = self.model.encode_target(clr["xs"], clr["flag_diffs"])
        z = rearrange(posterior.mode(), "b d n -> b n d")
        preds = self.model.decode_latent(z)
        preds_wf = self.model.reconstruct(
            preds, recon_curves=True, num_points=self.hparams.eval_num_per_edge
        )
        self.val_metrics.update(preds_wf, _gt_wireframes(batch))

    def on_validation_epoch_end(self) -> None:
        res = self.val_metrics.compute()
        self.log_dict(
            {f"val/{k}": v for k, v in res.items()},
            prog_bar=False, sync_dist=False,
        )
        self.log("val/score", res["score"], prog_bar=True, sync_dist=False)
        self.val_metrics.reset()


# ----------------------------------------------------------------------
# stage 3: full reconstruction (wireframe VAE + curve VAE frozen)
# ----------------------------------------------------------------------
class PC2WireframeModule(_BaseModule):
    """Stage 3 -- train point cloud -> latent; both VAEs are frozen.

    Two complementary signals to the (only trainable) point-cloud encoder:
      1. latent regression: pull the predicted latent toward the **frozen**
         CLR-Wire wireframe-VAE posterior of the GT wireframe (a fixed teacher,
         no moving target now that the VAE is frozen);
      2. decode-through: decode the predicted latent through the **frozen**
         decoder and supervise the wireframe heads against GT -- gradients flow
         back to the point-cloud encoder only.
    """

    def __init__(
        self,
        # ----- model config (nested, fully overridable from YAML) -----
        pc_encoder: dict[str, Any] | None = None,
        wireframe_vae: dict[str, Any] | None = None,
        curve_vae: dict[str, Any] | None = None,
        # ----- frozen decoder warm-start -----
        wireframe_vae_ckpt: str | None = None,  # stage-2 ckpt
        curve_vae_ckpt: str | None = None,      # stage-1 ckpt
        # ----- loss weights -----
        latent_recon_weight: float = 1.0,
        kl_weight: float = 1e-4,
        decode_through_weight: float = 1.0,
        # ----- eval metric (CCD / TA / VPE -> weighted final score) -----
        eval_w_ccd: float = 0.3,
        eval_w_ta: float = 0.4,
        eval_w_vpe: float = 0.3,
        eval_ccd_tau: float = 0.1,
        eval_vpe_tau: float = 0.1,
        eval_match_thresh: float = 0.1,
        eval_num_per_edge: int = 32,
        # ----- optimization -----
        lr: float = 1.5e-4,
        weight_decay: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 500,
        max_steps: int = 100_000,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = PC2WireframeModel(
            pc_encoder=pc_encoder or _default_pc_encoder(),
            wireframe_vae=wireframe_vae or _default_wireframe_vae(),
            curve_vae=curve_vae or _default_curve_vae(),
        )
        if wireframe_vae_ckpt:
            _load_submodule(
                self.model.wireframe_vae, wireframe_vae_ckpt,
                ["model.wireframe_vae", "wireframe_vae"],
            )
        if curve_vae_ckpt:
            _load_submodule(
                self.model.curve_vae, curve_vae_ckpt,
                ["curve_vae", "model.curve_vae"],
            )
        _freeze(self.model.wireframe_vae)
        _freeze(self.model.curve_vae)

        self.val_metrics = WireframeScore(
            w_ccd=eval_w_ccd, w_ta=eval_w_ta, w_vpe=eval_w_vpe,
            ccd_tau=eval_ccd_tau, vpe_tau=eval_vpe_tau,
            match_thresh=eval_match_thresh, num_per_edge=eval_num_per_edge,
        )

    def frozen_modules(self):
        return [self.model.wireframe_vae, self.model.curve_vae]

    # ------------------------------------------------------------------
    def forward(self, point_cloud: torch.Tensor, sample: bool = False):
        return self.model(point_cloud, sample=sample)

    def _compute_losses(
        self, batch: dict[str, Any], out: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        from einops import rearrange

        vae = self.model.wireframe_vae
        clr = self.model.graph_to_clr_inputs(batch)
        xs, flag_diffs = clr["xs"], clr["flag_diffs"]

        losses: dict[str, torch.Tensor] = {}
        total = xs.new_zeros(())

        # 1) latent regression toward the frozen teacher posterior mean.
        with torch.no_grad():
            posterior = self.model.encode_target(xs=xs, flag_diffs=flag_diffs)
            teacher_mu = rearrange(posterior.mode(), "b d n -> b n d")
        latent_recon = F.smooth_l1_loss(out["mu"], teacher_mu)
        losses["latent_recon"] = latent_recon
        total = total + self.hparams.latent_recon_weight * latent_recon

        # KL on the PC-encoder latent (only when variational).
        if out["logvar"] is not None:
            mu, logvar = out["mu"], out["logvar"]
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            losses["kl"] = kl
            total = total + self.hparams.kl_weight * kl

        # 2) decode-through: supervise the (frozen) decode of the PC latent.
        xs_mask = flag_diffs[..., 0] > 0.5
        (cls_ce, seg_mse, col_ce, row_ce, curve_mse) = vae.loss(
            gt_segment_coords=xs[..., :6],
            gt_flag_diffs=flag_diffs,
            gt_curve_latent=xs[..., 6:],
            xs_mask=xs_mask,
            preds=out["preds"],
        )
        decode_through = (
            vae.cls_loss_weight * cls_ce
            + vae.segment_loss_weight * seg_mse
            + vae.col_diff_loss_weight * col_ce
            + vae.row_diff_loss_weight * row_ce
            + vae.curve_latent_loss_weight * curve_mse
        )
        losses["decode_through"] = decode_through
        total = total + self.hparams.decode_through_weight * decode_through

        losses["total"] = total
        return losses

    def _step(self, batch, stage):
        point_cloud = batch["point_cloud"]
        out = self.model(point_cloud, sample=(stage == "train"))
        losses = self._compute_losses(batch, out)
        total = losses["total"]
        for key, value in losses.items():
            self.log(
                f"{stage}/{key}",
                value,
                batch_size=point_cloud.shape[0],
                sync_dist=True,
                prog_bar=(key == "total"),
            )
        return out, total

    def training_step(self, batch, batch_idx):
        _, total = self._step(batch, "train")
        return total

    def validation_step(self, batch, batch_idx):
        out, _ = self._step(batch, "val")
        preds_wf = self.model.reconstruct(
            out["preds"],
            recon_curves=True,
            num_points=self.hparams.eval_num_per_edge,
        )
        self.val_metrics.update(preds_wf, _gt_wireframes(batch))

    def on_validation_epoch_end(self) -> None:
        res = self.val_metrics.compute()
        self.log_dict(
            {f"val/{k}": v for k, v in res.items()},
            prog_bar=False, sync_dist=False,
        )
        self.log("val/score", res["score"], prog_bar=True, sync_dist=False)
        self.val_metrics.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """Per-shape latent + decoded wireframe for submission."""
        out = self.model(batch["point_cloud"], sample=False)
        wireframes = self.model.reconstruct(out["preds"], recon_curves=True)
        return {
            "shape_id": batch.get("shape_id"),
            "z": out["z"],
            "wireframes": wireframes,
        }


__all__ = ["CurveVAEModule", "WireframeVAEModule", "PC2WireframeModule"]
