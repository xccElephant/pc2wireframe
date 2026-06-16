"""Lightning data module for the PC2Wireframe task.

Wires the wireframe datasets (:mod:`src.data.dataset`) into a
``pytorch_lightning.LightningDataModule`` that exposes train / val / predict
dataloaders, following the same structure as the reference
``Manual-Assembly`` data modules (``save_hyperparameters`` + ``setup`` +
``*_dataloader``).

Directory layout (relative to ``data_root``)::

    <data_root>/train/sample_edge/*.npz         (per-edge GT wireframes)
    <data_root>/train/sample_pointcloud/*.npz   (matching surface points)
    <data_root>/test/sample_pointcloud/*.npz    (inference-only point clouds)

Because the ``test`` split ships no ground-truth edges, the validation set is
carved out of the training edge files via a deterministic 9:1 split that is
computed **once** and saved to ``split_path`` (run ``scripts/make_split.py``).
Training then just reads that file instead of re-splitting on the fly.

Set ``use_val=False`` to train on *all* data (train + val combined) with no
held-out validation loader; ``use_val=True`` (default) trains on the saved
``train`` list and validates on ``val``. The ``predict`` dataloader serves the
point-cloud-only ``test`` split for submission generation.
"""
from __future__ import annotations

import os
from typing import Literal

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import (
    PointCloudDataset,
    WireframeGraphDataset,
    collate_rf_batch,
)


class WireframeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        # ----- paths -----
        data_root: str = "data",
        train_edge_subdir: str = "train/sample_edge",
        train_pointcloud_subdir: str = "train/sample_pointcloud",
        predict_pointcloud_subdir: str = "test/sample_pointcloud",
        split_path: str = "data/split.json",
        recursive_glob: bool = False,
        # ----- train/val split -----
        # use_val=True : train on the saved "train" split, validate on "val".
        # use_val=False: train on ALL data (train + val), no validation loader.
        use_val: bool = True,
        auto_build_split: bool = True,
        train_ratio: float = 0.9,
        split_seed: int = 42,
        # ----- graph / RF-target format -----
        vertex_merge_tol: float = 1e-4,
        # Caps <= 0 disable oversize skipping (RF-branch default: full data).
        max_vertices: int = 0,
        max_edges: int = 0,
        num_edge_points: int = 32,
        # Input point cloud is variable size (packed for PTv3); 0 = keep all.
        max_pc_points: int = 0,
        wf_num_points: int = 8192,
        min_edges: int = 1,
        max_load_retries: int = 64,
        # ----- data loader -----
        shuffle: bool = True,
        num_workers: int = 8,
        batch_size: int = 16,
        pin_memory: bool = True,
        prefetch_factor: int | None = 2,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["_class_path"]
        )  # HACK: https://github.com/Lightning-AI/pytorch-lightning/issues/20182

    # ------------------------------------------------------------------
    # path helpers
    # ------------------------------------------------------------------
    def _path(self, subdir: str) -> str:
        return os.path.join(self.hparams.data_root, subdir)

    @property
    def _pointcloud_dirs(self) -> list[str]:
        return [self._path(self.hparams.train_pointcloud_subdir)]

    def _make_graph_dataset(self, split: str) -> WireframeGraphDataset:
        return WireframeGraphDataset(
            split=split,
            split_path=self.hparams.split_path,
            edge_dir=self._path(self.hparams.train_edge_subdir),
            pointcloud_dirs=self._pointcloud_dirs,
            train_ratio=self.hparams.train_ratio,
            split_seed=self.hparams.split_seed,
            recursive_glob=self.hparams.recursive_glob,
            auto_build_split=self.hparams.auto_build_split,
            vertex_merge_tol=self.hparams.vertex_merge_tol,
            max_vertices=self.hparams.max_vertices,
            max_edges=self.hparams.max_edges,
            num_edge_points=self.hparams.num_edge_points,
            max_pc_points=self.hparams.max_pc_points,
            wf_num_points=self.hparams.wf_num_points,
            min_edges=self.hparams.min_edges,
            max_load_retries=self.hparams.max_load_retries,
        )

    # ------------------------------------------------------------------
    # lightning hooks
    # ------------------------------------------------------------------
    def setup(
        self,
        stage: Literal["fit", "validate", "test", "predict"] | None = None,
    ) -> None:
        use_val = self.hparams.use_val
        if stage in ("fit",) or stage is None:
            # use_val=False -> train on everything (train + val combined).
            self.train_dataset = self._make_graph_dataset(
                "train" if use_val else "all")
        if stage in ("fit", "validate") or stage is None:
            self.val_dataset = (
                self._make_graph_dataset("val") if use_val else None)
        if stage in ("predict", "test") or stage is None:
            self.predict_dataset = PointCloudDataset(
                pointcloud_dir=self._path(
                    self.hparams.predict_pointcloud_subdir),
                max_pc_points=self.hparams.max_pc_points,
                recursive_glob=self.hparams.recursive_glob,
            )

    # ------------------------------------------------------------------
    # dataloaders
    # ------------------------------------------------------------------
    def _loader_kwargs(self) -> dict:
        nw = self.hparams.num_workers
        return dict(
            num_workers=nw,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor if nw > 0 else None,
            persistent_workers=self.hparams.persistent_workers and nw > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            drop_last=True,
            collate_fn=collate_rf_batch,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader | None:
        # When use_val=False there is no held-out split; skip validation.
        if getattr(self, "val_dataset", None) is None:
            return None
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_rf_batch,
            **self._loader_kwargs(),
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_rf_batch,
            **self._loader_kwargs(),
        )

    # test reuses the predict (point-cloud-only) split
    def test_dataloader(self) -> DataLoader:
        return self.predict_dataloader()


__all__ = ["WireframeDataModule"]
