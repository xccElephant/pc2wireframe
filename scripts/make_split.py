#!/usr/bin/env python3
"""Build and save the train/val split for the PC2Wireframe task.

Globs the training edge NPZ files, performs a deterministic 9:1 (by default)
train/val split, and writes the result to a JSON file so training can reuse the
exact same split across runs / processes / DDP ranks without re-shuffling.

Usage (from the project root)::

    python scripts/make_split.py
    python scripts/make_split.py --train-ratio 0.9 --seed 42 \
        --edge-dir data/train/sample_edge --out data/split.json

The output JSON has the shape::

    {"train": [...paths...], "val": [...paths...], "meta": {...}}
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

# Load ``src/data/dataset.py`` directly so building a split does not pull in
# the package ``__init__`` (and thus pytorch_lightning, which is unneeded here).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_PATH = os.path.join(_PROJECT_ROOT, "src", "data", "dataset.py")
_spec = importlib.util.spec_from_file_location("_pc2w_dataset", _DATASET_PATH)
_dataset = importlib.util.module_from_spec(_spec)
# Register before exec so dataclass annotation resolution can find the module.
sys.modules[_spec.name] = _dataset
_spec.loader.exec_module(_dataset)
make_split = _dataset.make_split
save_split = _dataset.save_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--edge-dir",
        default="data/train/sample_edge",
        help="Directory of training edge NPZ files (relative to cwd).",
    )
    p.add_argument(
        "--out",
        default="data/split.json",
        help="Output split JSON path.",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of files assigned to the train split.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the deterministic shuffle.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Glob edge files recursively.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing split file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if os.path.isfile(args.out) and not args.force:
        raise SystemExit(
            f"Split file {args.out!r} already exists. Use --force to overwrite."
        )

    split = make_split(
        args.edge_dir,
        train_ratio=args.train_ratio,
        split_seed=args.seed,
        recursive_glob=args.recursive,
    )
    meta = split["meta"]
    if meta["num_total"] == 0:
        raise SystemExit(f"No .npz files found under {args.edge_dir!r}.")

    save_split(split, args.out)
    print(
        f"[make_split] total={meta['num_total']} "
        f"train={meta['num_train']} val={meta['num_val']} "
        f"(ratio={args.train_ratio}, seed={args.seed}) -> {args.out}"
    )


if __name__ == "__main__":
    main()
