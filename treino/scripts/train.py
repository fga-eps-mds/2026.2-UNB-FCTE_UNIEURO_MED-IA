#!/usr/bin/env python
"""Train a single train/val/test split.

Usage:
    python scripts/train.py --config configs/original.yaml --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mci.run import run_single_split
from mci.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    save_dir = Path(cfg["output_dir"]) / f"seed_{args.seed}"
    metrics = run_single_split(cfg, args.seed, save_dir)
    print(f"\nTest metrics (seed={args.seed}): {metrics}")


if __name__ == "__main__":
    main()
