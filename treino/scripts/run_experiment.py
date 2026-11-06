#!/usr/bin/env python
"""Run all repetitions (seed_list) for a config and report mean +/- std, matching
the paper's Table 1 style.

Usage:
    python scripts/run_experiment.py --config configs/original.yaml
    python scripts/run_experiment.py --config configs/original.yaml --seeds 0 1   # subset
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from mci.run import run_single_split
from mci.utils import load_config, mean_std_table, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Override config's seed_list")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg["seed_list"]

    print(f"Running experiment '{cfg['experiment_name']}' over seeds {seeds}")
    all_metrics = []
    t_start = time.time()
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        save_dir = Path(cfg["output_dir"]) / f"seed_{seed}"
        t0 = time.time()
        metrics = run_single_split(cfg, seed, save_dir)
        print(f"seed {seed} done in {(time.time() - t0) / 60:.1f} min -> {metrics}")
        all_metrics.append(metrics)

    summary = mean_std_table(all_metrics)
    print(f"\n=== Summary over {len(seeds)} repetitions ({cfg['experiment_name']}) ===")
    for k, v in summary.items():
        print(f"{k:>10s}: {v}")
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min")

    save_json(
        {"experiment_name": cfg["experiment_name"], "seeds": seeds, "per_seed": all_metrics, "summary": summary},
        Path(cfg["output_dir"]) / "summary.json",
    )


if __name__ == "__main__":
    main()
