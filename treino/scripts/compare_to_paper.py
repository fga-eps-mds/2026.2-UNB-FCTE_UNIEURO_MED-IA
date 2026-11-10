#!/usr/bin/env python
"""Print a Table-1-style comparison: paper's published numbers vs. our reproduction(s).

Usage:
    python scripts/compare_to_paper.py --runs outputs/original outputs/improved
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Table 1, "Multi-input Conv-Att with soft label (proposed)" row.
PAPER_PROPOSED = {"accuracy": (0.8116, 0.0103), "f1": (0.6539, 0.0097), "auc": (0.8375, 0.0116)}


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["outputs/original", "outputs/improved"])
    args = parser.parse_args()

    rows = [("Paper (Table 1, proposed)", {k: fmt(*v) for k, v in PAPER_PROPOSED.items()})]
    for run_dir in args.runs:
        summary_path = Path(run_dir) / "summary.json"
        if not summary_path.exists():
            print(f"(skipping {run_dir}: no summary.json yet)")
            continue
        summary = json.loads(summary_path.read_text())
        rows.append((summary["experiment_name"], summary["summary"]))

    cols = ["accuracy", "f1", "auc"]
    name_w = max(len(r[0]) for r in rows) + 2
    header = f"{'Model':<{name_w}}" + "".join(f"{c:>22}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, metrics in rows:
        print(f"{name:<{name_w}}" + "".join(f"{metrics[c]:>22}" for c in cols))


if __name__ == "__main__":
    main()
