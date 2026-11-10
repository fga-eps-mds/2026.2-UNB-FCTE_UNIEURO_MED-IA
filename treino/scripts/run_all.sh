#!/usr/bin/env bash
# Runs both configs' 5-repetition experiments sequentially (single GPU, avoid VRAM contention).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== [$(date)] Starting ORIGINAL (paper reproduction) ==="
python scripts/run_experiment.py --config configs/original.yaml
echo "=== [$(date)] ORIGINAL done ==="

echo "=== [$(date)] Starting IMPROVED ==="
python scripts/run_experiment.py --config configs/improved.yaml
echo "=== [$(date)] IMPROVED done ==="

echo "=== [$(date)] ALL RUNS COMPLETE ==="
