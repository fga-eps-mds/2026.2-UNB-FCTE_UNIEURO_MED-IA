#!/usr/bin/env python
"""Sanity-check the pipeline before committing to a full run:
  1) a few real-size training steps -> checks it fits in VRAM and measures step time
  2) a tiny 2-epoch end-to-end run (small subset) -> checks the full run_single_split path

Usage:
    python scripts/smoke_test.py --config configs/original.yaml
"""
from __future__ import annotations

import argparse
import copy
import shutil
import time
from pathlib import Path

import torch

from mci.dataset import MCIDataset, build_transforms, collate_fn, load_records, make_splits
from mci.engine import train_one_epoch
from mci.model import build_model
from mci.run import run_single_split
from mci.utils import load_config, set_seed


def timing_probe(cfg: dict, n_micro_steps: int = 12):
    print("=== 1) Timing / VRAM probe (real micro-batch size + grad-accum, real image size) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0)
    dcfg, tcfg, acfg = cfg["data"], cfg["training"], cfg["augmentation"]
    micro_bs = min(tcfg.get("micro_batch_size", tcfg["batch_size"]), tcfg["batch_size"])
    grad_accum_steps = max(tcfg["batch_size"] // micro_bs, 1)
    records = load_records(dcfg["label_csv"], dcfg["moca_cutoff"], dcfg["soft_label_center"])
    train_tf, _ = build_transforms(acfg["pad"], acfg["crop_size"], acfg.get("extra", "none"))
    ds = MCIDataset(records[: micro_bs * (n_micro_steps + 1)], dcfg["images_dir"], dcfg["tasks"], train_tf)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=micro_bs, shuffle=True, collate_fn=collate_fn, num_workers=dcfg.get("num_workers", 4)
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))
    scaler = torch.amp.GradScaler("cuda", enabled=tcfg.get("amp", False) and device.type == "cuda")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: {n_params / 1e6:.1f}M total, {n_trainable / 1e6:.1f}M trainable")
    print(f"micro_batch_size={micro_bs}, grad_accum_steps={grad_accum_steps} (effective batch={tcfg['batch_size']})")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    from mci.losses import soft_cross_entropy

    times = []
    optimizer.zero_grad(set_to_none=True)
    it = iter(loader)
    for step in range(n_micro_steps):
        batch = next(it)
        images = [img.to(device) for img in batch["images"]]
        target = batch["soft_label"].to(device)
        t0 = time.time()
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(images)
            loss = soft_cross_entropy(logits, target) / grad_accum_steps
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % grad_accum_steps == 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)
        print(f"  micro-step {step + 1}/{n_micro_steps}: loss={loss.item() * grad_accum_steps:.4f} time={dt:.3f}s")

    warmup = min(2, len(times) - 1)
    avg_micro_step = sum(times[warmup:]) / max(len(times) - warmup, 1)
    n_train = int(cfg["data"]["train_frac"] * 918)
    micro_steps_per_epoch = max(round(n_train / micro_bs), 1)
    est_epoch_s = avg_micro_step * micro_steps_per_epoch
    est_total_h = est_epoch_s * tcfg["epochs"] * len(cfg["seed_list"]) / 3600
    print(f"\navg micro-step time (steady state): {avg_micro_step:.3f}s")
    print(f"~{micro_steps_per_epoch} micro-steps/epoch -> ~{est_epoch_s:.0f}s/epoch")
    print(
        f"estimated total for {tcfg['epochs']} epochs x {len(cfg['seed_list'])} seeds: "
        f"~{est_total_h:.2f} GPU-hours"
    )
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak GPU memory: {peak_gb:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def functional_probe(cfg: dict, tmp_dir: str = "outputs/_smoke_test"):
    print("\n=== 2) Functional probe (tiny subset, 2 epochs, full run_single_split path) ===")
    tiny_cfg = copy.deepcopy(cfg)
    tiny_cfg["training"]["epochs"] = 2
    tiny_cfg["training"]["batch_size"] = 8
    tiny_cfg["training"]["micro_batch_size"] = 8
    tiny_cfg["data"]["num_workers"] = 2

    # Monkey-patch load_records to truncate, via a tiny label csv copy.
    import pandas as pd

    df = pd.read_csv(cfg["data"]["label_csv"], dtype={"patient_id": str})
    tiny_csv = Path(tmp_dir) / "tiny_label.csv"
    tiny_csv.parent.mkdir(parents=True, exist_ok=True)
    df.iloc[:60].to_csv(tiny_csv, index=False)
    tiny_cfg["data"]["label_csv"] = str(tiny_csv)
    tiny_cfg["output_dir"] = str(Path(tmp_dir) / "run")

    metrics = run_single_split(tiny_cfg, seed=0, save_dir=Path(tiny_cfg["output_dir"]) / "seed_0")
    print(f"functional probe finished OK. test metrics on tiny subset: {metrics}")
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/original.yaml")
    parser.add_argument("--skip-functional", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    timing_probe(cfg)
    if not args.skip_functional:
        functional_probe(cfg)
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
