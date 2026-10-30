"""Orchestrates one train/val/test run for a given config + seed. Shared by
scripts/train.py (single run) and scripts/run_experiment.py (N repetitions)."""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import MCIDataset, build_transforms, collate_fn, load_records, make_splits
from .engine import evaluate, train_one_epoch
from .model import build_model
from .utils import save_json, set_seed


def _build_optimizer(model: torch.nn.Module, tcfg: dict) -> torch.optim.Optimizer:
    kwargs = dict(
        lr=float(tcfg["lr"]),
        betas=tuple(tcfg.get("betas", [0.9, 0.999])),
        eps=float(tcfg.get("eps", 1e-8)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    if tcfg.get("optimizer", "adam") == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    return torch.optim.Adam(model.parameters(), **kwargs)


def _build_scheduler(optimizer, tcfg: dict, epochs: int):
    schedule = tcfg.get("lr_schedule", "none")
    if schedule == "none":
        return None
    if schedule == "cosine_warmup":
        warmup = tcfg.get("warmup_epochs", 0)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / max(warmup, 1)
            progress = (epoch - warmup) / max(epochs - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unknown lr_schedule: {schedule}")


def _make_loader(dataset, batch_size, shuffle, num_workers, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )


def run_single_split(cfg: dict, seed: int, save_dir: str | Path, quiet: bool = False) -> dict:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dcfg, tcfg, acfg = cfg["data"], cfg["training"], cfg["augmentation"]
    label_mode = cfg.get("label_mode", "soft")

    records = load_records(dcfg["label_csv"], dcfg["moca_cutoff"], dcfg["soft_label_center"])
    train_recs, val_recs, test_recs = make_splits(
        records, seed, dcfg["train_frac"], dcfg["val_frac"], dcfg["test_frac"]
    )
    train_tf, eval_tf = build_transforms(acfg["pad"], acfg["crop_size"], acfg.get("extra", "none"))

    train_ds = MCIDataset(train_recs, dcfg["images_dir"], dcfg["tasks"], train_tf)
    val_ds = MCIDataset(val_recs, dcfg["images_dir"], dcfg["tasks"], eval_tf)
    test_ds = MCIDataset(test_recs, dcfg["images_dir"], dcfg["tasks"], eval_tf)

    sampler = None
    if tcfg.get("class_balanced_sampling", False):
        hard = [r.hard_label for r in train_recs]
        n_pos = sum(hard)
        n_neg = len(hard) - n_pos
        weight_pos = len(hard) / (2 * max(n_pos, 1))
        weight_neg = len(hard) / (2 * max(n_neg, 1))
        weights = [weight_pos if h == 1 else weight_neg for h in hard]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    micro_bs = min(tcfg.get("micro_batch_size", tcfg["batch_size"]), tcfg["batch_size"])
    grad_accum_steps = max(tcfg["batch_size"] // micro_bs, 1)
    num_workers = dcfg.get("num_workers", 4)

    train_loader = _make_loader(train_ds, micro_bs, sampler is None, num_workers, sampler)
    val_loader = _make_loader(val_ds, micro_bs, False, num_workers)
    test_loader = _make_loader(test_ds, micro_bs, False, num_workers)

    model = build_model(cfg).to(device)
    optimizer = _build_optimizer(model, tcfg)
    scheduler = _build_scheduler(optimizer, tcfg, tcfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=tcfg.get("amp", False) and device.type == "cuda")

    model_selection = tcfg.get("model_selection", "last")
    best_val_auc = -1.0
    best_state = None
    history = []

    for epoch in range(tcfg["epochs"]):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, label_mode,
            scaler if scaler.is_enabled() else None, grad_accum_steps,
        )
        val_metrics, _, _ = evaluate(model, val_loader, device, label_mode)
        if scheduler is not None:
            scheduler.step()
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "epoch_seconds": dt, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if not quiet:
            print(
                f"[seed {seed}] epoch {epoch + 1:3d}/{tcfg['epochs']} "
                f"train_loss={train_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f} "
                f"({dt:.1f}s)"
            )
        if model_selection == "best_val_auc" and val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())

    if model_selection == "best_val_auc" and best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, test_probs, test_hard = evaluate(model, test_loader, device, label_mode)

    torch.save(model.state_dict(), save_dir / "model.pt")
    save_json(history, save_dir / "history.json")
    save_json(
        {
            "seed": seed,
            "n_train": len(train_recs),
            "n_val": len(val_recs),
            "n_test": len(test_recs),
            "n_test_mci": int(sum(r.hard_label for r in test_recs)),
            "n_test_healthy": int(sum(1 - r.hard_label for r in test_recs)),
            "test_metrics": test_metrics,
        },
        save_dir / "result.json",
    )
    return test_metrics
