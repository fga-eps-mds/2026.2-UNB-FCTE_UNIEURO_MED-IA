"""Training / evaluation loops."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import soft_cross_entropy
from .utils import compute_metrics


def _target_key(label_mode: str) -> str:
    return "soft_label" if label_mode == "soft" else "hard_label"


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_mode: str = "soft",
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    target_key = _target_key(label_mode)
    for step, batch in enumerate(loader):
        images = [img.to(device, non_blocking=True) for img in batch["images"]]
        target = batch[target_key].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss = soft_cross_entropy(logits, target) / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * grad_accum_steps
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_mode: str = "soft",
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    target_key = _target_key(label_mode)
    all_probs, all_hard, total_loss, n_batches = [], [], 0.0, 0
    for batch in loader:
        images = [img.to(device, non_blocking=True) for img in batch["images"]]
        target = batch[target_key].to(device, non_blocking=True)
        logits = model(images)
        loss = soft_cross_entropy(logits, target)
        total_loss += loss.item()
        n_batches += 1
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_hard.append(batch["hard_label"].numpy())
    all_probs = np.concatenate(all_probs)
    all_hard = np.concatenate(all_hard)
    metrics = compute_metrics(all_hard, all_probs)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics, all_probs, all_hard
