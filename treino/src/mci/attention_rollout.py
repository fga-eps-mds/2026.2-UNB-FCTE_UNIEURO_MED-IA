"""Attention rollout (Abnar & Zuidema, 2020), exactly as described in the paper's
"Attention rollout" subsection:

    A = 0.5 * W_att + 0.5 * I                      (per layer, heads averaged)
    Ã(l_i) = A(l_i) · Ã(l_{i-1})   for i > j,   Ã(l_j) = A(l_j)

The [CLS] row of the final rolled-out matrix gives, per patch, how much it
contributed to the final classification decision -> reshaped to the last
feature map's (H, W) grid -> resized to the input image -> heat map (Figure 1).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


@torch.no_grad()
def compute_rollout(attns: list[torch.Tensor]) -> torch.Tensor:
    """attns: list (per layer, in order) of (B, heads, N, N) post-softmax attention
    weights. Returns the rolled-out (B, N, N) matrix."""
    B, _, N, _ = attns[0].shape
    device = attns[0].device
    eye = torch.eye(N, device=device).unsqueeze(0)
    rollout = eye.expand(B, -1, -1).clone()
    for attn in attns:
        avg_heads = attn.mean(dim=1)  # (B, N, N)
        a_hat = 0.5 * avg_heads + 0.5 * eye
        rollout = torch.bmm(a_hat, rollout)
    return rollout


def cls_relevance_map(rollout: torch.Tensor, grid_hw: tuple[int, int]) -> np.ndarray:
    """Extract the [CLS]->patch row (dropping the CLS->CLS entry) and reshape to (B,H,W)."""
    H, W = grid_hw
    cls_row = rollout[:, 0, 1:]  # (B, H*W)
    return cls_row.reshape(-1, H, W).cpu().numpy()


def overlay_heatmap(image_rgb_uint8: np.ndarray, relevance: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Resize a (h,w) relevance map to the image size, colorize (jet) and alpha-blend."""
    h, w = image_rgb_uint8.shape[:2]
    rel = relevance - relevance.min()
    rel = rel / (rel.max() + 1e-8)
    rel_resized = cv2.resize(rel, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_uint8 = np.uint8(255 * np.clip(rel_resized, 0, 1))
    color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return np.uint8(image_rgb_uint8 * (1 - alpha) + color * alpha)


@torch.no_grad()
def explain_sample(model, images: list[torch.Tensor], device) -> tuple[torch.Tensor, list[np.ndarray]]:
    """images: list of (1,3,H,W) preprocessed tensors (one per task, batch size 1).
    Returns (logits, [relevance_map_per_task])."""
    model.eval()
    images = [img.to(device) for img in images]
    logits, attn_maps = model(images, return_attn=True)
    maps = []
    for tower_out in attn_maps:
        rollout = compute_rollout(tower_out["attns"])
        rel = cls_relevance_map(rollout, tower_out["grid_hw"])[0]
        maps.append(rel)
    return logits, maps
