#!/usr/bin/env python
"""Reproduce Figure 1's style of visual explanation for a trained multi-input Conv-Att
model: original drawing + attention-rollout heat map, for each of the 3 tasks.

Usage:
    python scripts/visualize.py --config configs/original.yaml --seed 0 --patient_id 4260692764019076784
    python scripts/visualize.py --config configs/original.yaml --seed 0 --pick mci   # random MCI test case
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch

from mci.attention_rollout import explain_sample, overlay_heatmap
from mci.dataset import MCIDataset, build_transforms, load_records, make_splits
from mci.model import build_model
from mci.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True, help="Which split's test set + checkpoint to use")
    parser.add_argument("--patient_id", default=None, help="Specific subject; default picks one from --pick")
    parser.add_argument("--pick", choices=["mci", "healthy", "random"], default="mci")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = load_records(dcfg["label_csv"], dcfg["moca_cutoff"], dcfg["soft_label_center"])
    _, _, test_recs = make_splits(records, args.seed, dcfg["train_frac"], dcfg["val_frac"], dcfg["test_frac"])

    if args.patient_id is not None:
        rec = next(r for r in test_recs if r.patient_id == args.patient_id)
    else:
        pool = test_recs
        if args.pick == "mci":
            pool = [r for r in test_recs if r.hard_label == 1]
        elif args.pick == "healthy":
            pool = [r for r in test_recs if r.hard_label == 0]
        rec = random.choice(pool)

    ckpt_path = Path(cfg["output_dir"]) / f"seed_{args.seed}" / "model.pt"
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    _, eval_tf = build_transforms(cfg["augmentation"]["pad"], cfg["augmentation"]["crop_size"])
    ds = MCIDataset([rec], dcfg["images_dir"], dcfg["tasks"], eval_tf)
    sample = ds[0]
    images = [img.unsqueeze(0) for img in sample["images"]]

    logits, rel_maps = explain_sample(model, images, device)
    prob_mci = torch.softmax(logits, dim=1)[0, 1].item()

    raw_imgs = []
    for task in dcfg["tasks"]:
        p = Path(dcfg["images_dir"]) / rec.patient_id / f"{task}.png"
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        raw_imgs.append(img)

    task_label_pt = {"clock": "relógio", "copy": "cubo", "trail": "trilha"}

    fig, axes = plt.subplots(len(dcfg["tasks"]), 2, figsize=(7.5, 3.3 * len(dcfg["tasks"])))
    if len(dcfg["tasks"]) == 1:
        axes = axes[None, :]
    for i, task in enumerate(dcfg["tasks"]):
        label = task_label_pt.get(task, task)
        axes[i, 0].imshow(raw_imgs[i])
        axes[i, 0].set_title(f"{label} (original)", fontsize=12)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(overlay_heatmap(raw_imgs[i], rel_maps[i]))
        axes[i, 1].set_title(f"{label} (atenção do modelo)", fontsize=12)
        axes[i, 1].axis("off")

    true_label_pt = "MCI" if rec.hard_label == 1 else "Saudável"
    pred_label_pt = "MCI" if prob_mci >= 0.5 else "Saudável"
    hit = "acertou" if (prob_mci >= 0.5) == bool(rec.hard_label) else "errou"
    fig.suptitle(
        f"MoCA {rec.moca:.0f}  ·  diagnóstico real: {true_label_pt}\n"
        f"previsto pelo modelo: {pred_label_pt}  (P(MCI)={prob_mci:.3f})  —  {hit}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = Path(args.out) if args.out else Path(cfg["output_dir"]) / f"seed_{args.seed}" / f"viz_{rec.patient_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, facecolor="white")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
