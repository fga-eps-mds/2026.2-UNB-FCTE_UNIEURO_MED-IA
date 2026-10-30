"""Data loading: label.csv + images/<patient_id>/{clock,copy,trail}.png.

Reproduces the paper's labeling and splitting exactly:
  - hard label: MCI (1) if MoCA < 25, healthy (0) otherwise (Petersen/MoCA cutoff, ref. 18).
  - soft label: y = 1 - sigmoid(m - 24.5)  (Materials & Methods, "Soft-label" subsection).
  - stratified 70/15/15 train/val/test split (on the hard label), repeated over several
    random seeds to reproduce the paper's "5 different random training-validation-test
    data splitting" evaluation protocol.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

TASKS = ["clock", "copy", "trail"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MOCA_CUTOFF = 25
SOFT_LABEL_CENTER = 24.5


def soft_label_from_moca(m: float, center: float = SOFT_LABEL_CENTER) -> float:
    """y = 1 - sigmoid(m - center); lower MoCA -> y closer to 1 (MCI)."""
    return 1.0 - 1.0 / (1.0 + math.exp(-(m - center)))


def hard_label_from_moca(m: float, cutoff: int = MOCA_CUTOFF) -> int:
    return 1 if m < cutoff else 0


@dataclass
class Record:
    patient_id: str
    moca: float
    hard_label: int
    soft_label: float


def load_records(label_csv: str | Path, moca_cutoff: int = MOCA_CUTOFF,
                  soft_label_center: float = SOFT_LABEL_CENTER) -> list[Record]:
    df = pd.read_csv(label_csv, dtype={"patient_id": str})
    records = []
    for _, row in df.iterrows():
        m = float(row["MoCA_score"])
        records.append(
            Record(
                patient_id=row["patient_id"],
                moca=m,
                hard_label=hard_label_from_moca(m, moca_cutoff),
                soft_label=soft_label_from_moca(m, soft_label_center),
            )
        )
    return records


def make_splits(
    records: list[Record],
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[list[Record], list[Record], list[Record]]:
    """Stratified 70/15/15 split on the hard label, one seed per repetition."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    labels = [r.hard_label for r in records]
    train_val, test = train_test_split(
        records, test_size=test_frac, stratify=labels, random_state=seed
    )
    tv_labels = [r.hard_label for r in train_val]
    val_relative = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        train_val, test_size=val_relative, stratify=tv_labels, random_state=seed
    )
    return train, val, test


def build_transforms(pad: int = 12, crop_size: int = 256, extra: str = "none"):
    """Train transform reproduces the paper's augmentation: zero-pad 256->280,
    then random-crop back to 256 (equivalent to RandomCrop(256, padding=12)).
    `extra="mild_affine"` is an opt-in addition used by configs/improved.yaml,
    documented in IMPROVEMENTS.md — not part of the original paper pipeline.
    """
    train_ops = [
        transforms.ToPILImage(),
        transforms.Pad(pad, fill=0, padding_mode="constant"),
        transforms.RandomCrop(crop_size),
    ]
    if extra == "mild_affine":
        train_ops.append(
            transforms.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.97, 1.03), fill=255)
        )
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    train_transform = transforms.Compose(train_ops)

    eval_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


class MCIDataset(Dataset):
    """One sample = the 3 drawings (clock, cube-copy, trail-making) for one subject."""

    def __init__(
        self,
        records: list[Record],
        images_dir: str | Path,
        tasks: list[str] = TASKS,
        transform=None,
    ):
        self.records = records
        self.images_dir = Path(images_dir)
        self.tasks = tasks
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, patient_id: str, task: str) -> np.ndarray:
        path = self.images_dir / patient_id / f"{task}.png"
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        images = []
        for task in self.tasks:
            img = self._load_image(rec.patient_id, task)
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
        return {
            "images": images,
            "hard_label": torch.tensor(rec.hard_label, dtype=torch.float32),
            "soft_label": torch.tensor(rec.soft_label, dtype=torch.float32),
            "moca": torch.tensor(rec.moca, dtype=torch.float32),
            "patient_id": rec.patient_id,
        }


def collate_fn(batch):
    n_tasks = len(batch[0]["images"])
    images = [torch.stack([b["images"][t] for b in batch]) for t in range(n_tasks)]
    return {
        "images": images,
        "hard_label": torch.stack([b["hard_label"] for b in batch]),
        "soft_label": torch.stack([b["soft_label"] for b in batch]),
        "moca": torch.stack([b["moca"] for b in batch]),
        "patient_id": [b["patient_id"] for b in batch],
    }
