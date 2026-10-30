"""Loss function: soft cross-entropy.

The paper trains a 2-node softmax with binary cross-entropy
    L = -(1/M) sum_i [ y_i log(p_i) + (1-y_i) log(1-p_i) ]
where p_i = P(MCI). Writing target_i = [1-y_i, y_i] and pred_i = softmax(logits_i),
this is exactly the standard (soft-label) cross-entropy below; it reduces to plain
cross-entropy when y_i in {0,1} (the hard-label ablation).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_cross_entropy(logits: torch.Tensor, target_mci_prob: torch.Tensor) -> torch.Tensor:
    """logits: (B, 2) raw scores [healthy, MCI]. target_mci_prob: (B,) in [0, 1]."""
    targets = torch.stack([1.0 - target_mci_prob, target_mci_prob], dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()
