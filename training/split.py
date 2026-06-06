"""Edge-level train/val/test split for RECEIVED_PAST (no leakage)."""

from __future__ import annotations

import random
from typing import Dict, Set, Tuple

import torch

EdgeTensor = torch.Tensor


def split_received_edges(
    edge_index: EdgeTensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, EdgeTensor]:
    """
    Split positive edges; stratify by researcher so all edges for a researcher
    stay in one split (prevents transductive leakage on same PI).
    """
    assert edge_index.shape[0] == 2
    n = edge_index.shape[1]
    if n == 0:
        empty = torch.zeros(2, 0, dtype=torch.long)
        return {"train": empty, "val": empty, "test": empty}

    rng = random.Random(seed)
    by_researcher: Dict[int, list] = {}
    for i in range(n):
        r = edge_index[0, i].item()
        by_researcher.setdefault(r, []).append(i)

    researchers = list(by_researcher.keys())
    rng.shuffle(researchers)

    n_r = len(researchers)
    n_train_r = int(n_r * train_ratio)
    n_val_r = int(n_r * val_ratio)

    train_r = set(researchers[:n_train_r])
    val_r = set(researchers[n_train_r : n_train_r + n_val_r])
    test_r = set(researchers[n_train_r + n_val_r :])

    def collect(r_set: Set[int]) -> EdgeTensor:
        idx = [i for r in r_set for i in by_researcher[r]]
        if not idx:
            return torch.zeros(2, 0, dtype=torch.long)
        return edge_index[:, idx].contiguous()

    return {
        "train": collect(train_r),
        "val": collect(val_r),
        "test": collect(test_r),
    }


def edge_set(edge_index: EdgeTensor) -> Set[Tuple[int, int]]:
    if edge_index.shape[1] == 0:
        return set()
    return set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
