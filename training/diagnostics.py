"""
Training diagnostics: degrees, cold-start, sample predictions, loss plots.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def cold_start_report(
    num_researchers: int,
    num_grants: int,
    received: torch.Tensor,
    researchers: List[dict],
) -> Dict[str, float]:
    r_with = set(received[0].tolist()) if received.shape[1] else set()
    g_with = set(received[1].tolist()) if received.shape[1] else set()
    r_cold = num_researchers - len(r_with)
    g_cold = num_grants - len(g_with)
    with_abstracts = sum(1 for r in researchers if r.get("recent_abstracts"))

    stats = {
        "researchers_no_received": r_cold,
        "researchers_no_received_pct": 100.0 * r_cold / max(num_researchers, 1),
        "grants_no_received": g_cold,
        "grants_no_received_pct": 100.0 * g_cold / max(num_grants, 1),
        "researchers_with_abstracts": with_abstracts,
        "abstract_coverage_pct": 100.0 * with_abstracts / max(num_researchers, 1),
    }

    print("\n" + "=" * 60)
    print("COLD-START ANALYSIS")
    print("=" * 60)
    print(f"  Researchers without received_past: {r_cold} ({stats['researchers_no_received_pct']:.1f}%)")
    print(f"  Grants without any PI link:        {g_cold} ({stats['grants_no_received_pct']:.1f}%)")
    print(f"  Researchers with abstracts:        {with_abstracts} ({stats['abstract_coverage_pct']:.1f}%)")
    print("  Cold-start strategy: use text-embedding cosine (baseline) for unlabeled researchers.")
    print("=" * 60 + "\n")
    return stats


def degree_distributions(data, save_prefix: str = "degrees"):
    print("\n--- Node degree distributions ---")
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        ei = data[edge_type].edge_index
        if src == dst:
            nodes = data[src].num_nodes
            deg = torch.zeros(nodes)
            for i in range(ei.shape[1]):
                deg[ei[0, i]] += 1
                deg[ei[1, i]] += 1
        else:
            deg = torch.zeros(data[src].num_nodes)
            for i in range(ei.shape[1]):
                deg[ei[0, i]] += 1
        d = deg.numpy()
        print(
            f"  {src}--{rel}--> {dst}: mean={d.mean():.2f} max={d.max():.0f} "
            f"zero_deg={(d == 0).sum()}"
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    for ntype in data.node_types:
        total = torch.zeros(data[ntype].num_nodes)
        for edge_type in data.edge_types:
            src, rel, dst = edge_type
            if src == ntype:
                ei = data[edge_type].edge_index
                for i in range(ei.shape[1]):
                    total[ei[0, i]] += 1
            if dst == ntype and src != ntype:
                ei = data[edge_type].edge_index
                for i in range(ei.shape[1]):
                    total[ei[1, i]] += 1
        ax.hist(total.numpy(), bins=30, alpha=0.5, label=ntype)
    ax.set_xlabel("Total degree (in+out)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Node degree distribution")
    path = REPORTS_DIR / f"{save_prefix}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def embedding_statistics(r_emb: torch.Tensor, g_emb: torch.Tensor):
    print("\n--- Embedding statistics ---")
    for name, emb in [("researcher", r_emb), ("grant", g_emb)]:
        norms = emb.norm(dim=1)
        print(
            f"  {name}: shape={tuple(emb.shape)} "
            f"norm mean={norms.mean():.3f} std={norms.std():.3f}"
        )


def plot_training_loss(history: Dict[str, List[float]], path: Optional[Path] = None):
    path = path or REPORTS_DIR / "training_loss.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train"], label="train")
    if history.get("val"):
        ax.plot(history["val"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Training / validation loss")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diagnostics] Loss plot saved to {path}")


def sample_predictions(
    pos_edge: torch.Tensor,
    score_fn,
    num_grants: int,
    grants: List[dict],
    researchers: List[dict],
    n_success: int = 3,
    n_fail: int = 3,
    k: int = 10,
):
    print("\n--- Sample predictions (test edges) ---")
    rng_idx = list(range(pos_edge.shape[1]))
    np.random.shuffle(rng_idx)

    successes, failures = [], []
    for i in rng_idx:
        r_id = pos_edge[0, i].item()
        true_g = pos_edge[1, i].item()
        scores = score_fn(r_id)
        top = scores.argsort(descending=True)[:k].tolist()
        hit = true_g in top
        entry = {
            "researcher": researchers[r_id].get("name", r_id),
            "true_grant": grants[true_g].get("title", true_g)[:80],
            "rank": top.index(true_g) + 1 if hit else None,
            "top3": [grants[g].get("title", g)[:60] for g in top[:3]],
        }
        if hit and len(successes) < n_success:
            successes.append(entry)
        elif not hit and len(failures) < n_fail:
            failures.append(entry)
        if len(successes) >= n_success and len(failures) >= n_fail:
            break

    print("  Successes:")
    for s in successes:
        print(f"    {s['researcher'][:40]} | rank {s['rank']} | true: {s['true_grant']}")
    print("  Failures:")
    for f in failures:
        print(f"    {f['researcher'][:40]} | true: {f['true_grant']}")
        print(f"      top3: {f['top3']}")


def save_report(payload: dict, name: str = "evaluation_report.json"):
    path = REPORTS_DIR / name
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Diagnostics] Report saved to {path}")
