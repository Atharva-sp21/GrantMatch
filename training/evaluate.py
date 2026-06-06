"""
Ranking metrics for link prediction: Hit@K, NDCG, MRR, Precision@K, Recall@K.

Supports edge-level (each positive edge) and researcher-level (any true grant in top-K).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

Pair = Tuple[int, int]


def _rank_all_grants(
    score_fn: Callable[[int], torch.Tensor],
    num_grants: int,
) -> torch.Tensor:
    """score_fn(r) -> [num_grants] scores."""
    return score_fn


def scores_for_researcher(
    r_id: int,
    z_r: torch.Tensor,
    z_g: torch.Tensor,
    predictor: Optional[torch.nn.Module] = None,
    r_text_emb: Optional[torch.Tensor] = None,
    g_text_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unified scoring: GNN predictor or cosine on text embeddings."""
    if predictor is not None:
        z_r_vec = z_r[r_id].unsqueeze(0).expand(z_g.shape[0], -1)
        with torch.no_grad():
            logits = predictor(z_r_vec, z_g)
            if logits.dim() == 0:
                return logits.unsqueeze(0)
            return logits.squeeze(-1)

    if r_text_emb is not None and g_text_emb is not None:
        r = torch.nn.functional.normalize(r_text_emb[r_id].float(), dim=0)
        g = torch.nn.functional.normalize(g_text_emb.float(), dim=1)
        return g @ r
    raise ValueError("Need predictor or text embeddings")


def evaluate_ranking(
    pos_edge: torch.Tensor,
    num_grants: int,
    score_fn: Callable[[int], torch.Tensor],
    ks: List[int] = (1, 5, 10),
    label: str = "eval",
) -> Dict[str, float]:
    """Edge-level metrics: each (r,g) pair ranked against all grants."""
    if pos_edge.shape[1] == 0:
        return {}

    ks = sorted(set(ks))
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    prec_sum = {k: 0.0 for k in ks}
    rec_sum = {k: 0.0 for k in ks}
    total = pos_edge.shape[1]

    for i in range(total):
        r_id = pos_edge[0, i].item()
        true_g = pos_edge[1, i].item()
        scores = score_fn(r_id)
        ranked = scores.argsort(descending=True).tolist()
        rank = ranked.index(true_g) + 1 if true_g in ranked else num_grants + 1

        for k in ks:
            if true_g in ranked[:k]:
                hits[k] += 1
            dcg = 1.0 / np.log2(rank + 1) if rank <= k else 0.0
            ndcg_sum[k] += dcg
            prec_sum[k] += 1.0 / k if rank <= k else 0.0
            rec_sum[k] += 1.0 if rank <= k else 0.0

        mrr_sum += 1.0 / rank if true_g in ranked else 0.0

    metrics = {}
    for k in ks:
        metrics[f"hit@{k}"] = hits[k] / total
        metrics[f"ndcg@{k}"] = ndcg_sum[k] / total
        metrics[f"precision@{k}"] = prec_sum[k] / total
        metrics[f"recall@{k}"] = rec_sum[k] / total
    metrics["mrr"] = mrr_sum / total

    print(f"\n--- {label} (n={total} edges) ---")
    for k in ks:
        print(
            f"  Hit@{k}: {metrics[f'hit@{k}']:.4f}  "
            f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}  "
            f"P@{k}: {metrics[f'precision@{k}']:.4f}  "
            f"R@{k}: {metrics[f'recall@{k}']:.4f}"
        )
    print(f"  MRR: {metrics['mrr']:.4f}")
    return metrics


def evaluate_researcher_level(
    pos_edge: torch.Tensor,
    num_grants: int,
    score_fn: Callable[[int], torch.Tensor],
    ks: List[int] = (5, 10),
    label: str = "researcher-level",
) -> Dict[str, float]:
    """One query per researcher: success if any of their grants appears in top-K."""
    by_r: Dict[int, Set[int]] = {}
    for i in range(pos_edge.shape[1]):
        r, g = pos_edge[0, i].item(), pos_edge[1, i].item()
        by_r.setdefault(r, set()).add(g)

    ks = sorted(set(ks))
    hits = {k: 0 for k in ks}
    n = len(by_r)

    for r_id, true_gs in by_r.items():
        scores = score_fn(r_id)
        top = scores.argsort(descending=True).tolist()
        for k in ks:
            if any(g in top[:k] for g in true_gs):
                hits[k] += 1

    metrics = {f"hit@{k}": hits[k] / max(n, 1) for k in ks}
    print(f"\n--- {label} (researchers={n}) ---")
    for k in ks:
        print(f"  Hit@{k}: {metrics[f'hit@{k}']:.4f}")
    return metrics


def gnn_score_fn(z_dict, predictor, num_grants):
    z_g = z_dict["grant"]

    def fn(r_id):
        return scores_for_researcher(r_id, z_dict["researcher"], z_g, predictor=predictor)

    return fn


def embedding_score_fn(r_emb, g_emb):
    def fn(r_id):
        return scores_for_researcher(
            r_id, None, None, r_text_emb=r_emb, g_text_emb=g_emb
        )

    return fn


# Backward-compatible helpers
def hit_rate_at_k(z_dict, predictor, pos_edge, num_grants, k=10):
    fn = gnn_score_fn(z_dict, predictor, num_grants)
    m = evaluate_ranking(pos_edge, num_grants, fn, ks=[k], label=f"Hit@{k}")
    return m.get(f"hit@{k}", 0.0)


def ndcg_at_k(z_dict, predictor, pos_edge, num_grants, k=10):
    fn = gnn_score_fn(z_dict, predictor, num_grants)
    m = evaluate_ranking(pos_edge, num_grants, fn, ks=[k], label=f"NDCG@{k}")
    return m.get(f"ndcg@{k}", 0.0)
