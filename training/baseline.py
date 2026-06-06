"""
Cosine-similarity baseline on text embeddings (before GNN).

Why: if GNN cannot beat this, graph training adds little over semantic matching.
"""

from __future__ import annotations

from typing import Dict

import torch

from training.evaluate import embedding_score_fn, evaluate_ranking, evaluate_researcher_level


def evaluate_embedding_baseline(
    researcher_emb: torch.Tensor,
    grant_emb: torch.Tensor,
    test_edge: torch.Tensor,
    val_edge: torch.Tensor = None,
    ks=(1, 5, 10),
) -> Dict[str, Dict[str, float]]:
    num_g = grant_emb.shape[0]
    score_fn = embedding_score_fn(researcher_emb, grant_emb)
    results = {}

    print("\n" + "=" * 60)
    print("EMBEDDING COSINE BASELINE (SPECTER2 / MiniLM)")
    print("=" * 60)

    if val_edge is not None and val_edge.shape[1] > 0:
        results["val"] = evaluate_ranking(
            val_edge, num_g, score_fn, ks=list(ks), label="Baseline VAL"
        )
        evaluate_researcher_level(val_edge, num_g, score_fn, ks=[5, 10], label="Baseline VAL")

    if test_edge.shape[1] > 0:
        results["test"] = evaluate_ranking(
            test_edge, num_g, score_fn, ks=list(ks), label="Baseline TEST"
        )
        evaluate_researcher_level(test_edge, num_g, score_fn, ks=[5, 10], label="Baseline TEST")

    return results
