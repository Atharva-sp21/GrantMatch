"""Hard + adaptive negative sampling using topics, agency, embeddings, career stage."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch

Pair = Tuple[int, int]


def build_hetero_maps(
    researches: torch.Tensor,
    funds_topic: torch.Tensor,
    provides: torch.Tensor,
    num_grants: int,
    grants_meta: Optional[List[dict]] = None,
):
    r_topics: Dict[int, Set[int]] = defaultdict(set)
    g_topics: Dict[int, Set[int]] = defaultdict(set)
    g_agency: Dict[int, int] = {}
    agency_grants: Dict[int, Set[int]] = defaultdict(set)
    g_career: Dict[int, str] = {}

    if researches.shape[1]:
        for s, d in zip(researches[0].tolist(), researches[1].tolist()):
            r_topics[s].add(d)

    if funds_topic.shape[1]:
        for g, t in zip(funds_topic[0].tolist(), funds_topic[1].tolist()):
            g_topics[g].add(t)

    if provides.shape[1]:
        for a, g in zip(provides[0].tolist(), provides[1].tolist()):
            g_agency[g] = a
            agency_grants[a].add(g)

    if grants_meta:
        for g in grants_meta:
            gid = g["id"]
            g_career[gid] = g.get("career_stage_eligibility", "any")

    return r_topics, g_topics, g_agency, agency_grants, g_career


def _embedding_hard_grants(
    r_id: int,
    g_id: int,
    grant_emb: torch.Tensor,
    num_g: int,
    pos_edge_set: Set[Pair],
    k: int = 20,
) -> List[int]:
    """Grants closest in embedding space to positive grant (excluding true pair)."""
    if grant_emb is None or grant_emb.shape[0] != num_g:
        return []
    r_vec = grant_emb[r_id]
    g_vec = grant_emb[g_id]
    # Hard: grants similar to the positive grant (not researcher) — confusable alternatives
    g_norm = torch.nn.functional.normalize(grant_emb, dim=1)
    sim = g_norm @ g_norm[g_id]
    sim[g_id] = -1.0
    _, idx = torch.topk(sim, k=min(k, num_g - 1))
    out = []
    for j in idx.tolist():
        if (r_id, j) not in pos_edge_set:
            out.append(j)
    return out


def sample_negatives(
    pos_src: torch.Tensor,
    pos_dst: torch.Tensor,
    pos_edge_set: Set[Pair],
    num_r: int,
    num_g: int,
    r_topics: Dict[int, Set[int]],
    g_topics: Dict[int, Set[int]],
    g_agency: Dict[int, int],
    agency_grants: Dict[int, Set[int]],
    g_career: Dict[int, str],
    researchers_meta: Optional[List[dict]] = None,
    grant_emb: Optional[torch.Tensor] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    neg_src, neg_dst = [], []
    all_grants = list(range(num_g))

    for i in range(pos_src.shape[0]):
        r = pos_src[i].item()
        g = pos_dst[i].item()
        neg_g = None
        r_career = "unknown"
        if researchers_meta and r < len(researchers_meta):
            r_career = researchers_meta[r].get("career_stage", "unknown")

        roll = rng.random()
        emb_hard = _embedding_hard_grants(r, g, grant_emb, num_g, pos_edge_set)

        if roll < 0.35 and emb_hard:
            neg_g = rng.choice(emb_hard)
        elif roll < 0.55:
            topics_g = g_topics.get(g, set())
            candidates = [
                gg for gg in all_grants
                if gg != g and (r, gg) not in pos_edge_set and g_topics.get(gg, set()) & topics_g
            ]
            if candidates:
                neg_g = rng.choice(candidates)
        elif roll < 0.75:
            agency = g_agency.get(g)
            if agency is not None:
                candidates = [
                    gg for gg in agency_grants.get(agency, set())
                    if gg != g and (r, gg) not in pos_edge_set
                ]
                if candidates:
                    neg_g = rng.choice(candidates)
        elif roll < 0.88:
            candidates = [
                gg for gg in all_grants
                if gg != g and (r, gg) not in pos_edge_set
                and g_career.get(gg, "any") in (r_career, "any", "unknown")
            ]
            if candidates:
                neg_g = rng.choice(candidates)

        attempts = 0
        while neg_g is None and attempts < 60:
            cand = rng.randint(0, num_g - 1)
            if (r, cand) not in pos_edge_set:
                neg_g = cand
            attempts += 1
        if neg_g is None:
            neg_g = rng.choice(all_grants)

        neg_src.append(r)
        neg_dst.append(neg_g)

    return (
        torch.tensor(neg_src, dtype=torch.long),
        torch.tensor(neg_dst, dtype=torch.long),
    )
