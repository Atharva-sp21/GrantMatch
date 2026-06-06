"""
Heterogeneous graph construction, validation, and optional researcher similarity edges.

Why validation: catches ID misalignment between JSON counts and edge tensors before training.
Why SIMILAR edges: connects cold-start / low-degree researchers to neighbors for message passing.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

import torch
from torch_geometric.data import HeteroData

EDGE_SPECS = [
    ("researcher", "AFFILIATED_WITH", "institution"),
    ("researcher", "RESEARCHES", "topic"),
    ("researcher", "RECEIVED_PAST", "grant"),
    ("grant", "FUNDS_TOPIC", "topic"),
    ("agency", "PROVIDES", "grant"),
    ("researcher", "SIMILAR", "researcher"),
]

EXPECTED_COUNTS = {
    "researcher": 1031,
    "grant": 1000,
    "agency": 131,
    "topic": 27,
    "institution": 1003,
}


def build_graph(
    researcher_x: torch.Tensor,
    institution_x: torch.Tensor,
    topic_x: torch.Tensor,
    grant_x: torch.Tensor,
    agency_x: torch.Tensor,
) -> HeteroData:
    data = HeteroData()
    if researcher_x.shape[0] > 0:
        data["researcher"].x = researcher_x
    if institution_x.shape[0] > 0:
        data["institution"].x = institution_x
    if topic_x.shape[0] > 0:
        data["topic"].x = topic_x
    if grant_x.shape[0] > 0:
        data["grant"].x = grant_x
    if agency_x.shape[0] > 0:
        data["agency"].x = agency_x
    return data


def add_edges(
    data: HeteroData,
    affiliated: torch.Tensor,
    researches: torch.Tensor,
    received: torch.Tensor,
    funds: torch.Tensor,
    provides: torch.Tensor,
    similar: Optional[torch.Tensor] = None,
    similar_attr: Optional[torch.Tensor] = None,
) -> HeteroData:
    if (
        "researcher" in data.node_types
        and "institution" in data.node_types
        and affiliated.shape[1] > 0
    ):
        data["researcher", "AFFILIATED_WITH", "institution"].edge_index = _sanitize(affiliated)
    if (
        "researcher" in data.node_types
        and "topic" in data.node_types
        and researches.shape[1] > 0
    ):
        data["researcher", "RESEARCHES", "topic"].edge_index = _sanitize(researches)
    if (
        "researcher" in data.node_types
        and "grant" in data.node_types
        and received.shape[1] > 0
    ):
        data["researcher", "RECEIVED_PAST", "grant"].edge_index = _sanitize(received)
    if "grant" in data.node_types and "topic" in data.node_types and funds.shape[1] > 0:
        data["grant", "FUNDS_TOPIC", "topic"].edge_index = _sanitize(funds)
    if (
        "agency" in data.node_types
        and "grant" in data.node_types
        and provides.shape[1] > 0
    ):
        data["agency", "PROVIDES", "grant"].edge_index = _sanitize(provides)
    if (
        similar is not None
        and similar.shape[1] > 0
        and "researcher" in data.node_types
    ):
        data["researcher", "SIMILAR", "researcher"].edge_index = _sanitize(similar)
        if similar_attr is not None and similar_attr.numel() == similar.shape[1]:
            data["researcher", "SIMILAR", "researcher"].edge_attr = similar_attr.float()
    return data


def export_graph_statistics(data: HeteroData, path) -> dict:
    import json
    from pathlib import Path

    stats = {"nodes": {}, "edges": {}}
    for nt in data.node_types:
        stats["nodes"][nt] = {
            "count": data[nt].num_nodes,
            "feature_dim": int(data[nt].x.shape[1]),
        }
    for et in data.edge_types:
        key = f"{et[0]}_{et[1]}_{et[2]}"
        stats["edges"][key] = {
            "count": int(data[et].edge_index.shape[1]),
            "has_edge_attr": hasattr(data[et], "edge_attr") and data[et].edge_attr is not None,
        }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def _sanitize(edge_index: torch.Tensor) -> torch.Tensor:
    return edge_index.long().contiguous()


def _max_index(edge_index: torch.Tensor) -> Tuple[int, int]:
    if edge_index.shape[1] == 0:
        return -1, -1
    return int(edge_index[0].max()), int(edge_index[1].max())


def validate_graph(
    data: HeteroData,
    expected: Optional[Dict[str, int]] = None,
    raw_edge_counts: Optional[Dict[str, int]] = None,
) -> None:
    """
    Print node/edge statistics and assert index ranges match node counts.
    """
    expected = expected or {}
    print("\n" + "=" * 60)
    print("GRAPH VALIDATION")
    print("=" * 60)

    print("\nNode counts:")
    for ntype in ["researcher", "grant", "institution", "topic", "agency"]:
        n = data[ntype].num_nodes if ntype in data.node_types else 0
        exp = expected.get(ntype, EXPECTED_COUNTS.get(ntype))
        flag = ""
        if exp is not None and n != exp:
            flag = f"  (expected {exp})"
        print(f"  {ntype:14s} {n:5d}{flag}")
        if ntype in data.node_types:
            assert data[ntype].x.shape[0] == n, f"{ntype} feature rows != num_nodes"

    print("\nEdge counts:")
    for src, rel, dst in EDGE_SPECS:
        key = (src, rel, dst)
        count = 0
        if key in data.edge_types:
            count = data[key].edge_index.shape[1]
        raw = raw_edge_counts.get(rel, raw_edge_counts.get(key, "")) if raw_edge_counts else ""
        extra = f"  (file: {raw})" if raw != "" else ""
        print(f"  {src} --{rel}--> {dst}: {count}{extra}")

        if count == 0:
            continue
        if src not in data.node_types or dst not in data.node_types:
            continue
        ei = data[key].edge_index
        max_s, max_d = _max_index(ei)
        assert max_s < data[src].num_nodes, f"{rel} source index {max_s} >= {src} nodes"
        assert max_d < data[dst].num_nodes, f"{rel} dest index {max_d} >= {dst} nodes"

    if raw_edge_counts:
        print("\n  File vs graph edge check:")
        for rel, file_n in raw_edge_counts.items():
            graph_n = 0
            for src, r, dst in EDGE_SPECS:
                if r == rel and (src, r, dst) in data.edge_types:
                    graph_n = data[(src, r, dst)].edge_index.shape[1]
                    break
            match = "OK" if graph_n == file_n else "MISMATCH"
            print(f"    {rel}: file={file_n} graph={graph_n} [{match}]")

    print("=" * 60 + "\n")


def build_similarity_edges(
    researcher_emb: torch.Tensor,
    researches: torch.Tensor,
    affiliated: torch.Tensor,
    researchers: Optional[list] = None,
    top_k: int = 8,
    min_weight: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted researcher--researcher edges using:
      0.45 embedding cosine + 0.25 shared topics + 0.20 same institution
      + 0.10 citation profile similarity
    Returns (edge_index [2,E], edge_attr [E]).
    """
    n = researcher_emb.shape[0]
    if n < 2:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)

    emb = torch.nn.functional.normalize(researcher_emb.float(), p=2, dim=1)
    emb_sim = emb @ emb.t()

    r_topics: Dict[int, Set[int]] = defaultdict(set)
    if researches.shape[1]:
        for r, t in zip(researches[0].tolist(), researches[1].tolist()):
            r_topics[r].add(t)

    r_inst: Dict[int, Set[int]] = defaultdict(set)
    if affiliated.shape[1]:
        for r, i in zip(affiliated[0].tolist(), affiliated[1].tolist()):
            r_inst[r].add(i)

    cites = None
    if researchers:
        cites = torch.tensor(
            [float(r.get("citation_count", r.get("cited_by_count", 0))) for r in researchers]
        )
        cite_std = cites.std() + 1e-6

    src_list, dst_list, weights = [], [], []
    for i in range(n):
        composite = emb_sim[i].clone()
        composite[i] = -1.0

        for j in range(n):
            if i == j:
                continue
            w = 0.45 * emb_sim[i, j].item()
            shared_topics = r_topics.get(i, set()) & r_topics.get(j, set())
            if shared_topics:
                w += 0.25 * min(1.0, len(shared_topics) / 3.0)
            if r_inst.get(i, set()) & r_inst.get(j, set()):
                w += 0.20
            if cites is not None:
                cite_sim = 1.0 - (cites[i] - cites[j]).abs().item() / cite_std
                w += 0.10 * max(0.0, min(1.0, cite_sim))
            composite[j] = w

        k = min(top_k, n - 1)
        vals, idx = torch.topk(composite, k=k)
        for j, w in zip(idx.tolist(), vals.tolist()):
            if w >= min_weight:
                src_list.extend([i, j])
                dst_list.extend([j, i])
                weights.extend([w, w])

    if not src_list:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float)
    print(
        f"  SIMILAR: {edge_index.shape[1]} directed edges, "
        f"weight mean={edge_attr.mean():.3f} max={edge_attr.max():.3f}"
    )
    return edge_index, edge_attr


def graph_for_training(
    data: HeteroData,
    train_received: torch.Tensor,
) -> HeteroData:
    """
    Training graph: only TRAIN received_past edges (val/test labels hidden from message passing).
    """
    try:
        out = data.clone()
    except AttributeError:
        out = HeteroData()
        for nt in data.node_types:
            out[nt].x = data[nt].x.clone()
        for et in data.edge_types:
            out[et].edge_index = data[et].edge_index.clone()

    if ("researcher", "RECEIVED_PAST", "grant") in out.edge_types:
        del out["researcher", "RECEIVED_PAST", "grant"]
    if train_received.shape[1] > 0:
        out["researcher", "RECEIVED_PAST", "grant"].edge_index = train_received
    return out
