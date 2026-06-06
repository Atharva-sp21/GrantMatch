"""Load JSON/CSV from data/raw/ for the ML pipeline."""

import csv
import json
from pathlib import Path

import torch

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"

EDGE_COLUMN_ALIASES = [
    ("source_id", "target_id"),
    ("researcher_id", "institution_id"),
    ("researcher_id", "topic_id"),
    ("researcher_id", "grant_id"),
    ("grant_id", "topic_id"),
    ("agency_id", "grant_id"),
]


def load_json(filename: str):
    path = RAW_DIR / filename
    if not path.exists():
        print(f"[WARN] Missing {path}")
        return []
    with open(path) as f:
        return json.load(f)


def _resolve_columns(fieldnames):
    if not fieldnames:
        return None, None
    norm = {n.strip().lower(): n for n in fieldnames}
    for src, dst in EDGE_COLUMN_ALIASES:
        if src in norm and dst in norm:
            return norm[src], norm[dst]
    return None, None


def load_csv_edges(filename: str) -> torch.Tensor:
    path = RAW_DIR / filename
    if not path.exists():
        return torch.zeros(2, 0, dtype=torch.long)

    edges = []
    with open(path) as f:
        reader = csv.DictReader(f)
        src_col, dst_col = _resolve_columns(reader.fieldnames)
        if not src_col:
            print(f"[WARN] Unrecognized columns in {filename}: {reader.fieldnames}")
            return torch.zeros(2, 0, dtype=torch.long)
        for row in reader:
            try:
                edges.append([int(row[src_col]), int(row[dst_col])])
            except (ValueError, KeyError):
                continue

    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def enrich_topics(topics, researches, funds):
    if not topics:
        return topics
    n = len(topics)
    rc, gc = [0] * n, [0] * n
    if researches.shape[1]:
        for d in researches[1].tolist():
            if 0 <= d < n:
                rc[d] += 1
    if funds.shape[1]:
        for d in funds[1].tolist():
            if 0 <= d < n:
                gc[d] += 1
    for i, t in enumerate(topics):
        if not t.get("researcher_count"):
            t["researcher_count"] = rc[i]
        if not t.get("grant_count"):
            t["grant_count"] = gc[i]
    return topics


def load_all_data():
    print(f"[INFO] Loading from {RAW_DIR}")
    researchers = load_json("researchers.json")
    institutions = load_json("institutions.json")
    grants = load_json("grants.json")
    topics = load_json("topics.json")
    agencies = load_json("agencies.json")

    affiliated = load_csv_edges("affiliated.csv")
    researches = load_csv_edges("researches.csv")
    received = load_csv_edges("received_past.csv")
    funds = load_csv_edges("funds_topic.csv")
    provides = load_csv_edges("provides.csv")

    topics = enrich_topics(topics, researches, funds)

    print(f"[OK] researchers={len(researchers)} institutions={len(institutions)} "
          f"grants={len(grants)} topics={len(topics)} agencies={len(agencies)}")
    print(f"[OK] edges: aff={affiliated.shape[1]} res={researches.shape[1]} "
          f"recv={received.shape[1]} funds={funds.shape[1]} prov={provides.shape[1]}")

    return {
        "researchers": researchers,
        "institutions": institutions,
        "grants": grants,
        "topics": topics,
        "agencies": agencies,
        "affiliated": affiliated,
        "researches": researches,
        "received": received,
        "funds": funds,
        "provides": provides,
    }
