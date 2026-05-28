import csv
import json
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"


def load_json(filename):
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Required data file missing: {filepath}")
    with open(filepath, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{filename} must contain a JSON array")
    return data


def load_csv_to_tensor(filename):
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Required data file missing: {filepath}")

    edges = []
    with open(filepath, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                edges.append([int(row["source_id"]), int(row["target_id"])])
            except (KeyError, ValueError):
                continue

    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def validate_id_sequence(records, filename):
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{filename} must contain objects at every position")
        if record.get("id") != index:
            raise ValueError(
                f"{filename} has inconsistent id at index {index}: expected {index}, found {record.get('id')}"
            )


def validate_edge_tensor(edge_tensor, source_count, target_count, filename):
    if edge_tensor.numel() == 0:
        return
    if source_count == 0 or target_count == 0:
        raise ValueError(f"{filename} contains edges but one endpoint node type is empty")
    src_min = int(edge_tensor[0].min().item())
    src_max = int(edge_tensor[0].max().item())
    dst_min = int(edge_tensor[1].min().item())
    dst_max = int(edge_tensor[1].max().item())
    if src_min < 0 or src_max >= source_count or dst_min < 0 or dst_max >= target_count:
        raise ValueError(
            f"{filename} references ids outside the loaded node ranges: source [0, {source_count - 1}], target [0, {target_count - 1}]"
        )


def load_all_data():
    print("[INFO] Loading data from data/raw...")

    researchers = load_json("researchers.json")
    institutions = load_json("institutions.json")
    agencies = load_json("agencies.json")
    grants = load_json("grants.json")
    topics = load_json("topics.json")

    validate_id_sequence(researchers, "researchers.json")
    validate_id_sequence(institutions, "institutions.json")
    validate_id_sequence(agencies, "agencies.json")
    validate_id_sequence(grants, "grants.json")
    validate_id_sequence(topics, "topics.json")

    affiliated = load_csv_to_tensor("affiliated.csv")
    researches = load_csv_to_tensor("researches.csv")
    received = load_csv_to_tensor("received_past.csv")
    funds = load_csv_to_tensor("funds_topic.csv")
    provides = load_csv_to_tensor("provides.csv")

    validate_edge_tensor(affiliated, len(researchers), len(institutions), "affiliated.csv")
    validate_edge_tensor(researches, len(researchers), len(topics), "researches.csv")
    validate_edge_tensor(received, len(researchers), len(grants), "received_past.csv")
    validate_edge_tensor(funds, len(grants), len(topics), "funds_topic.csv")
    validate_edge_tensor(provides, len(agencies), len(grants), "provides.csv")

    print(f"[OK] Loaded {len(researchers)} researchers")
    print(f"[OK] Loaded {len(institutions)} institutions")
    print(f"[OK] Loaded {len(agencies)} agencies")
    print(f"[OK] Loaded {len(grants)} grants")
    print(f"[OK] Loaded {len(topics)} topics")
    print(f"[OK] Loaded {affiliated.shape[1]} affiliated edges")
    print(f"[OK] Loaded {researches.shape[1]} researches edges")
    print(f"[OK] Loaded {received.shape[1]} received edges")
    print(f"[OK] Loaded {funds.shape[1]} funds edges")
    print(f"[OK] Loaded {provides.shape[1]} provides edges")

    return {
        "researchers": researchers,
        "institutions": institutions,
        "agencies": agencies,
        "grants": grants,
        "topics": topics,
        "affiliated": affiliated,
        "researches": researches,
        "received": received,
        "funds": funds,
        "provides": provides,
    }
