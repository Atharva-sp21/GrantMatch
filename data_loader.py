import json
import csv
import torch
from pathlib import Path

DATA_DIR = Path('data/raw')

def load_json(filename):
    """Load JSON file from data/raw/"""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"[WARN] {filename} not found, returning empty list")
        return []
    with open(filepath, 'r') as f:
        return json.load(f)

def load_csv_to_tensor(filename):
    """Load CSV and convert to edge tensor [2, num_edges]"""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"[WARN] {filename} not found, returning empty tensor")
        return torch.zeros(2, 0, dtype=torch.long)

    edges = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                src = int(row['source_id'])
                dst = int(row['target_id'])
                edges.append([src, dst])
            except (ValueError, KeyError):
                continue

    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)

    edges_array = torch.tensor(edges, dtype=torch.long).t()
    return edges_array

def load_all_data():
    """Load all JSON and CSV files into proper format"""
    print("[INFO] Loading data from data/raw/...")

    researchers = load_json('researchers.json')
    institutions = load_json('institutions.json')
    agencies = load_json('agencies.json')
    grants = load_json('grants.json')
    topics = load_json('topics.json')

    print(f"[OK] Loaded {len(researchers)} researchers")
    print(f"[OK] Loaded {len(institutions)} institutions")
    print(f"[OK] Loaded {len(agencies)} agencies")
    print(f"[OK] Loaded {len(grants)} grants")
    print(f"[OK] Loaded {len(topics)} topics")

    affiliated = load_csv_to_tensor('affiliated.csv')
    researches = load_csv_to_tensor('researches.csv')
    received = load_csv_to_tensor('received_past.csv')
    funds = load_csv_to_tensor('funds_topic.csv')
    provides = load_csv_to_tensor('provides.csv')

    print(f"[OK] Loaded {affiliated.shape[1]} affiliated edges")
    print(f"[OK] Loaded {researches.shape[1]} researches edges")
    print(f"[OK] Loaded {received.shape[1]} received edges")
    print(f"[OK] Loaded {funds.shape[1]} funds edges")
    print(f"[OK] Loaded {provides.shape[1]} provides edges")

    return {
        'researchers': researchers,
        'institutions': institutions,
        'agencies': agencies,
        'grants': grants,
        'topics': topics,
        'affiliated': affiliated,
        'researches': researches,
        'received': received,
        'funds': funds,
        'provides': provides,
    }
