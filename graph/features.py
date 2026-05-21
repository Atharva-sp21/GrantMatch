import torch
import numpy as np

def normalize_features(features_tensor):
    """Standardize features to mean=0, std=1."""
    features = features_tensor.float()
    mean = features.mean(dim=0)
    std = features.std(dim=0)
    std[std == 0] = 1.0
    normalized = (features - mean) / (std + 1e-8)
    normalized[torch.isnan(normalized)] = 0.0
    normalized[torch.isinf(normalized)] = 0.0
    return torch.clamp(normalized, -5.0, 5.0)

def build_researcher_features(researchers):
    """Build researcher features from citation/publication metrics."""
    rows = []
    for r in researchers:
        row = [
            r.get('h_index', 0),
            r.get('works_count', 0),
            r.get('cited_by_count', 0),
            r.get('i10_index', 0),
            1.0 if r.get('institution_id') is not None else 0.0,
            r.get('h_index', 0) / 50.0,
            r.get('cited_by_count', 0) / 1000.0,
            r.get('works_count', 0) / 100.0,
            1.0 if r.get('h_index', 0) > 0 else 0.0,
        ]
        rows.append(row)
    tensor = torch.tensor(rows, dtype=torch.float)
    return normalize_features(tensor)

def build_institution_features(institutions):
    """Build institution features."""
    rows = []
    for i in institutions:
        rows.append([
            i.get('researcher_count', 0) / 100.0,
            1.0 if i.get('name') else 0.0,
            i.get('total_works', 0) / 1000.0,
            i.get('researcher_count', 0) / 100.0,
            i.get('researcher_count', 0) / 500.0,
        ])
    if not rows:
        return torch.zeros(0, 5, dtype=torch.float)
    tensor = torch.tensor(rows, dtype=torch.float)
    return normalize_features(tensor)

def build_topic_features(topics):
    """Build topic features."""
    rows = []
    for t in topics:
        rows.append([
            t.get('grant_count', 0) / 10.0,
            t.get('researcher_count', 0) / 10.0,
            1.0 if t.get('name') else 0.0,
            encode_field(t.get('name', 'unknown')),
            t.get('grant_count', 0) / 50.0,
        ])
    tensor = torch.tensor(rows, dtype=torch.float)
    return normalize_features(tensor)

def build_grant_features(grants):
    """Build grant features emphasizing funding amount and relevance."""
    rows = []
    for g in grants:
        row = [
            g.get('amount', 0) / 100000.0,
            1.0,
            1.0,
            2.0,
            0.5 + (g.get('amount', 0) / 1000000.0),
            g.get('amount', 0) / 50000.0,
            1.0 if g.get('title') else 0.0,
        ]
        rows.append(row)
    tensor = torch.tensor(rows, dtype=torch.float)
    return normalize_features(tensor)

def build_agency_features(agencies):
    """Build agency features."""
    rows = []
    for a in agencies:
        rows.append([
            1.0,
            a.get('total_funding', 0) / 1000000.0,
            a.get('total_funding', 0) / max(a.get('grant_count', 1), 1) / 100000.0,
            a.get('grant_count', 0) / 50.0,
        ])
    tensor = torch.tensor(rows, dtype=torch.float)
    return normalize_features(tensor)

def encode_field(s):
    """Encode research field."""
    mapping = {'AI': 0, 'Machine Learning': 0, 'Biology': 1, 'Chemistry': 1,
               'Physics': 2, 'Engineering': 3, 'Medicine': 4, 'Climate': 4,
               'Energy': 3, 'Quantum': 2, 'unknown': 5}
    for key in mapping:
        if key.lower() in s.lower():
            return float(mapping[key])
    return 5.0