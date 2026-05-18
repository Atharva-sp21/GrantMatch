import torch
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('allenai-specter')

# ── Researcher tensor [N, 9] ─────────────────────────────────────────
def build_researcher_features(researchers):
    rows = []
    for r in researchers:
        abstract_emb = embedder.encode(r.get('abstract', ''))
        row = [
            r.get('h_index', 0),
            r.get('publication_count', 0),
            r.get('citation_count', 0),
            r.get('career_age', 0),
            encode_career_stage(r.get('career_stage', 'unknown')),
            r.get('grant_success_rate', 0.0),
            r.get('total_funding', 0),
        ]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float)

# ── Institution tensor [N, 5] ────────────────────────────────────────
def build_institution_features(institutions):
    rows = []
    for i in institutions:
        rows.append([
            i.get('world_ranking', 999),
            encode_inst_type(i.get('institution_type', 'unknown')),
            i.get('rd_spend', 0),
            encode_country(i.get('country', 'unknown')),
            i.get('faculty_size', 0),
        ])
    return torch.tensor(rows, dtype=torch.float)

# ── Topic tensor [N, 5] ──────────────────────────────────────────────
def build_topic_features(topics):
    rows = []
    for t in topics:
        rows.append([
            t.get('funding_frequency', 0),
            t.get('researcher_count', 0),
            t.get('trend_score', 0),
            encode_field(t.get('field_of_study', 'unknown')),
            0,  # placeholder for topic embedding index
        ])
    return torch.tensor(rows, dtype=torch.float)

# ── Grant tensor [N, 7] ──────────────────────────────────────────────
def build_grant_features(grants):
    rows = []
    for g in grants:
        rows.append([
            g.get('funding_amount', 0),
            encode_grant_type(g.get('grant_type', 'unknown')),
            g.get('deadline_days_remaining', 0),
            encode_career_stage(g.get('career_stage_eligibility', 'any')),
            g.get('acceptance_rate', 0.0),
            encode_country(g.get('country_restriction', 'any')),
            0,  # placeholder for guidelines embedding index
        ])
    return torch.tensor(rows, dtype=torch.float)

# ── Agency tensor [N, 4] ─────────────────────────────────────────────
def build_agency_features(agencies):
    rows = []
    for a in agencies:
        rows.append([
            encode_agency_type(a.get('agency_type', 'unknown')),
            a.get('total_annual_budget', 0),
            a.get('avg_grant_size', 0),
            0,  # placeholder for focus area embedding index
        ])
    return torch.tensor(rows, dtype=torch.float)

# ── Encoders ─────────────────────────────────────────────────────────
def encode_career_stage(s):
    return {'phd': 0, 'postdoc': 1, 'assistant_prof': 2,
            'full_prof': 3, 'any': 4, 'unknown': 4}.get(s, 4)

def encode_inst_type(s):
    return {'public': 0, 'private': 1, 'hospital': 2,
            'national_lab': 3, 'unknown': 4}.get(s, 4)

def encode_grant_type(s):
    return {'fellowship': 0, 'project': 1, 'equipment': 2,
            'travel': 3, 'collaborative': 4, 'unknown': 5}.get(s, 5)

def encode_agency_type(s):
    return {'government': 0, 'private': 1,
            'corporate': 2, 'international': 3, 'unknown': 4}.get(s, 4)

def encode_country(s):
    mapping = {'US': 0, 'UK': 1, 'EU': 2, 'any': 3, 'unknown': 4}
    return mapping.get(s, 4)

def encode_field(s):
    mapping = {'AI': 0, 'Biology': 1, 'Physics': 2,
               'Chemistry': 3, 'Engineering': 4, 'unknown': 5}
    return mapping.get(s, 5)