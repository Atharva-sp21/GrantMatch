"""
Tabular features (log1p + standard scale) + optional text embeddings + grant TF-IDF stats.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from graph.embeddings import grant_text, researcher_text


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def log1p_transform(values: List[float]) -> List[float]:
    return [math.log1p(max(0.0, v)) for v in values]


def standard_scale(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    x = matrix.float()
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    std[std < 1e-6] = 1.0
    out = (x - mean) / std
    out[torch.isnan(out)] = 0.0
    return torch.clamp(out, -5.0, 5.0)


def encode_career_stage(s) -> float:
    return float(
        {"phd": 0, "postdoc": 1, "assistant_prof": 2, "full_prof": 3, "any": 4, "unknown": 4}.get(s, 4)
    )


def encode_inst_type(s) -> float:
    return float(
        {"public": 0, "private": 1, "hospital": 2, "national_lab": 3, "unknown": 4}.get(s, 4)
    )


def encode_grant_type(s) -> float:
    return float(
        {"fellowship": 0, "project": 1, "equipment": 2, "travel": 3, "collaborative": 4, "unknown": 5}.get(s, 5)
    )


def encode_agency_type(s) -> float:
    return float(
        {"government": 0, "private": 1, "corporate": 2, "international": 3, "unknown": 4}.get(s, 4)
    )


def encode_country(s) -> float:
    return float({"US": 0, "UK": 1, "EU": 2, "any": 3, "unknown": 4}.get(s, 4))


def encode_field(s) -> float:
    return float(
        {"AI": 0, "Biology": 1, "Physics": 2, "Chemistry": 3, "Engineering": 4, "unknown": 5}.get(s, 5)
    )


def _scale_columns(raw: torch.Tensor, cols: List[int]) -> torch.Tensor:
    for col in cols:
        raw[:, col] = torch.tensor(log1p_transform(raw[:, col].tolist()))
    return standard_scale(raw)


def build_researcher_features(
    researchers: List[dict],
    text_embeddings: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    rows = []
    for r in researchers:
        rows.append(
            [
                _safe_float(r.get("h_index", 0)),
                _safe_float(r.get("publication_count", r.get("works_count", 0))),
                _safe_float(r.get("citation_count", r.get("cited_by_count", 0))),
                _safe_float(r.get("career_age", 0)),
                _safe_float(r.get("grant_success_rate", 0)),
                _safe_float(r.get("total_funding", 0)),
                _safe_float(r.get("publication_velocity", 0)),
                _safe_float(r.get("avg_citations_per_paper", 0)),
                _safe_float(r.get("years_since_first_publication", 0)),
                _safe_float(r.get("years_since_latest_publication", 0)),
                _safe_float(r.get("collaboration_count", 0)),
                _safe_float(r.get("institution_prestige", 0)),
                _safe_float(r.get("topic_diversity_score", 0)),
                encode_career_stage(r.get("career_stage", "unknown")),
            ]
        )
    if not rows:
        return torch.zeros(0, 0)
    raw = torch.tensor(rows, dtype=torch.float)
    tabular = _scale_columns(raw, list(range(13)))

    if text_embeddings is not None and text_embeddings.shape[0] == len(researchers):
        return torch.cat([tabular, standard_scale(text_embeddings)], dim=1)
    return tabular


def build_institution_features(institutions: List[dict]) -> torch.Tensor:
    rows = []
    for i in institutions:
        rows.append(
            [
                _safe_float(i.get("world_ranking", 999)),
                encode_inst_type(i.get("institution_type", "unknown")),
                _safe_float(i.get("rd_spend", 0)),
                encode_country(i.get("country", "unknown")),
                _safe_float(i.get("faculty_size", 0)),
            ]
        )
    if not rows:
        return torch.zeros(0, 5)
    return _scale_columns(torch.tensor(rows, dtype=torch.float), [0, 2, 4])


def build_topic_features(topics: List[dict]) -> torch.Tensor:
    rows = []
    for t in topics:
        rows.append(
            [
                _safe_float(t.get("funding_frequency", 0)),
                _safe_float(t.get("researcher_count", 0)),
                _safe_float(t.get("trend_score", 0)),
                encode_field(t.get("field_of_study", t.get("name", "unknown"))),
            ]
        )
    if not rows:
        return torch.zeros(0, 4)
    return _scale_columns(torch.tensor(rows, dtype=torch.float), [0, 1, 2])


def _grant_extra_features(grants: List[dict], agencies: List[dict]) -> torch.Tensor:
    amounts = [_safe_float(g.get("funding_amount", g.get("amount", 0))) for g in grants]
    sorted_a = sorted(amounts)
    agency_budget = {a["id"]: _safe_float(a.get("total_annual_budget", a.get("total_funding", 0))) for a in agencies}
    max_b = max(agency_budget.values()) if agency_budget else 1.0

    texts = [grant_text(g) for g in grants]
    tfidf = TfidfVectorizer(max_features=16, stop_words="english")
    try:
        tfidf_mat = tfidf.fit_transform(texts).toarray()
    except ValueError:
        tfidf_mat = [[0.0] * 16 for _ in grants]

    rows = []
    for i, g in enumerate(grants):
        amt = amounts[i]
        pct = 0.5
        if sorted_a:
            rank = sum(1 for a in sorted_a if a <= amt) / len(sorted_a)
            pct = rank
        agency_id = g.get("agency_id")
        agency_rep = agency_budget.get(agency_id, 0) / max(max_b, 1.0)
        topic_div = len(g.get("topic_ids") or [])
        text_len = len(texts[i]) / 5000.0
        dl = _safe_float(g.get("deadline_days_remaining", 0))
        rows.append([pct, agency_rep, topic_div, text_len, dl] + tfidf_mat[i].tolist()[:16])

    raw = torch.tensor(rows, dtype=torch.float)
    return _scale_columns(raw, [0, 2, 3, 4])


def _deadline_features(grant: dict) -> List[float]:
    days = _safe_float(grant.get("deadline_days_remaining", 0))
    has_deadline = 1.0 if grant.get("deadline") else 0.0
    return [days, has_deadline]


def build_grant_features(
    grants: List[dict],
    text_embeddings: Optional[torch.Tensor] = None,
    agencies: Optional[List[dict]] = None,
) -> torch.Tensor:
    rows = []
    for g in grants:
        dl = _deadline_features(g)
        rows.append(
            [
                _safe_float(g.get("funding_amount", g.get("amount", 0))),
                _safe_float(g.get("acceptance_rate", 0)),
                dl[0],
                dl[1],
                encode_grant_type(g.get("grant_type", "unknown")),
                encode_career_stage(g.get("career_stage_eligibility", "any")),
                encode_country(g.get("country_restriction", "any")),
            ]
        )
    if not rows:
        return torch.zeros(0, 0)
    tabular = _scale_columns(torch.tensor(rows, dtype=torch.float), [0, 2])

    extra = _grant_extra_features(grants, agencies or [])
    tabular = torch.cat([tabular, extra], dim=1)

    if text_embeddings is not None and text_embeddings.shape[0] == len(grants):
        return torch.cat([tabular, standard_scale(text_embeddings)], dim=1)
    return tabular


def build_agency_features(agencies: List[dict]) -> torch.Tensor:
    rows = []
    for a in agencies:
        rows.append(
            [
                encode_agency_type(a.get("agency_type", "unknown")),
                _safe_float(a.get("total_annual_budget", a.get("total_funding", 0))),
                _safe_float(a.get("avg_grant_size", 0)),
                _safe_float(a.get("grant_count", 0)),
            ]
        )
    if not rows:
        return torch.zeros(0, 4)
    return _scale_columns(torch.tensor(rows, dtype=torch.float), [1, 2, 3])


def build_all_node_features(
    researchers: List[dict],
    institutions: List[dict],
    topics: List[dict],
    grants: List[dict],
    agencies: List[dict],
    researcher_emb: Optional[torch.Tensor] = None,
    grant_emb: Optional[torch.Tensor] = None,
):
    return {
        "researcher": build_researcher_features(researchers, researcher_emb),
        "institution": build_institution_features(institutions),
        "topic": build_topic_features(topics),
        "grant": build_grant_features(grants, grant_emb, agencies),
        "agency": build_agency_features(agencies),
    }
