import hashlib
import math
import re
from functools import lru_cache
from typing import Iterable, List

import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class _HashingTextEncoder:
    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True):
        vectors = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for row_index, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9]+", str(text).lower())
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                bucket = int(digest[:8], 16) % EMBEDDING_DIM
                vectors[row_index, bucket] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(vectors[row_index])
                if norm > 0:
                    vectors[row_index] /= norm
        return vectors


@lru_cache(maxsize=1)
def _load_encoder():
    if SentenceTransformer is None:
        return _HashingTextEncoder()
    return SentenceTransformer(EMBEDDING_MODEL)


def normalize_features(features_tensor: torch.Tensor) -> torch.Tensor:
    if features_tensor.numel() == 0:
        return features_tensor.float()
    features = features_tensor.float()
    mean = features.mean(dim=0)
    std = features.std(dim=0)
    std[std == 0] = 1.0
    normalized = (features - mean) / (std + 1e-8)
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(normalized, -5.0, 5.0)


def _embed_texts(texts: List[str]) -> torch.Tensor:
    if not texts:
        return torch.zeros(0, EMBEDDING_DIM, dtype=torch.float)

    encoder = _load_encoder()
    vectors = encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    return torch.from_numpy(vectors)


def _combine_text_and_numeric(texts: List[str], numeric_rows: List[List[float]]) -> torch.Tensor:
    text_tensor = _embed_texts(texts)
    if numeric_rows:
        numeric_tensor = normalize_features(torch.tensor(numeric_rows, dtype=torch.float))
    else:
        numeric_tensor = torch.zeros(len(texts), 0, dtype=torch.float)

    if text_tensor.shape[0] == 0:
        return numeric_tensor
    if numeric_tensor.shape[0] == 0:
        return text_tensor
    return torch.cat([text_tensor, numeric_tensor], dim=-1)


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_safe_text(item) for item in value if _safe_text(item))
    if isinstance(value, dict):
        for key in ("name", "display_name", "title"):
            if value.get(key):
                return _safe_text(value.get(key))
    return str(value).strip()


def build_researcher_features(researchers):
    texts = []
    numeric_rows = []
    for researcher in researchers:
        concepts = ", ".join(
            _safe_text(concept.get("name") if isinstance(concept, dict) else concept)
            for concept in researcher.get("concepts", [])
            if _safe_text(concept.get("name") if isinstance(concept, dict) else concept)
        )
        if not concepts:
            concepts = researcher.get("name", "")
        texts.append(concepts)
        numeric_rows.append(
            [
                math.log1p(max(float(researcher.get("h_index", 0) or 0), 0.0)),
                math.log1p(max(float(researcher.get("works_count", 0) or 0), 0.0)),
                math.log1p(max(float(researcher.get("cited_by_count", 0) or 0), 0.0)),
                math.log1p(max(float(researcher.get("i10_index", 0) or 0), 0.0)),
                1.0 if researcher.get("institution_id") is not None else 0.0,
                float(len(researcher.get("concepts", [])) or 0),
            ]
        )
    if not texts:
        return torch.zeros(0, EMBEDDING_DIM + 6, dtype=torch.float)
    return _combine_text_and_numeric(texts, numeric_rows)


def build_institution_features(institutions):
    texts = []
    numeric_rows = []
    for institution in institutions:
        texts.append(_safe_text(institution.get("name", "")) or "institution")
        researcher_count = float(institution.get("researcher_count", 0) or 0)
        total_works = float(institution.get("total_works", 0) or 0)
        total_citations = float(institution.get("total_citations", 0) or 0)
        numeric_rows.append(
            [
                math.log1p(researcher_count),
                math.log1p(total_works),
                math.log1p(total_citations),
                total_works / max(researcher_count, 1.0),
            ]
        )
    if not texts:
        return torch.zeros(0, EMBEDDING_DIM + 4, dtype=torch.float)
    return _combine_text_and_numeric(texts, numeric_rows)


def build_topic_features(topics):
    texts = []
    numeric_rows = []
    for topic in topics:
        texts.append(_safe_text(topic.get("name", "")) or "topic")
        numeric_rows.append(
            [
                float(topic.get("researcher_count", 0) or 0),
                float(topic.get("grant_count", 0) or 0),
            ]
        )
    if not texts:
        return torch.zeros(0, EMBEDDING_DIM + 2, dtype=torch.float)
    return _combine_text_and_numeric(texts, numeric_rows)


def build_grant_features(grants):
    texts = []
    numeric_rows = []
    for grant in grants:
        title = _safe_text(grant.get("title", ""))
        abstract = _safe_text(grant.get("abstract", ""))
        texts.append(" ".join(part for part in [title, abstract] if part) or "grant")
        amount = float(grant.get("amount", 0) or 0)
        numeric_rows.append(
            [
                math.log1p(max(amount, 0.0)),
                float(grant.get("title_length", 0) or 0),
                float(grant.get("abstract_length", 0) or 0),
                float(grant.get("investigator_count", 0) or 0),
                1.0 if amount > 0 else 0.0,
                float(_date_span_days(grant.get("start_date", ""), grant.get("end_date", ""))),
            ]
        )
    if not texts:
        return torch.zeros(0, EMBEDDING_DIM + 6, dtype=torch.float)
    return _combine_text_and_numeric(texts, numeric_rows)


def build_agency_features(agencies):
    texts = []
    numeric_rows = []
    for agency in agencies:
        texts.append(_safe_text(agency.get("name", "")) or "agency")
        total_funding = float(agency.get("total_funding", 0) or 0)
        grant_count = float(agency.get("grant_count", 0) or 0)
        numeric_rows.append(
            [
                math.log1p(grant_count),
                math.log1p(total_funding),
                total_funding / max(grant_count, 1.0),
            ]
        )
    if not texts:
        return torch.zeros(0, EMBEDDING_DIM + 3, dtype=torch.float)
    return _combine_text_and_numeric(texts, numeric_rows)


def _date_span_days(start_date: str, end_date: str) -> float:
    start_year = _extract_year(start_date)
    end_year = _extract_year(end_date)
    if start_year is None or end_year is None:
        return 0.0
    return float(max((end_year - start_year) * 365, 0))


def _extract_year(value: str):
    match = re.search(r"(\d{4})", str(value or ""))
    return int(match.group(1)) if match else None