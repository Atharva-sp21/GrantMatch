"""
PI / researcher name matching with confidence scoring (OpenAlex).

Reduces noisy received_past labels from naive string equality.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

OPENALEX = "https://api.openalex.org"
MAILTO = "grantmatch@example.com"


@dataclass
class MatchResult:
    query_name: str
    openalex_id: Optional[str]
    matched_name: Optional[str]
    confidence: float
    method: str
    duplicate_candidates: int


def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def name_tokens(name: str) -> Tuple[str, str, set]:
    n = normalize_name(name)
    parts = n.split()
    if not parts:
        return "", "", set()
    last = parts[-1]
    first = parts[0]
    return first, last, set(parts)


def score_name_match(query: str, candidate: str) -> float:
    q_first, q_last, q_set = name_tokens(query)
    c_first, c_last, c_set = name_tokens(candidate)
    if not q_last or not c_last:
        return 0.0
    if q_last != c_last:
        return 0.0
    if normalize_name(query) == normalize_name(candidate):
        return 1.0
    # First initial match
    if q_first and c_first and q_first[0] == c_first[0]:
        base = 0.75
    else:
        base = 0.5
    overlap = len(q_set & c_set) / max(len(q_set | c_set), 1)
    return min(0.95, base + 0.2 * overlap)


def search_openalex_candidates(name: str, limit: int = 5) -> List[dict]:
    try:
        r = requests.get(
            f"{OPENALEX}/authors",
            params={"search": name, "per_page": limit, "mailto": MAILTO},
            timeout=30,
        )
        if r.status_code != 200:
            return []
        return r.json().get("results", [])
    except requests.RequestException:
        return []


def match_pi_to_openalex(
    pi_name: str,
    institution_hint: Optional[str] = None,
) -> MatchResult:
    """Score best OpenAlex author for a PI name."""
    if not pi_name or len(pi_name.strip()) < 2:
        return MatchResult(pi_name, None, None, 0.0, "empty", 0)

    candidates = search_openalex_candidates(pi_name, limit=5)
    if not candidates:
        return MatchResult(pi_name, None, None, 0.0, "no_candidates", 0)

    scored = []
    for c in candidates:
        cname = c.get("display_name", "")
        s = score_name_match(pi_name, cname)
        if institution_hint and c.get("last_known_institutions"):
            inst_names = [
                (i.get("display_name") or "").lower()
                for i in c["last_known_institutions"]
            ]
            if institution_hint.lower() in " ".join(inst_names):
                s += 0.1
        scored.append((s, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    dup = sum(1 for s, _ in scored if s >= best_score - 0.05)

    conf = best_score
    if dup > 1:
        conf *= 0.85  # penalize ambiguity

    method = "openalex_search"
    if conf >= 0.9:
        method = "high_confidence"
    elif conf >= 0.7:
        method = "medium_confidence"
    else:
        method = "low_confidence"

    return MatchResult(
        query_name=pi_name,
        openalex_id=best.get("id"),
        matched_name=best.get("display_name"),
        confidence=round(conf, 4),
        method=method,
        duplicate_candidates=dup,
    )


def refine_researcher_records(
    researchers: List[dict],
    min_confidence: float = 0.72,
) -> Tuple[List[dict], dict]:
    """Add match_confidence / fix openalex_id for name-only records."""
    report = {
        "total": len(researchers),
        "had_openalex": 0,
        "matched_new": 0,
        "low_confidence": 0,
        "high_confidence": 0,
        "samples": [],
    }

    for r in researchers:
        if r.get("openalex_id"):
            r["match_confidence"] = r.get("match_confidence", 1.0)
            report["had_openalex"] += 1
            continue

        m = match_pi_to_openalex(r.get("name", ""))
        r["match_confidence"] = m.confidence
        r["openalex_match_name"] = m.matched_name

        if m.openalex_id and m.confidence >= min_confidence:
            r["openalex_id"] = m.openalex_id
            report["matched_new"] += 1
            report["high_confidence"] += 1
        elif m.confidence < min_confidence:
            report["low_confidence"] += 1

        if len(report["samples"]) < 20:
            report["samples"].append(m.__dict__)

    return researchers, report


def save_matching_report(report: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
