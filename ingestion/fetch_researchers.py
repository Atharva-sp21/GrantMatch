import html
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

OPENALEX_URL = "https://api.openalex.org/authors"
PAGE_SIZE = 200
REQUEST_TIMEOUT = 30
DEFAULT_LIMIT = 3000


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _name_variants(name: str) -> List[str]:
    normalized = _clean_text(name)
    if not normalized:
        return []

    variants = {normalized}
    if "," in normalized:
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
        if len(parts) >= 2:
            variants.add(_clean_text(" ".join(parts[1:] + parts[:1])))

    tokens = normalized.split()
    if len(tokens) >= 2:
        variants.add(" ".join(tokens[:2]))
        variants.add(" ".join(tokens[-2:]))
        variants.add(tokens[-1])
        variants.add(f"{tokens[0][0]} {tokens[-1]}")

    return sorted(variant for variant in variants if variant)


def _extract_aliases(result: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []
    raw_aliases = result.get("display_name_alternatives") or []
    if isinstance(raw_aliases, str):
        raw_aliases = [raw_aliases]

    for alias in [result.get("display_name") or "", *raw_aliases]:
        aliases.extend(_name_variants(alias))

    unique_aliases: List[str] = []
    seen = set()
    for alias in aliases:
        normalized = alias.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_aliases.append(normalized)
    return unique_aliases


def _extract_concepts(result: Dict[str, Any], limit: int = 12) -> List[Dict[str, Any]]:
    concepts: List[Dict[str, Any]] = []
    for concept in result.get("x_concepts", [])[:limit]:
        if not isinstance(concept, dict):
            continue
        name = concept.get("display_name") or concept.get("name")
        if not name:
            continue
        concepts.append(
            {
                "name": _clean_text(name),
                "score": float(concept.get("score", 0.0) or 0.0),
            }
        )
    return concepts


def _iter_institution_candidates(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    def _append(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                _append(item)
            return
        if not isinstance(value, dict):
            return
        nested = value.get("institution") if isinstance(value.get("institution"), dict) else None
        candidate = nested or value
        if isinstance(candidate, dict):
            candidates.append(candidate)

    _append(result.get("affiliations"))
    _append(result.get("last_known_institution"))
    _append(result.get("last_known_institutions"))
    _append(result.get("institutions"))

    unique: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        candidate_id = candidate.get("id") or candidate.get("ror") or candidate.get("display_name")
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        unique.append(candidate)
    return unique


def _register_institution(mapping: Dict[str, Dict[str, Any]], candidate: Dict[str, Any]) -> Optional[int]:
    institution_id = candidate.get("id") or candidate.get("openalex_id") or candidate.get("ror")
    if not institution_id:
        return None

    if institution_id not in mapping:
        mapping[institution_id] = {
            "id": len(mapping),
            "openalex_id": candidate.get("id") or "",
            "ror": candidate.get("ror") or "",
            "name": _clean_text(candidate.get("display_name") or candidate.get("name") or ""),
            "country_code": candidate.get("country_code") or "",
            "type": candidate.get("type") or "",
        }

    return mapping[institution_id]["id"]


def _choose_primary_institution(result: Dict[str, Any], institution_mapping: Dict[str, Dict[str, Any]]) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    for candidate in _iter_institution_candidates(result):
        internal_id = _register_institution(institution_mapping, candidate)
        if internal_id is not None:
            return internal_id, candidate.get("id") or candidate.get("ror"), _clean_text(candidate.get("display_name") or candidate.get("name") or "")
    return None, None, None


def fetch_researchers(limit: int = 1000):
    """Fetch real researcher profiles from OpenAlex."""
    researchers: List[Dict[str, Any]] = []
    institution_mapping: Dict[str, Dict[str, Any]] = {}

    session = requests.Session()
    params = {
        "per-page": PAGE_SIZE,
        "cursor": "*",
        "sort": "cited_by_count:desc",
    }

    print("[INFO] Fetching researchers from OpenAlex...")

    while len(researchers) < limit:
        response = session.get(OPENALEX_URL, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenAlex request failed with {response.status_code}: {response.text[:500]}"
            )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            break

        for result in results:
            if len(researchers) >= limit:
                break

            institution_id, institution_external_id, institution_name = _choose_primary_institution(result, institution_mapping)
            summary_stats = result.get("summary_stats") or {}
            aliases = _extract_aliases(result)

            researcher = {
                "id": len(researchers),
                "openalex_id": result.get("id") or "",
                "name": _clean_text(result.get("display_name") or ""),
                "aliases": aliases,
                "institution_id": institution_id,
                "institution_openalex_id": institution_external_id or "",
                "institution_name": institution_name or "",
                "works_count": int(result.get("works_count", 0) or 0),
                "cited_by_count": int(result.get("cited_by_count", 0) or 0),
                "h_index": int(summary_stats.get("h_index", 0) or 0),
                "i10_index": int(summary_stats.get("i10_index", 0) or 0),
                "concepts": _extract_concepts(result),
            }
            researchers.append(researcher)

        next_cursor = (data.get("meta") or {}).get("next_cursor")
        if not next_cursor:
            break
        params["cursor"] = next_cursor
        time.sleep(0.05)

    if not researchers:
        raise RuntimeError("OpenAlex returned no researcher records.")

    print(f"[OK] Fetched {len(researchers)} researchers")
    print(f"[OK] Found {len(institution_mapping)} institutions")
    return researchers, institution_mapping


if __name__ == "__main__":
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    os.makedirs(output_dir, exist_ok=True)

    researchers, institution_mapping = fetch_researchers(limit=DEFAULT_LIMIT)

    researchers_path = os.path.join(output_dir, "researchers_raw.json")
    institutions_path = os.path.join(output_dir, "institution_mapping.json")

    with open(researchers_path, "w", encoding="utf-8") as handle:
        json.dump(researchers, handle, indent=2)

    with open(institutions_path, "w", encoding="utf-8") as handle:
        json.dump(institution_mapping, handle, indent=2)

    print(f"[OK] Saved {researchers_path}")
    print(f"[OK] Saved {institutions_path}")
