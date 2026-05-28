import html
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "it", "its", "of", "on", "or", "our", "the",
    "to", "with", "using", "via", "new", "study", "project", "research",
    "method", "methods", "approach", "approaches", "based", "toward", "towards",
    "analysis", "data", "model", "models", "system", "systems", "article",
    "paper", "papers", "jats", "xml", "title", "abstract", "introduction",
    "results", "discussion", "conclusion", "conclusions", "that", "this",
    "these", "those", "were", "was", "been", "have", "has", "had", "may",
    "can", "could", "will", "would", "should", "among", "between", "within",
}


def normalize_text(text: Any) -> str:
    cleaned = html.unescape(str(text or "")).lower()
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_text(text: Any) -> str:
    return normalize_text(text)


def token_terms(text: Any) -> List[str]:
    words = [
        word
        for word in re.findall(r"[a-z0-9]+", normalize_text(text))
        if len(word) > 2 and word not in STOPWORDS and not word.isdigit()
    ]
    terms = set(words)
    for index in range(len(words) - 1):
        bigram = f"{words[index]} {words[index + 1]}"
        if words[index] not in STOPWORDS and words[index + 1] not in STOPWORDS:
            terms.add(bigram)
    return sorted(terms)


def concept_names(researcher: Dict[str, Any]) -> List[str]:
    concepts: List[str] = []
    for concept in researcher.get("concepts", []):
        if isinstance(concept, dict):
            name = concept.get("name") or concept.get("display_name")
        else:
            name = str(concept)
        normalized = normalize_text(name)
        if normalized:
            concepts.append(normalized)
    return concepts


def load_raw_data():
    """Load intermediate data from fetch scripts."""
    base_raw = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))

    def _read_optional(path: str):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return None

    researchers_raw = _read_optional(os.path.join(base_raw, "researchers_raw.json")) or []
    grants_raw = _read_optional(os.path.join(base_raw, "grants_raw.json")) or []
    institution_mapping = _read_optional(os.path.join(base_raw, "institution_mapping.json")) or {}
    agency_mapping = _read_optional(os.path.join(base_raw, "agency_mapping.json")) or {}

    return researchers_raw, grants_raw, institution_mapping, agency_mapping


def _clean_abstract(text: Any) -> str:
    cleaned = html.unescape(str(text or ""))
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def build_researchers_json(researchers_raw: List[Dict[str, Any]]):
    researchers = []
    for index, researcher_raw in enumerate(researchers_raw):
        researchers.append(
            {
                "id": index,
                "name": researcher_raw.get("name", ""),
                "aliases": researcher_raw.get("aliases", []),
                "openalex_id": researcher_raw.get("openalex_id", ""),
                "institution_id": researcher_raw.get("institution_id"),
                "institution_openalex_id": researcher_raw.get("institution_openalex_id", ""),
                "institution_name": researcher_raw.get("institution_name", ""),
                "h_index": int(researcher_raw.get("h_index", 0) or 0),
                "i10_index": int(researcher_raw.get("i10_index", 0) or 0),
                "works_count": int(researcher_raw.get("works_count", 0) or 0),
                "cited_by_count": int(researcher_raw.get("cited_by_count", 0) or 0),
                "concepts": concept_names(researcher_raw),
                "concept_count": len(researcher_raw.get("concepts", [])),
            }
        )

    print(f"[OK] Built {len(researchers)} researcher records")
    return researchers


def build_institutions_json(institution_mapping: Dict[str, Dict[str, Any]], researchers_raw: List[Dict[str, Any]]):
    institutions: Dict[int, Dict[str, Any]] = {}
    metadata_by_id = {value["id"]: value for value in institution_mapping.values() if isinstance(value, dict) and "id" in value}

    for researcher in researchers_raw:
        institution_id = researcher.get("institution_id")
        if institution_id is None:
            continue
        if institution_id not in institutions:
            meta = metadata_by_id.get(institution_id, {})
            institutions[institution_id] = {
                "id": institution_id,
                "name": meta.get("name") or researcher.get("institution_name", ""),
                "openalex_id": meta.get("openalex_id") or researcher.get("institution_openalex_id", ""),
                "ror": meta.get("ror") or "",
                "country_code": meta.get("country_code") or "",
                "type": meta.get("type") or "",
                "researcher_count": 0,
                "total_works": 0,
                "total_citations": 0,
            }
        institutions[institution_id]["researcher_count"] += 1
        institutions[institution_id]["total_works"] += int(researcher.get("works_count", 0) or 0)
        institutions[institution_id]["total_citations"] += int(researcher.get("cited_by_count", 0) or 0)

    institutions_list = sorted(institutions.values(), key=lambda item: item["id"])
    print(f"[OK] Built {len(institutions_list)} institution records")
    return institutions_list


def build_agencies_json(agency_mapping: Dict[str, Any]):
    agencies = []
    for agency_name, agency_value in sorted(agency_mapping.items(), key=lambda item: item[1] if isinstance(item[1], int) else item[1].get("id", 0)):
        agency_id = agency_value if isinstance(agency_value, int) else int(agency_value.get("id", 0))
        agencies.append(
            {
                "id": agency_id,
                "name": agency_name,
                "grant_count": 0,
                "total_funding": 0.0,
                "average_grant_size": 0.0,
            }
        )

    print(f"[OK] Built {len(agencies)} agency records")
    return agencies


def build_grants_json(grants_raw: List[Dict[str, Any]], agencies: List[Dict[str, Any]]):
    grants = []
    agency_lookup = {agency["id"]: agency for agency in agencies}

    for index, grant_raw in enumerate(grants_raw):
        amount = float(grant_raw.get("amount", 0) or 0)
        start_date = grant_raw.get("start_date", "")
        end_date = grant_raw.get("end_date", "")
        title = _clean_abstract(grant_raw.get("title", ""))
        abstract = _clean_abstract(grant_raw.get("abstract", ""))
        investigators = [name for name in grant_raw.get("investigators", []) if normalize_text(name)]

        grants.append(
            {
                "id": index,
                "source": grant_raw.get("source", ""),
                "nih_project_number": grant_raw.get("nih_project_number", ""),
                "title": title,
                "agency_id": int(grant_raw.get("agency_id", 0) or 0),
                "amount": amount,
                "start_date": start_date,
                "end_date": end_date,
                "investigators": investigators,
                "abstract": abstract,
                "abstract_length": len(abstract.split()),
                "title_length": len(title.split()),
                "investigator_count": len(investigators),
            }
        )

        agency_id = int(grant_raw.get("agency_id", 0) or 0)
        if agency_id in agency_lookup:
            agency_lookup[agency_id]["grant_count"] += 1
            agency_lookup[agency_id]["total_funding"] += amount

    for agency in agencies:
        count = max(int(agency.get("grant_count", 0) or 0), 1)
        agency["average_grant_size"] = agency.get("total_funding", 0.0) / count if agency.get("grant_count", 0) else 0.0

    print(f"[OK] Built {len(grants)} grant records")
    return grants


def build_topics_json(researchers_raw: List[Dict[str, Any]], grants_raw: List[Dict[str, Any]]):
    topic_counts = Counter()

    for researcher in researchers_raw:
        for concept in concept_names(researcher):
            topic_counts.update(token_terms(concept))

    for grant in grants_raw:
        grant_text = " ".join(filter(None, [grant.get("title", ""), grant.get("abstract", "")]))
        topic_counts.update(token_terms(grant_text))

    if not topic_counts:
        raise RuntimeError("Could not derive any topics from API data.")

    ranked_topics = [
        term
        for term, _ in sorted(topic_counts.items(), key=lambda item: (-item[1], item[0]))
        if len(term) > 2 and term not in STOPWORDS
    ]

    topics = [
        {
            "id": index,
            "name": name,
            "researcher_count": 0,
            "grant_count": 0,
        }
        for index, name in enumerate(ranked_topics[:75])
    ]

    print(f"[OK] Built {len(topics)} topic records")
    return topics


def save_json_files(researchers, institutions, agencies, grants, topics):
    output_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw")))

    files = {
        "researchers.json": researchers,
        "institutions.json": institutions,
        "agencies.json": agencies,
        "grants.json": grants,
        "topics.json": topics,
    }

    for filename, data in files.items():
        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        print(f"[OK] Saved {filepath}")


if __name__ == "__main__":
    print("[INFO] Building normalized JSON files...")

    researchers_raw, grants_raw, institution_mapping, agency_mapping = load_raw_data()
    if not researchers_raw:
        raise RuntimeError("researchers_raw.json is missing or empty. Run ingestion/fetch_researchers.py first.")
    if not grants_raw:
        raise RuntimeError("grants_raw.json is missing or empty. Run ingestion/fetch_grants.py first.")

    researchers = build_researchers_json(researchers_raw)
    institutions = build_institutions_json(institution_mapping, researchers_raw)
    agencies = build_agencies_json(agency_mapping)
    grants = build_grants_json(grants_raw, agencies)
    topics = build_topics_json(researchers_raw, grants_raw)

    save_json_files(researchers, institutions, agencies, grants, topics)

    print("\n[OK] Normalized JSON files created successfully")
    print(f"  Researchers: {len(researchers)}")
    print(f"  Institutions: {len(institutions)}")
    print(f"  Agencies: {len(agencies)}")
    print(f"  Grants: {len(grants)}")
    print(f"  Topics: {len(topics)}")
