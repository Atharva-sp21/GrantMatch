import csv
import difflib
import html
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


def normalize_text(text: Any) -> str:
    cleaned = html.unescape(str(text or "")).lower()
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def token_terms(text: Any) -> Set[str]:
    words = [word for word in re.findall(r"[a-z0-9]+", normalize_text(text)) if len(word) > 2]
    terms = set(words)
    for index in range(len(words) - 1):
        terms.add(f"{words[index]} {words[index + 1]}")
    return terms


def load_json_files():
    data_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw")))

    def _read(name: str):
        with open(data_dir / name, "r", encoding="utf-8") as handle:
            return json.load(handle)

    return _read("researchers.json"), _read("institutions.json"), _read("agencies.json"), _read("grants.json"), _read("topics.json")


def _name_variants(name: str) -> Set[str]:
    normalized = normalize_text(name)
    variants = {normalized}
    if "," in name:
        parts = [part.strip() for part in name.split(",") if part.strip()]
        if len(parts) >= 2:
            variants.add(normalize_text(" ".join(parts[1:] + parts[:1])))
    tokens = normalized.split()
    if len(tokens) >= 2:
        variants.add(f"{tokens[0]} {tokens[-1]}")
        variants.add(" ".join(tokens[:2]))
        variants.add(" ".join(tokens[-2:]))
        variants.add(tokens[-1])
        variants.add(f"{tokens[0][0]} {tokens[-1]}")
    return {variant for variant in variants if variant}


def _build_researcher_name_index(researchers: Sequence[Dict[str, Any]]) -> Dict[str, Set[int]]:
    index: Dict[str, Set[int]] = defaultdict(set)
    for researcher in researchers:
        names = [researcher.get("name", ""), *researcher.get("aliases", [])]
        for name in names:
            for variant in _name_variants(name):
                index[variant].add(researcher["id"])
    return index


def _match_researcher_ids(name: str, name_index: Dict[str, Set[int]]) -> Set[int]:
    variants = list(_name_variants(name))
    matches: Set[int] = set()
    for variant in variants:
        if variant in name_index:
            matches.update(name_index[variant])

    if matches:
        return matches

    candidate_names = list(name_index.keys())
    for variant in variants:
        close_matches = difflib.get_close_matches(variant, candidate_names, n=5, cutoff=0.78)
        for close_match in close_matches:
            matches.update(name_index[close_match])
    return matches


def build_affiliated_edges(researchers):
    edges = []
    for researcher in researchers:
        institution_id = researcher.get("institution_id")
        if institution_id is not None:
            edges.append({"source_id": researcher["id"], "target_id": institution_id})

    print(f"[OK] Built {len(edges)} AFFILIATED_WITH edges")
    return edges


def build_researches_edges(researchers, topics):
    edges = []
    topic_lookup = {normalize_text(topic["name"]): topic["id"] for topic in topics}

    for researcher in researchers:
        matched_topic_ids = set()
        for concept in researcher.get("concepts", []):
            if isinstance(concept, dict):
                concept_name = concept.get("name") or concept.get("display_name") or ""
            else:
                concept_name = str(concept)
            for term in token_terms(concept_name):
                topic_id = topic_lookup.get(normalize_text(term))
                if topic_id is not None:
                    matched_topic_ids.add(topic_id)

        for topic_id in sorted(matched_topic_ids):
            edges.append({"source_id": researcher["id"], "target_id": topic_id})

    print(f"[OK] Built {len(edges)} RESEARCHES edges")
    return edges


def build_received_past_edges(researchers, grants):
    edges = []
    researcher_index = _build_researcher_name_index(researchers)
    seen_pairs = set()

    for grant in grants:
        matched_ids = set()
        for investigator in grant.get("investigators", []):
            matched_ids.update(_match_researcher_ids(investigator, researcher_index))

        for researcher_id in sorted(matched_ids):
            pair = (researcher_id, grant["id"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edges.append({"source_id": researcher_id, "target_id": grant["id"]})

    print(f"[OK] Built {len(edges)} RECEIVED_PAST edges")
    return edges


def build_funds_topic_edges(grants, topics):
    edges = []
    topic_lookup = {normalize_text(topic["name"]): topic["id"] for topic in topics}

    for grant in grants:
        grant_text = " ".join(filter(None, [grant.get("title", ""), grant.get("abstract", "")]))
        matched_topic_ids = set()
        for term in token_terms(grant_text):
            topic_id = topic_lookup.get(normalize_text(term))
            if topic_id is not None:
                matched_topic_ids.add(topic_id)

        for topic_id in sorted(matched_topic_ids):
            edges.append({"source_id": grant["id"], "target_id": topic_id})

    print(f"[OK] Built {len(edges)} FUNDS_TOPIC edges")
    return edges


def build_provides_edges(agencies, grants):
    edges = []
    for grant in grants:
        edges.append({"source_id": grant["agency_id"], "target_id": grant["id"]})

    print(f"[OK] Built {len(edges)} PROVIDES edges")
    return edges


def save_edge_csvs(affiliated, researches, received, funds, provides):
    output_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw")))

    edge_files = {
        "affiliated.csv": affiliated,
        "researches.csv": researches,
        "received_past.csv": received,
        "funds_topic.csv": funds,
        "provides.csv": provides,
    }

    for filename, edges in edge_files.items():
        filepath = output_dir / filename
        with open(filepath, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_id", "target_id"])
            writer.writeheader()
            writer.writerows(edges)
        print(f"[OK] Saved {filepath} ({len(edges)} edges)")


if __name__ == "__main__":
    print("[INFO] Building edge CSV files...")

    researchers, institutions, agencies, grants, topics = load_json_files()

    affiliated = build_affiliated_edges(researchers)
    researches = build_researches_edges(researchers, topics)
    received = build_received_past_edges(researchers, grants)
    funds = build_funds_topic_edges(grants, topics)
    provides = build_provides_edges(agencies, grants)

    save_edge_csvs(affiliated, researches, received, funds, provides)

    print("\n[OK] All edge CSVs created successfully")
    print(f"  AFFILIATED_WITH: {len(affiliated)}")
    print(f"  RESEARCHES: {len(researches)}")
    print(f"  RECEIVED_PAST: {len(received)}")
    print(f"  FUNDS_TOPIC: {len(funds)}")
    print(f"  PROVIDES: {len(provides)}")
    print(f"  Total edges: {sum(len(group) for group in [affiliated, researches, received, funds, provides])}")
