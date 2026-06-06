"""
Enrich existing data/raw/ from OpenAlex: more abstracts, topics, researcher profiles.

Run (from repo root):  python data/enrich_dataset.py

Reports before/after edge counts and abstract coverage.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

from fetch_data import (
    MAILTO,
    OPENALEX,
    TOPIC_KEYWORDS,
    fetch_author_works,
    get_json,
    invert_abstract,
    norm_name,
)

RAW = Path(__file__).resolve().parent / "raw"
WORKS_PER_RESEARCHER = 25
TOPICS_PER_RESEARCHER = 10
MIN_TOPIC_SCORE = 0.2


def load_json(name):
    with open(RAW / name) as f:
        return json.load(f)


def save_json(name, obj):
    with open(RAW / name, "w") as f:
        json.dump(obj, f, indent=2)


def write_edges(name: str, edges: List[Tuple[int, int]]):
    with open(RAW / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "target_id"])
        w.writerows(sorted(set(edges)))


def parse_year(date_str) -> int:
    if not date_str:
        return 0
    try:
        return int(str(date_str)[:4])
    except ValueError:
        return 0


def enrich_works_for_researcher(oa_id: str) -> dict:
    works = fetch_author_works(oa_id, WORKS_PER_RESEARCHER)
    abstracts = []
    topic_counts: Counter = Counter()
    coauthor_ids: Set[str] = set()
    years = []
    total_cites = 0

    for w in works:
        ab = invert_abstract(w.get("abstract_inverted_index"))
        if ab:
            abstracts.append(ab[:2500])
        total_cites += w.get("cited_by_count") or 0
        pub = w.get("publication_date") or ""
        y = parse_year(pub)
        if y:
            years.append(y)

        for topic in w.get("topics") or []:
            tname = (topic.get("display_name") or "").strip().lower()
            score = topic.get("score") or 1.0
            if tname and score >= MIN_TOPIC_SCORE:
                topic_counts[tname] += float(score)
        for concept in w.get("concepts") or []:
            if (concept.get("score") or 0) >= 0.35:
                cname = (concept.get("display_name") or "").strip().lower()
                if cname:
                    topic_counts[cname] += float(concept.get("score", 0.35))

        for auth in w.get("authorships") or []:
            aid = (auth.get("author") or {}).get("id")
            if aid:
                coauthor_ids.add(aid)

    return {
        "abstracts": abstracts[:30],
        "topic_counts": topic_counts,
        "coauthor_count": max(0, len(coauthor_ids) - 1),
        "years": years,
        "works_fetched": len(works),
        "avg_cites_per_work": total_cites / max(len(works), 1),
    }


def build_topic_vocab(all_topic_counts: Counter, grants: List[dict], max_topics: int = 50):
    names = set(TOPIC_KEYWORDS)
    for t, _ in all_topic_counts.most_common(max_topics):
        if t and len(t) < 80:
            names.add(t)
    for g in grants:
        text = (g.get("title", "") + " " + g.get("guidelines_text", "")).lower()
        for kw in TOPIC_KEYWORDS:
            if kw in text:
                names.add(kw)
    sorted_names = sorted(names)[:max_topics]
    topics = [
        {
            "id": i,
            "name": n,
            "field_of_study": n.split()[0].title(),
            "funding_frequency": 0.0,
            "researcher_count": 0,
            "trend_score": 0.0,
        }
        for i, n in enumerate(sorted_names)
    ]
    return topics, {t["name"]: t["id"] for t in topics}


def compute_profile_fields(r: dict, work_data: dict, current_year: int = 2025) -> None:
    pubs = r.get("publication_count") or r.get("works_count") or 0
    cites = r.get("citation_count") or r.get("cited_by_count") or 0
    years = work_data.get("years") or []
    n_works = max(work_data.get("works_fetched", 0), 1)

    if years:
        first_y, last_y = min(years), max(years)
        r["years_since_first_publication"] = current_year - first_y
        r["years_since_latest_publication"] = current_year - last_y
        span = max(last_y - first_y, 1)
        r["publication_velocity"] = len(years) / span
    else:
        r["years_since_first_publication"] = 0
        r["years_since_latest_publication"] = 0
        r["publication_velocity"] = 0.0

    r["avg_citations_per_paper"] = cites / max(pubs, 1)
    r["collaboration_count"] = work_data.get("coauthor_count", 0)
    r["topic_diversity_score"] = len(work_data.get("topic_counts", {}))
    r["recent_abstracts"] = work_data.get("abstracts", r.get("recent_abstracts", []))


def main():
    t0 = time.time()
    researchers = load_json("researchers.json")
    grants = load_json("grants.json")
    institutions = load_json("institutions.json")

    inst_prestige = {
        i["id"]: 1.0 / (i.get("world_ranking", 500) + 1) for i in institutions
    }

    # Researchers to enrich: have openalex_id OR appear in received_past
    received = []
    with open(RAW / "received_past.csv") as f:
        for row in csv.DictReader(f):
            received.append((int(row["source_id"]), int(row["target_id"])))
    labeled_r = {s for s, _ in received}

    before = {
        "researches_edges": sum(1 for _ in open(RAW / "researches.csv")) - 1,
        "with_abstracts": sum(1 for r in researchers if r.get("recent_abstracts")),
    }

    print(f"[Enrich] Researchers={len(researchers)} labeled={len(labeled_r)}")
    print(f"[Before] researches edges≈{before['researches_edges']} abstracts={before['with_abstracts']}")

    all_topic_counts: Counter = Counter()
    researches_edges: List[Tuple[int, int]] = []
    enriched = 0

    for i, r in enumerate(researchers):
        rid = r["id"]
        oa_id = r.get("openalex_id")
        if not oa_id and rid not in labeled_r:
            continue

        if not oa_id:
            continue

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(researchers)}] {r.get('name','')[:40]}")

        wd = enrich_works_for_researcher(oa_id)
        compute_profile_fields(r, wd)
        r["institution_prestige"] = inst_prestige.get(r.get("institution_id"), 0.0)
        enriched += 1

        for tname, score in wd["topic_counts"].most_common(TOPICS_PER_RESEARCHER):
            all_topic_counts[tname] += score

        r["_topic_names"] = [t for t, _ in wd["topic_counts"].most_common(TOPICS_PER_RESEARCHER)]

    topics, topic_name_to_id = build_topic_vocab(all_topic_counts, grants)

    for r in researchers:
        rid = r["id"]
        for tname in r.pop("_topic_names", []):
            if tname in topic_name_to_id:
                researches_edges.append((rid, topic_name_to_id[tname]))

    # Grant-derived topic links for labeled researchers
    grant_topics = {g["id"]: g.get("topic_ids", []) for g in grants}
    for s, g in received:
        for tid in grant_topics.get(g, []):
            if isinstance(tid, int) and tid < len(topics):
                researches_edges.append((s, tid))

    # Rebuild funds_topic from grants + new topic ids
    funds_edges = []
    for g in grants:
        text = (g.get("title", "") + " " + g.get("guidelines_text", "")).lower()
        for t in topics:
            if t["name"] in text:
                funds_edges.append((g["id"], t["id"]))
        if not g.get("topic_ids"):
            g["topic_ids"] = [t["id"] for t in topics if t["name"] in text][:3]

    topic_r_count = Counter(d for _, d in researches_edges)
    topic_g_count = Counter(s for s, _ in funds_edges)
    for t in topics:
        t["researcher_count"] = topic_r_count.get(t["id"], 0)
        t["grant_count"] = topic_g_count.get(t["id"], 0)

    save_json("researchers.json", researchers)
    save_json("topics.json", topics)
    save_json("grants.json", grants)
    write_edges("researches.csv", researches_edges)
    write_edges("funds_topic.csv", funds_edges)

    after = {
        "researches_edges": len(set(researches_edges)),
        "with_abstracts": sum(1 for r in researchers if r.get("recent_abstracts")),
        "enriched_from_openalex": enriched,
        "topics": len(topics),
    }

    stats = {
        "before": before,
        "after": after,
        "delta_researches_edges": after["researches_edges"] - before["researches_edges"],
        "delta_abstracts": after["with_abstracts"] - before["with_abstracts"],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    cache = Path(__file__).resolve().parent / "cache"
    cache.mkdir(exist_ok=True)
    with open(cache / "enrichment_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n[After]")
    print(f"  researches edges: {after['researches_edges']} (+{stats['delta_researches_edges']})")
    print(f"  researchers w/ abstracts: {after['with_abstracts']} (+{stats['delta_abstracts']})")
    print(f"  topics: {after['topics']}")
    print(f"  Done in {stats['elapsed_sec']}s")


if __name__ == "__main__":
    main()
