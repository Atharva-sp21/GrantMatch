"""
Phase 2: Aggressive OpenAlex Enrichment
"""

import sys
import os
import json
import time
from collections import Counter
from pathlib import Path
import csv

# Import from existing scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fetch_data import fetch_author_works, get_json, invert_abstract, TOPIC_KEYWORDS

RAW = Path(__file__).resolve().parent / "raw"
PROCESSED = Path(__file__).resolve().parent / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS = Path(__file__).resolve().parent.parent / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

WORKS_PER_RESEARCHER = 50
TOPICS_PER_RESEARCHER = 15

def load_json(name):
    path = RAW / name
    if not path.exists(): return []
    with open(path) as f:
        return json.load(f)

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def write_edges(path, edges):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "target_id"])
        w.writerows(sorted(set(edges)))

def parse_year(date_str) -> int:
    if not date_str: return 0
    try: return int(str(date_str)[:4])
    except ValueError: return 0

def enrich_works_for_researcher(oa_id: str):
    works = fetch_author_works(oa_id, WORKS_PER_RESEARCHER)
    abstracts = []
    paper_titles = []
    keywords = []
    topic_counts = Counter()
    concept_counts = Counter()
    coauthor_ids = set()
    years = []
    total_cites = 0

    for w in works:
        title = w.get("title")
        if title: paper_titles.append(title)
        
        ab = invert_abstract(w.get("abstract_inverted_index"))
        if ab: abstracts.append(ab[:2500])
        
        total_cites += w.get("cited_by_count") or 0
        pub = w.get("publication_date") or ""
        y = parse_year(pub)
        if y: years.append(y)

        # Keyword dict processing based on OpenAlex changes
        for kw in w.get("keywords") or []:
            kw_name = kw.get("display_name")
            if kw_name: keywords.append(kw_name.lower())

        for topic in w.get("topics") or []:
            tname = (topic.get("display_name") or "").strip().lower()
            if tname: topic_counts[tname] += float(topic.get("score") or 1.0)

        for concept in w.get("concepts") or []:
            cname = (concept.get("display_name") or "").strip().lower()
            if cname: concept_counts[cname] += float(concept.get("score") or 0.35)

        for auth in w.get("authorships") or []:
            aid = (auth.get("author") or {}).get("id")
            if aid: coauthor_ids.add(aid)

    # Combine topics and concepts for "topics" list
    combined_topics = Counter()
    for k, v in topic_counts.items(): combined_topics[k] += v
    for k, v in concept_counts.items(): combined_topics[k] += v

    return {
        "abstracts": abstracts,
        "paper_titles": paper_titles,
        "keywords": list(set(keywords)),
        "topic_counts": combined_topics,
        "coauthor_count": max(0, len(coauthor_ids) - 1),
        "years": years,
        "works_fetched": len(works),
        "avg_cites_per_work": total_cites / max(len(works), 1),
    }

def main():
    t0 = time.time()
    print("Loading initial data...")
    researchers = load_json("researchers.json")
    grants = load_json("grants.json")
    institutions = load_json("institutions.json")
    old_topics = load_json("topics.json")
    
    # Existing graph degree calculations
    try:
        old_researches = []
        with open(RAW / "researches.csv") as f:
            for row in csv.DictReader(f):
                old_researches.append((int(row["source_id"]), int(row["target_id"])))
    except Exception:
        old_researches = []

    old_r_topic_deg = Counter(r for r, t in old_researches)
    
    # Calculate connected components (simplified, nodes = researchers + grants)
    old_isolated_nodes = 0
    
    before_stats = {
        "researchers_with_abstracts": sum(1 for r in researchers if r.get("recent_abstracts") and len(r["recent_abstracts"]) > 0 and any(len(a.strip()) > 0 for a in r["recent_abstracts"])),
        "total_topic_edges": len(old_researches),
        "average_topics_per_researcher": len(old_researches) / max(len(researchers), 1),
        "researchers_with_zero_topics": sum(1 for r in researchers if old_r_topic_deg.get(r["id"], 0) == 0),
    }

    inst_prestige = {i["id"]: 1.0 / (i.get("world_ranking", 500) + 1) for i in institutions}

    all_topic_counts = Counter()
    enriched_profiles = []
    new_researches_edges = []
    
    print(f"Enriching {len(researchers)} researchers...")
    for i, r in enumerate(researchers):
        oa_id = r.get("openalex_id")
        if not oa_id:
            continue
            
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(researchers)}] {r.get('name','')[:40]}")

        wd = enrich_works_for_researcher(oa_id)
        
        # Build researcher_profile
        top_topics = [t for t, _ in wd["topic_counts"].most_common(TOPICS_PER_RESEARCHER)]
        profile = {
            "researcher_id": r["id"],
            "openalex_id": oa_id,
            "topics": top_topics,
            "paper_titles": wd["paper_titles"],
            "abstracts": wd["abstracts"],
            "keywords": wd["keywords"]
        }
        enriched_profiles.append(profile)
        
        # Compute fields
        pubs = r.get("publication_count", 0)
        cites = r.get("citation_count", 0)
        years = wd.get("years", [])
        
        if years:
            first_y, last_y = min(years), max(years)
            r["years_since_first_publication"] = 2025 - first_y
            r["years_since_latest_publication"] = 2025 - last_y
            span = max(last_y - first_y, 1)
            r["publication_velocity"] = len(years) / span
        else:
            r["years_since_first_publication"] = 0
            r["years_since_latest_publication"] = 0
            r["publication_velocity"] = 0.0

        r["avg_citations_per_paper"] = cites / max(pubs, 1)
        r["collaboration_count"] = wd["coauthor_count"]
        r["topic_diversity_score"] = len(wd["topic_counts"])
        r["recent_abstracts"] = wd["abstracts"]
        r["institution_prestige"] = inst_prestige.get(r.get("institution_id"), 0.0)
        
        for tname, score in wd["topic_counts"].most_common(TOPICS_PER_RESEARCHER):
            all_topic_counts[tname] += score
            
        r["_topic_names"] = top_topics
        
    # Build unified topics
    print("Building new topics...")
    topic_names = set(TOPIC_KEYWORDS)
    for t, _ in all_topic_counts.most_common(100):
        if t and len(t) < 80:
            topic_names.add(t)
            
    for g in grants:
        text = (g.get("title", "") + " " + g.get("guidelines_text", "")).lower()
        for kw in TOPIC_KEYWORDS:
            if kw in text:
                topic_names.add(kw)
                
    sorted_names = sorted(topic_names)[:100]
    new_topics = [
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
    topic_name_to_id = {t["name"]: t["id"] for t in new_topics}
    
    for r in researchers:
        rid = r["id"]
        tns = r.pop("_topic_names", [])
        for tname in tns:
            if tname in topic_name_to_id:
                new_researches_edges.append((rid, topic_name_to_id[tname]))
                
    # Also inherit grant topics for existing received labels to keep connectivity
    try:
        received = []
        with open(RAW / "received_past.csv") as f:
            for row in csv.DictReader(f):
                received.append((int(row["source_id"]), int(row["target_id"])))
                
        grant_topics = {g["id"]: g.get("topic_ids", []) for g in grants}
        for s, g in received:
            for tid in grant_topics.get(g, []):
                # Only inherit if it was an old topic ID that we mapped, or just skip it. 
                # Better to assign topics to grants from new vocab, then inherit
                pass
    except Exception:
        received = []
        
    funds_edges = []
    for g in grants:
        text = (g.get("title", "") + " " + g.get("guidelines_text", "")).lower()
        g_topics = []
        for t in new_topics:
            if t["name"] in text:
                g_topics.append(t["id"])
                funds_edges.append((g["id"], t["id"]))
        g["topic_ids"] = g_topics[:5]

    for s, gid in received:
        for tid in grants[gid].get("topic_ids", []):
            new_researches_edges.append((s, tid))
            
    # Update topic counts
    topic_r_count = Counter(d for _, d in new_researches_edges)
    topic_g_count = Counter(s for s, _ in funds_edges)
    for t in new_topics:
        t["researcher_count"] = topic_r_count.get(t["id"], 0)
        t["grant_count"] = topic_g_count.get(t["id"], 0)

    # Save all files
    save_json(PROCESSED / "researchers_enriched.json", researchers)
    save_json(PROCESSED / "researcher_profile.json", enriched_profiles)
    save_json(PROCESSED / "topics.json", new_topics)
    save_json(PROCESSED / "grants.json", grants) # Updated with new topic_ids
    
    write_edges(PROCESSED / "researches.csv", new_researches_edges)
    write_edges(PROCESSED / "funds_topic.csv", funds_edges)
    
    # Overwrite raw files as requested by user
    save_json(RAW / "researchers.json", researchers)
    save_json(RAW / "topics.json", new_topics)
    write_edges(RAW / "researches.csv", new_researches_edges)
    
    # After stats
    new_r_topic_deg = Counter(r for r, t in new_researches_edges)
    
    after_stats = {
        "researchers_with_abstracts": sum(1 for r in researchers if r.get("recent_abstracts") and len(r["recent_abstracts"]) > 0 and any(len(a.strip()) > 0 for a in r["recent_abstracts"])),
        "total_topic_edges": len(set(new_researches_edges)),
        "average_topics_per_researcher": len(set(new_researches_edges)) / max(len(researchers), 1),
        "researchers_with_zero_topics": sum(1 for r in researchers if new_r_topic_deg.get(r["id"], 0) == 0),
    }

    report = {
        "before": before_stats,
        "after": after_stats,
        "delta": {
            "abstract_coverage": after_stats["researchers_with_abstracts"] - before_stats["researchers_with_abstracts"],
            "topic_edges": after_stats["total_topic_edges"] - before_stats["total_topic_edges"],
            "zero_topic_researchers": before_stats["researchers_with_zero_topics"] - after_stats["researchers_with_zero_topics"]
        },
        "time_elapsed_seconds": round(time.time() - t0, 1)
    }
    
    save_json(REPORTS / "enrichment_report.json", report)
    print("\nPhase 2 Complete!")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
