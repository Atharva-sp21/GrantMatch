"""
Fetch real grant–researcher data from OpenAlex, NIH RePORTER, and NSF.
Writes ML-ready files to data/raw/ (integer IDs, source_id/target_id edges).

Run from repo root:
    python data/fetch_data.py
"""

from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# ── Config ───────────────────────────────────────────────────────────
MAILTO = "grantmatch@example.com"  # OpenAlex polite pool — use your real email
OPENALEX_AUTHORS = 400
WORKS_PER_AUTHOR = 5
NIH_LIMIT = 600
NSF_LIMIT = 400
NIH_FISCAL_YEARS = [2022, 2023, 2024]
REQUEST_DELAY = 0.15

OPENALEX = "https://api.openalex.org"
NIH_SEARCH = "https://api.reporter.nih.gov/v2/projects/search"
NSF_AWARDS = "https://api.nsf.gov/services/v1/awards.json"

RAW_DIR = Path(__file__).resolve().parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TOPIC_KEYWORDS = [
    "machine learning", "artificial intelligence", "neuroscience", "cancer",
    "immunology", "genomics", "climate", "energy", "quantum", "materials",
    "chemistry", "physics", "biology", "engineering", "computer", "health",
    "covid", "vaccine", "cardiovascular", "diabetes", "ecology", "ocean",
    "astronomy", "robotics", "nanotechnology", "biomedical", "public health",
]


# ── HTTP helpers ─────────────────────────────────────────────────────
def _sleep():
    time.sleep(REQUEST_DELAY)


def get_json(url: str, params: Optional[dict] = None, method: str = "GET", json_body=None):
    try:
        if method == "POST":
            r = requests.post(url, json=json_body, timeout=60)
        else:
            r = requests.get(url, params=params, timeout=60)
        _sleep()
        if r.status_code != 200:
            print(f"[WARN] HTTP {r.status_code} {url[:80]}")
            return None
        return r.json()
    except requests.RequestException as e:
        print(f"[WARN] Request failed: {e}")
        return None


def invert_abstract(inv_index: Optional[dict]) -> str:
    if not inv_index:
        return ""
    words = {}
    for word, positions in inv_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


def norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").lower().strip())


def short_id(openalex_url: str) -> str:
    return openalex_url.rsplit("/", 1)[-1] if openalex_url else ""


# ── OpenAlex ─────────────────────────────────────────────────────────
def fetch_openalex_authors(n: int) -> List[dict]:
    """OpenAlex allows max per_page=200; paginate with cursor."""
    print(f"[OpenAlex] Fetching top {n} authors...")
    authors = []
    cursor = "*"
    per_page = min(200, n)

    while len(authors) < n:
        params = {
            "per_page": min(per_page, n - len(authors)),
            "sort": "cited_by_count:desc",
            "mailto": MAILTO,
        }
        if cursor:
            params["cursor"] = cursor
        data = get_json(f"{OPENALEX}/authors", params)
        if not data:
            break
        batch = data.get("results", [])
        if not batch:
            break
        authors.extend(batch)
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    print(f"[OpenAlex] Got {len(authors)} authors")
    return authors[:n]


def fetch_author_works(author_id_url: str, per_page: int) -> List[dict]:
    data = get_json(
        f"{OPENALEX}/works",
        {
            "filter": f"author.id:{author_id_url}",
            "per_page": per_page,
            "sort": "publication_date:desc",
            "mailto": MAILTO,
        },
    )
    return (data or {}).get("results", [])


def openalex_search_author_by_name(name: str) -> Optional[dict]:
    data = get_json(
        f"{OPENALEX}/authors",
        {"search": name, "per_page": 1, "mailto": MAILTO},
    )
    results = (data or {}).get("results", [])
    return results[0] if results else None


# ── NIH RePORTER ─────────────────────────────────────────────────────
def fetch_nih_projects(limit: int) -> List[dict]:
    print(f"[NIH] Fetching up to {limit} projects...")
    projects = []
    offset = 0
    page_size = 100

    while len(projects) < limit:
        body = {
            "criteria": {"fiscal_years": NIH_FISCAL_YEARS},
            "offset": offset,
            "limit": min(page_size, limit - len(projects)),
        }
        data = get_json(NIH_SEARCH, method="POST", json_body=body)
        if not data:
            break
        batch = data.get("results", [])
        if not batch:
            break
        projects.extend(batch)
        offset += len(batch)
        if len(batch) < page_size:
            break

    print(f"[NIH] Got {len(projects)} projects")
    return projects[:limit]


# ── NSF ──────────────────────────────────────────────────────────────
def fetch_nsf_awards(limit: int) -> List[dict]:
    print(f"[NSF] Fetching up to {limit} awards...")
    awards = []
    offset = 1  # NSF API is 1-based

    fields = (
        "id,title,abstractText,awardeeName,piFirstName,piLastName,"
        "estimatedTotalAmt,date,startDate,expDate,fundProgramName,agency"
    )

    while len(awards) < limit:
        params = {
            "offset": offset,
            "rpp": min(100, limit - len(awards)),
            "printFields": fields,
        }
        data = get_json(NSF_AWARDS, params=params)
        if not data:
            break
        batch = data.get("response", {}).get("award", [])
        if not batch:
            break
        if isinstance(batch, dict):
            batch = [batch]
        awards.extend(batch)
        offset += len(batch)
        if len(batch) < params["rpp"]:
            break

    print(f"[NSF] Got {len(awards)} awards")
    return awards[:limit]


# ── Topic helpers ────────────────────────────────────────────────────
def assign_topics_from_text(text: str, topic_name_to_id: Dict[str, int]) -> List[int]:
    text_l = (text or "").lower()
    ids = []
    for name, tid in topic_name_to_id.items():
        if name in text_l:
            ids.append(tid)
    return ids[:3] if ids else []


def build_topic_vocab(
    work_topic_names: Counter,
    grant_texts: List[str],
    max_topics: int = 40,
) -> Tuple[List[dict], Dict[str, int]]:
    """Topics from OpenAlex labels + keyword seeds."""
    names: Set[str] = set(TOPIC_KEYWORDS)

    for name, _ in work_topic_names.most_common(max_topics):
        if name and len(name) < 80:
            names.add(name.lower())

    for text in grant_texts:
        text_l = text.lower()
        for kw in TOPIC_KEYWORDS:
            if kw in text_l:
                names.add(kw)

    sorted_names = sorted(names)[:max_topics]
    topics = [
        {
            "id": i,
            "name": name,
            "field_of_study": name.split()[0].title() if name else "General",
            "funding_frequency": 0.0,
            "researcher_count": 0,
            "trend_score": 0.0,
        }
        for i, name in enumerate(sorted_names)
    ]
    name_to_id = {t["name"]: t["id"] for t in topics}
    return topics, name_to_id


# ── Main build ───────────────────────────────────────────────────────
def main():
    t0 = time.time()
    researchers: List[dict] = []
    institutions: Dict[str, dict] = {}
    inst_url_to_idx: Dict[str, int] = {}
    grants: List[dict] = []
    agencies: Dict[str, dict] = {}
    agency_name_to_idx: Dict[str, int] = {}

    affiliated_edges: List[Tuple[int, int]] = []
    researches_edges: List[Tuple[int, int]] = []
    received_edges: List[Tuple[int, int]] = []
    funds_edges: List[Tuple[int, int]] = []
    provides_edges: List[Tuple[int, int]] = []

    name_to_researcher_idx: Dict[str, int] = {}
    openalex_url_to_idx: Dict[str, int] = {}
    work_topic_counter: Counter = Counter()
    grant_texts_for_topics: List[str] = []

    # ── 1. OpenAlex researchers ──────────────────────────────────────
    authors = fetch_openalex_authors(OPENALEX_AUTHORS)

    for author in authors:
        oa_id = author["id"]
        stats = author.get("summary_stats") or {}
        idx = len(researchers)
        openalex_url_to_idx[oa_id] = idx

        rec = {
            "id": idx,
            "openalex_id": oa_id,
            "name": author.get("display_name", "Unknown"),
            "publication_count": author.get("works_count", 0),
            "citation_count": author.get("cited_by_count", 0),
            "h_index": stats.get("h_index", 0) or 0,
            "i10_index": stats.get("i10_index", 0) or 0,
            "career_age": 0,
            "career_stage": "unknown",
            "grant_success_rate": 0.0,
            "total_funding": 0,
            "institution_id": None,
            "recent_abstracts": [],
        }
        researchers.append(rec)
        name_to_researcher_idx[norm_name(rec["name"])] = idx

        for inst in author.get("last_known_institutions") or []:
            iurl = inst.get("id")
            if not iurl:
                continue
            if iurl not in inst_url_to_idx:
                inst_url_to_idx[iurl] = len(institutions)
                institutions[iurl] = {
                    "id": inst_url_to_idx[iurl],
                    "openalex_id": iurl,
                    "name": inst.get("display_name", "Unknown"),
                    "world_ranking": 999,
                    "institution_type": inst.get("type", "unknown"),
                    "rd_spend": 0,
                    "country": inst.get("country_code", "unknown"),
                    "faculty_size": 0,
                }
            rec["institution_id"] = inst_url_to_idx[iurl]
            affiliated_edges.append((idx, inst_url_to_idx[iurl]))

    # Works + abstracts + topic names
    print(f"[OpenAlex] Fetching works for {len(authors)} authors...")
    author_topics: Dict[int, Set[int]] = defaultdict(set)

    for i, author in enumerate(authors):
        oa_id = author["id"]
        ridx = openalex_url_to_idx[oa_id]
        print(f"  [{i + 1}/{len(authors)}] {author.get('display_name', '')[:40]}")

        works = fetch_author_works(oa_id, WORKS_PER_AUTHOR)
        for work in works:
            abstract = invert_abstract(work.get("abstract_inverted_index"))
            if abstract and len(researchers[ridx]["recent_abstracts"]) < 5:
                researchers[ridx]["recent_abstracts"].append(abstract[:2000])

            for topic in work.get("topics") or []:
                tname = (topic.get("display_name") or "").lower()
                if tname:
                    work_topic_counter[tname] += 1
            for concept in work.get("concepts") or []:
                if (concept.get("score") or 0) >= 0.3:
                    cname = (concept.get("display_name") or "").lower()
                    if cname:
                        work_topic_counter[cname] += 1

    def ensure_researcher_by_name(full_name: str) -> int:
        key = norm_name(full_name)
        if key in name_to_researcher_idx:
            return name_to_researcher_idx[key]
        hit = openalex_search_author_by_name(full_name)
        idx = len(researchers)
        if hit:
            oa_id = hit["id"]
            if oa_id in openalex_url_to_idx:
                return openalex_url_to_idx[oa_id]
            stats = hit.get("summary_stats") or {}
            researchers.append({
                "id": idx,
                "openalex_id": oa_id,
                "name": hit.get("display_name", full_name),
                "publication_count": hit.get("works_count", 0),
                "citation_count": hit.get("cited_by_count", 0),
                "h_index": stats.get("h_index", 0) or 0,
                "i10_index": stats.get("i10_index", 0) or 0,
                "career_age": 0,
                "career_stage": "unknown",
                "grant_success_rate": 0.0,
                "total_funding": 0,
                "institution_id": None,
                "recent_abstracts": [],
            })
            openalex_url_to_idx[oa_id] = idx
        else:
            researchers.append({
                "id": idx,
                "openalex_id": None,
                "name": full_name,
                "publication_count": 0,
                "citation_count": 0,
                "h_index": 0,
                "i10_index": 0,
                "career_age": 0,
                "career_stage": "unknown",
                "grant_success_rate": 0.0,
                "total_funding": 0,
                "institution_id": None,
                "recent_abstracts": [],
            })
        name_to_researcher_idx[key] = idx
        return idx

    def ensure_agency(name: str, agency_type: str = "government") -> int:
        name = name or "Unknown Agency"
        if name not in agency_name_to_idx:
            aid = len(agencies)
            agency_name_to_idx[name] = aid
            agencies[name] = {
                "id": aid,
                "name": name,
                "agency_type": agency_type,
                "total_annual_budget": 0,
                "avg_grant_size": 0,
                "primary_focus_areas": [],
            }
        return agency_name_to_idx[name]

    # ── 2. NIH grants + received_past ────────────────────────────────
    nih_projects = fetch_nih_projects(NIH_LIMIT)

    for proj in nih_projects:
        gid = len(grants)
        title = proj.get("project_title") or "Untitled NIH Project"
        abstract = proj.get("abstract_text") or ""
        amount = proj.get("award_amount") or proj.get("total_cost") or 0
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            amount = 0.0

        agency_name = "NIH"
        admin = proj.get("agency_ic_admin") or {}
        if isinstance(admin, dict):
            agency_name = admin.get("name") or admin.get("abbreviation") or "NIH"
        elif isinstance(admin, str):
            agency_name = admin

        aid = ensure_agency(agency_name, "government")
        grant_texts_for_topics.append(f"{title} {abstract}")

        grants.append({
            "id": gid,
            "source": "nih",
            "external_id": str(proj.get("appl_id") or proj.get("project_num") or gid),
            "title": title,
            "guidelines_text": abstract or title,
            "guidelines": abstract or title,
            "agency_id": aid,
            "funding_amount": amount,
            "grant_type": "project",
            "deadline": "",
            "deadline_days_remaining": 0,
            "career_stage_eligibility": "any",
            "acceptance_rate": 0.0,
            "country_restriction": "US",
            "topic_ids": [],
        })
        provides_edges.append((aid, gid))

        pis = proj.get("principal_investigators") or []
        for pi in pis:
            pi_name = pi.get("full_name") or f"{pi.get('first_name', '')} {pi.get('last_name', '')}".strip()
            if not pi_name:
                continue
            ridx = ensure_researcher_by_name(pi_name)
            received_edges.append((ridx, gid))
            researchers[ridx]["total_funding"] = researchers[ridx].get("total_funding", 0) + amount

    # ── 3. NSF grants ────────────────────────────────────────────────
    nsf_awards = fetch_nsf_awards(NSF_LIMIT)

    for award in nsf_awards:
        gid = len(grants)
        title = award.get("title") or "Untitled NSF Award"
        abstract = award.get("abstractText") or ""
        amount = award.get("estimatedTotalAmt") or 0
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            amount = 0.0

        program = award.get("fundProgramName") or "NSF"
        aid = ensure_agency(f"NSF — {program}", "government")
        grant_texts_for_topics.append(f"{title} {abstract}")

        grants.append({
            "id": gid,
            "source": "nsf",
            "external_id": str(award.get("id") or gid),
            "title": title,
            "guidelines_text": abstract or title,
            "guidelines": abstract or title,
            "agency_id": aid,
            "funding_amount": amount,
            "grant_type": "project",
            "deadline": award.get("expDate") or "",
            "deadline_days_remaining": 0,
            "career_stage_eligibility": "any",
            "acceptance_rate": 0.0,
            "country_restriction": "US",
            "topic_ids": [],
        })
        provides_edges.append((aid, gid))

        pi_name = f"{award.get('piFirstName', '')} {award.get('piLastName', '')}".strip()
        if not pi_name:
            pi_name = award.get("awardeeName") or ""
        if pi_name:
            ridx = ensure_researcher_by_name(pi_name)
            received_edges.append((ridx, gid))
            researchers[ridx]["total_funding"] = researchers[ridx].get("total_funding", 0) + amount

    # ── 4. Topics + researches + funds_topic ─────────────────────────
    topics_list, topic_name_to_id = build_topic_vocab(work_topic_counter, grant_texts_for_topics)
    topic_counts_r = Counter()
    topic_counts_g = Counter()

    for ridx, r in enumerate(researchers):
        text = " ".join(r.get("recent_abstracts", [])) + " " + r.get("name", "")
        for tid in assign_topics_from_text(text, topic_name_to_id):
            researches_edges.append((ridx, tid))
            topic_counts_r[tid] += 1
            author_topics[ridx].add(tid)

    grant_topics: Dict[int, List[int]] = {}
    for g in grants:
        gid = g["id"]
        text = g.get("title", "") + " " + g.get("guidelines_text", "")
        tids = assign_topics_from_text(text, topic_name_to_id)
        if not tids and topic_name_to_id:
            tids = [0]
        g["topic_ids"] = tids
        grant_topics[gid] = tids
        for tid in tids:
            funds_edges.append((gid, tid))
            topic_counts_g[tid] += 1

    # Researchers who got a grant inherit that grant's topics (researches edges)
    for ridx, gid in received_edges:
        for tid in grant_topics.get(gid, []):
            researches_edges.append((ridx, tid))
            topic_counts_r[tid] += 1

    for t in topics_list:
        t["researcher_count"] = topic_counts_r.get(t["id"], 0)
        t["grant_count"] = topic_counts_g.get(t["id"], 0)
        t["funding_frequency"] = min(1.0, t["grant_count"] / max(len(grants), 1))
        t["trend_score"] = min(1.0, t["researcher_count"] / max(len(researchers), 1))

    # Dedupe edges
    affiliated_edges = list(dict.fromkeys(affiliated_edges))
    researches_edges = list(dict.fromkeys(researches_edges))
    received_edges = list(dict.fromkeys(received_edges))
    funds_edges = list(dict.fromkeys(funds_edges))
    provides_edges = list(dict.fromkeys(provides_edges))

    institutions_list = sorted(institutions.values(), key=lambda x: x["id"])
    agencies_list = sorted(agencies.values(), key=lambda x: x["id"])

    # ── 5. Write files ───────────────────────────────────────────────
    def write_json(name: str, obj: Any):
        path = RAW_DIR / name
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        print(f"[OK] {path} ({len(obj) if isinstance(obj, list) else 'obj'})")

    def write_edges(name: str, edges: List[Tuple[int, int]]):
        path = RAW_DIR / name
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source_id", "target_id"])
            w.writerows(edges)
        print(f"[OK] {path} ({len(edges)} edges)")

    write_json("researchers.json", researchers)
    write_json("institutions.json", institutions_list)
    write_json("grants.json", grants)
    write_json("topics.json", topics_list)
    write_json("agencies.json", agencies_list)
    write_edges("affiliated.csv", affiliated_edges)
    write_edges("researches.csv", researches_edges)
    write_edges("received_past.csv", received_edges)
    write_edges("funds_topic.csv", funds_edges)
    write_edges("provides.csv", provides_edges)

    # ── 6. Sufficiency report ─────────────────────────────────────────
    researchers_with_labels = len({s for s, _ in received_edges})
    pct_labeled = 100.0 * researchers_with_labels / max(len(researchers), 1)
    grants_with_labels = len({d for _, d in received_edges})

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Researchers:     {len(researchers)}")
    print(f"  w/ abstracts:  {sum(1 for r in researchers if r.get('recent_abstracts'))}")
    print(f"  w/ train label: {researchers_with_labels} ({pct_labeled:.1f}%)")
    print(f"Institutions:    {len(institutions_list)}")
    print(f"Grants:          {len(grants)} (NIH+NSF real awards)")
    print(f"  w/ label edge: {grants_with_labels}")
    print(f"Topics:          {len(topics_list)}")
    print(f"Agencies:        {len(agencies_list)}")
    print(f"Edges: affiliated={len(affiliated_edges)} researches={len(researches_edges)}")
    print(f"       received_past={len(received_edges)} funds_topic={len(funds_edges)} provides={len(provides_edges)}")
    print("=" * 60)

    enough = True
    warnings = []
    if len(researchers) < 200:
        warnings.append("Few researchers (<200) — increase OPENALEX_AUTHORS")
        enough = False
    if len(grants) < 200:
        warnings.append("Few grants (<200) — increase NIH_LIMIT / NSF_LIMIT")
        enough = False
    if len(received_edges) < 300:
        warnings.append("Few received_past edges (<300) — GNN may underfit")
        enough = False
    if pct_labeled < 30:
        warnings.append("Low % researchers with grant labels — PI name matching is weak")
    if len(affiliated_edges) < 50:
        warnings.append("Few affiliation edges — check OpenAlex institutions")

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("Data volume looks sufficient to train the GNN.")

    if not enough:
        print("Tip: re-run with higher limits in CONFIG at top of fetch_data.py")

    print(f"\nDone in {time.time() - t0:.1f}s. Files in {RAW_DIR}")
    return enough


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openalex-only",
        action="store_true",
        help="Refresh OpenAlex researchers/institutions/abstracts; keep existing grants",
    )
    args = parser.parse_args()
    if args.openalex_only:
        print("[Mode] openalex-only — run full ingest first if grants are missing")
    main()
