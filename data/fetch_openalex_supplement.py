"""Add OpenAlex researchers + affiliations + abstracts to existing data/raw/."""

import json
import csv
from pathlib import Path

# Reuse fetch helpers
from fetch_data import (
    MAILTO,
    OPENALEX_AUTHORS,
    WORKS_PER_AUTHOR,
    RAW_DIR,
    fetch_openalex_authors,
    fetch_author_works,
    invert_abstract,
    norm_name,
    get_json,
)


def main():
    raw = RAW_DIR
    researchers = json.loads((raw / "researchers.json").read_text())
    grants = json.loads((raw / "grants.json").read_text())

    existing_oa = {r.get("openalex_id"): r for r in researchers if r.get("openalex_id")}
    name_to_idx = {norm_name(r["name"]): r["id"] for r in researchers}
    next_id = max((r["id"] for r in researchers), default=-1) + 1

    institutions = {}
    inst_list = json.loads((raw / "institutions.json").read_text())
    for inst in inst_list:
        institutions[inst.get("openalex_id", str(inst["id"]))] = inst

    affiliated = set()
    with open(raw / "affiliated.csv") as f:
        for row in csv.DictReader(f):
            affiliated.add((int(row["source_id"]), int(row["target_id"])))

    authors = fetch_openalex_authors(OPENALEX_AUTHORS)
    print(f"Processing {len(authors)} OpenAlex authors...")

    for i, author in enumerate(authors):
        oa_id = author["id"]
        print(f"  [{i+1}/{len(authors)}] {author.get('display_name','')[:50]}")

        if oa_id in existing_oa:
            rec = existing_oa[oa_id]
        else:
            stats = author.get("summary_stats") or {}
            rec = {
                "id": next_id,
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
            existing_oa[oa_id] = rec
            name_to_idx[norm_name(rec["name"])] = rec["id"]
            next_id += 1

        works = fetch_author_works(oa_id, WORKS_PER_AUTHOR)
        for work in works:
            abstract = invert_abstract(work.get("abstract_inverted_index"))
            if abstract and len(rec.get("recent_abstracts", [])) < 5:
                rec.setdefault("recent_abstracts", []).append(abstract[:2000])

        for inst in author.get("last_known_institutions") or []:
            iurl = inst.get("id")
            if not iurl:
                continue
            if iurl not in institutions:
                institutions[iurl] = {
                    "id": len(institutions),
                    "openalex_id": iurl,
                    "name": inst.get("display_name", "Unknown"),
                    "world_ranking": 999,
                    "institution_type": inst.get("type", "unknown"),
                    "rd_spend": 0,
                    "country": inst.get("country_code", "unknown"),
                    "faculty_size": 0,
                }
            iidx = institutions[iurl]["id"]
            rec["institution_id"] = iidx
            affiliated.add((rec["id"], iidx))

    inst_values = sorted(institutions.values(), key=lambda x: x["id"])
    for i, inst in enumerate(inst_values):
        inst["id"] = i

    id_remap = {old["id"]: i for i, old in enumerate(inst_values)}
    for r in researchers:
        if r.get("institution_id") is not None and r["institution_id"] in id_remap:
            r["institution_id"] = id_remap[r["institution_id"]]
    affiliated = {(r, id_remap.get(i, i)) for r, i in affiliated if i in id_remap}

    with open(raw / "researchers.json", "w") as f:
        json.dump(researchers, f, indent=2)
    with open(raw / "institutions.json", "w") as f:
        json.dump(inst_values, f, indent=2)
    with open(raw / "affiliated.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "target_id"])
        w.writerows(sorted(affiliated))

    print(f"Done: {len(researchers)} researchers, {len(inst_values)} institutions, {len(affiliated)} affiliations")
    print(f"  with abstracts: {sum(1 for r in researchers if r.get('recent_abstracts'))}")


if __name__ == "__main__":
    main()
