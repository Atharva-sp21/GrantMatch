"""
Phase 3: PI Matching Quality Improvement

Audits PI matching by re-evaluating all researchers involved in received_past edges.
Produces researcher_matching_report.json and a cleaned high_confidence_received_past.csv.
"""

import sys
import os
import json
import csv
from pathlib import Path
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pi_matching import match_pi_to_openalex

RAW = Path(__file__).resolve().parent / "raw"
PROCESSED = Path(__file__).resolve().parent / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS = Path(__file__).resolve().parent.parent / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.72

def main():
    print("Loading data for Phase 3 PI Matching...")
    
    with open(RAW / "researchers.json") as f:
        researchers = json.load(f)
        
    received = []
    with open(RAW / "received_past.csv") as f:
        for row in csv.DictReader(f):
            received.append((int(row["source_id"]), int(row["target_id"])))
            
    researcher_ids_with_grants = {s for s, g in received}
    
    audit_results = []
    low_confidence_count = 0
    high_confidence_count = 0
    duplicate_candidates = 0
    
    high_confidence_researcher_ids = set()
    
    print(f"Auditing {len(researcher_ids_with_grants)} researchers with grant history...")
    
    t0 = time.time()
    for i, r_id in enumerate(researcher_ids_with_grants):
        r = researchers[r_id]
        pi_name = r.get("name", "")
        
        # We re-evaluate all of them to be safe
        m = match_pi_to_openalex(pi_name)
        
        res = {
            "researcher_id": r_id,
            "source_pi_name": pi_name,
            "openalex_id": m.openalex_id,
            "matched_author_name": m.matched_name,
            "confidence_score": m.confidence,
            "ambiguity_score": m.duplicate_candidates,
            "status": "high_confidence" if m.confidence >= CONFIDENCE_THRESHOLD else "low_confidence"
        }
        audit_results.append(res)
        
        if m.confidence >= CONFIDENCE_THRESHOLD:
            high_confidence_count += 1
            high_confidence_researcher_ids.add(r_id)
        else:
            low_confidence_count += 1
            
        if m.duplicate_candidates > 1:
            duplicate_candidates += 1
            
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(researcher_ids_with_grants)}] Evaluated {pi_name[:30]}")
            
    report = {
        "total_evaluated": len(researcher_ids_with_grants),
        "high_confidence_matches": high_confidence_count,
        "low_confidence_matches": low_confidence_count,
        "matches_with_duplicates": duplicate_candidates,
        "threshold_used": CONFIDENCE_THRESHOLD,
        "time_elapsed_seconds": round(time.time() - t0, 1),
        "audit_details": audit_results
    }
    
    out_report = REPORTS / "researcher_matching_report.json"
    with open(out_report, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\nSaved matching report to {out_report}")
    
    # Generate cleaned received_past
    high_conf_received = [(s, g) for s, g in received if s in high_confidence_researcher_ids]
    
    out_csv = PROCESSED / "high_confidence_received_past.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "target_id"])
        w.writerows(high_conf_received)
        
    print(f"Original received_past edges: {len(received)}")
    print(f"High-confidence received_past edges: {len(high_conf_received)} (removed {len(received) - len(high_conf_received)} noisy edges)")
    print(f"Saved to {out_csv}")
    
    # Overwrite the raw one so it's used by the pipeline
    out_raw = RAW / "received_past.csv"
    with open(out_raw, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "target_id"])
        w.writerows(high_conf_received)
        
    print("Overwrote data/raw/received_past.csv for subsequent training steps.")

if __name__ == "__main__":
    main()
