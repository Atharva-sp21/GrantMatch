"""Detailed error analysis for ranking failures."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def run_error_analysis(
    pos_edge,
    score_fn: Callable,
    num_grants: int,
    researchers: List[dict],
    grants: List[dict],
    topics: List[dict],
    institutions: List[dict],
    received_train_researchers: set,
    k: int = 10,
) -> dict:
    by_r: Dict[int, set] = defaultdict(set)
    for i in range(pos_edge.shape[1]):
        by_r[pos_edge[0, i].item()].add(pos_edge[1, i].item())

    successes, failures = [], []
    cold_hits, warm_hits = 0, 0
    cold_n, warm_n = 0, 0
    by_topic = Counter()
    by_topic_total = Counter()
    by_inst = Counter()
    by_inst_total = Counter()
    by_career = Counter()
    by_career_total = Counter()

    topic_names = {t["id"]: t.get("name", str(t["id"])) for t in topics}
    inst_names = {i["id"]: i.get("name", "") for i in institutions}

    for r_id, true_gs in by_r.items():
        r = researchers[r_id] if r_id < len(researchers) else {}
        is_cold = r_id not in received_train_researchers
        scores = score_fn(r_id)
        top = scores.argsort(descending=True)[:k].tolist()
        hit = any(g in top for g in true_gs)

        if is_cold:
            cold_n += 1
            cold_hits += int(hit)
        else:
            warm_n += 1
            warm_hits += int(hit)

        career = r.get("career_stage", "unknown")
        by_career_total[career] += 1
        if hit:
            by_career[career] += 1

        inst = r.get("institution_id")
        if inst is not None:
            by_inst_total[inst] += 1
            if hit:
                by_inst[inst] += 1

        for g in true_gs:
            grec = grants[g] if g < len(grants) else {}
            for tid in grec.get("topic_ids") or []:
                by_topic_total[tid] += 1
                if hit:
                    by_topic[tid] += 1

        entry = {
            "researcher_id": r_id,
            "name": r.get("name", "")[:60],
            "cold_start": is_cold,
            "num_true_grants": len(true_gs),
            "hit": hit,
            "top3_titles": [grants[g].get("title", "")[:70] for g in top[:3] if g < len(grants)],
            "true_titles": [grants[g].get("title", "")[:70] for g in true_gs if g < len(grants)],
        }
        if hit and len(successes) < 15:
            successes.append(entry)
        elif not hit and len(failures) < 15:
            failures.append(entry)

    topic_perf = {
        topic_names.get(tid, str(tid)): {
            "hit_rate": by_topic[tid] / c if c else 0,
            "count": c,
        }
        for tid, c in by_topic_total.items()
    }
    inst_perf = {
        inst_names.get(iid, str(iid))[:40]: {
            "hit_rate": by_inst[iid] / c if c else 0,
            "count": c,
        }
        for iid, c in sorted(by_inst_total.items(), key=lambda x: -x[1])[:15]
    }

    report = {
        "researcher_level_hit_rate": (cold_hits + warm_hits) / max(len(by_r), 1),
        "cold_start": {
            "researchers": cold_n,
            "hit_rate": cold_hits / max(cold_n, 1),
        },
        "warm_labeled": {
            "researchers": warm_n,
            "hit_rate": warm_hits / max(warm_n, 1),
        },
        "by_career_stage": {
            c: {"hit_rate": by_career[c] / by_career_total[c]}
            for c in by_career_total
        },
        "by_topic_top": dict(
            sorted(topic_perf.items(), key=lambda x: -x[1]["count"])[:12]
        ),
        "by_institution_top": inst_perf,
        "top_successes": successes,
        "top_failures": failures,
        "largest_error_sources": _infer_bottlenecks(
            cold_hits / max(cold_n, 1),
            warm_hits / max(warm_n, 1),
            topic_perf,
        ),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "error_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def _infer_bottlenecks(cold_hr, warm_hr, topic_perf) -> List[str]:
    issues = []
    if cold_hr < warm_hr * 0.5:
        issues.append("Cold-start researchers rank much worse — need better text/semantic fallback.")
    if warm_hr < 0.05:
        issues.append("Even warm researchers show near-random ranking — label noise or weak features.")
    low_topics = [t for t, v in topic_perf.items() if v["count"] >= 5 and v["hit_rate"] < 0.02]
    if low_topics:
        issues.append(f"Poor performance on topics: {low_topics[:5]}")
    if not issues:
        issues.append("Errors spread across topics; focus on PI–grant label quality and hard negatives.")
    return issues
