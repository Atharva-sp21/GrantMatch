#!/usr/bin/env python3
"""
Full experiment suite: enrich (optional) → features → baselines → model comparison → reports.

  python run_experiments.py              # use existing data/raw
  python run_experiments.py --enrich     # refresh OpenAlex topics/abstracts (slow)
  python run_experiments.py --quick      # fewer epochs for smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_loader import load_all_data
from graph.embeddings import build_node_embeddings
from graph.features import build_all_node_features
from graph.schema import (
    add_edges,
    build_graph,
    build_similarity_edges,
    export_graph_statistics,
    graph_for_training,
    validate_graph,
)
from training.baseline import evaluate_embedding_baseline
from training.diagnostics import cold_start_report, embedding_statistics
from training.error_analysis import run_error_analysis
from training.evaluate import (
    evaluate_ranking,
    evaluate_researcher_level,
    embedding_score_fn,
    gnn_score_fn,
)
from training.split import split_received_edges
from training.train_gnn import train

REPORTS = ROOT / "reports"


def build_full_graph(data_dict, researchers, grants, agencies, institutions, topics, r_emb, g_emb):
    feats = build_all_node_features(
        researchers, institutions, topics, grants, agencies, r_emb, g_emb
    )
    similar, similar_attr = build_similarity_edges(
        r_emb,
        data_dict["researches"],
        data_dict["affiliated"],
        researchers=researchers,
        top_k=8,
    )
    data = build_graph(
        feats["researcher"],
        feats["institution"],
        feats["topic"],
        feats["grant"],
        feats["agency"],
    )
    return add_edges(
        data,
        data_dict["affiliated"],
        data_dict["researches"],
        data_dict["received"],
        data_dict["funds"],
        data_dict["provides"],
        similar=similar,
        similar_attr=similar_attr,
    ), similar.shape[1]


def build_graph_no_similar(data_dict, researchers, grants, agencies, institutions, topics, r_emb, g_emb):
    feats = build_all_node_features(
        researchers, institutions, topics, grants, agencies, r_emb, g_emb
    )
    data = build_graph(
        feats["researcher"],
        feats["institution"],
        feats["topic"],
        feats["grant"],
        feats["agency"],
    )
    return add_edges(
        data,
        data_dict["affiliated"],
        data_dict["researches"],
        data_dict["received"],
        data_dict["funds"],
        data_dict["provides"],
        similar=None,
    ), 0


def eval_model(name, z_or_emb, predictor, test_pos, val_pos, num_g, mode="gnn"):
    if mode == "emb":
        score_fn = embedding_score_fn(z_or_emb[0], z_or_emb[1])
    else:
        score_fn = gnn_score_fn(z_or_emb, predictor, num_g)

    metrics = {}
    if val_pos.shape[1]:
        metrics["val"] = evaluate_ranking(val_pos, num_g, score_fn, ks=[1, 5, 10], label=f"{name} VAL")
        evaluate_researcher_level(val_pos, num_g, score_fn, ks=[5, 10], label=f"{name} VAL (researcher)")
    metrics["test"] = evaluate_ranking(test_pos, num_g, score_fn, ks=[1, 5, 10], label=f"{name} TEST")
    evaluate_researcher_level(test_pos, num_g, score_fn, ks=[1, 5, 10], label=f"{name} TEST (researcher)")
    m = metrics.get("test", {})
    return {
        "hit@1": m.get("hit@1", 0),
        "hit@5": m.get("hit@5", 0),
        "hit@10": m.get("hit@10", 0),
        "ndcg@10": m.get("ndcg@10", 0),
        "mrr": m.get("mrr", 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrich", action="store_true", help="Run data/enrich_dataset.py first")
    parser.add_argument("--quick", action="store_true", help="30 epochs, skip GraphSAGE")
    args = parser.parse_args()

    REPORTS.mkdir(exist_ok=True)
    epochs = 40 if args.quick else 100

    if args.enrich:
        print("[*] Running OpenAlex enrichment (may take 30+ min)...")
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "data" / "enrich_dataset.py")], check=False)

    print("=" * 60)
    print("GrantMatch Experiment Suite")
    print("=" * 60)

    data_dict = load_all_data()
    researchers = data_dict["researchers"]
    grants = data_dict["grants"]
    topics = data_dict["topics"]
    agencies = data_dict["agencies"]
    institutions = data_dict["institutions"]

    before_edges = data_dict["researches"].shape[1]
    before_abs = sum(1 for r in researchers if r.get("recent_abstracts"))

    # PI matching quality report (sample for speed; full run via data/pi_matching.py)
    matching_report = {"total_researchers": len(researchers)}
    try:
        from data.pi_matching import match_pi_to_openalex, save_matching_report

        samples = []
        for r in researchers:
            if r.get("openalex_id"):
                continue
            if len(samples) < 80 and r.get("name"):
                m = match_pi_to_openalex(r["name"])
                samples.append(m.__dict__)
        matching_report["unmatched_samples"] = samples
        matching_report["low_confidence_count"] = sum(
            1 for s in samples if s.get("confidence", 0) < 0.72
        )
        save_matching_report(matching_report, REPORTS / "researcher_matching_report.json")
    except Exception as e:
        matching_report["error"] = str(e)
        with open(REPORTS / "researcher_matching_report.json", "w") as f:
            json.dump(matching_report, f, indent=2)

    r_emb, g_emb, _ = build_node_embeddings(researchers, grants, use_aggregate_abstracts=True)
    after_abs = sum(1 for r in researchers if r.get("recent_abstracts"))
    embedding_statistics(r_emb, g_emb)

    print(f"\n[Coverage] abstracts: {before_abs} -> {after_abs} (with text emb: {len(researchers)})")

    data_full, n_similar = build_full_graph(
        data_dict, researchers, grants, agencies, institutions, topics, r_emb, g_emb
    )
    after_edges = data_dict["researches"].shape[1]

    validate_graph(
        data_full,
        expected={
            "researcher": len(researchers),
            "grant": len(grants),
            "topic": len(topics),
            "institution": len(institutions),
            "agency": len(agencies),
        },
        raw_edge_counts={
            "RESEARCHES": after_edges,
            "RECEIVED_PAST": data_dict["received"].shape[1],
        },
    )
    export_graph_statistics(data_full, REPORTS / "graph_statistics.json")

    cold_start_report(len(researchers), len(grants), data_dict["received"], researchers)

    splits = split_received_edges(data_dict["received"], seed=42)
    train_pos, val_pos, test_pos = splits["train"], splits["val"], splits["test"]
    data_train = graph_for_training(data_full, train_pos)
    train_r_set = set(train_pos[0].tolist())
    num_g = data_full["grant"].num_nodes

    comparison = {}
    loss_comparison = {}

    # 1. Baseline
    t0 = time.time()
    baseline_res = evaluate_embedding_baseline(r_emb, g_emb, test_pos, val_pos, ks=[1, 5, 10])
    comparison["embedding_cosine"] = {
        **eval_model("Baseline", (r_emb, g_emb), None, test_pos, val_pos, num_g, mode="emb"),
        "train_sec": time.time() - t0,
    }

    # 2. Loss comparison on HGT+similar
    for loss in ["bce", "margin", "bpr"]:
        model, pred, z, hist, sec = train(
            data_train,
            train_pos,
            val_pos,
            data_dict["researches"],
            data_dict["funds"],
            data_dict["provides"],
            epochs=epochs,
            loss_type=loss,
            model_type="hgt",
            grant_emb=g_emb,
            researchers_meta=researchers,
            grants_meta=grants,
            checkpoint_tag=f"loss_{loss}",
        )
        if model is None:
            continue
        m = eval_model(f"HGT_{loss}", z, pred, test_pos, val_pos, num_g)
        m["val_loss_final"] = hist["val"][-1] if hist.get("val") else None
        m["train_sec"] = sec
        loss_comparison[loss] = m

    # 3. HGT without similar
    data_nosim, _ = build_graph_no_similar(
        data_dict, researchers, grants, agencies, institutions, topics, r_emb, g_emb
    )
    data_train_ns = graph_for_training(data_nosim, train_pos)
    model, pred, z, _, sec = train(
        data_train_ns,
        train_pos,
        val_pos,
        data_dict["researches"],
        data_dict["funds"],
        data_dict["provides"],
        epochs=epochs,
        loss_type="bpr",
        model_type="hgt",
        grant_emb=g_emb,
        researchers_meta=researchers,
        grants_meta=grants,
        checkpoint_tag="hgt_no_similar",
    )
    if model:
        comparison["hgt_no_similar"] = {**eval_model("HGT no SIM", z, pred, test_pos, val_pos, num_g), "train_sec": sec}

    # 4. HGT + similar + BPR (primary)
    model, pred, z, _, sec = train(
        data_train,
        train_pos,
        val_pos,
        data_dict["researches"],
        data_dict["funds"],
        data_dict["provides"],
        epochs=epochs,
        loss_type="bpr",
        model_type="hgt",
        grant_emb=g_emb,
        researchers_meta=researchers,
        grants_meta=grants,
        checkpoint_tag="hgt_similar_bpr",
    )
    if model:
        comparison["hgt_similar_bpr"] = {
            **eval_model("HGT+SIM+BPR", z, pred, test_pos, val_pos, num_g),
            "train_sec": sec,
        }
        score_fn = gnn_score_fn(z, pred, num_g)
        run_error_analysis(
            test_pos,
            score_fn,
            num_g,
            researchers,
            grants,
            topics,
            institutions,
            train_r_set,
        )

    # 5. GraphSAGE
    if not args.quick:
        model, pred, z, _, sec = train(
            data_train,
            train_pos,
            val_pos,
            data_dict["researches"],
            data_dict["funds"],
            data_dict["provides"],
            epochs=epochs,
            loss_type="bpr",
            model_type="graphsage",
            grant_emb=g_emb,
            researchers_meta=researchers,
            grants_meta=grants,
            checkpoint_tag="graphsage",
        )
        if model:
            comparison["graphsage_bpr"] = {
                **eval_model("GraphSAGE", z, pred, test_pos, val_pos, num_g),
                "train_sec": sec,
            }

    # Best model
    best_name = max(comparison, key=lambda k: comparison[k].get("hit@10", 0))
    best_loss = max(loss_comparison, key=lambda k: loss_comparison[k].get("hit@10", 0)) if loss_comparison else "bpr"

    enrichment_summary = {
        "researches_edges_before": before_edges,
        "researches_edges_after": after_edges,
        "abstracts_before": before_abs,
        "abstracts_after": after_abs,
        "similar_edges": int(n_similar),
    }

    final = {
        "enrichment": enrichment_summary,
        "model_comparison": comparison,
        "loss_comparison": loss_comparison,
        "recommended_model": best_name,
        "recommended_loss": best_loss,
        "largest_gain_hypothesis": _largest_gain(comparison, loss_comparison, enrichment_summary),
    }

    with open(REPORTS / "model_comparison.json", "w") as f:
        json.dump(final, f, indent=2)
    with open(REPORTS / "evaluation_report.json", "w") as f:
        json.dump(final, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Best model (Hit@10): {best_name} -> {comparison[best_name].get('hit@10', 0):.4f}")
    print(f"  Best loss (Hit@10):  {best_loss} -> {loss_comparison.get(best_loss, {}).get('hit@10', 0):.4f}")
    print(f"  Reports: {REPORTS}")
    print("=" * 60)


def _largest_gain(comparison, loss_comparison, enrich):
    candidates = []
    if loss_comparison.get("bpr", {}).get("hit@10", 0) > loss_comparison.get("bce", {}).get("hit@10", 0):
        candidates.append("Switching BCE -> BPR ranking loss")
    if comparison.get("hgt_similar_bpr", {}).get("hit@10", 0) > comparison.get("embedding_cosine", {}).get("hit@10", 0):
        candidates.append("GNN structure over raw embeddings")
    if enrich["researches_edges_after"] > enrich["researches_edges_before"] * 1.2:
        candidates.append("Denser researcher-topic edges from OpenAlex")
    if enrich["abstracts_after"] > enrich["abstracts_before"]:
        candidates.append("More abstracts + aggregated embeddings")
    return candidates[0] if candidates else "Further label cleaning (PI matching) likely highest impact"


if __name__ == "__main__":
    main()
