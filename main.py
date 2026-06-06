"""
GrantMatch end-to-end pipeline:
  load → embed → features → validate graph → split → baseline → train GNN → evaluate → diagnostics
"""

import sys

import torch

from data_loader import load_all_data
from graph.embeddings import build_node_embeddings
from graph.features import build_all_node_features
from graph.schema import (
    add_edges,
    build_graph,
    build_similarity_edges,
    graph_for_training,
    validate_graph,
)
from training.baseline import evaluate_embedding_baseline
from training.diagnostics import (
    cold_start_report,
    degree_distributions,
    embedding_statistics,
    sample_predictions,
    save_report,
)
from training.evaluate import (
    evaluate_ranking,
    evaluate_researcher_level,
    gnn_score_fn,
)
from training.split import split_received_edges
from training.train_gnn import train


def main():
    print("=" * 60)
    print("GrantMatch ML Pipeline (improved)")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────────
    data_dict = load_all_data()
    researchers = data_dict["researchers"]
    grants = data_dict["grants"]
    institutions = data_dict["institutions"]
    topics = data_dict["topics"]
    agencies = data_dict["agencies"]

    if not researchers or not grants:
        print("[ERROR] Missing data/raw/. Run: python data/fetch_data.py")
        sys.exit(1)

    expected = {
        "researcher": len(researchers),
        "grant": len(grants),
        "agency": len(agencies),
        "topic": len(topics),
        "institution": len(institutions),
    }
    raw_edges = {
        "AFFILIATED_WITH": data_dict["affiliated"].shape[1],
        "RESEARCHES": data_dict["researches"].shape[1],
        "RECEIVED_PAST": data_dict["received"].shape[1],
        "FUNDS_TOPIC": data_dict["funds"].shape[1],
        "PROVIDES": data_dict["provides"].shape[1],
    }

    # ── 2. Text embeddings ───────────────────────────────────────────
    print("\n[STEP 2] Text embeddings (SPECTER2 → MiniLM fallback)...")
    r_emb, g_emb, emb_dim = build_node_embeddings(researchers, grants)
    embedding_statistics(r_emb, g_emb)

    # ── 3. Features (tabular + text) ─────────────────────────────────
    print("\n[STEP 3] Feature engineering (log1p + standard scale + concat embeddings)...")
    feats = build_all_node_features(
        researchers,
        institutions,
        topics,
        grants,
        agencies,
        researcher_emb=r_emb,
        grant_emb=g_emb,
    )
    print(
        f"  researcher {feats['researcher'].shape} "
        f"(7 tabular + {emb_dim} text)"
    )
    print(f"  grant {feats['grant'].shape} (7 tabular + {emb_dim} text)")

    # ── 4. Full graph + similarity edges ─────────────────────────────
    print("\n[STEP 4] Building graph + researcher similarity edges...")
    similar, similar_attr = build_similarity_edges(
        r_emb,
        data_dict["researches"],
        data_dict["affiliated"],
        researchers=researchers,
        top_k=8,
    )
    print(f"  SIMILAR edges: {similar.shape[1]}")

    data_full = build_graph(
        feats["researcher"],
        feats["institution"],
        feats["topic"],
        feats["grant"],
        feats["agency"],
    )
    data_full = add_edges(
        data_full,
        affiliated=data_dict["affiliated"],
        researches=data_dict["researches"],
        received=data_dict["received"],
        funds=data_dict["funds"],
        provides=data_dict["provides"],
        similar=similar,
        similar_attr=similar_attr,
    )

    validate_graph(data_full, expected=expected, raw_edge_counts=raw_edges)

    # ── 5. Cold-start ──────────────────────────────────────────────────
    cold_stats = cold_start_report(
        len(researchers),
        len(grants),
        data_dict["received"],
        researchers,
    )
    degree_distributions(data_full)

    # ── 6. Train / val / test split (researcher-stratified) ────────────
    print("[STEP 6] Splitting received_past edges (70/15/15, by researcher)...")
    splits = split_received_edges(data_dict["received"], seed=42)
    train_pos, val_pos, test_pos = splits["train"], splits["val"], splits["test"]
    print(
        f"  train={train_pos.shape[1]} val={val_pos.shape[1]} test={test_pos.shape[1]}"
    )

    data_train = graph_for_training(data_full, train_pos)

    # ── 7. Baseline (embeddings only) ──────────────────────────────────
    print("\n[STEP 7] Embedding cosine baseline (before GNN)...")
    baseline_metrics = evaluate_embedding_baseline(
        r_emb, g_emb, test_pos, val_pos, ks=(1, 5, 10)
    )

    # ── 8. Train GNN ───────────────────────────────────────────────────
    print("\n[STEP 8] Training HGT (hidden=128, heads=4, hard negatives)...")
    model, predictor, z_dict, history, _ = train(
        data_train,
        train_pos=train_pos,
        val_pos=val_pos,
        researches=data_dict["researches"],
        funds=data_dict["funds"],
        provides=data_dict["provides"],
        epochs=120,
        loss_type="bpr",
        model_type="hgt",
        grant_emb=g_emb,
        researchers_meta=researchers,
        grants_meta=grants,
        checkpoint_tag="main",
    )
    if model is None:
        sys.exit(1)

    # ── 9. GNN evaluation on held-out test ─────────────────────────────
    print("\n[STEP 9] GNN evaluation on TEST split...")
    num_g = data_full["grant"].num_nodes
    score_fn = gnn_score_fn(z_dict, predictor, num_g)

    gnn_test = evaluate_ranking(
        test_pos, num_g, score_fn, ks=[1, 5, 10], label="GNN TEST (edge-level)"
    )
    evaluate_researcher_level(
        test_pos, num_g, score_fn, ks=[5, 10], label="GNN TEST (researcher-level)"
    )

    if val_pos.shape[1]:
        gnn_val = evaluate_ranking(
            val_pos, num_g, score_fn, ks=[1, 5, 10], label="GNN VAL"
        )
    else:
        gnn_val = {}

    sample_predictions(
        test_pos,
        score_fn,
        num_g,
        grants,
        researchers,
        n_success=3,
        n_fail=3,
    )

    report = {
        "dataset": expected,
        "raw_edges": raw_edges,
        "splits": {
            "train": int(train_pos.shape[1]),
            "val": int(val_pos.shape[1]),
            "test": int(test_pos.shape[1]),
        },
        "cold_start": cold_stats,
        "baseline": baseline_metrics,
        "gnn_test": gnn_test,
        "gnn_val": gnn_val,
        "final_train_loss": history["train"][-1] if history.get("train") else None,
        "final_val_loss": history["val"][-1] if history.get("val") else None,
    }
    save_report(report)

    print("\n" + "=" * 60)
    print("Pipeline complete. See reports/ for plots and evaluation_report.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
