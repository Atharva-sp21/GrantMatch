import sys

import torch

from data_loader import load_all_data
from graph.features import (
    build_agency_features,
    build_grant_features,
    build_institution_features,
    build_researcher_features,
    build_topic_features,
)
from graph.schema import add_edges, build_graph
from training.train_gnn import train


def main():
    print("=" * 60)
    print("GrantMatch API-First GNN Pipeline")
    print("=" * 60)

    print("\n[STEP 1] Loading normalized data...")
    data_dict = load_all_data()

    researchers = data_dict["researchers"]
    institutions = data_dict["institutions"]
    agencies = data_dict["agencies"]
    grants = data_dict["grants"]
    topics = data_dict["topics"]

    if not researchers or not grants:
        print("[ERROR] Missing researchers or grants. Run ingestion/run_pipeline.py first.")
        sys.exit(1)

    print("\n[STEP 2] Building features...")
    r_feat = build_researcher_features(researchers) if researchers else torch.zeros(0, 390)
    i_feat = build_institution_features(institutions) if institutions else torch.zeros(0, 388)
    t_feat = build_topic_features(topics) if topics else torch.zeros(0, 386)
    g_feat = build_grant_features(grants) if grants else torch.zeros(0, 390)
    a_feat = build_agency_features(agencies) if agencies else torch.zeros(0, 387)

    print(f"[OK] researcher features: {tuple(r_feat.shape)}")
    print(f"[OK] institution features: {tuple(i_feat.shape)}")
    print(f"[OK] topic features: {tuple(t_feat.shape)}")
    print(f"[OK] grant features: {tuple(g_feat.shape)}")
    print(f"[OK] agency features: {tuple(a_feat.shape)}")

    print("\n[STEP 3] Building heterogeneous graph...")
    data = build_graph(r_feat, i_feat, t_feat, g_feat, a_feat)
    data = add_edges(
        data,
        affiliated=data_dict["affiliated"],
        researches=data_dict["researches"],
        received=data_dict["received"],
        funds=data_dict["funds"],
        provides=data_dict["provides"],
    )
    print(f"[OK] Graph built with {len(data.node_types)} node types and {len(data.edge_types)} edge types")

    print("\n[STEP 4] Training GNN...")
    model, predictor, z_dict, metrics = train(
        data,
        epochs=120,
        hidden=256,
        heads=2,
        lr=1e-3,
        seed=42,
    )

    if model is None:
        print("[WARN] Training skipped because no positive grant edges were available")
        return

    print("\n[STEP 5] Final metrics")
    val_metrics = metrics.get("val", {})
    test_metrics = metrics.get("test", {})
    if val_metrics:
        print(
            f"[VAL] auc {val_metrics.get('auc', 0.0):.4f} | ap {val_metrics.get('average_precision', 0.0):.4f} | "
            f"acc {val_metrics.get('accuracy', 0.0):.4f} | mrr {val_metrics.get('mrr', 0.0):.4f} | "
            f"hits@10 {val_metrics.get('hits@10', 0.0):.4f}"
        )
    if test_metrics:
        print(
            f"[TEST] auc {test_metrics.get('auc', 0.0):.4f} | ap {test_metrics.get('average_precision', 0.0):.4f} | "
            f"acc {test_metrics.get('accuracy', 0.0):.4f} | mrr {test_metrics.get('mrr', 0.0):.4f} | "
            f"hits@10 {test_metrics.get('hits@10', 0.0):.4f}"
        )

    if test_metrics:
        print(
            f"[DIAG] pos_prob_mean {test_metrics.get('positive_prob_mean', 0.0):.4f} | "
            f"neg_prob_mean {test_metrics.get('negative_prob_mean', 0.0):.4f}"
        )

    print("\n" + "=" * 60)
    print("GrantMatch pipeline complete")
    print("=" * 60)


if __name__ == "__main__":
    main()