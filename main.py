import torch
import sys
from pathlib import Path

from data_loader import load_all_data
from graph.schema import build_graph, add_edges
from graph.features import (
    build_researcher_features, build_institution_features,
    build_topic_features, build_grant_features, build_agency_features
)
from models.rag import setup_collection, embed_grants
from retrieval.fallback import build_tfidf_index
from training.train_gnn import train
from training.evaluate import hit_rate_at_k, ndcg_at_k

def main():
    print("="*60)
    print("GrantMatch ML Pipeline")
    print("="*60)

    # ── Step 1: Load raw data ────────────────────────────────────────
    print("\n[STEP 1] Loading data from data/raw/...")
    data_dict = load_all_data()

    researchers = data_dict['researchers']
    institutions = data_dict['institutions']
    agencies = data_dict['agencies']
    grants = data_dict['grants']
    topics = data_dict['topics']

    if not researchers or not grants:
        print("[ERROR] No data loaded. Check data/raw/ folder.")
        sys.exit(1)

    # ── Step 2: Build feature tensors ────────────────────────────────
    print("\n[STEP 2] Building feature tensors...")

    # Only build features for non-empty entities
    r_feat = build_researcher_features(researchers) if len(researchers) > 0 else torch.zeros(0, 9)
    i_feat = build_institution_features(institutions) if len(institutions) > 0 else torch.zeros(0, 5)
    t_feat = build_topic_features(topics) if len(topics) > 0 else torch.zeros(0, 5)
    g_feat = build_grant_features(grants) if len(grants) > 0 else torch.zeros(0, 7)
    a_feat = build_agency_features(agencies) if len(agencies) > 0 else torch.zeros(0, 4)

    print(f"[OK] Researcher features: {r_feat.shape}")
    print(f"[OK] Institution features: {i_feat.shape}")
    print(f"[OK] Topic features: {t_feat.shape}")
    print(f"[OK] Grant features: {g_feat.shape}")
    print(f"[OK] Agency features: {a_feat.shape}")

    # ── Step 3: Build graph ──────────────────────────────────────────
    print("\n[STEP 3] Building heterogeneous graph...")
    data = build_graph(r_feat, i_feat, t_feat, g_feat, a_feat)

    data = add_edges(
        data,
        affiliated=data_dict['affiliated'],
        researches=data_dict['researches'],
        received=data_dict['received'],
        funds=data_dict['funds'],
        provides=data_dict['provides'],
    )
    print("[OK] Graph built with edges")

    # ── Step 4: RAG setup ────────────────────────────────────────────
    print("\n[STEP 4] Setting up RAG system...")
    setup_collection()
    embed_grants(grants)
    build_tfidf_index(grants)

    # ── Step 5: Train GNN ────────────────────────────────────────────
    print("\n[STEP 5] Training GNN model...")
    print("(This may take a few minutes...)")
    model, predictor, z_dict = train(data, epochs=200, use_wandb=False)

    if model is None:
        print("[WARN] Training skipped (no edges to train on)")
        print("\n" + "="*60)
        print("Pipeline complete (no training data)")
        print("="*60)
        return

    # ── Step 6: Evaluate ─────────────────────────────────────────────
    print("\n[STEP 6] Evaluating model...")

    # Only evaluate if RECEIVED_PAST edges exist
    if ('researcher', 'RECEIVED_PAST', 'grant') in data.edge_types:
        pos_edge = data['researcher', 'RECEIVED_PAST', 'grant'].edge_index
        if pos_edge.shape[1] > 0:
            num_g = data['grant'].x.shape[0]
            hit_rate_at_k(z_dict, predictor, pos_edge, num_g, k=10)
            ndcg_at_k(z_dict, predictor, pos_edge, num_g, k=10)
        else:
            print("[WARN] No RECEIVED_PAST edges to evaluate")
    else:
        print("[WARN] No RECEIVED_PAST edge type in graph")

    print("\n" + "="*60)
    print("GrantMatch pipeline complete!")
    print("="*60)
    print(f"\nModels saved to checkpoints/")
    print(f"Trained on {len(researchers)} researchers and {len(grants)} grants")
    print("Ready for inference!")

if __name__ == "__main__":
    main()