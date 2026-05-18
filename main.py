import torch
from graph.schema import build_graph, add_edges
from graph.features import (
    build_researcher_features, build_institution_features,
    build_topic_features, build_grant_features, build_agency_features
)
from models.rag import setup_collection, embed_grants
from retrieval.fallback import build_tfidf_index
from training.train_gnn import train
from training.evaluate import hit_rate_at_k, ndcg_at_k

# ── Step 1: Load raw data (from Person B's ingestion) ────────────────
# Replace these with actual loaded JSON/CSV from data/raw/
researchers  = []   # list of dicts
institutions = []
topics       = []
grants       = []
agencies     = []

# ── Step 2: Build feature tensors ───────────────────────────────────
r_feat = build_researcher_features(researchers)
i_feat = build_institution_features(institutions)
t_feat = build_topic_features(topics)
g_feat = build_grant_features(grants)
a_feat = build_agency_features(agencies)

# ── Step 3: Build graph ──────────────────────────────────────────────
data = build_graph(r_feat, i_feat, t_feat, g_feat, a_feat)

# Edge indices — build these from relationship data (src/dst node IDs)
# Example: torch.tensor([[0,1,2],[3,4,5]], dtype=torch.long)
data = add_edges(
    data,
    affiliated = torch.zeros(2, 0, dtype=torch.long),  # replace
    researches = torch.zeros(2, 0, dtype=torch.long),
    received   = torch.zeros(2, 0, dtype=torch.long),
    funds      = torch.zeros(2, 0, dtype=torch.long),
    provides   = torch.zeros(2, 0, dtype=torch.long),
)

# ── Step 4: RAG setup ────────────────────────────────────────────────
setup_collection()
embed_grants(grants)
build_tfidf_index(grants)

# ── Step 5: Train GNN ────────────────────────────────────────────────
model, predictor, z_dict = train(data, epochs=200, use_wandb=False)

# ── Step 6: Evaluate ─────────────────────────────────────────────────
pos_edge = data['researcher', 'RECEIVED_PAST', 'grant'].edge_index
num_g    = data['grant'].x.shape[0]
hit_rate_at_k(z_dict, predictor, pos_edge, num_g, k=10)
ndcg_at_k(z_dict, predictor, pos_edge, num_g, k=10)

print("\nGrantMatch pipeline complete.")