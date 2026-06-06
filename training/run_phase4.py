import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_all_data
from graph.embeddings import build_node_embeddings
from graph.features import build_all_node_features
from graph.schema import build_graph, add_edges, graph_for_training
from training.split import split_received_edges
from training.train_gnn import train
from training.evaluate import evaluate_ranking, scores_for_researcher

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

def compute_retrieval_scores(researchers, grants, r_emb, g_emb, researches_edges, funds_edges, val_pos, test_pos):
    num_r = len(researchers)
    num_g = len(grants)
    
    # 1. Semantic Cosine
    print("Computing semantic cosine scores...")
    r_norm = torch.nn.functional.normalize(r_emb.float(), dim=1)
    g_norm = torch.nn.functional.normalize(g_emb.float(), dim=1)
    semantic_scores = r_norm @ g_norm.t()  # [num_r, num_g]
    
    # 2. Topic Overlap
    print("Computing topic overlap scores...")
    r_topics = {i: set() for i in range(num_r)}
    if researches_edges.shape[1] > 0:
        for r, t in zip(researches_edges[0].tolist(), researches_edges[1].tolist()):
            r_topics[r].add(t)
            
    g_topics = {i: set() for i in range(num_g)}
    if funds_edges.shape[1] > 0:
        for g, t in zip(funds_edges[0].tolist(), funds_edges[1].tolist()):
            g_topics[g].add(t)
            
    # Also grants have 'topic_ids' in JSON
    for i, g in enumerate(grants):
        for t in g.get("topic_ids", []):
            g_topics[i].add(t)
            
    topic_scores = torch.zeros((num_r, num_g))
    for i in range(num_r):
        rt = r_topics[i]
        if not rt:
            continue
        for j in range(num_g):
            gt = g_topics[j]
            if gt:
                intersection = len(rt & gt)
                union = len(rt | gt)
                topic_scores[i, j] = intersection / union
                
    # 3. Metadata Similarity (Career stage)
    print("Computing metadata similarity...")
    meta_scores = torch.zeros((num_r, num_g))
    for i, r in enumerate(researchers):
        r_stage = r.get("career_stage", "unknown").lower()
        for j, g in enumerate(grants):
            g_stage = g.get("career_stage_eligibility", "any").lower()
            if g_stage == "any" or r_stage == g_stage:
                meta_scores[i, j] = 1.0
            else:
                meta_scores[i, j] = 0.0
                
    # Normalize scores to [0, 1]
    semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-9)
    
    # Hybrid Retrieval Score
    w_sem, w_top, w_meta = 0.6, 0.3, 0.1
    hybrid_scores = w_sem * semantic_scores + w_top * topic_scores + w_meta * meta_scores
    
    return hybrid_scores

def evaluate_retrieval_recall(hybrid_scores, test_pos):
    print("\n--- Evaluating Candidate Recall ---")
    ks = [25, 50, 100]
    hits = {k: 0 for k in ks}
    total = test_pos.shape[1]
    
    for i in range(total):
        r_id = test_pos[0, i].item()
        true_g = test_pos[1, i].item()
        
        scores = hybrid_scores[r_id]
        ranked = scores.argsort(descending=True).tolist()
        
        for k in ks:
            if true_g in ranked[:k]:
                hits[k] += 1
                
    metrics = {f"hit@{k}": hits[k] / total for k in ks}
    for k in ks:
        print(f"  Recall@{k}: {metrics[f'hit@{k}']:.4f}")
        
    out_path = REPORTS_DIR / "retrieval_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics

def evaluate_hybrid_ranking(pos_edge, num_grants, hybrid_scores, gnn_scores_matrix, alpha, ks=(1, 5, 10)):
    total = pos_edge.shape[1]
    hits = {k: 0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    prec_sum = {k: 0.0 for k in ks}
    rec_sum = {k: 0.0 for k in ks}
    
    for i in range(total):
        r_id = pos_edge[0, i].item()
        true_g = pos_edge[1, i].item()
        
        # Get top 100 candidates from retrieval
        h_scores = hybrid_scores[r_id]
        top_100_idx = h_scores.argsort(descending=True)[:100]
        
        # Score only the top 100 using GNN
        final_scores = []
        for g_idx in top_100_idx:
            g_idx = g_idx.item()
            g_score = gnn_scores_matrix[r_id, g_idx]
            r_score = h_scores[g_idx]
            final_scores.append((alpha * r_score + (1 - alpha) * g_score, g_idx))
            
        final_scores.sort(key=lambda x: x[0], reverse=True)
        ranked = [x[1] for x in final_scores]
        
        rank = ranked.index(true_g) + 1 if true_g in ranked else num_grants + 1
        
        for k in ks:
            if true_g in ranked[:k]:
                hits[k] += 1
            dcg = 1.0 / np.log2(rank + 1) if rank <= k else 0.0
            ndcg_sum[k] += dcg
            prec_sum[k] += 1.0 / k if rank <= k else 0.0
            rec_sum[k] += 1.0 if rank <= k else 0.0

        mrr_sum += 1.0 / rank if true_g in ranked else 0.0
        
    metrics = {}
    for k in ks:
        metrics[f"hit@{k}"] = hits[k] / total
        metrics[f"ndcg@{k}"] = ndcg_sum[k] / total
        metrics[f"precision@{k}"] = prec_sum[k] / total
        metrics[f"recall@{k}"] = rec_sum[k] / total
    metrics["mrr"] = mrr_sum / total
    
    print(f"\n--- Alpha {alpha:.2f} (Retrieval) / {1-alpha:.2f} (GNN) ---")
    print(f"  Hit@10: {metrics['hit@10']:.4f}  NDCG@10: {metrics['ndcg@10']:.4f}  MRR: {metrics['mrr']:.4f}")
    return metrics


def run_phase4():
    print("Loading data...")
    data_dict = load_all_data()
    researchers = data_dict["researchers"]
    grants = data_dict["grants"]
    institutions = data_dict["institutions"]
    topics = data_dict["topics"]
    agencies = data_dict["agencies"]
    
    # 1. Embeddings & Features
    print("Building embeddings & features...")
    r_emb, g_emb, _ = build_node_embeddings(researchers, grants)
    feats = build_all_node_features(researchers, institutions, topics, grants, agencies, r_emb, g_emb)
    
    # 2. Base graph (No Similarity Edges for best config)
    data = build_graph(feats["researcher"], feats["institution"], feats["topic"], feats["grant"], feats["agency"])
    data = add_edges(
        data.clone(),
        affiliated=data_dict["affiliated"], researches=data_dict["researches"],
        received=data_dict["received"], funds=data_dict["funds"], provides=data_dict["provides"],
        similar=torch.zeros(2,0, dtype=torch.long), similar_attr=torch.zeros(0)
    )
    
    # 3. Splits
    splits = split_received_edges(data_dict["received"], seed=42)
    train_pos, val_pos, test_pos = splits["train"], splits["val"], splits["test"]
    num_g = len(grants)
    num_r = len(researchers)
    
    # 4. Retrieval Stage
    hybrid_scores = compute_retrieval_scores(
        researchers, grants, r_emb, g_emb, 
        data_dict["researches"], data_dict["funds"],
        val_pos, test_pos
    )
    evaluate_retrieval_recall(hybrid_scores, test_pos)
    
    # 5. Train best GNN (HGT NoSim BPR)
    print("\nTraining HGT model for re-ranking (reduced epochs to 30 for speed)...")
    data_train = graph_for_training(data, train_pos)
    model, predictor, z_dict, history, _ = train(
        data_train, train_pos=train_pos, val_pos=val_pos,
        researches=data_dict["researches"], funds=data_dict["funds"], provides=data_dict["provides"],
        epochs=30, loss_type="bpr", model_type="hgt",
        grant_emb=g_emb, researchers_meta=researchers, grants_meta=grants,
        use_similar_edges=False, checkpoint_tag="phase4_reranker"
    )
    
    # Pre-compute all GNN scores for fast hybrid evaluation
    print("\nPre-computing GNN scores...")
    gnn_scores_matrix = torch.zeros((num_r, num_g))
    model.eval()
    predictor.eval()
    with torch.no_grad():
        z_r = z_dict["researcher"]
        z_g = z_dict["grant"]
        for r_id in range(num_r):
            z_r_vec = z_r[r_id].unsqueeze(0).expand(num_g, -1)
            logits = predictor(z_r_vec, z_g)
            gnn_scores_matrix[r_id] = logits.squeeze(-1)
            
    # Normalize GNN scores to [0,1] per researcher to match hybrid score scale
    for r_id in range(num_r):
        s = gnn_scores_matrix[r_id]
        if s.max() > s.min():
            gnn_scores_matrix[r_id] = (s - s.min()) / (s.max() - s.min())
            
    # 6. Evaluate Combinations
    alphas = [1.0, 0.75, 0.50, 0.25, 0.0]
    hybrid_results = {}
    for a in alphas:
        metrics = evaluate_hybrid_ranking(test_pos, num_g, hybrid_scores, gnn_scores_matrix, alpha=a)
        hybrid_results[f"Retrieval_{int(a*100)}_GNN_{int((1-a)*100)}"] = metrics
        
    out_path = REPORTS_DIR / "hybrid_comparison.json"
    with open(out_path, "w") as f:
        json.dump(hybrid_results, f, indent=4)
        
    print(f"\nSaved hybrid comparison to {out_path}")

if __name__ == "__main__":
    run_phase4()
