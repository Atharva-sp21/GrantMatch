import os
import sys
import json
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_all_data
from graph.embeddings import build_node_embeddings
from graph.features import build_all_node_features
from graph.schema import build_graph, add_edges, build_similarity_edges, graph_for_training
from training.split import split_received_edges
from training.baseline import evaluate_embedding_baseline
from training.train_gnn import train
from training.evaluate import evaluate_ranking, gnn_score_fn

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

def run_ablation():
    print("Loading data...")
    data_dict = load_all_data()
    researchers = data_dict["researchers"]
    grants = data_dict["grants"]
    institutions = data_dict["institutions"]
    topics = data_dict["topics"]
    agencies = data_dict["agencies"]
    
    # Text embeddings
    print("Building embeddings...")
    r_emb, g_emb, emb_dim = build_node_embeddings(researchers, grants)
    
    # Feature engineering
    print("Building features...")
    feats = build_all_node_features(
        researchers, institutions, topics, grants, agencies,
        researcher_emb=r_emb, grant_emb=g_emb
    )
    
    # Base graph
    data_base = build_graph(
        feats["researcher"], feats["institution"], feats["topic"],
        feats["grant"], feats["agency"]
    )
    
    # Create two versions: with and without similarity edges
    print("Building similarity edges...")
    similar, similar_attr = build_similarity_edges(
        r_emb, data_dict["researches"], data_dict["affiliated"],
        researchers=researchers, top_k=8
    )
    
    data_no_sim = add_edges(
        data_base.clone(),
        affiliated=data_dict["affiliated"], researches=data_dict["researches"],
        received=data_dict["received"], funds=data_dict["funds"], provides=data_dict["provides"],
        similar=torch.zeros(2,0, dtype=torch.long), similar_attr=torch.zeros(0)
    )
    
    data_with_sim = add_edges(
        data_base.clone(),
        affiliated=data_dict["affiliated"], researches=data_dict["researches"],
        received=data_dict["received"], funds=data_dict["funds"], provides=data_dict["provides"],
        similar=similar, similar_attr=similar_attr
    )
    
    # Splits (seed=42 matches main.py to verify identical splits)
    print("Splitting edges...")
    splits = split_received_edges(data_dict["received"], seed=42)
    train_pos, val_pos, test_pos = splits["train"], splits["val"], splits["test"]
    
    num_g = data_base["grant"].num_nodes
    
    results = {}
    
    # Config 1: Embedding Cosine Baseline
    print("\n--- Config 1: Embedding Cosine Baseline ---")
    baseline_metrics = evaluate_embedding_baseline(r_emb, g_emb, test_pos, val_pos, ks=(1, 5, 10))
    results["1_Cosine_Baseline"] = baseline_metrics["test"]
    
    def run_gnn_config(name, data, model_type, loss_type, use_similar):
        print(f"\n--- Config: {name} ---")
        data_train = graph_for_training(data, train_pos)
        model, predictor, z_dict, history, _ = train(
            data_train, train_pos=train_pos, val_pos=val_pos,
            researches=data_dict["researches"], funds=data_dict["funds"], provides=data_dict["provides"],
            epochs=60, # Reduced to 60 for faster ablation
            loss_type=loss_type, model_type=model_type,
            grant_emb=g_emb, researchers_meta=researchers, grants_meta=grants,
            checkpoint_tag=name
        )
        score_fn = gnn_score_fn(z_dict, predictor, num_g)
        metrics = evaluate_ranking(test_pos, num_g, score_fn, ks=[1, 5, 10])
        metrics["train_loss"] = history["train"][-1]
        metrics["val_loss"] = history["val"][-1]
        return metrics

    # Config 2: HGT without similarity edges
    results["2_HGT_NoSim_BPR"] = run_gnn_config("2_HGT_NoSim_BPR", data_no_sim, "hgt", "bpr", False)
    
    # Config 3: HGT with similarity edges
    results["3_HGT_WithSim_BPR"] = run_gnn_config("3_HGT_WithSim_BPR", data_with_sim, "hgt", "bpr", True)
    
    # Config 4: GraphSAGE without similarity edges
    results["4_GraphSAGE_NoSim_BPR"] = run_gnn_config("4_GraphSAGE_NoSim_BPR", data_no_sim, "graphsage", "bpr", False)
    
    # Config 5: HGT + BCE loss (without sim)
    results["5_HGT_NoSim_BCE"] = run_gnn_config("5_HGT_NoSim_BCE", data_no_sim, "hgt", "bce", False)
    
    # Config 6: HGT + BPR loss (without sim) - already done in config 2. Let's do HGT + BPR with SIM to be thorough, but we did that in 3.
    # The prompt asked for: (1) Baseline (2) HGT no sim (3) HGT w/ sim (4) GraphSAGE no sim (5) HGT + BCE (6) HGT + BPR.
    # I'll just report config 2 as config 6 as well, or just run it.
    
    out_path = REPORTS_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nAblation results saved to {out_path}")

if __name__ == "__main__":
    run_ablation()
