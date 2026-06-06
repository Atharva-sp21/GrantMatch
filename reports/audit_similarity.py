import os
import sys
import json
import torch
from pathlib import Path
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_all_data
from graph.embeddings import build_node_embeddings
from graph.schema import build_similarity_edges

def main():
    data_dict = load_all_data()
    researchers = data_dict["researchers"]
    grants = data_dict["grants"]
    
    r_emb, g_emb, _ = build_node_embeddings(researchers, grants)
    
    similar, similar_attr = build_similarity_edges(
        r_emb,
        data_dict["researches"],
        data_dict["affiliated"],
        researchers=researchers,
        top_k=8
    )
    
    edges = list(zip(similar[0].tolist(), similar[1].tolist()))
    if len(edges) > 500:
        sampled_edges = random.sample(edges, 500)
    else:
        sampled_edges = edges
        
    topic_overlaps = []
    inst_overlaps = []
    emb_sims = []
    
    # Pre-compute topic and inst sets
    r_topics = {i: set() for i in range(len(researchers))}
    for r, t in zip(data_dict["researches"][0].tolist(), data_dict["researches"][1].tolist()):
        r_topics[r].add(t)
        
    r_inst = {i: set() for i in range(len(researchers))}
    for r, i in zip(data_dict["affiliated"][0].tolist(), data_dict["affiliated"][1].tolist()):
        r_inst[r].add(i)
        
    emb_normalized = torch.nn.functional.normalize(r_emb.float(), p=2, dim=1)
    
    for u, v in sampled_edges:
        u_topics = r_topics.get(u, set())
        v_topics = r_topics.get(v, set())
        if len(u_topics) > 0 or len(v_topics) > 0:
            overlap = len(u_topics & v_topics) / max(len(u_topics | v_topics), 1)
        else:
            overlap = 0.0
        topic_overlaps.append(overlap)
        
        u_inst = r_inst.get(u, set())
        v_inst = r_inst.get(v, set())
        if len(u_inst) > 0 or len(v_inst) > 0:
            inst_overlap = len(u_inst & v_inst) / max(len(u_inst | v_inst), 1)
        else:
            inst_overlap = 0.0
        inst_overlaps.append(inst_overlap)
        
        sim = (emb_normalized[u] * emb_normalized[v]).sum().item()
        emb_sims.append(sim)
        
    avg_topic_overlap = sum(topic_overlaps) / len(topic_overlaps) if topic_overlaps else 0
    avg_inst_overlap = sum(inst_overlaps) / len(inst_overlaps) if inst_overlaps else 0
    avg_emb_sim = sum(emb_sims) / len(emb_sims) if emb_sims else 0
    
    audit = {
        "sampled_edges": len(sampled_edges),
        "total_similarity_edges": len(edges),
        "average_topic_overlap_jaccard": avg_topic_overlap,
        "average_institution_overlap_jaccard": avg_inst_overlap,
        "average_embedding_cosine_similarity": avg_emb_sim,
        "zero_topic_overlap_count": sum(1 for x in topic_overlaps if x == 0.0),
        "zero_inst_overlap_count": sum(1 for x in inst_overlaps if x == 0.0)
    }
    
    out_path = Path(__file__).resolve().parent / "similarity_edge_audit.json"
    with open(out_path, "w") as f:
        json.dump(audit, f, indent=4)
        
    print(f"Saved audit to {out_path}")

if __name__ == "__main__":
    main()
