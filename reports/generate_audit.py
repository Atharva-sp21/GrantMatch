import os
import sys
import json
import numpy as np
from pathlib import Path
import scipy.sparse as sp

# Add parent directory to path to import data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_all_data

def generate_audit():
    data = load_all_data()
    
    researchers = data["researchers"]
    grants = data["grants"]
    institutions = data["institutions"]
    topics = data["topics"]
    agencies = data["agencies"]
    
    # Edges are shape (2, num_edges)
    received = data["received"].numpy()
    researches = data["researches"].numpy()
    affiliated = data["affiliated"].numpy()
    funds = data["funds"].numpy()
    provides = data["provides"].numpy()
    
    num_researchers = len(researchers)
    num_grants = len(grants)
    num_topics = len(topics)
    num_institutions = len(institutions)
    num_agencies = len(agencies)
    
    report = {}
    
    # -------------------------
    # 1. Label Audit (received)
    # -------------------------
    received_edges = list(zip(received[0], received[1]))
    unique_received = set(received_edges)
    duplicates = len(received_edges) - len(unique_received)
    
    suspicious = 0
    for r, g in received_edges:
        if r < 0 or r >= num_researchers or g < 0 or g >= num_grants:
            suspicious += 1
            
    # Degree counts for received
    r_deg = np.zeros(num_researchers)
    g_deg = np.zeros(num_grants)
    for r, g in unique_received:
        if 0 <= r < num_researchers: r_deg[r] += 1
        if 0 <= g < num_grants: g_deg[g] += 1
        
    res_one_grant = int(np.sum(r_deg == 1))
    grant_one_res = int(np.sum(g_deg == 1))
    
    # Disconnected in full graph
    all_r_deg = np.zeros(num_researchers)
    for edges in [received, researches, affiliated]:
        for r in edges[0]:
            if 0 <= r < num_researchers: all_r_deg[r] += 1
            
    all_g_deg = np.zeros(num_grants)
    for edges in [received]:
        for g in edges[1]:
            if 0 <= g < num_grants: all_g_deg[g] += 1
    for edges in [funds]:
        for g in edges[0]:
            if 0 <= g < num_grants: all_g_deg[g] += 1
    for edges in [provides]:
        for g in edges[1]:
            if 0 <= g < num_grants: all_g_deg[g] += 1
            
    disconnected_res = int(np.sum(all_r_deg == 0))
    disconnected_grants = int(np.sum(all_g_deg == 0))
    
    report["label_audit"] = {
        "total_received_past_edges": len(received_edges),
        "duplicate_edges": duplicates,
        "suspicious_edges": suspicious,
        "researchers_with_only_one_grant": res_one_grant,
        "grants_with_only_one_recipient": grant_one_res,
        "disconnected_researchers": disconnected_res,
        "disconnected_grants": disconnected_grants
    }
    
    # -------------------------
    # 2. Researcher Audit
    # -------------------------
    res_with_abstracts = 0
    for res in researchers:
        abs_list = res.get("recent_abstracts", [])
        if abs_list and len(abs_list) > 0 and any(len(a.strip()) > 0 for a in abs_list if isinstance(a, str)):
            res_with_abstracts += 1
            
    res_without_abstracts = num_researchers - res_with_abstracts
    
    res_topic_deg = np.zeros(num_researchers)
    for r in researches[0]:
        if 0 <= r < num_researchers: res_topic_deg[r] += 1
        
    res_with_topics = int(np.sum(res_topic_deg > 0))
    res_without_topics = int(np.sum(res_topic_deg == 0))
    
    res_with_history = int(np.sum(r_deg > 0))
    res_without_history = int(np.sum(r_deg == 0))
    
    report["researcher_audit"] = {
        "researchers_with_abstracts": res_with_abstracts,
        "researchers_without_abstracts": res_without_abstracts,
        "researchers_with_topics": res_with_topics,
        "researchers_without_topics": res_without_topics,
        "researchers_with_grant_history": res_with_history,
        "researchers_without_grant_history": res_without_history
    }
    
    # -------------------------
    # 3. Topic Audit
    # -------------------------
    if num_researchers > 0:
        avg_topics = float(np.mean(res_topic_deg))
        med_topics = float(np.median(res_topic_deg))
    else:
        avg_topics = 0.0
        med_topics = 0.0
        
    res_lt_3_topics = int(np.sum(res_topic_deg < 3))
    res_gt_10_topics = int(np.sum(res_topic_deg > 10))
    
    topic_freq = np.zeros(num_topics)
    for t in researches[1]:
        if 0 <= t < num_topics: topic_freq[t] += 1
        
    topic_freq_dist = {int(i): int(freq) for i, freq in enumerate(topic_freq)}
    
    report["topic_audit"] = {
        "average_topics_per_researcher": avg_topics,
        "median_topics_per_researcher": med_topics,
        "researchers_with_fewer_than_3_topics": res_lt_3_topics,
        "researchers_with_more_than_10_topics": res_gt_10_topics,
        "topic_frequency_distribution": topic_freq_dist
    }
    
    # -------------------------
    # 4. Graph Audit
    # -------------------------
    # We will build a unified graph to compute connected components
    total_nodes = num_researchers + num_grants + num_topics + num_institutions + num_agencies
    
    # Offsets:
    # R: 0
    # G: num_researchers
    # T: num_researchers + num_grants
    # I: num_researchers + num_grants + num_topics
    # A: num_researchers + num_grants + num_topics + num_institutions
    
    off_g = num_researchers
    off_t = num_researchers + num_grants
    off_i = num_researchers + num_grants + num_topics
    off_a = num_researchers + num_grants + num_topics + num_institutions
    
    edges_src = []
    edges_dst = []
    
    def add_edges(arr, off1, off2):
        if arr.shape[1] > 0:
            for i in range(arr.shape[1]):
                u = int(arr[0, i]) + off1
                v = int(arr[1, i]) + off2
                edges_src.extend([u, v])
                edges_dst.extend([v, u])
                
    add_edges(received, 0, off_g)
    add_edges(researches, 0, off_t)
    add_edges(affiliated, 0, off_i)
    add_edges(funds, off_g, off_t)
    add_edges(provides, off_a, off_g)
    
    if len(edges_src) > 0:
        adj = sp.coo_matrix((np.ones(len(edges_src)), (edges_src, edges_dst)), shape=(total_nodes, total_nodes))
        n_components, labels = sp.csgraph.connected_components(csgraph=adj, directed=False, return_labels=True)
    else:
        n_components = total_nodes
        
    report["graph_audit"] = {
        "node_counts": {
            "researchers": num_researchers,
            "grants": num_grants,
            "topics": num_topics,
            "institutions": num_institutions,
            "agencies": num_agencies,
            "total": total_nodes
        },
        "edge_counts": {
            "received_past": len(received_edges),
            "researches": researches.shape[1],
            "affiliated": affiliated.shape[1],
            "funds_topic": funds.shape[1],
            "provides": provides.shape[1]
        },
        "degree_distributions": {
            "researcher_grant_history_mean": float(np.mean(r_deg)),
            "researcher_grant_history_median": float(np.median(r_deg)),
            "grant_recipient_mean": float(np.mean(g_deg)),
            "grant_recipient_median": float(np.median(g_deg))
        },
        "isolated_nodes": disconnected_res + disconnected_grants, # approximation
        "connected_components": int(n_components)
    }
    
    out_path = Path(__file__).resolve().parent / "data_quality_audit.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Report successfully saved to {out_path}")

if __name__ == "__main__":
    generate_audit()
