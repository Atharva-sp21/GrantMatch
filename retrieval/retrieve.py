import torch
from models.rag import rag_retrieve, rag_score_single, THRESHOLD
from retrieval.fallback import keyword_retrieve

GNN_WEIGHT = 0.6
RAG_WEIGHT = 0.4

def combined_score(researcher, researcher_id, grant, grant_id,
                   z_dict, predictor):
    # ── GNN score (structural match probability) ─────────────────────
    z_r     = z_dict['researcher'][researcher_id].unsqueeze(0)
    z_g     = z_dict['grant'][grant_id].unsqueeze(0)
    gnn_sc  = predictor(z_r, z_g).item()

    # ── RAG score (semantic text similarity) ─────────────────────────
    grant_text = grant.get('title', '') + ' ' + grant.get('guidelines', '')
    rag_sc     = rag_score_single(researcher, grant_text)

    return round(GNN_WEIGHT * gnn_sc + RAG_WEIGHT * rag_sc, 4)


def retrieve_top_grants(researcher, researcher_id, grants_list,
                        z_dict, predictor, top_k=10):
    # Try RAG first
    rag_results = rag_retrieve(researcher, top_k=top_k)

    # If top RAG score is below threshold, fall back to keyword
    if not rag_results or rag_results[0][1] < THRESHOLD:
        print("[Fallback] Low confidence — using keyword retrieval")
        from models.rag import get_researcher_text
        return keyword_retrieve(get_researcher_text(researcher), top_k)

    # Re-rank using combined GNN + RAG score
    scored = []
    grant_id_to_obj = {g['id']: g for g in grants_list}
    for grant_id, _ in rag_results:
        if grant_id not in grant_id_to_obj:
            continue
        grant  = grant_id_to_obj[grant_id]
        score  = combined_score(
            researcher, researcher_id,
            grant, grant_id,
            z_dict, predictor
        )
        scored.append((grant_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]