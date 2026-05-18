import torch
import numpy as np

def hit_rate_at_k(z_dict, predictor, pos_edge, num_grants, k=10):
    """What % of real matches appear in the top-K predictions."""
    hits = 0
    total = pos_edge.shape[1]

    predictor.eval()
    with torch.no_grad():
        for i in range(total):
            r_id  = pos_edge[0][i].item()
            true_g = pos_edge[1][i].item()

            # Score against all grants
            z_r    = z_dict['researcher'][r_id].unsqueeze(0).repeat(num_grants, 1)
            z_g    = z_dict['grant']
            scores = predictor(z_r, z_g).squeeze()
            top_k  = scores.topk(k).indices.tolist()

            if true_g in top_k:
                hits += 1

    hr = hits / total
    print(f"Hit Rate @{k}: {hr:.4f}  ({hits}/{total})")
    return hr


def ndcg_at_k(z_dict, predictor, pos_edge, num_grants, k=10):
    """Normalized Discounted Cumulative Gain — rewards top ranking."""
    scores_list = []

    predictor.eval()
    with torch.no_grad():
        for i in range(pos_edge.shape[1]):
            r_id   = pos_edge[0][i].item()
            true_g = pos_edge[1][i].item()

            z_r    = z_dict['researcher'][r_id].unsqueeze(0).repeat(num_grants, 1)
            scores = predictor(z_r, z_dict['grant']).squeeze()
            ranked = scores.argsort(descending=True).tolist()

            rank   = ranked.index(true_g) + 1 if true_g in ranked else k + 1
            dcg    = 1.0 / np.log2(rank + 1) if rank <= k else 0.0
            idcg   = 1.0  # ideal: correct answer at rank 1
            scores_list.append(dcg / idcg)

    ndcg = np.mean(scores_list)
    print(f"NDCG @{k}: {ndcg:.4f}")
    return ndcg