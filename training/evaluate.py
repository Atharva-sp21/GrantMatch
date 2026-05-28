from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def _score_pairs(z_dict, predictor, pairs: Sequence[Tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.zeros(0, dtype=torch.float)

    researcher_ids = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    grant_ids = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    researcher_emb = z_dict["researcher"][researcher_ids]
    grant_emb = z_dict["grant"][grant_ids]
    return predictor(researcher_emb, grant_emb)


def _sample_negative_pairs(positive_pairs, num_researchers, num_grants, count, seed):
    rng = np.random.default_rng(seed)
    positive_set = set(positive_pairs)
    negatives = []
    attempts = 0
    max_attempts = max(count * 50, 100)
    while len(negatives) < count and attempts < max_attempts:
        attempts += 1
        pair = (int(rng.integers(0, num_researchers)), int(rng.integers(0, num_grants)))
        if pair in positive_set or pair in negatives:
            continue
        negatives.append(pair)
    return negatives


def evaluate_link_prediction(z_dict, predictor, positive_pairs, num_researchers, num_grants, negative_pairs=None, k_values=(1, 5, 10)):
    predictor.eval()
    positive_pairs = list(positive_pairs)
    if negative_pairs is None:
        negative_pairs = _sample_negative_pairs(positive_pairs, num_researchers, num_grants, len(positive_pairs), seed=17)
    negative_pairs = list(negative_pairs)

    with torch.no_grad():
        positive_logits = _score_pairs(z_dict, predictor, positive_pairs)
        negative_logits = _score_pairs(z_dict, predictor, negative_pairs)
        logits = torch.cat([positive_logits, negative_logits], dim=0)
        labels = torch.cat(
            [torch.ones(len(positive_logits), dtype=torch.float), torch.zeros(len(negative_logits), dtype=torch.float)],
            dim=0,
        )
        probabilities = torch.sigmoid(logits).cpu().numpy()
        label_array = labels.cpu().numpy()
        predictions = (probabilities >= 0.5).astype(np.float32)

        if len(np.unique(label_array)) > 1:
            auc = float(roc_auc_score(label_array, probabilities))
            ap = float(average_precision_score(label_array, probabilities))
        else:
            auc = 0.5
            ap = float(label_array.mean()) if len(label_array) else 0.0

        accuracy = float((predictions == label_array).mean()) if len(label_array) else 0.0
        positive_prob_mean = float(probabilities[: len(positive_logits)].mean()) if len(positive_logits) else 0.0
        negative_prob_mean = float(probabilities[len(positive_logits) :].mean()) if len(negative_logits) else 0.0

        hits_at_k = {k: 0 for k in k_values}
        reciprocal_ranks = []
        grouped_positive_pairs = defaultdict(list)
        for researcher_id, grant_id in positive_pairs:
            grouped_positive_pairs[researcher_id].append(grant_id)

        for researcher_id, grant_ids in grouped_positive_pairs.items():
            researcher_vector = z_dict["researcher"][researcher_id].unsqueeze(0).repeat(num_grants, 1)
            scores = predictor(researcher_vector, z_dict["grant"]).detach().cpu()
            ranking = torch.argsort(scores, descending=True).tolist()
            rank_lookup = {grant_id: rank + 1 for rank, grant_id in enumerate(ranking)}

            for grant_id in grant_ids:
                rank = rank_lookup.get(grant_id, num_grants + 1)
                reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)
                for k in k_values:
                    if rank <= k:
                        hits_at_k[k] += 1

        total_hits = max(len(positive_pairs), 1)
        metrics = {
            "auc": auc,
            "average_precision": ap,
            "accuracy": accuracy,
            "positive_prob_mean": positive_prob_mean,
            "negative_prob_mean": negative_prob_mean,
            "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
            **{f"hits@{k}": hits_at_k[k] / total_hits for k in k_values},
            "positive_count": len(positive_pairs),
            "negative_count": len(negative_pairs),
        }

    return metrics


def hit_rate_at_k(z_dict, predictor, pos_edge, num_grants, k=10):
    positive_pairs = list(zip(pos_edge[0].tolist(), pos_edge[1].tolist()))
    metrics = evaluate_link_prediction(z_dict, predictor, positive_pairs, z_dict["researcher"].shape[0], num_grants, k_values=(k,))
    hit_rate = metrics[f"hits@{k}"]
    print(f"Hit Rate @{k}: {hit_rate:.4f}")
    return hit_rate


def ndcg_at_k(z_dict, predictor, pos_edge, num_grants, k=10):
    positive_pairs = list(zip(pos_edge[0].tolist(), pos_edge[1].tolist()))
    predictor.eval()
    scores_list = []
    with torch.no_grad():
        for researcher_id, grant_id in positive_pairs:
            researcher_vector = z_dict["researcher"][researcher_id].unsqueeze(0).repeat(num_grants, 1)
            scores = predictor(researcher_vector, z_dict["grant"]).detach().cpu().numpy()
            ranking = np.argsort(-scores)
            rank = int(np.where(ranking == grant_id)[0][0]) + 1 if grant_id in ranking else k + 1
            if rank <= k:
                scores_list.append(1.0 / np.log2(rank + 1))
            else:
                scores_list.append(0.0)
    ndcg = float(np.mean(scores_list)) if scores_list else 0.0
    print(f"NDCG @{k}: {ndcg:.4f}")
    return ndcg