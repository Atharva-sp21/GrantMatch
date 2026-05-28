import copy
import os
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.gnn import GrantMatchGNN, LinkPredictor
from training.evaluate import evaluate_link_prediction


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _split_pairs(pairs: List[Tuple[int, int]], seed: int):
    if not pairs:
        return [], [], []

    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    if total == 1:
        return shuffled, [], []
    if total == 2:
        return shuffled[:1], shuffled[1:], []

    train_end = max(int(total * 0.8), total - 2)
    val_end = max(train_end + 1, int(total * 0.9))
    if val_end >= total:
        val_end = total - 1

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    if not val:
        val = train[-1:]
        train = train[:-1]
    if not test:
        test = val[-1:]
        val = val[:-1] or test[:0]

    return train, val, test


def _build_topic_structures(data):
    researcher_topics = defaultdict(set)
    grant_topics = defaultdict(set)

    if ("researcher", "RESEARCHES", "topic") in data.edge_types:
        for researcher_id, topic_id in data["researcher", "RESEARCHES", "topic"].edge_index.t().tolist():
            researcher_topics[researcher_id].add(topic_id)

    if ("grant", "FUNDS_TOPIC", "topic") in data.edge_types:
        for grant_id, topic_id in data["grant", "FUNDS_TOPIC", "topic"].edge_index.t().tolist():
            grant_topics[topic_id].add(grant_id)

    return researcher_topics, grant_topics


def _build_hard_negative_candidates(researcher_topics, grant_topics, positive_pairs):
    positive_lookup = defaultdict(set)
    for researcher_id, grant_id in positive_pairs:
        positive_lookup[researcher_id].add(grant_id)

    candidates = {}
    for researcher_id, topic_ids in researcher_topics.items():
        grant_ids = set()
        for topic_id in topic_ids:
            grant_ids.update(grant_topics.get(topic_id, set()))
        grant_ids.difference_update(positive_lookup.get(researcher_id, set()))
        if grant_ids:
            candidates[researcher_id] = sorted(grant_ids)
    return candidates


def _sample_negative_pairs(positive_pairs, num_researchers, num_grants, count, seed, hard_candidates=None):
    rng = random.Random(seed)
    positive_set = set(positive_pairs)
    negatives = []
    attempts = 0
    max_attempts = max(count * 50, 100)

    candidate_researchers = list(hard_candidates.keys()) if hard_candidates else []

    while len(negatives) < count and attempts < max_attempts:
        attempts += 1
        if hard_candidates and candidate_researchers and rng.random() < 0.6:
            researcher_id = rng.choice(candidate_researchers)
            grant_id = rng.choice(hard_candidates[researcher_id])
        else:
            researcher_id = rng.randrange(num_researchers)
            grant_id = rng.randrange(num_grants)

        pair = (researcher_id, grant_id)
        if pair in positive_set or pair in negatives:
            continue
        negatives.append(pair)

    while len(negatives) < count:
        pair = (rng.randrange(num_researchers), rng.randrange(num_grants))
        if pair in positive_set or pair in negatives:
            continue
        negatives.append(pair)

    return negatives


def _replace_received_edges(data, edge_pairs):
    train_data = copy.deepcopy(data)
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    train_data["researcher", "RECEIVED_PAST", "grant"].edge_index = edge_index
    train_data["grant", "REV_RECEIVED_PAST", "researcher"].edge_index = edge_index.flip(0) if edge_index.numel() else torch.zeros(2, 0, dtype=torch.long)
    return train_data


def _state_dicts_to_cpu(state_dicts):
    return {key: value.cpu() for key, value in state_dicts.items()}


def train(data, epochs=120, hidden=256, heads=2, lr=1e-3, use_wandb=False, seed=42):
    del use_wandb
    set_seed(seed)

    if ("researcher", "RECEIVED_PAST", "grant") not in data.edge_types:
        print("[WARN] No RECEIVED_PAST edges - skipping training")
        return None, None, {}, {}

    pos_edge = data["researcher", "RECEIVED_PAST", "grant"].edge_index
    if pos_edge.shape[1] == 0:
        print("[WARN] Empty RECEIVED_PAST edges - skipping training")
        return None, None, {}, {}

    all_positive_pairs = list(zip(pos_edge[0].tolist(), pos_edge[1].tolist()))
    train_pos, val_pos, test_pos = _split_pairs(all_positive_pairs, seed)
    if not train_pos:
        print("[WARN] Not enough RECEIVED_PAST edges for training")
        return None, None, {}, {}

    train_data = _replace_received_edges(data, train_pos)
    model = GrantMatchGNN(hidden=hidden, heads=heads, metadata=train_data.metadata())
    predictor = LinkPredictor(hidden=hidden)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)
    criterion = nn.BCEWithLogitsLoss()

    num_researchers = train_data["researcher"].x.shape[0]
    num_grants = train_data["grant"].x.shape[0]
    positive_set = set(all_positive_pairs)
    researcher_topics, grant_topics = _build_topic_structures(train_data)
    hard_candidates = _build_hard_negative_candidates(researcher_topics, grant_topics, positive_set)

    os.makedirs("checkpoints", exist_ok=True)

    best_val_auc = float("-inf")
    best_state = None
    best_predictor_state = None
    best_val_metrics = {}
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        predictor.train()
        optimizer.zero_grad()

        z_dict = model(train_data.x_dict, train_data.edge_index_dict)

        train_neg = _sample_negative_pairs(
            train_pos,
            num_researchers,
            num_grants,
            len(train_pos),
            seed + epoch,
            hard_candidates=hard_candidates,
        )

        pos_src = torch.tensor([src for src, _ in train_pos], dtype=torch.long)
        pos_dst = torch.tensor([dst for _, dst in train_pos], dtype=torch.long)
        neg_src = torch.tensor([src for src, _ in train_neg], dtype=torch.long)
        neg_dst = torch.tensor([dst for _, dst in train_neg], dtype=torch.long)

        pos_logits = predictor(z_dict["researcher"][pos_src], z_dict["grant"][pos_dst])
        neg_logits = predictor(z_dict["researcher"][neg_src], z_dict["grant"][neg_dst])

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat(
            [torch.ones(len(pos_logits), dtype=torch.float), torch.zeros(len(neg_logits), dtype=torch.float)],
            dim=0,
        )

        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(predictor.parameters()), max_norm=1.0)
        optimizer.step()

        train_prob_mean = torch.sigmoid(logits).mean().item()
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train loss {loss.item():.4f} | mean prob {train_prob_mean:.4f}")

        model.eval()
        predictor.eval()
        with torch.no_grad():
            val_neg = _sample_negative_pairs(
                val_pos,
                num_researchers,
                num_grants,
                len(val_pos),
                seed + 10_000 + epoch,
                hard_candidates=hard_candidates,
            ) if val_pos else []

            if val_pos:
                val_metrics = evaluate_link_prediction(
                    z_dict,
                    predictor,
                    val_pos,
                    num_researchers,
                    num_grants,
                    negative_pairs=val_neg,
                    k_values=(1, 5, 10),
                )
                val_auc = val_metrics["auc"]
                scheduler.step(val_auc)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_val_metrics = val_metrics
                    best_state = _state_dicts_to_cpu(model.state_dict())
                    best_predictor_state = _state_dicts_to_cpu(predictor.state_dict())
                    torch.save(best_state, "checkpoints/gnn.pt")
                    torch.save(best_predictor_state, "checkpoints/predictor.pt")
                    patience = 0
                else:
                    patience += 1

                if epoch % 5 == 0:
                    print(
                        f"Epoch {epoch:03d} | val auc {val_auc:.4f} | val ap {val_metrics['average_precision']:.4f} | "
                        f"acc {val_metrics['accuracy']:.4f} | hits@10 {val_metrics['hits@10']:.4f}"
                    )
            else:
                scheduler.step(-loss.item())

        if patience >= 15:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    if best_state is not None and best_predictor_state is not None:
        model.load_state_dict(best_state)
        predictor.load_state_dict(best_predictor_state)

    with torch.no_grad():
        best_z_dict = model(train_data.x_dict, train_data.edge_index_dict)

    test_metrics = {}
    if test_pos:
        test_neg = _sample_negative_pairs(
            test_pos,
            num_researchers,
            num_grants,
            len(test_pos),
            seed + 20_000,
            hard_candidates=hard_candidates,
        )
        test_metrics = evaluate_link_prediction(
            best_z_dict,
            predictor,
            test_pos,
            num_researchers,
            num_grants,
            negative_pairs=test_neg,
            k_values=(1, 5, 10),
        )
        print(
            f"[OK] Test auc {test_metrics['auc']:.4f} | test ap {test_metrics['average_precision']:.4f} | "
            f"acc {test_metrics['accuracy']:.4f} | mrr {test_metrics['mrr']:.4f} | hits@10 {test_metrics['hits@10']:.4f}"
        )

    return model, predictor, best_z_dict, {"val": best_val_metrics, "test": test_metrics}