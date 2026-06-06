"""Ranking-oriented losses for link prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss(
    loss_type: str,
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
) -> torch.Tensor:
    """
    loss_type: 'bce' | 'margin' | 'bpr'
    """
    loss_type = loss_type.lower()
    if loss_type == "bce":
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat(
            [
                torch.ones(pos_logits.shape[0], device=logits.device),
                torch.zeros(neg_logits.shape[0], device=logits.device),
            ]
        )
        return F.binary_cross_entropy_with_logits(logits, labels)

    if loss_type == "margin":
        # pos should score higher than neg
        return F.margin_ranking_loss(
            pos_logits,
            neg_logits,
            torch.ones(pos_logits.shape[0], device=pos_logits.device),
            margin=0.3,
        )

    if loss_type == "bpr":
        # Bayesian Personalized Ranking: -log sigmoid(pos - neg)
        return -F.logsigmoid(pos_logits - neg_logits).mean()

    raise ValueError(f"Unknown loss_type: {loss_type}")
