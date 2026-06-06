"""
Train link predictor with configurable backbone (HGT / GraphSAGE) and loss (BCE / margin / BPR).
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.gnn import GrantMatchGNN, LinkPredictor
from models.graphsage import HeteroGraphSAGE
from training.diagnostics import plot_training_loss
from training.evaluate import evaluate_ranking, gnn_score_fn
from training.losses import compute_loss
from training.sampling import build_hetero_maps, sample_negatives
from training.split import edge_set

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def _build_model(model_type: str, hidden, heads, metadata, dropout):
    if model_type == "graphsage":
        return HeteroGraphSAGE(hidden=hidden, metadata=metadata, dropout=dropout)
    return GrantMatchGNN(hidden=hidden, heads=heads, metadata=metadata, dropout=dropout)


def train(
    data,
    train_pos: torch.Tensor,
    val_pos: torch.Tensor,
    researches: torch.Tensor,
    funds: torch.Tensor,
    provides: torch.Tensor,
    epochs: int = 120,
    hidden: int = 128,
    heads: int = 4,
    lr: float = 5e-4,
    dropout: float = 0.2,
    loss_type: str = "bpr",
    model_type: str = "hgt",
    grant_emb: Optional[torch.Tensor] = None,
    researchers_meta: Optional[list] = None,
    grants_meta: Optional[list] = None,
    use_similar_edges: bool = True,
    use_wandb: bool = False,
    checkpoint_tag: str = "",
) -> Tuple[Optional[nn.Module], Optional[LinkPredictor], dict, Dict, float]:
    t0 = time.time()

    if train_pos.shape[1] == 0:
        print("[ERROR] No training positive edges")
        return None, None, {}, {}, 0.0

    metadata = data.metadata()
    model = _build_model(model_type, hidden, heads, metadata, dropout)
    predictor = LinkPredictor(hidden=hidden, dropout=dropout)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    num_r = data["researcher"].x.shape[0]
    num_g = data["grant"].x.shape[0]
    train_pos_set = edge_set(train_pos)
    maps = build_hetero_maps(researches, funds, provides, num_g, grants_meta)
    r_topics, g_topics, g_agency, agency_grants, g_career = maps

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    tag = f"_{checkpoint_tag}" if checkpoint_tag else ""
    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        model.train()
        predictor.train()
        optimizer.zero_grad()

        z_dict = model(data.x_dict, data.edge_index_dict)
        pos_src, pos_dst = train_pos[0], train_pos[1]
        pos_logits = predictor(z_dict["researcher"][pos_src], z_dict["grant"][pos_dst])

        neg_src, neg_dst = sample_negatives(
            pos_src,
            pos_dst,
            train_pos_set,
            num_r,
            num_g,
            r_topics,
            g_topics,
            g_agency,
            agency_grants,
            g_career,
            researchers_meta=researchers_meta,
            grant_emb=grant_emb,
            seed=epoch,
        )
        neg_logits = predictor(z_dict["researcher"][neg_src], z_dict["grant"][neg_dst])
        loss = compute_loss(loss_type, pos_logits, neg_logits)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(predictor.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        history["train"].append(loss.item())
        val_loss = _val_loss(
            model,
            predictor,
            data,
            val_pos,
            loss_type,
            num_r,
            num_g,
            train_pos_set,
            r_topics,
            g_topics,
            g_agency,
            agency_grants,
            g_career,
            grant_emb,
            researchers_meta,
        )
        history["val"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "predictor": copy.deepcopy(predictor.state_dict()),
                "epoch": epoch,
            }
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  [{model_type}/{loss_type}] Ep {epoch:>3} "
                f"train {loss.item():.4f} val {val_loss:.4f}"
            )

        if patience_counter >= 20:
            break

    if best_state:
        model.load_state_dict(best_state["model"])
        predictor.load_state_dict(best_state["predictor"])

    torch.save(model.state_dict(), CHECKPOINT_DIR / f"gnn{tag}.pt")
    torch.save(predictor.state_dict(), CHECKPOINT_DIR / f"predictor{tag}.pt")
    plot_training_loss(history, path=CHECKPOINT_DIR.parent / "reports" / f"loss_{checkpoint_tag or 'default'}.png")

    model.eval()
    predictor.eval()
    with torch.no_grad():
        z_dict = model(data.x_dict, data.edge_index_dict)

    elapsed = time.time() - t0
    return model, predictor, z_dict, history, elapsed


def _val_loss(
    model, predictor, data, val_pos, loss_type, num_r, num_g, pos_set,
    r_topics, g_topics, g_agency, agency_grants, g_career, grant_emb, researchers_meta,
) -> float:
    if val_pos.shape[1] == 0:
        return 0.0
    model.eval()
    predictor.eval()
    with torch.no_grad():
        z_dict = model(data.x_dict, data.edge_index_dict)
        pos_logits = predictor(
            z_dict["researcher"][val_pos[0]], z_dict["grant"][val_pos[1]]
        )
        neg_src, neg_dst = sample_negatives(
            val_pos[0], val_pos[1], pos_set, num_r, num_g,
            r_topics, g_topics, g_agency, agency_grants, g_career,
            researchers_meta=researchers_meta, grant_emb=grant_emb, seed=0,
        )
        neg_logits = predictor(
            z_dict["researcher"][neg_src], z_dict["grant"][neg_dst]
        )
        return compute_loss(loss_type, pos_logits, neg_logits).item()
