import torch
import torch.nn as nn
import wandb
from models.gnn import GrantMatchGNN, LinkPredictor

def train(data, epochs=200, hidden=256, heads=4, lr=5e-4, use_wandb=True):

    if use_wandb:
        wandb.init(project='grantmatch', config={
            'hidden': hidden, 'heads': heads,
            'lr': lr, 'epochs': epochs
        })

    model     = GrantMatchGNN(hidden=hidden, heads=heads,
                              metadata=data.metadata())
    predictor = LinkPredictor(hidden=hidden)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )

    # BCELoss works better than MarginRankingLoss for this problem
    criterion = nn.BCELoss()

    if ('researcher', 'RECEIVED_PAST', 'grant') not in data.edge_types:
        print("[WARN] No RECEIVED_PAST edges - skipping training")
        return None, None, {}

    pos_edge = data['researcher', 'RECEIVED_PAST', 'grant'].edge_index

    if pos_edge.shape[1] == 0:
        print("[WARN] Empty RECEIVED_PAST edges - skipping training")
        return None, None, {}

    num_r    = data['researcher'].x.shape[0]
    num_g    = data['grant'].x.shape[0]

    import os
    os.makedirs('checkpoints', exist_ok=True)

    pos_edge_set = set(zip(pos_edge[0].tolist(), pos_edge[1].tolist()))

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        predictor.train()
        optimizer.zero_grad()

        z_dict = model(data.x_dict, data.edge_index_dict)

        if epoch == 1:
            print(f"[DEBUG] z_dict keys: {z_dict.keys()}")
            if 'researcher' not in z_dict:
                print("[ERROR] 'researcher' missing from model output!")

        pos_src = pos_edge[0]
        pos_dst = pos_edge[1]

        z_r_pos = z_dict['researcher'][pos_src]
        z_g_pos = z_dict['grant'][pos_dst]
        pos_pred = predictor(z_r_pos, z_g_pos)

        # Safe negative sampling
        neg_src, neg_dst = [], []
        max_attempts = len(pos_src) * 5
        attempts = 0
        while len(neg_src) < len(pos_src) and attempts < max_attempts:
            candidate_src = torch.randint(0, num_r, (1,)).item()
            candidate_dst = torch.randint(0, num_g, (1,)).item()
            if (candidate_src, candidate_dst) not in pos_edge_set:
                neg_src.append(candidate_src)
                neg_dst.append(candidate_dst)
            attempts += 1

        if len(neg_src) == 0:
            print("[WARN] Could not generate negatives, skipping epoch")
            continue

        neg_src = torch.tensor(neg_src, dtype=torch.long)
        neg_dst = torch.tensor(neg_dst, dtype=torch.long)

        z_r_neg = z_dict['researcher'][neg_src]
        z_g_neg = z_dict['grant'][neg_dst]
        neg_pred = predictor(z_r_neg, z_g_neg)

        labels = torch.cat([torch.ones(len(pos_pred)),
                            torch.zeros(len(neg_pred))])
        preds  = torch.cat([pos_pred, neg_pred])
        loss   = criterion(preds, labels)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(predictor.parameters()),
            max_norm=1.0
        )

        # Check for NaN gradients
        has_nan_grad = False
        for param in list(model.parameters()) + list(predictor.parameters()):
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan_grad = True
                    break

        if has_nan_grad:
            print(f"[WARN] NaN gradients at epoch {epoch}, skipping")
            optimizer.zero_grad()
        else:
            optimizer.step()

        # Track loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            print(f"Epoch {epoch:>3} | Loss: {loss.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            if use_wandb:
                wandb.log({'loss': loss.item(), 'epoch': epoch})

        scheduler.step(loss.item())

        if patience_counter > 25:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    # Save checkpoint
    torch.save(model.state_dict(),     'checkpoints/gnn.pt')
    torch.save(predictor.state_dict(), 'checkpoints/predictor.pt')
    print("Models saved to checkpoints/")

    if use_wandb:
        wandb.finish()

    return model, predictor, z_dict