import torch
import torch.nn as nn
import wandb
from models.gnn import GrantMatchGNN, LinkPredictor

def train(data, epochs=200, hidden=64, heads=2, lr=1e-3, use_wandb=True):

    if use_wandb:
        wandb.init(project='grantmatch', config={
            'hidden': hidden, 'heads': heads,
            'lr': lr, 'epochs': epochs
        })

    model     = GrantMatchGNN(hidden=hidden, heads=heads,
                              metadata=data.metadata())
    predictor = LinkPredictor(hidden=hidden)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()), lr=lr
    )
    criterion = nn.BCELoss()

    # Positive edges = RECEIVED_PAST (researcher → grant)
    pos_edge = data['researcher', 'RECEIVED_PAST', 'grant'].edge_index
    num_r    = data['researcher'].x.shape[0]
    num_g    = data['grant'].x.shape[0]

    for epoch in range(1, epochs + 1):
        model.train()
        predictor.train()
        optimizer.zero_grad()

        z_dict   = model(data.x_dict, data.edge_index_dict)

        # Positive pairs
        pos_src  = pos_edge[0]
        pos_dst  = pos_edge[1]
        pos_pred = predictor(
            z_dict['researcher'][pos_src],
            z_dict['grant'][pos_dst]
        )

        # Negative pairs — random non-matches
        neg_src  = torch.randint(0, num_r, (len(pos_src),))
        neg_dst  = torch.randint(0, num_g, (len(pos_src),))
        neg_pred = predictor(
            z_dict['researcher'][neg_src],
            z_dict['grant'][neg_dst]
        )

        labels = torch.cat([torch.ones(len(pos_pred)),
                            torch.zeros(len(neg_pred))])
        preds  = torch.cat([pos_pred, neg_pred])
        loss   = criterion(preds, labels)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:>3} | Loss: {loss.item():.4f}")
            if use_wandb:
                wandb.log({'loss': loss.item(), 'epoch': epoch})

    # Save checkpoint
    torch.save(model.state_dict(),     'checkpoints/gnn.pt')
    torch.save(predictor.state_dict(), 'checkpoints/predictor.pt')
    print("Models saved to checkpoints/")

    if use_wandb:
        wandb.finish()

    return model, predictor, z_dict