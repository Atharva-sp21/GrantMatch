import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear

class GrantMatchGNN(nn.Module):
    def __init__(self, hidden=64, heads=2, metadata=None):
        super().__init__()
        # Project every node type to the same hidden dimension
        self.lin_dict = nn.ModuleDict({
            node_type: Linear(-1, hidden)
            for node_type in metadata[0]
        })
        self.conv1 = HGTConv(hidden, hidden, metadata, heads)
        self.conv2 = HGTConv(hidden, hidden, metadata, heads)
        self.hidden = hidden

    def forward(self, x_dict, edge_index_dict):
        # Step 1 — project all node types to same space
        x_dict = {
            nt: self.lin_dict[nt](x).relu()
            for nt, x in x_dict.items()
        }
        # Step 2 — message passing layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items()}

        # Step 3 — message passing layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


class LinkPredictor(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, z_researcher, z_grant):
        z = torch.cat([z_researcher, z_grant], dim=-1)
        return self.mlp(z).squeeze(-1)