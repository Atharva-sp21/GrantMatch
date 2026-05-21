import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear

class GrantMatchGNN(nn.Module):
    def __init__(self, hidden=256, heads=4, metadata=None):
        super().__init__()
        self.hidden = hidden

        # Project all node types to hidden dimension
        self.lin_dict = nn.ModuleDict({
            node_type: Linear(-1, hidden)
            for node_type in metadata[0]
        })

        # 4 conv layers for deeper message passing
        self.conv1 = HGTConv(hidden, hidden, metadata, heads)
        self.conv2 = HGTConv(hidden, hidden, metadata, heads)
        self.conv3 = HGTConv(hidden, hidden, metadata, heads)
        self.conv4 = HGTConv(hidden, hidden, metadata, heads)

        # Layer normalization after each conv
        self.norm_dict = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden)
            for node_type in metadata[0]
        })

    def forward(self, x_dict, edge_index_dict):
        # Project all node types to hidden space
        x_dict = {
            nt: self.lin_dict[nt](x).relu()
            for nt, x in x_dict.items()
            if x.shape[0] > 0
        }

        if len(x_dict) == 0:
            return x_dict

        x_dict_prev = x_dict.copy()

        # Conv layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: self.norm_dict[k](v).relu() for k, v in x_dict.items()}
        for nt in x_dict_prev:
            if nt not in x_dict:
                x_dict[nt] = x_dict_prev[nt]
        x_dict_prev = x_dict.copy()

        # Conv layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: self.norm_dict[k](v).relu() for k, v in x_dict.items()}
        for nt in x_dict_prev:
            if nt not in x_dict:
                x_dict[nt] = x_dict_prev[nt]
        x_dict_prev = x_dict.copy()

        # Conv layer 3
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {k: self.norm_dict[k](v).relu() for k, v in x_dict.items()}
        for nt in x_dict_prev:
            if nt not in x_dict:
                x_dict[nt] = x_dict_prev[nt]
        x_dict_prev = x_dict.copy()

        # Conv layer 4
        x_dict = self.conv4(x_dict, edge_index_dict)
        x_dict = {k: self.norm_dict[k](v).relu() for k, v in x_dict.items()}
        for nt in x_dict_prev:
            if nt not in x_dict:
                x_dict[nt] = x_dict_prev[nt]

        return x_dict


class LinkPredictor(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z_researcher, z_grant):
        z = torch.cat([z_researcher, z_grant], dim=-1)
        return self.mlp(z).squeeze(-1)


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