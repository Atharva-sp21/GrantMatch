import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


class GrantMatchGNN(nn.Module):
    def __init__(self, hidden=256, heads=2, metadata=None, dropout=0.2):
        super().__init__()
        if metadata is None:
            raise ValueError("metadata is required to build GrantMatchGNN")

        self.hidden = hidden
        self.dropout = dropout
        node_types = metadata[0]

        self.input_proj = nn.ModuleDict({node_type: Linear(-1, hidden) for node_type in node_types})
        self.norm1 = nn.ModuleDict({node_type: nn.LayerNorm(hidden) for node_type in node_types})
        self.norm2 = nn.ModuleDict({node_type: nn.LayerNorm(hidden) for node_type in node_types})
        self.conv1 = HGTConv(hidden, hidden, metadata, heads=heads)
        self.conv2 = HGTConv(hidden, hidden, metadata, heads=heads)

    def _project(self, x_dict):
        projected = {}
        for node_type, features in x_dict.items():
            if node_type in self.input_proj:
                projected[node_type] = F.relu(self.input_proj[node_type](features))
        return projected

    def _apply_block(self, conv, norm, x_dict, edge_index_dict):
        residuals = x_dict
        updated = conv(x_dict, edge_index_dict)
        merged = {}
        for node_type, residual in residuals.items():
            current = updated.get(node_type, residual)
            if current.shape == residual.shape:
                current = current + residual
            merged[node_type] = F.relu(norm[node_type](current))
        return {node_type: F.dropout(features, p=self.dropout, training=self.training) for node_type, features in merged.items()}

    def forward(self, x_dict, edge_index_dict):
        x_dict = self._project(x_dict)
        if not x_dict:
            return x_dict

        x_dict = self._apply_block(self.conv1, self.norm1, x_dict, edge_index_dict)
        x_dict = self._apply_block(self.conv2, self.norm2, x_dict, edge_index_dict)
        return x_dict


class LinkPredictor(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z_researcher, z_grant):
        features = torch.cat(
            [z_researcher, z_grant, torch.abs(z_researcher - z_grant), z_researcher * z_grant],
            dim=-1,
        )
        return self.mlp(features).squeeze(-1)
