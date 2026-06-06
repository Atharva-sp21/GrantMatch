"""
HGT encoder + link predictor.

Architecture choices (why they help):
- hidden=128, heads=4: more capacity than 64/2 for 1k nodes / heterogeneous types.
- 2 HGT layers: enough propagation without oversmoothing on sparse graphs.
- BatchNorm + residual: stabilizes training when loss was diverging.
- Dropout 0.2: regularizes on limited positive edges.
- LinkPredictor outputs logits + BCEWithLogitsLoss: numerically stable vs Sigmoid+BCE.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear


class GrantMatchGNN(nn.Module):
    def __init__(self, hidden=128, heads=4, metadata=None, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout

        self.lin_dict = nn.ModuleDict(
            {nt: Linear(-1, hidden) for nt in metadata[0]}
        )
        self.norm_in = nn.ModuleDict(
            {nt: nn.BatchNorm1d(hidden) for nt in metadata[0]}
        )

        self.conv1 = HGTConv(hidden, hidden, metadata, heads)
        self.norm1 = nn.ModuleDict(
            {nt: nn.BatchNorm1d(hidden) for nt in metadata[0]}
        )
        self.conv2 = HGTConv(hidden, hidden, metadata, heads)
        self.norm2 = nn.ModuleDict(
            {nt: nn.BatchNorm1d(hidden) for nt in metadata[0]}
        )

    def _apply_conv(self, conv, norm_dict, x_dict, edge_index_dict):
        prev = {k: v for k, v in x_dict.items()}
        out = conv(x_dict, edge_index_dict)
        merged = {}
        for nt in prev:
            if nt in out:
                h = norm_dict[nt](out[nt])
                h = torch.relu(h)
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
                # Residual when shapes match
                if prev[nt].shape == h.shape:
                    h = h + prev[nt]
                merged[nt] = h
            else:
                merged[nt] = prev[nt]
        return merged

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            nt: self.norm_in[nt](torch.relu(self.lin_dict[nt](x)))
            for nt, x in x_dict.items()
            if x.shape[0] > 0
        }
        if not x_dict:
            return x_dict

        x_dict = self._apply_conv(self.conv1, self.norm1, x_dict, edge_index_dict)
        x_dict = self._apply_conv(self.conv2, self.norm2, x_dict, edge_index_dict)
        return x_dict


class LinkPredictor(nn.Module):
    def __init__(self, hidden=128, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_researcher, z_grant):
        z = torch.cat([z_researcher, z_grant], dim=-1)
        return self.mlp(z).squeeze(-1)
