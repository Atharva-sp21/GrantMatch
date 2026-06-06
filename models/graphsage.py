"""Heterogeneous GraphSAGE baseline (2 layers) for comparison with HGT."""

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, Linear


class HeteroGraphSAGE(nn.Module):
    def __init__(self, hidden=128, metadata=None, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout
        node_types = metadata[0]
        edge_types = metadata[1]

        self.lin_dict = nn.ModuleDict({nt: Linear(-1, hidden) for nt in node_types})

        conv1, conv2 = {}, {}
        for et in edge_types:
            src, rel, dst = et
            conv1[et] = SAGEConv((-1, -1), hidden)
            conv2[et] = SAGEConv((-1, -1), hidden)
        self.conv1 = HeteroConv(conv1, aggr="sum")
        self.conv2 = HeteroConv(conv2, aggr="sum")
        self.norm1 = nn.ModuleDict({nt: nn.BatchNorm1d(hidden) for nt in node_types})
        self.norm2 = nn.ModuleDict({nt: nn.BatchNorm1d(hidden) for nt in node_types})

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            nt: torch.relu(self.lin_dict[nt](x))
            for nt, x in x_dict.items()
            if x.shape[0] > 0
        }
        prev = dict(x_dict)
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {
            k: torch.relu(self.norm1[k](v))
            for k, v in x_dict.items()
            if k in self.norm1
        }
        for k in prev:
            if k not in x_dict:
                x_dict[k] = prev[k]
            elif prev[k].shape == x_dict[k].shape:
                x_dict[k] = x_dict[k] + prev[k]

        prev = dict(x_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)
        for k, v in x_dict.items():
            if k in self.norm2:
                x_dict[k] = torch.relu(self.norm2[k](v))
        for k in prev:
            if k not in x_dict:
                x_dict[k] = prev[k]
        return x_dict
