import torch
from torch_geometric.data import HeteroData


def build_graph(researchers, institutions, topics, grants, agencies):
    data = HeteroData()
    data["researcher"].x = researchers
    data["institution"].x = institutions
    data["topic"].x = topics
    data["grant"].x = grants
    data["agency"].x = agencies
    return data


def _empty_edge_index():
    return torch.zeros(2, 0, dtype=torch.long)


def _set_edge_pair(data, src_type, relation, dst_type, edge_index):
    if edge_index is None:
        edge_index = _empty_edge_index()
    data[(src_type, relation, dst_type)].edge_index = edge_index
    reverse_relation = f"REV_{relation}"
    data[(dst_type, reverse_relation, src_type)].edge_index = edge_index.flip(0) if edge_index.numel() else _empty_edge_index()


def add_edges(data, affiliated, researches, received, funds, provides):
    _set_edge_pair(data, "researcher", "AFFILIATED_WITH", "institution", affiliated)
    _set_edge_pair(data, "researcher", "RESEARCHES", "topic", researches)
    _set_edge_pair(data, "researcher", "RECEIVED_PAST", "grant", received)
    _set_edge_pair(data, "grant", "FUNDS_TOPIC", "topic", funds)
    _set_edge_pair(data, "agency", "PROVIDES", "grant", provides)
    return data