import torch
from torch_geometric.data import HeteroData

def build_graph(researchers, institutions, topics, grants, agencies):
    data = HeteroData()

    # ── Node features ────────────────────────────────────────────────
    data['researcher'].x  = researchers   # [N_r, 9]
    data['institution'].x = institutions  # [N_i, 5]
    data['topic'].x       = topics        # [N_t, 5]
    data['grant'].x       = grants        # [N_g, 7]
    data['agency'].x      = agencies      # [N_a, 4]

    return data

def add_edges(data, affiliated, researches, received, funds, provides):
    data['researcher', 'AFFILIATED_WITH', 'institution'].edge_index = affiliated
    data['researcher', 'RESEARCHES',      'topic'].edge_index       = researches
    data['researcher', 'RECEIVED_PAST',   'grant'].edge_index       = received
    data['grant',      'FUNDS_TOPIC',     'topic'].edge_index       = funds
    data['agency',     'PROVIDES',        'grant'].edge_index       = provides
    return data