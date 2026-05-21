import torch
from torch_geometric.data import HeteroData

def build_graph(researchers, institutions, topics, grants, agencies):
    data = HeteroData()

    # ── Node features (only add non-empty node types) ──────────────────
    if len(researchers) > 0:
        data['researcher'].x = researchers   # [N_r, 9]
    if len(institutions) > 0:
        data['institution'].x = institutions  # [N_i, 5]
    if len(topics) > 0:
        data['topic'].x = topics        # [N_t, 5]
    if len(grants) > 0:
        data['grant'].x = grants        # [N_g, 7]
    if len(agencies) > 0:
        data['agency'].x = agencies      # [N_a, 4]

    return data

def add_edges(data, affiliated, researches, received, funds, provides):
    # Only add edges if both node types exist
    if 'researcher' in data.node_types and 'institution' in data.node_types:
        if affiliated.shape[1] > 0:
            data['researcher', 'AFFILIATED_WITH', 'institution'].edge_index = affiliated

    if 'researcher' in data.node_types and 'topic' in data.node_types:
        if researches.shape[1] > 0:
            data['researcher', 'RESEARCHES', 'topic'].edge_index = researches

    if 'researcher' in data.node_types and 'grant' in data.node_types:
        if received.shape[1] > 0:
            data['researcher', 'RECEIVED_PAST', 'grant'].edge_index = received

    if 'grant' in data.node_types and 'topic' in data.node_types:
        if funds.shape[1] > 0:
            data['grant', 'FUNDS_TOPIC', 'topic'].edge_index = funds

    if 'agency' in data.node_types and 'grant' in data.node_types:
        if provides.shape[1] > 0:
            data['agency', 'PROVIDES', 'grant'].edge_index = provides

    return data