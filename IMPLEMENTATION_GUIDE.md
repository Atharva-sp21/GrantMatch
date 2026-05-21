# Implementation Guide: 5 Fixes for GNN Hit Rate

This document provides specific code changes for each of the 5 recommended fixes.

---

## FIX #1: Add Topic Features (Highest Impact +35-40%)

### File: `/graph/features.py`

**Current Problem:**
- Grant features 2,3,4,5,7 are hardcoded constants
- No topic information captured
- Researchers indistinguishable for 52% with h_index=0

**Solution:**

Add helper functions:

```python
def get_researcher_topic_distribution(researcher_id, researcher_topic_edges, num_topics=5):
    """Returns topic distribution for a researcher as one-hot or smooth vector."""
    topic_counts = [0] * num_topics
    for edge in researcher_topic_edges:
        if edge[0] == researcher_id:  # source is researcher
            topic_counts[edge[1]] += 1
    
    total = sum(topic_counts)
    if total == 0:
        return [1.0 / num_topics] * num_topics  # Uniform if no connections
    
    return [float(c) / total for c in topic_counts]  # Normalize to probability

def get_grant_topic_distribution(grant_id, grant_topic_edges, num_topics=5):
    """Returns topic distribution for a grant."""
    topic_counts = [0] * num_topics
    for edge in grant_topic_edges:
        if edge[0] == grant_id:  # source is grant
            topic_counts[edge[1]] += 1
    
    total = sum(topic_counts)
    if total == 0:
        return [1.0 / num_topics] * num_topics  # Uniform if no connections
    
    return [float(c) / total for c in topic_counts]
```

Modify `build_researcher_features`:

```python
def build_researcher_features(researchers, researcher_topic_edges=None):
    """Researcher features: [impact, volume, citations, i10_index, institution_flag, + topics]"""
    rows = []
    for r in researchers:
        row = [
            r.get('h_index', 0),                    # impact
            r.get('works_count', 0),               # total works
            r.get('cited_by_count', 0),            # citation count
            r.get('i10_index', 0),                 # i10-index
            1.0 if r.get('institution_id') is not None else 0.0,
        ]
        
        # Add topic distribution (5 dims)
        if researcher_topic_edges is not None:
            topic_dist = get_researcher_topic_distribution(
                r.get('id'), researcher_topic_edges, num_topics=5
            )
            row.extend(topic_dist)
        else:
            row.extend([0.2, 0.2, 0.2, 0.2, 0.2])  # Uniform default
        
        rows.append(row)
    
    tensor = torch.tensor(rows, dtype=torch.float)
    return normalize_features(tensor)  # [N, 10] now instead of [N, 9]
```

Modify `build_grant_features`:

```python
def build_grant_features(grants, grant_topic_edges=None):
    """Grant features: [amount (2 scales), agency_preference, career_stage, title_flag, + topics]"""
    rows = []
    for g in grants:
        row = [
            g.get('amount', 0) / 100000.0,      # amount (primary scale)
            g.get('amount', 0) / 1000000.0,     # amount (secondary scale)
            0.8,                                 # agency credibility (or from agency data)
            0.7,                                 # avg competitiveness score
            1.0 if g.get('title') else 0.0,     # has description flag
        ]
        
        # Add topic distribution (5 dims) - REPLACES hardcoded constants
        if grant_topic_edges is not None:
            topic_dist = get_grant_topic_distribution(
                g.get('id'), grant_topic_edges, num_topics=5
            )
            row.extend(topic_dist)
        else:
            row.extend([0.2, 0.2, 0.2, 0.2, 0.2])  # Uniform default
        
        rows.append(row)
    
    tensor = torch.tensor(rows, dtype=torch.float)
    return normalize_features(tensor)  # [N, 10] now instead of [N, 7]
```

Update `data_loader.py` to return edges:

```python
def load_all_data():
    """Load all JSON and CSV files into proper format"""
    # ... existing code ...
    
    researches = load_csv_to_tensor('researches.csv')  # [2, N_edges]
    funds = load_csv_to_tensor('funds_topic.csv')      # [2, N_edges]
    
    # Convert to edge list for feature engineering
    researches_list = researches.t().tolist() if researches.shape[1] > 0 else []
    funds_list = funds.t().tolist() if funds.shape[1] > 0 else []
    
    return {
        'researchers': researchers,
        'institutions': institutions,
        'agencies': agencies,
        'grants': grants,
        'topics': topics,
        'affiliated': affiliated,
        'researches': researches,
        'researches_list': researches_list,  # NEW
        'received': received,
        'funds': funds,
        'funds_list': funds_list,            # NEW
        'provides': provides,
    }
```

Update `main.py`:

```python
# In main.py, Step 2:
r_feat = build_researcher_features(
    researchers, 
    researcher_topic_edges=data_dict.get('researches_list')  # NEW
) if len(researchers) > 0 else torch.zeros(0, 10)  # Changed from 9 to 10

g_feat = build_grant_features(
    grants,
    grant_topic_edges=data_dict.get('funds_list')  # NEW
) if len(grants) > 0 else torch.zeros(0, 10)  # Changed from 7 to 10

# Update print statements
print(f"[OK] Researcher features: {r_feat.shape}")  # Now [N, 10]
print(f"[OK] Grant features: {g_feat.shape}")       # Now [N, 10]
```

---

## FIX #2: Hard Negative Mining (Impact +20-25%)

### File: `/training/train_gnn.py`

Add hard negative mining function:

```python
def compute_topic_similarity(z_dict, grant_topic_edges, num_grants, num_topics=5):
    """Compute topic similarity matrix between grants (from embedding indices 5:10)."""
    # Extract topic dimensions from grant embeddings
    grant_embeddings = z_dict['grant']  # [N_g, hidden]
    
    # Simplified: use normalized amount + topic info from original features
    # Better approach: use last layers that learned topic representations
    # For now, use a simple heuristic
    
    similarity = torch.zeros(num_grants, num_grants)
    # To be refined based on learned embeddings
    
    return similarity

def mine_hard_negatives(pos_edge, grant_topic_edges, num_grants, 
                       hard_neg_ratio=1.0, num_similar_grants=5):
    """
    Mine hard negatives: grants topically similar to positive but different researcher.
    
    Args:
        pos_edge: [2, num_pos] tensor
        grant_topic_edges: list of [grant_id, topic_id] edges
        hard_neg_ratio: what fraction of negatives should be hard (0.0-1.0)
    
    Returns:
        hard_neg_src, hard_neg_dst: hard negative edges
    """
    if not grant_topic_edges:
        return [], []  # Fall back to random if no topic edges
    
    # Build topic lookup for grants
    grant_topics = {}
    for src_id, dst_id in grant_topic_edges:
        if src_id not in grant_topics:
            grant_topics[src_id] = set()
        grant_topics[src_id].add(dst_id)
    
    hard_neg_src = []
    hard_neg_dst = []
    
    for i in range(pos_edge.shape[1]):
        r_id = pos_edge[0][i].item()
        true_g_id = pos_edge[1][i].item()
        
        true_topics = grant_topics.get(true_g_id, set())
        
        if not true_topics:
            continue  # Skip if no topic info for this grant
        
        # Find grants with topic overlap
        similar_grants = []
        for g_id in range(num_grants):
            if g_id == true_g_id:
                continue
            g_topics = grant_topics.get(g_id, set())
            
            # Jaccard similarity
            if len(true_topics | g_topics) > 0:
                overlap = len(true_topics & g_topics) / len(true_topics | g_topics)
                if overlap > 0.3:  # Topic similarity threshold
                    similar_grants.append((g_id, overlap))
        
        # Sample hard negatives from similar grants
        if similar_grants:
            similar_grants.sort(key=lambda x: x[1], reverse=True)
            sampled = min(num_similar_grants, len(similar_grants))
            for j in range(sampled):
                hard_neg_dst.append(similar_grants[j][0])
                hard_neg_src.append(r_id)
    
    return hard_neg_src, hard_neg_dst


def train(data, epochs=200, hidden=64, heads=2, lr=1e-3, 
          use_wandb=True, hard_neg_ratio=0.5):
    """
    Args:
        hard_neg_ratio: fraction of negatives that should be hard (0.0-1.0)
    """
    # ... existing initialization code ...
    
    if ('researcher', 'RECEIVED_PAST', 'grant') not in data.edge_types:
        print("[WARN] No RECEIVED_PAST edges - skipping training")
        return None, None, {}
    
    pos_edge = data['researcher', 'RECEIVED_PAST', 'grant'].edge_index
    num_r = data['researcher'].x.shape[0]
    num_g = data['grant'].x.shape[0]
    
    # Extract grant-topic edges if available
    grant_topic_edges = []
    if ('grant', 'FUNDS_TOPIC', 'topic') in data.edge_types:
        gt_edge_index = data['grant', 'FUNDS_TOPIC', 'topic'].edge_index
        grant_topic_edges = gt_edge_index.t().tolist()
    
    # ... training loop ...
    for epoch in range(1, epochs + 1):
        # ... existing code up to line 66 ...
        
        # NEW: Hard negative mining
        hard_neg_src_list, hard_neg_dst_list = mine_hard_negatives(
            pos_edge, grant_topic_edges, num_g, 
            hard_neg_ratio=hard_neg_ratio
        )
        
        # Random negatives for remaining slots
        num_random_negatives = max(0, len(pos_src) - len(hard_neg_src_list))
        pos_edge_set = set(zip(pos_edge[0].tolist(), pos_edge[1].tolist()))
        
        rand_neg_src, rand_neg_dst = [], []
        max_attempts = len(pos_src) * 10
        attempts = 0
        while len(rand_neg_src) < num_random_negatives and attempts < max_attempts:
            candidate_src = torch.randint(0, num_r, (1,)).item()
            candidate_dst = torch.randint(0, num_g, (1,)).item()
            if (candidate_src, candidate_dst) not in pos_edge_set:
                rand_neg_src.append(candidate_src)
                rand_neg_dst.append(candidate_dst)
            attempts += 1
        
        # Combine hard and random negatives
        neg_src = torch.tensor(hard_neg_src_list + rand_neg_src, dtype=torch.long)
        neg_dst = torch.tensor(hard_neg_dst_list + rand_neg_dst, dtype=torch.long)
        
        if len(neg_src) == 0:
            print("[WARN] Could not generate negative samples, skipping epoch")
            continue
        
        # ... rest of existing training code ...
```

---

## FIX #3: Increase Model Capacity (Impact +15-20%)

### File: `/models/gnn.py`

```python
import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear

class GrantMatchGNN(nn.Module):
    def __init__(self, hidden=256, heads=4, num_layers=4, metadata=None):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        
        # Project input features to hidden space
        self.lin_dict = nn.ModuleDict({
            node_type: Linear(-1, hidden)
            for node_type in metadata[0]
        })
        
        # Multiple HGT layers
        self.convs = nn.ModuleList([
            HGTConv(hidden, hidden, metadata, heads)
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each layer
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(hidden)
                for node_type in metadata[0]
            })
            for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict):
        # Project all node types to hidden space
        x_dict = {
            nt: self.lin_dict[nt](x).relu()
            for nt, x in x_dict.items()
            if x.shape[0] > 0
        }

        x_dict_initial = x_dict.copy()

        # Apply multiple HGT layers with residual connections
        for layer_idx in range(self.num_layers):
            x_dict_before = x_dict.copy()
            
            if len(x_dict) > 0:
                x_dict = self.convs[layer_idx](x_dict, edge_index_dict)
                x_dict = {
                    k: self.norms[layer_idx][k](v).relu() 
                    for k, v in x_dict.items()
                }
                
                # Restore missing node types
                for nt in x_dict_before:
                    if nt not in x_dict:
                        x_dict[nt] = x_dict_before[nt]

        return x_dict


class LinkPredictor(nn.Module):
    def __init__(self, hidden=256, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.33),
            
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, z_researcher, z_grant):
        z = torch.cat([z_researcher, z_grant], dim=-1)
        return self.mlp(z).squeeze(-1)
```

### File: `/training/train_gnn.py`

Update training call:

```python
def main():
    # ... existing code ...
    
    print("\n[STEP 5] Training GNN model...")
    print("(This may take a few minutes...)")
    model, predictor, z_dict = train(
        data, 
        epochs=200, 
        hidden=256,      # Increased from 64
        heads=4,         # Increased from 2
        num_layers=4,    # Increased from 2
        use_wandb=False
    )
```

Update train function signature:

```python
def train(data, epochs=200, hidden=256, heads=4, num_layers=4, 
          lr=1e-3, use_wandb=True, hard_neg_ratio=0.5):
    
    if use_wandb:
        wandb.init(project='grantmatch', config={
            'hidden': hidden, 'heads': heads, 'num_layers': num_layers,
            'lr': lr, 'epochs': epochs
        })

    model = GrantMatchGNN(hidden=hidden, heads=heads, 
                         num_layers=num_layers, metadata=data.metadata())
    predictor = LinkPredictor(hidden=hidden)
    # ... rest of training ...
```

---

## FIX #4: Use Better Loss Function (Impact +10-15%)

### File: `/training/train_gnn.py`

Replace BCELoss with MarginRankingLoss or ContrastiveLoss:

```python
class MarginRankingLoss(nn.Module):
    """Penalizes when negative score > positive score by margin."""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, pos_scores, neg_scores):
        # pos_scores should be > neg_scores - margin
        # Loss = max(0, margin + neg_scores - pos_scores)
        loss = torch.clamp(self.margin + neg_scores - pos_scores, min=0)
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, pos_scores, neg_scores):
        # pos_scores: [batch_size]
        # neg_scores: [batch_size]
        
        # Create logits matrix
        batch_size = len(pos_scores)
        pos_scores_expanded = pos_scores.unsqueeze(1)  # [batch, 1]
        neg_scores_expanded = neg_scores.unsqueeze(1)  # [batch, 1]
        
        logits = torch.cat([pos_scores_expanded, neg_scores_expanded], dim=1)
        logits = logits / self.temperature
        
        labels = torch.zeros(batch_size, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        return loss


# In train function:
def train(data, epochs=200, hidden=256, heads=4, num_layers=4,
          lr=1e-3, use_wandb=True, hard_neg_ratio=0.5, 
          loss_type='margin'):  # NEW parameter
    
    # ... initialization ...
    
    if loss_type == 'margin':
        criterion = MarginRankingLoss(margin=1.0)
    elif loss_type == 'contrastive':
        criterion = ContrastiveLoss(temperature=0.1)
    else:  # bce (old)
        criterion = nn.BCELoss()
    
    # ... training loop ...
    
    # Modify loss computation (depends on loss type)
    if loss_type in ['margin', 'contrastive']:
        loss = criterion(pos_pred, neg_pred)
    else:  # BCE
        labels = torch.cat([torch.ones(len(pos_pred)),
                           torch.zeros(len(neg_pred))])
        preds = torch.cat([pos_pred, neg_pred])
        loss = criterion(preds, labels)
```

Call with new loss:

```python
model, predictor, z_dict = train(
    data, 
    epochs=200, 
    hidden=256,
    heads=4,
    num_layers=4,
    loss_type='margin',  # Use margin loss instead of BCE
    use_wandb=False
)
```

---

## FIX #5: Add Researcher Similarity Edges (Impact +5-10%)

### File: `/graph/schema.py`

Add similarity edge creation:

```python
import torch
from torch_geometric.data import HeteroData

def add_researcher_similarity_edges(data, researcher_topic_edges=None, k=5):
    """
    Add edges between topically similar researchers.
    Helps isolated researchers by connecting them through similarity.
    """
    if 'researcher' not in data.node_types:
        return data
    
    num_researchers = data['researcher'].x.shape[0]
    
    if researcher_topic_edges is None:
        return data  # No topic info, skip
    
    # Build topic vectors for each researcher
    topic_vectors = torch.zeros(num_researchers, 5)
    for src_id, dst_id in researcher_topic_edges:
        if src_id < num_researchers:
            topic_vectors[src_id, dst_id] = 1.0
    
    # Normalize
    norms = topic_vectors.norm(dim=1, keepdim=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    topic_vectors = topic_vectors / norms
    
    # Compute cosine similarity
    similarity = torch.mm(topic_vectors, topic_vectors.t())
    
    # Find top-k similar researchers for each researcher
    edges_src = []
    edges_dst = []
    
    for r_i in range(num_researchers):
        # Get top-k+1 (excluding self)
        top_similarities, top_indices = torch.topk(
            similarity[r_i], 
            k=min(k + 1, num_researchers)
        )
        
        for j, r_j in enumerate(top_indices):
            r_j = r_j.item()
            if r_i == r_j:  # Skip self
                continue
            if r_i < r_j:  # Avoid duplicates (undirected)
                edges_src.append(r_i)
                edges_dst.append(r_j)
    
    if edges_src:
        # Create edge tensor [2, num_edges]
        edge_index = torch.tensor(
            [edges_src, edges_dst], 
            dtype=torch.long
        )
        
        # Add edges to graph
        data['researcher', 'SIMILAR_TO', 'researcher'].edge_index = edge_index
        
        # Add reverse edges for undirected graph
        data['researcher', 'rev_SIMILAR_TO', 'researcher'].edge_index = \
            torch.stack([edge_index[1], edge_index[0]])
    
    return data


def build_graph(researchers, institutions, topics, grants, agencies):
    data = HeteroData()

    # ... existing node features code ...
    
    return data


def add_edges(data, affiliated, researches, received, funds, provides,
              researcher_topic_edges=None):
    """Add edges including similarity edges."""
    
    # ... existing edges code ...
    
    # NEW: Add researcher similarity edges
    data = add_researcher_similarity_edges(data, researcher_topic_edges, k=5)
    
    return data
```

Update `main.py`:

```python
# In main.py, Step 3:
data = add_edges(
    data,
    affiliated=data_dict['affiliated'],
    researches=data_dict['researches'],
    received=data_dict['received'],
    funds=data_dict['funds'],
    provides=data_dict['provides'],
    researcher_topic_edges=data_dict.get('researches_list'),  # NEW
)
print("[OK] Graph built with edges (including similarity edges)")
```

---

## Summary of File Changes

| File | Fix # | Change | Impact |
|------|-------|--------|--------|
| `graph/features.py` | 1 | Add topic distribution to researcher/grant features | +35-40% |
| `models/gnn.py` | 3 | Increase hidden (64→256), heads (2→4), layers (2→4) | +15-20% |
| `training/train_gnn.py` | 2,4 | Add hard negative mining, replace BCELoss | +30-30% |
| `graph/schema.py` | 5 | Add researcher similarity edges | +5-10% |
| `data_loader.py` | 1 | Return researcher/grant topic edges | - |
| `main.py` | 1,5 | Pass topic edges to functions | - |

---

## Testing Strategy

After implementing each fix:

1. **After Fix #1:** Verify feature dimensions changed (9→10 and 7→10)
2. **After Fix #3:** Check that model trains (may need learning rate adjustment)
3. **After Fix #2:** Monitor hard negative sampling ratio in logs
4. **After Fix #4:** Compare loss curves (margin vs BCE)
5. **After Fix #5:** Check researcher similarity edges were added

Expected trajectory:
- Baseline: 6.65% Hit Rate
- After 1+3: 30-35%
- After 2+4: 45-50%
- After 5: 50-55%

---
