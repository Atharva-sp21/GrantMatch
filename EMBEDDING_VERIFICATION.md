# Embedding Quality Verification Plan

## How to Debug Whether Root Causes Are Correct

After training, verify the root causes with embedding analysis:

### 1. Check Feature Quality

```python
import torch
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load data
researchers = json.load(open('data/raw/researchers.json'))
grants = json.load(open('data/raw/grants.json'))

# Check researcher feature variance
h_indices = [r.get('h_index', 0) for r in researchers]
citations = [r.get('cited_by_count', 0) for r in researchers]

print(f"Researcher h_index: std={np.std(h_indices):.2f}")
print(f"Researcher citations: std={np.std(citations):.0f}")
print(f"Researchers with h_index=0: {sum(1 for h in h_indices if h == 0)}")

# Check grant amount variance
amounts = [g.get('amount', 0) for g in grants]
print(f"\nGrant amounts: unique={len(set(amounts))}, std={np.std(amounts):.0f}")

# Expected: Low std and many zero-valued researchers = poor features
```

### 2. Check Embedding Quality (After Training)

```python
import torch
import torch.nn.functional as F

# Load trained model and get embeddings
model = GrantMatchGNN.load_state_dict(torch.load('checkpoints/gnn.pt'))
predictor = LinkPredictor.load_state_dict(torch.load('checkpoints/predictor.pt'))

# Forward pass to get z_dict
z_dict = model(data.x_dict, data.edge_index_dict)

z_r = z_dict['researcher']  # [N_r, hidden]
z_g = z_dict['grant']       # [N_g, hidden]

# === METRIC 1: Positive vs Negative Pair Distances ===
pos_edge = data['researcher', 'RECEIVED_PAST', 'grant'].edge_index

# Sample positive pairs
pos_distances = []
for i in range(min(100, pos_edge.shape[1])):
    r_id, g_id = pos_edge[0, i].item(), pos_edge[1, i].item()
    dist = F.cosine_distance(
        z_r[r_id].unsqueeze(0),
        z_g[g_id].unsqueeze(0)
    ).item()
    pos_distances.append(dist)

# Sample negative pairs
neg_distances = []
np.random.seed(42)
for _ in range(100):
    r_id = np.random.randint(0, len(researchers))
    g_id = np.random.randint(0, len(grants))
    if (r_id, g_id) not in pos_edge_set:
        dist = F.cosine_distance(
            z_r[r_id].unsqueeze(0),
            z_g[g_id].unsqueeze(0)
        ).item()
        neg_distances.append(dist)

print("EMBEDDING QUALITY TEST #1: Positive vs Negative Distances")
print(f"Positive pair distances: mean={np.mean(pos_distances):.3f}, std={np.std(pos_distances):.3f}")
print(f"Negative pair distances: mean={np.mean(neg_distances):.3f}, std={np.std(neg_distances):.3f}")
print(f"Separation ratio: {np.mean(neg_distances) / (np.mean(pos_distances) + 1e-6):.2f}")

# INTERPRETATION:
# If separation ratio < 1.2: embeddings don't distinguish positive from negative -> ROOT CAUSE CONFIRMED
# If separation ratio > 2.0: embeddings are learning -> ROOT CAUSE NOT IT

# === METRIC 2: Similar Researcher Distances ===
researcher_topics = build_researcher_topic_matrix()
topic_similarity = torch.cosine_similarity(
    researcher_topics.unsqueeze(1),
    researcher_topics.unsqueeze(0)
)

# Find pairs with similar topics
similar_pairs_embedding_dist = []
dissimilar_pairs_embedding_dist = []

for r_i in range(min(50, len(researchers))):
    for r_j in range(r_i+1, min(50, len(researchers))):
        topic_sim = topic_similarity[r_i, r_j].item()
        emb_dist = F.cosine_distance(
            z_r[r_i].unsqueeze(0),
            z_r[r_j].unsqueeze(0)
        ).item()
        
        if topic_sim > 0.5:  # Similar topics
            similar_pairs_embedding_dist.append(emb_dist)
        elif topic_sim < 0.2:  # Dissimilar topics
            dissimilar_pairs_embedding_dist.append(emb_dist)

print("\nEMBEDDING QUALITY TEST #2: Topic Similarity vs Embedding Similarity")
print(f"Similar topic pairs: embedding distance={np.mean(similar_pairs_embedding_dist):.3f}")
print(f"Dissimilar topic pairs: embedding distance={np.mean(dissimilar_pairs_embedding_dist):.3f}")

# INTERPRETATION:
# If distances are similar: model didn't learn topic differences -> ROOT CAUSE CONFIRMED
# If different topics have larger distances: model learned topics -> ROOT CAUSE NOT IT

# === METRIC 3: Researcher Differentiation ===
low_h_index_researchers = [i for i, r in enumerate(researchers) if r.get('h_index', 0) <= 5]
high_h_index_researchers = [i for i, r in enumerate(researchers) if r.get('h_index', 0) >= 20]

# Compute distances within groups
low_h_distances = []
for i in range(min(20, len(low_h_index_researchers))):
    for j in range(i+1, min(20, len(low_h_index_researchers))):
        r_i = low_h_index_researchers[i]
        r_j = low_h_index_researchers[j]
        dist = F.cosine_distance(z_r[r_i].unsqueeze(0), z_r[r_j].unsqueeze(0)).item()
        low_h_distances.append(dist)

high_h_distances = []
for i in range(min(20, len(high_h_index_researchers))):
    for j in range(i+1, min(20, len(high_h_index_researchers))):
        r_i = high_h_index_researchers[i]
        r_j = high_h_index_researchers[j]
        dist = F.cosine_distance(z_r[r_i].unsqueeze(0), z_r[r_j].unsqueeze(0)).item()
        high_h_distances.append(dist)

print("\nEMBEDDING QUALITY TEST #3: Researcher Homogeneity")
print(f"Low h-index researchers: avg distance={np.mean(low_h_distances):.3f} (should be SMALL)")
print(f"High h-index researchers: avg distance={np.mean(high_h_distances):.3f} (should be SMALL)")

# INTERPRETATION:
# If both are similar: can't differentiate researchers -> ROOT CAUSE CONFIRMED
# If there's variance: model is capturing differences -> MAY NEED OTHER FIXES

# === METRIC 4: Grant Homogeneity ===
low_budget_grants = [i for i, g in enumerate(grants) if g.get('amount', 0) <= 500000]
high_budget_grants = [i for i, g in enumerate(grants) if g.get('amount', 0) >= 5000000]

low_g_distances = []
for i in range(min(20, len(low_budget_grants))):
    for j in range(i+1, min(20, len(low_budget_grants))):
        g_i = low_budget_grants[i]
        g_j = low_budget_grants[j]
        dist = F.cosine_distance(z_g[g_i].unsqueeze(0), z_g[g_j].unsqueeze(0)).item()
        low_g_distances.append(dist)

high_g_distances = []
for i in range(min(20, len(high_budget_grants))):
    for j in range(i+1, min(20, len(high_budget_grants))):
        g_i = high_budget_grants[i]
        g_j = high_budget_grants[j]
        dist = F.cosine_distance(z_g[g_i].unsqueeze(0), z_g[g_j].unsqueeze(0)).item()
        high_g_distances.append(dist)

print("\nEMBEDDING QUALITY TEST #4: Grant Homogeneity")
print(f"Low budget grants: avg distance={np.mean(low_g_distances):.3f} (should be SMALL)")
print(f"High budget grants: avg distance={np.mean(high_g_distances):.3f} (should be SMALL)")

# INTERPRETATION:
# If both are similar: can't differentiate grants -> ROOT CAUSE CONFIRMED
# If there's variance: model is capturing differences -> FEATURE ENGINEERING WORKING

# === METRIC 5: Ranking Quality ===
def ranking_quality(z_r, z_g, pos_edge, predictor, k=10):
    """Measure how well positives are ranked."""
    predictor.eval()
    with torch.no_grad():
        num_correct_in_topk = 0
        
        for i in range(min(100, pos_edge.shape[1])):
            r_id = pos_edge[0, i].item()
            true_g_id = pos_edge[1, i].item()
            
            # Score against all grants
            z_r_expanded = z_r[r_id].unsqueeze(0).repeat(len(grants), 1)
            scores = predictor(z_r_expanded, z_g)
            
            # Get rank of true grant
            rank = (scores > scores[true_g_id]).sum().item() + 1
            
            if rank <= k:
                num_correct_in_topk += 1
        
        return num_correct_in_topk / min(100, pos_edge.shape[1])

hr_k10 = ranking_quality(z_r, z_g, pos_edge, predictor, k=10)
print(f"\nEMBEDDING QUALITY TEST #5: Hit Rate @10")
print(f"Hit Rate: {hr_k10:.4f}")

# INTERPRETATION:
# If HR < 0.10: embeddings don't rank well -> ROOT CAUSE IS REAL
# If HR > 0.30: embeddings rank well -> PROBLEM MAY BE IN OTHER AREAS
```

### 3. Interpretation Guide

**Test Results That Confirm Root Causes:**

✓ **Root Cause #1 (Hardcoded Features) Confirmed If:**
- Feature variance is very low (std of grant features < 0.5)
- All grants look nearly identical in PCA/TSNE plot
- Grant embeddings cluster into very few groups

✓ **Root Cause #2 (Indistinguishable Researchers) Confirmed If:**
- Researchers with h_index=0 and h_index>20 have similar embedding distances
- Low variance in researcher embedding space
- Similar topic researchers don't have closer embeddings

✓ **Root Cause #3 (Graph Disconnection) Confirmed If:**
- Isolated researchers have very similar embeddings to connected ones
- No clear separation in embedding space
- Random embeddings for unconnected nodes

✓ **Root Cause #4 (Small Model) Confirmed If:**
- Positive and negative pairs have overlapping distances
- Model can't separate easy positives from easy negatives
- Embedding manifold is "folded" (poor use of space)

✓ **Root Cause #5 (Loss Function) Confirmed If:**
- Positive pair distances close to negative distances
- Model has low training loss but high ranking error
- Predictions have poor ranking quality despite good classification

---

## What "Should" Happen After Fixes

After implementing fixes, you should see:

**After Fix #1 (Topic Features):**
- Grant embeddings separate by topic
- Researcher embeddings group by specialty
- Separation ratio: positive distances < negative distances
- Hit Rate: 15-20%

**After Fix #3 (Increased Capacity):**
- Better learned embeddings with more structure
- Clearer separation of positive vs negative
- Hit Rate: 30-35%

**After Fix #2 (Hard Negatives):**
- Model learns to rank within similar grants
- Better discrimination between hard negatives
- Hit Rate: 40-45%

**Final After Fix #4 (Better Loss):**
- Ranking quality improves
- Hit Rate: 50-55%

---

## Quick Debug Script

```python
import torch
import json
import numpy as np
from pathlib import Path

# Run this after training to verify root causes

DATA_DIR = Path('data/raw')
researchers = json.load(open(DATA_DIR / 'researchers.json'))
grants = json.load(open(DATA_DIR / 'grants.json'))

# 1. Check features
h_indices = [r.get('h_index', 0) for r in researchers]
amounts = [g.get('amount', 0) for g in grants]

print("FEATURE QUALITY")
print(f"  Researcher h-index: {sum(1 for h in h_indices if h == 0)}/{len(researchers)} zeros")
print(f"  Grant amounts: {len(set(amounts))}/{len(grants)} unique values")

# 2. Check graph connectivity
with open(DATA_DIR / 'received_past.csv') as f:
    reader = csv.DictReader(f)
    edges = list(reader)

degree = {}
for e in edges:
    src = int(e['source_id'])
    degree[src] = degree.get(src, 0) + 1

print("\nGRAPH CONNECTIVITY")
print(f"  Connected researchers: {len(degree)}/{len(researchers)}")
print(f"  Isolated researchers: {len(researchers) - len(degree)}")

# 3. Check embedding quality (after training)
model.eval()
with torch.no_grad():
    z_dict = model(data.x_dict, data.edge_index_dict)
    z_r = z_dict['researcher']
    z_g = z_dict['grant']

pos_edge = data['researcher', 'RECEIVED_PAST', 'grant'].edge_index

# Positive distances
pos_dists = []
for i in range(min(50, pos_edge.shape[1])):
    r_id, g_id = pos_edge[0, i].item(), pos_edge[1, i].item()
    dist = torch.nn.functional.cosine_distance(
        z_r[r_id].unsqueeze(0), z_g[g_id].unsqueeze(0)
    ).item()
    pos_dists.append(dist)

# Negative distances
neg_dists = []
np.random.seed(42)
for _ in range(50):
    r_id = np.random.randint(0, len(researchers))
    g_id = np.random.randint(0, len(grants))
    if (r_id, g_id) not in pos_edge_set:
        dist = torch.nn.functional.cosine_distance(
            z_r[r_id].unsqueeze(0), z_g[g_id].unsqueeze(0)
        ).item()
        neg_dists.append(dist)

print("\nEMBEDDING QUALITY")
print(f"  Positive distances: {np.mean(pos_dists):.3f} ± {np.std(pos_dists):.3f}")
print(f"  Negative distances: {np.mean(neg_dists):.3f} ± {np.std(neg_dists):.3f}")
print(f"  Separation: {np.mean(neg_dists) / np.mean(pos_dists):.2f}x")

if np.mean(neg_dists) / np.mean(pos_dists) < 1.3:
    print("  ⚠️  Poor separation - embeddings not discriminative")
else:
    print("  ✓ Good separation - embeddings are learning")
```

---
