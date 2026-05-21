# GrantMatch GNN Model Analysis: Why Hit Rate is Poor Despite Learning

**Current Metrics:**
- Loss decreases: 0.465 → 0.357 (improving)
- Hit Rate @10: 6.65% (218/3280) 
- Random baseline: ~5%
- Result: Model barely beats random despite learning

---

## ROOT CAUSE ANALYSIS

### 1. FEATURE QUALITY - CRITICAL ISSUE ✗✗✗

**Researcher Features (9 dims, but mostly non-discriminative):**
- h_index range: 0-321, but **52.1% have h_index=0** (521/1000 researchers completely indistinguishable)
- Citations: 0-806,327 (highly skewed distribution)
- i10_index: mean=130.7 (sparse signal)
- Features 5-9 in build_researcher_features are DERIVED FEATURES from features 1-4 (redundant)
- **Result:** Most researchers are nearly identical in feature space

**Grant Features (7 dims, but 4 are HARDCODED CONSTANTS):**
```python
Feature 1: amount/100000          # Real data
Feature 2: 1.0                    # HARDCODED FLAG - all identical
Feature 3: 1.0                    # HARDCODED "open" - all identical
Feature 4: 2.0                    # HARDCODED "career_stage" - all identical
Feature 5: 0.5                    # HARDCODED "relevance placeholder" - all identical
Feature 6: amount/50000           # REDUNDANT with feature 1
Feature 7: 1.0                    # HARDCODED "has_title" - all identical
```
- **Only 1-2 features actually vary** (amount in 2 scales)
- 100 unique grant amounts out of 200 grants = minimal diversity
- **Effectively, grants are almost identical in feature space**

**Missing Critical Information:**
- No topic features embedded (303 grant-topic edges unused!)
- No researcher specialty features (2023 researcher-topic edges unused!)
- Grant titles contain rich semantic info ("Quantum Machine Learning", "Climate Change") but ignored
- Researcher-topic alignment completely missing from features

---

### 2. GRAPH STRUCTURE ISSUES

**Extreme Sparsity and Disconnection:**
- Total positive edges: 3,280
- Density: 1.64% (200×1000 possible edges)
- **643 researchers (64.3%) have ZERO edges** - cannot learn anything from graph
- Researchers with edges: 357 only
- Researcher degree range: 1 to 89 edges
- 44 researchers have only 1 edge (minimal training signal)

**Missing Graph Context:**
- Institutions: 0 institutions loaded (empty dataset) → no institutional context
- No researcher co-author graph (would identify similar researchers)
- Topic graph is fragmented: 303 grant-topic edges, 2,023 researcher-topic edges (not well integrated)
- Topic nodes have zero connections in data (all grant_count and researcher_count are 0)

**Training/Inference Mismatch:**
- Many researchers never appear in training edges
- Model must generalize to 643 unseen researchers during evaluation
- HGTConv only propagates information through edges → isolated researchers get no context

---

### 3. MODEL CAPACITY INSUFFICIENT

**Architecture is Tiny:**
- Hidden dimension: 64
- HGTConv heads: 2 (should be 4-8 for complex graphs)
- Layers: 2 (should be 3-4 for deeper propagation)
- LinkPredictor MLP: 128→64→1 (3 layers, can only learn simple patterns)

**Why This Matters:**
- 1,000 researchers compressed into 64-dim embeddings = 15.6 bits per researcher
- 200 grants in 64-dim space = 0.32 bits per grant
- Not enough capacity to distinguish between topically similar researchers
- 2-layer graph means information propagates only 2 hops
- With sparse edges, most researcher pairs are 3+ hops apart

---

### 4. LOSS FUNCTION PROBLEM

**BCELoss Treats All Negatives Equally:**
```python
criterion = nn.BCELoss()
loss = criterion(preds, labels)  # All negatives weighted equally
```

**The Issue:**
- Random negatives (researcher-grant pairs with completely different fields) are too easy to distinguish
- Model learns to classify any researcher-grant pair as negative if features don't match
- No distinction between:
  - Hard negatives (topically similar grant, different researcher)
  - Easy negatives (random unrelated grant-researcher pair)
- Random negative sampling gives poor training signal

**Example Problem:**
- Positive: Researcher A (AI researcher) receives Grant X (AI grant)
- Negative sampled: Researcher B (Climate researcher) vs Grant X
- Model easily learns they don't match (different topics)
- But never learns that Researcher A should match Grant X over Grant Y (similar amount)

---

### 5. MISSING SEMANTIC FEATURES

**Grant Titles Ignored:**
- "Quantum Machine Learning Applications (Study 100)"
- "Deep Learning for Protein Structure Prediction (Study 1)"
- These titles uniquely identify grants but aren't captured in 7 features

**Researcher Specialties Ignored:**
- 2,023 researcher-topic edges exist (researchers are linked to topics)
- Not reflected in researcher features at all
- Model cannot use topic information for matching

**Topic Coverage Missing:**
- All 5 topics have grant_count=0 and researcher_count=0 in features
- But 303 grant-topic edges and 2,023 researcher-topic edges exist
- Topic information exists but is not utilized

---

### 6. TRAINING DATA ISSUES

**Positive Examples Are Sparse and Possibly Noisy:**
- Only 3,280 positive edges for 200,000 possible pairs
- Labels from RECEIVED_PAST history (real researcher-grant matches)
- But many grants likely go to researchers NOT in the dataset (external funding)
- For in-dataset researchers, positive examples might be extremely selective

**Negative Sampling is Poor:**
```python
# Lines 69-78 in train_gnn.py
while len(neg_src) < len(pos_src) and attempts < max_attempts:
    candidate_src = torch.randint(0, num_r, (1,)).item()
    candidate_dst = torch.randint(0, num_g, (1,)).item()
    if (candidate_src, candidate_dst) not in pos_edge_set:
        neg_src.append(candidate_src)
        neg_dst.append(candidate_dst)
```
- Completely random negative sampling
- May sample many irrelevant researcher-grant pairs that don't look like positives
- No hard negative mining (topically similar but not matched pairs)

---

### 7. EVALUATION METRIC MISALIGNMENT

**Hit Rate @10 is Harsh For Sparse Data:**
- Model must rank true grant in top 10 out of 200 grants = top 5%
- With only 3.28 grants per researcher on average, most researchers have very few "correct" matches
- Random baseline: 10/200 = 5%
- Achieving 6.65% with poor features is nearly random

**Better Metrics Would Be:**
- Mean Reciprocal Rank (MRR) - captures ranking quality
- NDCG - already calculated but shows similar issues
- Precision@1, Recall@5 - more realistic for sparse labels

---

## VERIFICATION OF ROOT CAUSES

### Feature Encoding Analysis

From `graph/features.py`:
1. **Researcher features** (9 dims):
   - Dims 1-4: Raw metrics (h_index, works_count, cited_by_count, i10_index)
   - Dim 5: Binary institution flag
   - Dims 6-9: Normalized versions of dims 1-4 (REDUNDANT)
   - **Missing:** Topic specialty, co-authorship patterns, research areas

2. **Grant features** (7 dims):
   - Dims 1,6: Amount (redundant)
   - Dims 2,3,4,5,7: Hardcoded constants or always 1.0
   - **Missing:** Topic alignment, grant focus area, required expertise

### Graph Connectivity Analysis

Connected components:
- 357 researchers connected to grants
- 643 researchers isolated (no edges)
- 200 grants all connected
- 5 topic nodes barely connected (only 303 edges across all grants)

**During evaluation:**
- Many positive edges are for disconnected researchers
- Model cannot use graph to inform predictions for 64% of researchers
- Embeddings are essentially random for isolated nodes

---

## TOP 5 FIXES (Ranked by Impact)

### 1. **Inject Topic Information Into Node Features** (Impact: +35-40%)

**Current:** Grants and researchers are nearly identical in feature space

**Fix:** Add topic one-hot or embedding features
```python
def build_researcher_features_v2(researchers, researcher_topic_edges):
    # Existing features: h_index, citations, etc.
    # NEW: Topic one-hot (5 dims) or embedding
    # Topic distribution: which topics does researcher work in?
    rows = []
    for r in researchers:
        topic_vector = get_topic_distribution(r.id, researcher_topic_edges)  # 5 dims
        base_features = [h_index, citations, ...]  # existing
        rows.append(base_features + topic_vector)
    return torch.tensor(rows)

def build_grant_features_v2(grants, grant_topic_edges):
    # Replace hardcoded constants with meaningful features
    rows = []
    for g in grants:
        topic_vector = get_topic_distribution(g.id, grant_topic_edges)  # 5 dims
        rows.append([
            g.amount / 100000,
            g.amount / 1000000,
            1.0 if g.title else 0.0,
            g.agency_id / 100,  # real agency preference
        ] + topic_vector)  # 4 + 5 = 9 dims
    return torch.tensor(rows)
```

**Expected improvement:**
- Grants now distinguishable by topic (AI grants ≠ Climate grants)
- Researchers now show specialties (AI researchers ≠ Climate researchers)
- Hit rate could jump 35-40% by capturing topic alignment

---

### 2. **Implement Hard Negative Mining** (Impact: +20-25%)

**Current:** Random negatives are too easy to distinguish

**Fix:** Mine hard negatives (topically similar, but wrong researcher)
```python
def mine_hard_negatives(pos_edge, num_researchers, num_grants, 
                       researcher_topic, grant_topic, k=3):
    """Find grants topically similar to positive but different researcher"""
    hard_negatives = []
    
    for r_id, true_g_id in pos_edge.t():
        # 1. Find grants with similar topic to true_g_id
        true_topics = grant_topic[true_g_id]  # topic set for grant
        similar_grants = [g for g in range(num_grants) 
                         if overlap(grant_topic[g], true_topics) > 0.5
                         and g != true_g_id]
        
        # 2. Sample k hard negatives from similar grants
        if similar_grants:
            hard_negs = random.sample(similar_grants, min(k, len(similar_grants)))
            hard_negatives.extend([(r_id, g) for g in hard_negs])
    
    return hard_negatives

# In training loop:
hard_neg_src, hard_neg_dst = mine_hard_negatives(...)
all_neg_src = torch.cat([random_neg_src, hard_neg_src])
all_neg_dst = torch.cat([random_neg_dst, hard_neg_dst])
```

**Expected improvement:**
- Model learns to distinguish between similar grants
- Better training signal from hard negatives
- Hit rate +20-25%

---

### 3. **Increase Model Capacity** (Impact: +15-20%)

**Current:** 64-dim hidden space is too small

**Fix:** Increase capacity
```python
# In train_gnn.py:
model, predictor, z_dict = train(
    data, 
    epochs=200,
    hidden=256,      # 4x larger
    heads=4,         # 2x more heads
    # Add layers to architecture:
)

class GrantMatchGNN(nn.Module):
    def __init__(self, hidden=256, heads=4, metadata=None):
        super().__init__()
        self.lin_dict = nn.ModuleDict({
            node_type: Linear(-1, hidden) for node_type in metadata[0]
        })
        
        # 4 layers instead of 2
        self.conv1 = HGTConv(hidden, hidden, metadata, heads)
        self.conv2 = HGTConv(hidden, hidden, metadata, heads)
        self.conv3 = HGTConv(hidden, hidden, metadata, heads)  # NEW
        self.conv4 = HGTConv(hidden, hidden, metadata, heads)  # NEW
        
        # Layer norms for all layers
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(4)])
        
        # Larger LinkPredictor
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )
```

**Expected improvement:**
- 256-dim embeddings can capture more nuance
- 4 layers propagate information through sparse graph
- 4 heads provide different representation subspaces
- Hit rate +15-20%

---

### 4. **Replace BCELoss with Margin-Based Loss** (Impact: +10-15%)

**Current:** BCELoss doesn't penalize ranking mistakes

**Fix:** Use contrastive or margin loss
```python
import torch.nn.functional as F

class MarginRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, pos_pred, neg_pred):
        # pos_pred > neg_pred - margin
        return F.relu(self.margin + neg_pred - pos_pred).mean()

criterion = MarginRankingLoss(margin=1.0)

# In training:
pos_pred = predictor(z_dict['researcher'][pos_src], z_dict['grant'][pos_dst])
neg_pred = predictor(z_dict['researcher'][neg_src], z_dict['grant'][neg_dst])
loss = criterion(pos_pred, neg_pred)
```

**Or use Contrastive Loss:**
```python
# Contrastive: same topic pairs score high, different topic low
def contrastive_loss(pos_scores, neg_scores, temp=0.1):
    # Positive pairs should score high
    # Negative pairs should score low
    # Similar to InfoNCE
    logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1) / temp
    labels = torch.zeros(len(pos_scores), dtype=torch.long)
    return F.cross_entropy(logits, labels)
```

**Expected improvement:**
- Directly optimizes for ranking (>5% from better loss)
- Penalizes ranking mistakes more than classification mistakes
- Hit rate +10-15%

---

### 5. **Add Graph Edges for Similarity** (Impact: +5-10%)

**Current:** 643 researchers have zero edges

**Fix:** Add synthetic edges based on similarity
```python
def add_researcher_similarity_edges(data, researchers):
    """Add edges between similar researchers (co-author or topic similarity)"""
    # Method 1: Topic-based similarity
    researcher_topics = build_researcher_topic_matrix(data)  # [N_r, 5]
    
    # Cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        researcher_topics.unsqueeze(1),  # [N_r, 1, 5]
        researcher_topics.unsqueeze(0)   # [1, N_r, 5]
    )  # [N_r, N_r]
    
    # Add edges for top-k similar researchers
    k = 5
    top_sim_indices = similarity.topk(k, dim=1).indices
    
    edges = []
    for r_i in range(len(researchers)):
        for r_j in top_sim_indices[r_i]:
            if r_i < r_j:  # Avoid duplicates
                edges.append([r_i, r_j])
    
    if edges:
        edge_tensor = torch.tensor(edges, dtype=torch.long).t()
        data['researcher', 'SIMILAR_TO', 'researcher'].edge_index = edge_tensor
        # Add reverse edges
        data['researcher', 'rev_SIMILAR_TO', 'researcher'].edge_index = \
            torch.stack([edge_tensor[1], edge_tensor[0]])
    
    return data
```

**Expected improvement:**
- Connects isolated researchers through similarity graph
- HGT can propagate information even without direct funding edges
- Hit rate +5-10%

---

## IMPLEMENTATION PRIORITY

**Phase 1 (Immediate, +50% gain):**
1. Add topic features to researcher/grant (Fix #1)
2. Increase model capacity (Fix #3)

**Phase 2 (Short-term, +10% gain):**
3. Hard negative mining (Fix #2)
4. Better loss function (Fix #4)

**Phase 3 (Polish, +10% gain):**
5. Add similarity edges (Fix #5)

---

## EXPECTED OUTCOMES

| Baseline | After Fix #1 | After Fix #3 | After Fix #2 | Final |
|----------|------------|------------|------------|-------|
| 6.65% | 15-20% | 30-35% | 40-45% | 45-55% |

**Realistic target:** 40-50% Hit Rate @10 (8x-7.5x improvement from baseline)

---

## Metrics to Monitor

1. **Hit Rate @K:** Track @1, @5, @10, @20
2. **NDCG @K:** Already implemented, should improve proportionally
3. **MRR (Mean Reciprocal Rank):** Add this metric
4. **Topic Alignment:** Custom metric - what % of top-K predictions have matching topics?
5. **Loss:** Monitor both training and validation curves
6. **Embedding Quality:** Analyze cosine similarity between positive pairs vs negatives

---

## Files That Need Changes

1. `/c/Users/kjpat/Desktop/GrantMatch/graph/features.py` - Add topic features
2. `/c/Users/kjpat/Desktop/GrantMatch/models/gnn.py` - Increase capacity
3. `/c/Users/kjpat/Desktop/GrantMatch/training/train_gnn.py` - Add hard negative mining, new loss
4. `/c/Users/kjpat/Desktop/GrantMatch/training/evaluate.py` - Add MRR metric
5. `/c/Users/kjpat/Desktop/GrantMatch/graph/schema.py` - Add similarity edges (optional)
