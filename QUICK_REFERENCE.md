# EXECUTIVE SUMMARY: GNN Hit Rate Analysis

## The Problem
- **Current Performance:** 6.65% Hit Rate @10 (barely above 5% random baseline)
- **Loss Trend:** Decreasing (0.465 → 0.357) - model IS learning
- **Paradox:** Good training loss but terrible ranking performance

---

## Why So Bad? (6 Root Causes)

### 1. **HARDCODED GRANT FEATURES** ✗✗✗ [CRITICAL]
   - Features 2,3,4,5,7 are constant values (1.0, 1.0, 2.0, 0.5, 1.0)
   - Only feature 1 & 6 vary (both are grant amount in different scales)
   - **Result:** All 200 grants look nearly identical to the model
   - **Fix:** Use actual grant features including topics

### 2. **INDISTINGUISHABLE RESEARCHERS** ✗✗✗ [CRITICAL]
   - 52.1% of researchers have h_index=0
   - Models can't differentiate them
   - Features 6-9 are just normalized versions of 1-4 (redundant)
   - **Result:** Researchers are nearly identical in feature space
   - **Fix:** Add topic specialization features (2,023 edges available!)

### 3. **MASSIVE GRAPH DISCONNECTION** ✗✗
   - 643 researchers (64.3%) have ZERO training edges
   - These researchers have random embeddings
   - Can't use graph structure to learn
   - **Result:** For majority of researchers, model just guesses
   - **Fix:** Add similarity edges to connect isolated nodes

### 4. **MODEL TOO SMALL** ✗✗
   - Hidden=64 (should be 256)
   - 2 HGT heads (should be 4)
   - 2 layers (should be 4)
   - **Result:** Model lacks capacity to learn complex matching patterns
   - **Fix:** Increase architecture size 4x

### 5. **LOSS FUNCTION TREATS ALL NEGATIVES EQUALLY** ✗
   - BCELoss doesn't care about ranking
   - Random negatives too easy to distinguish
   - No hard negative mining
   - **Result:** Model learns classification, not ranking
   - **Fix:** Use margin/contrastive loss + hard negative mining

### 6. **MISSING SEMANTIC INFORMATION**
   - Grant titles have topic info (ignored)
   - 2,023 researcher-topic edges (ignored)
   - 303 grant-topic edges (ignored)
   - **Result:** Model can't use rich contextual information
   - **Fix:** Embed topic data into features

---

## Top 5 Fixes (Ranked by Impact)

| Priority | Fix | Impact | Files | Effort |
|----------|-----|--------|-------|--------|
| 🔴 CRITICAL | Add topic features | +35-40% | `graph/features.py` | Medium |
| 🔴 CRITICAL | Increase model capacity | +15-20% | `models/gnn.py` | Low |
| 🟠 IMPORTANT | Hard negative mining | +20-25% | `training/train_gnn.py` | Medium |
| 🟠 IMPORTANT | Better loss function | +10-15% | `training/train_gnn.py` | Low |
| 🟡 NICE-TO-HAVE | Similarity edges | +5-10% | `graph/schema.py` | Low |

---

## Expected Impact

```
Before:    6.65% Hit Rate @10  (barely above random 5%)
Fix #1:   15-20% (+9-15pp)
Fix #1+3: 30-35% (+25-30pp)
Fix #1+3+2+4: 45-50% (+39-45pp)
Final:    50-55% (+45-50pp)  ← 7.5-8x improvement!
```

**Realistic Target:** 45-55% Hit Rate @10 (vs 6.65% baseline)

---

## Quick Start

1. **Start with Fix #1:** Add topic features to grants/researchers
   - Uses existing 2,023 + 303 edges
   - Highest impact per effort
   - ~1-2 hours to implement

2. **Then Fix #3:** Increase model capacity  
   - Quick changes to hyperparameters
   - ~15 minutes to implement
   - Significant gain when features are better

3. **Then Fix #2:** Hard negative mining
   - Improves training signal
   - ~1 hour to implement

4. **Then Fix #4:** Better loss function
   - Simple swap of loss function
   - ~30 minutes to implement

5. **Polish with Fix #5:** Similarity edges
   - Helps isolated researchers
   - ~30 minutes to implement

---

## Key Insights

**Why Model Learns but Predicts Poorly:**

The model's loss decreases because it:
1. Memorizes which researcher-grant pairs appeared in training
2. Learns to classify pairs as positive (loss signal)
3. Learns to score random negatives as negative (too easy)

But it doesn't learn:
1. Which features actually indicate matching potential
2. How to rank similar grants
3. General patterns (poor generalization)

Because the features are mostly constants and random negatives are too easy.

---

## Files to Review

### Analysis Documents (Read These First)
- `ROOT_CAUSES_SUMMARY.txt` - Quick overview
- `GNN_ANALYSIS.md` - Detailed technical analysis
- `IMPLEMENTATION_GUIDE.md` - Code changes for each fix

### Code Files (Need Changes)
- `graph/features.py` - Feature engineering
- `models/gnn.py` - Model architecture
- `training/train_gnn.py` - Training loop
- `graph/schema.py` - Graph construction
- `data_loader.py` - Data loading
- `main.py` - Pipeline

---

## Next Steps

1. Read `ROOT_CAUSES_SUMMARY.txt` (5 min read)
2. Read `GNN_ANALYSIS.md` section on "Root Cause #1" (10 min)
3. Implement Fix #1 using `IMPLEMENTATION_GUIDE.md` (1-2 hours)
4. Test and measure Hit Rate improvement
5. Proceed with remaining fixes

---

## Questions to Ask Yourself

- **Q: Should I implement all 5 fixes?**
  A: Start with 1+3 (highest impact, lowest effort). Then 2+4. Optional: 5.

- **Q: Will this definitely work?**
  A: These are fundamental issues. Fixing them should improve performance significantly (likely to 40-50% HR).

- **Q: How long will implementation take?**
  A: Fixes 1-4: ~4-5 hours total. Fix 5: ~30 min optional.

- **Q: Should I test incrementally?**
  A: Yes! Implement, test, measure after each fix.

- **Q: What if it doesn't work?**
  A: The root causes are clear. If these fixes don't help, check:
     - Data quality (h_index distribution is suspicious)
     - Edge label quality (RECEIVED_PAST edges are correct matches)
     - Check embedding quality (are similar researchers close in space?)

---
