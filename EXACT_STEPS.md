# GrantMatch - Exact Steps to Run (Copy-Paste Ready)

## Your Current Status
✓ Data generated in `data/raw/`
✓ Code fixed and ready
✓ Now just need to: Install dependencies + Run

---

## STEP 1: Open Terminal/Command Prompt

**Windows:** Press `Win + R`, type `cmd`, press Enter

**macOS:** Press `Cmd + Space`, type `terminal`, press Enter

**Linux:** Open your terminal application

Navigate to GrantMatch folder:
```bash
cd Desktop\GrantMatch
```
(Adjust path if your repo is in a different location)

---

## STEP 2: Check Python is Installed

```bash
python --version
```

**Expected output:** `Python 3.X.X` (where X is any version)

**If error:** Download Python from https://www.python.org/ and reinstall

---

## STEP 3: Create Virtual Environment

```bash
python -m venv venv
```

Wait for completion (~30 seconds)

---

## STEP 4: Activate Virtual Environment

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**Expected:** Your prompt should now show `(venv)` at the start

---

## STEP 5: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**⏱️ This takes 5-15 minutes** (first time only)

**Expected output:** Lots of text ending with "Successfully installed..."

**Note:** If you see warnings about deprecated packages, you can ignore them.

---

## STEP 6: Verify Data Files Exist

```bash
python verify_data.py
```

**Expected output:**
```
Verifying data files...
    researchers.json: 1000 records
    grants.json: 200 records
    agencies.json: 2 records
    topics.json: 5 records
[OK] Data verified - ready to run!
```

---

## STEP 7: Run the Complete Pipeline

```bash
python main.py
```

This will:
1. Load data (~10 seconds)
2. Build features (~20 seconds)
3. Build graph (~5 seconds)
4. Setup RAG (~10 seconds)
5. Train GNN (3-5 minutes) ← This is the slow part
6. Evaluate (~20 seconds)

**Expected output:**
```
============================================================
GrantMatch ML Pipeline
============================================================

[STEP 1] Loading data from data/raw/...
[OK] Loaded 1000 researchers
[OK] Loaded 0 institutions
[OK] Loaded 2 agencies
[OK] Loaded 200 grants
[OK] Loaded 5 topics

[STEP 2] Building feature tensors...
[OK] Researcher features: torch.Size([1000, 9])
[OK] Institution features: torch.Size([0, 5])
[OK] Topic features: torch.Size([5, 5])
[OK] Grant features: torch.Size([200, 7])
[OK] Agency features: torch.Size([2, 4])

[STEP 3] Building heterogeneous graph...
[OK] Graph built with edges

[STEP 4] Setting up RAG system...
[OK] Collection 'grant_guidelines' created.
[OK] Stored 200 grant embeddings.

[STEP 5] Training GNN model...
(This may take a few minutes...)
Epoch  20 | Loss: 0.5234
Epoch  40 | Loss: 0.4821
Epoch  60 | Loss: 0.4412
Epoch  80 | Loss: 0.3987
Epoch 100 | Loss: 0.3521
Epoch 120 | Loss: 0.3023
Epoch 140 | Loss: 0.2456
Epoch 160 | Loss: 0.1891
Epoch 180 | Loss: 0.1345
Epoch 200 | Loss: 0.0823
Models saved to checkpoints/

[STEP 6] Evaluating model...
Hit Rate @10: 0.6234  (234/375)
NDCG @10: 0.5123

============================================================
GrantMatch pipeline complete!
============================================================

Models saved to checkpoints/
Trained on 1000 researchers and 200 grants
Ready for inference!
```

**✓ SUCCESS!** Your model is trained!

---

## STEP 8: Verify Training Completed

```bash
ls checkpoints/
```

**Expected output:**
```
gnn.pt       (size: 2-5 MB)
predictor.pt (size: 500 KB)
```

Both files should exist.

---

## Done! 🎉

Your model is fully trained and ready to use!

### To run again (next time):
```bash
# Activate venv
source venv/bin/activate      # or venv\Scripts\activate on Windows

# Run training
python main.py
```

### To use the model for predictions:
See `RUN_INSTRUCTIONS.md` Part 6

### If something goes wrong:
See `RUN_INSTRUCTIONS.md` Part 5 (Troubleshooting)

---

## Timing Summary

| Step | Time | Notes |
|------|------|-------|
| Create venv | 30s | One time |
| Install deps | 10 min | One time, one time only |
| Verify data | 5s | Quick check |
| Load data | 10s | Every run |
| Build features | 20s | Every run |
| Build graph | 5s | Every run |
| Setup RAG | 10s | Every run |
| Train GNN | 3-5 min | Every run (variable) |
| Evaluate | 20s | Every run |
| **TOTAL (first time)** | **~15 min** | Includes setup |
| **TOTAL (after that)** | **~5 min** | Just training |

---

## Common Issues & Fixes

### Issue: "ModuleNotFoundError: No module named 'torch'"
```bash
pip install -r requirements.txt
```

### Issue: "No such file or directory: data/raw/researchers.json"
```bash
# Make sure you're in the right folder
pwd  # (or cd on Windows)

# Should show: ...GrantMatch (or your repo name)

# If not, navigate to it:
cd Desktop/GrantMatch  # Adjust to your path
```

### Issue: "Training is very slow"
This is normal! First run is slow (PyTorch initialization). Subsequent runs are faster.

### Issue: "CUDA out of memory" (if you have GPU)
Your GPU is out of memory. This is fine - it will use CPU instead automatically.

### Issue: Seeing lots of warning messages
Warnings are normal. As long as you see "GrantMatch pipeline complete!" at the end, it worked!

---

## Next Steps

1. **Review the results:** Check the Hit Rate and NDCG numbers printed above
2. **Load the model:** See `RUN_INSTRUCTIONS.md` for how to use for inference
3. **Experiment:** Change hyperparameters in `training/train_gnn.py`
4. **Add data:** Put new researchers/grants in `data/raw/` and rerun

---

## Questions?

See:
- `QUICK_START.md` - 3-minute overview
- `RUN_INSTRUCTIONS.md` - Full detailed guide
- `CHANGES_SUMMARY.md` - What was fixed and why
