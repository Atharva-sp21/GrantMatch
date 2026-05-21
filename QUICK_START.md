# GrantMatch - Quick Start Guide

## 🚀 Get Running in 3 Minutes

### On Windows:
```bash
setup.bat
```
Then wait for the "Setup Complete!" message, then run:
```bash
python main.py
```

### On macOS/Linux:
```bash
bash setup.sh
python main.py
```

---

## What You'll See

**Installation (first time only):** ~5-10 minutes

**Training (every time):** ~5 minutes
```
[STEP 1] Loading data from data/raw/...
[OK] Loaded 1000 researchers
[OK] Loaded 200 grants
...
[STEP 5] Training GNN model...
Epoch  20 | Loss: 0.6234
Epoch  40 | Loss: 0.5821
...
[STEP 6] Evaluating model...
Hit Rate @10: 0.6543
NDCG @10: 0.5678
```

---

## Output

After training completes:
- **Models saved:** `checkpoints/gnn.pt`, `checkpoints/predictor.pt`
- **Performance metrics:** Hit Rate and NDCG scores printed to console

---

## If Something Goes Wrong

**Error: "ModuleNotFoundError"**
```bash
pip install -r requirements.txt
```

**Error: "No such file or directory: data/raw/"**
```bash
cd ingestion
python run_pipeline.py
cd ..
python main.py
```

**Running very slowly?**
This is normal for the first PyTorch run. Subsequent runs are faster.

---

## Next: Use the Model

See `RUN_INSTRUCTIONS.md` for detailed guidance on:
- Using the trained model for inference
- Retraining with new data
- Modifying hyperparameters
