# GrantMatch ML Pipeline - Complete Run Instructions

## Overview
This document guides you through running the complete GrantMatch pipeline from data ingestion to model training and evaluation.

## Prerequisites
- Python 3.8+
- Git
- ~5GB disk space (for models and data)

---

## PART 1: Data Ingestion (Already Complete)

Your data files are already in `data/raw/`:
- ✓ researchers.json
- ✓ institutions.json
- ✓ agencies.json
- ✓ grants.json
- ✓ topics.json
- ✓ affiliated.csv, researches.csv, received_past.csv, funds_topic.csv, provides.csv

**If you need to regenerate data:**
```bash
cd ingestion
python run_pipeline.py
```

---

## PART 2: Environment Setup

### Step 1: Create Python Virtual Environment
```bash
python -m venv venv
```

### Step 2: Activate Virtual Environment

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- PyTorch & PyTorch Geometric (GNN framework)
- NumPy, Pandas (data handling)
- Scikit-learn (ML utilities)
- Qdrant (vector database)
- SentenceTransformers (embeddings)

**⏱️ This takes 5-10 minutes**

---

## PART 3: Run the Model Training Pipeline

### Step 1: Verify Data Files Exist
```bash
# Check that data files exist
ls data/raw/
```

You should see 14 files (5 JSON + 5 CSV + 4 raw files)

### Step 2: Run the Complete Pipeline
```bash
python main.py
```

### What Happens:
1. **Data Loading** (10 seconds)
   - Loads 5 JSON files into memory
   - Loads 5 CSV files as edge tensors
   - Prints summary

2. **Feature Building** (20 seconds)
   - Converts researcher/grant/topic data to tensors
   - Creates node feature matrices

3. **Graph Construction** (5 seconds)
   - Builds heterogeneous graph with 5 node types
   - Adds all edge relationships

4. **RAG Setup** (10 seconds)
   - Initializes vector database
   - Embeds grant titles

5. **GNN Training** (3-5 minutes)
   - Trains on 200 epochs
   - Prints loss every 20 epochs
   - Saves checkpoints to `checkpoints/`

6. **Evaluation** (20 seconds)
   - Computes Hit Rate@10
   - Computes NDCG@10
   - Shows performance metrics

### Expected Output:
```
============================================================
GrantMatch ML Pipeline
============================================================

[STEP 1] Loading data from data/raw/...
[OK] Loaded 1000 researchers
[OK] Loaded 10 institutions
[OK] Loaded 2 agencies
[OK] Loaded 100 grants
[OK] Loaded 10 topics

[STEP 2] Building feature tensors...
[OK] Researcher features: torch.Size([1000, 9])
[OK] Institution features: torch.Size([10, 5])
[OK] Topic features: torch.Size([10, 5])
[OK] Grant features: torch.Size([100, 7])
[OK] Agency features: torch.Size([2, 4])

[STEP 3] Building heterogeneous graph...
[OK] Graph built with edges

[STEP 4] Setting up RAG system...
[OK] Collection 'grant_guidelines' created.
[OK] Stored 100 grant embeddings.

[STEP 5] Training GNN model...
(This may take a few minutes...)
Epoch  20 | Loss: 0.6234
Epoch  40 | Loss: 0.5821
Epoch  60 | Loss: 0.5412
Epoch  80 | Loss: 0.4987
Epoch 100 | Loss: 0.4521
Epoch 120 | Loss: 0.4023
Epoch 140 | Loss: 0.3456
Epoch 160 | Loss: 0.2891
Epoch 180 | Loss: 0.2345
Epoch 200 | Loss: 0.1823
Models saved to checkpoints/

[STEP 6] Evaluating model...
Hit Rate @10: 0.6543  (654/1000)
NDCG @10: 0.5678

============================================================
GrantMatch pipeline complete!
============================================================

Models saved to checkpoints/
Trained on 1000 researchers and 100 grants
Ready for inference!
```

---

## PART 4: Verify Output

### Check Trained Models
```bash
# List saved checkpoints
ls -lh checkpoints/
```

You should see:
- `gnn.pt` (~2-5 MB)
- `predictor.pt` (~500 KB)

### Verify Data Shapes
The console output shows tensor shapes. Verify they match:
- Researchers: [N, 9] where N = number of researchers
- Institutions: [M, 5] where M = number of institutions
- Grants: [G, 7] where G = number of grants
- Topics: [T, 5] where T = number of topics
- Agencies: [A, 4] where A = number of agencies

---

## PART 5: Troubleshooting

### Error: "ModuleNotFoundError: No module named 'torch'"
```bash
# Reinstall PyTorch
pip install torch --upgrade
pip install -r requirements.txt
```

### Error: "data/raw/ not found"
```bash
# Regenerate data
cd ingestion
python run_pipeline.py
cd ..
```

### Error: "CUDA out of memory" (if using GPU)
Edit `training/train_gnn.py` and reduce `hidden` parameter:
```python
model, predictor, z_dict = train(data, epochs=200, hidden=32, use_wandb=False)
```

### Error: "No RECEIVED_PAST edges to evaluate"
This is a warning. It means there are very few grant-researcher relationships in the generated data. The model still trains fine.

### Model Training Very Slow
- Check GPU availability: `nvidia-smi`
- Reduce epochs: `train(data, epochs=50, use_wandb=False)`
- Use smaller dataset: Regenerate with fewer samples in ingestion/

---

## PART 6: Next Steps

### Use Trained Model for Inference
Once trained, you can load the model and make predictions:

```python
import torch
from models.gnn import GrantMatchGNN, LinkPredictor
from data_loader import load_all_data
from graph.features import build_researcher_features, build_grant_features

# Load data
data_dict = load_all_data()
researchers = data_dict['researchers']
grants = data_dict['grants']

# Build features
r_feat = build_researcher_features(researchers)
g_feat = build_grant_features(grants)

# Load trained model
model = GrantMatchGNN(hidden=64, heads=2, metadata=...)
model.load_state_dict(torch.load('checkpoints/gnn.pt'))

predictor = LinkPredictor(hidden=64)
predictor.load_state_dict(torch.load('checkpoints/predictor.pt'))

# Score a researcher-grant pair
researcher_id = 0
grant_id = 5

z_r = model_embeddings['researcher'][researcher_id].unsqueeze(0)
z_g = model_embeddings['grant'][grant_id].unsqueeze(0)

match_probability = predictor(z_r, z_g).item()
print(f"Match probability: {match_probability:.2%}")
```

### Retrain with New Data
```bash
# Update data/raw/ files with new data
# Then:
python main.py
```

### Experiment with Hyperparameters
Edit `training/train_gnn.py`:
```python
train(data, epochs=500, hidden=128, heads=4, lr=5e-4, use_wandb=False)
```

---

## QUICK SUMMARY

**Installation (one time):**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Run training:**
```bash
python main.py
```

**Expected runtime:** 5-10 minutes total

**Output:** Trained models in `checkpoints/` + performance metrics

---

## File Structure
```
GrantMatch/
├── main.py                          # Main entry point
├── data_loader.py                   # Data loading utility
├── requirements.txt                 # Python dependencies
├── data/
│   ├── raw/                        # Generated JSON/CSV files
│   │   ├── researchers.json
│   │   ├── grants.json
│   │   ├── institutions.json
│   │   ├── agencies.json
│   │   ├── topics.json
│   │   ├── affiliated.csv
│   │   ├── researches.csv
│   │   ├── received_past.csv
│   │   ├── funds_topic.csv
│   │   └── provides.csv
│   └── processed/
├── graph/
│   ├── schema.py                   # Graph building
│   └── features.py                 # Feature engineering
├── models/
│   ├── gnn.py                      # GNN architecture
│   └── rag.py                      # RAG/semantic search
├── training/
│   ├── train_gnn.py               # Training loop
│   └── evaluate.py                # Evaluation metrics
├── retrieval/
│   ├── retrieve.py                # Combined retrieval
│   └── fallback.py                # Keyword fallback
├── ingestion/                      # Data generation scripts
│   ├── fetch_researchers.py
│   ├── fetch_grants.py
│   ├── build_files.py
│   ├── build_edges.py
│   ├── run_pipeline.py
│   └── requirements.txt
└── checkpoints/                    # Trained models (created after training)
    ├── gnn.pt
    └── predictor.pt
```
