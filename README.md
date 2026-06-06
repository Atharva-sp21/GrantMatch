# GrantMatch

GrantMatch is a researcher-to-grant recommendation pipeline built on real API data from NIH Reporter and OpenAlex. The rewrite removes the old Crossref, RAG, and TF-IDF fallback paths from the training flow and keeps the project centered on one data contract:

- NIH Reporter for grant records
- OpenAlex Authors API for researchers and affiliations
- Sentence-transformer embeddings for semantic features
- HGT-based link prediction for researcher -> grant matching

## Layout

- `ingestion/` pulls and normalizes API data
- `graph/` builds features and the heterogeneous graph
- `models/` contains the GNN and link predictor
- `training/` handles splitting, training, validation, and evaluation
- `main.py` runs the end-to-end training pipeline

## Run Order

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Regenerate data from the APIs:

```bash
python ingestion/run_pipeline.py
```

3. Train and evaluate the model:

```bash
python main.py
```

## What Gets Produced

The ingestion pipeline writes clean JSON and CSV files into `data/raw/`:

- `researchers.json`
- `institutions.json`
- `agencies.json`
- `grants.json`
- `topics.json`
- `affiliated.csv`
- `researches.csv`
- `received_past.csv`
- `funds_topic.csv`
- `provides.csv`

Training saves the best checkpoints to `checkpoints/`.

## Notes

- The project is designed to run on CPU first.
- `sentence-transformers/all-MiniLM-L6-v2` is used locally for text embeddings.
- The primary prediction task is researcher -> grant recommendation.
- If `received_past.csv` is empty, the trainer will stop instead of training on bad labels.

## Troubleshooting

- If data files are missing, run `python ingestion/run_pipeline.py` first.
- If dependencies are missing, reinstall with `pip install -r requirements.txt`.
- If embeddings cannot download on first run, check network access or pre-cache the model.
