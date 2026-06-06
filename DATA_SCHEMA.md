# GrantMatch Data Schema

This document reconciles the **Week 2 handoff PDF** (Person B deliverables) with the **ingestion pipeline** (OpenAlex + NIH/NSF) that Person A's code actually consumes.

## Two Valid Formats

| Aspect | PDF handoff | Ingestion (current `data/raw/`) |
|--------|-------------|----------------------------------|
| Entity IDs | Strings: `R001`, `G201` | Integers `0..N-1` matching JSON array index |
| Edge CSV columns | `researcher_id`, `grant_id`, etc. | `source_id`, `target_id` (also accepted) |
| Researcher metrics | `publication_count`, `citation_count` | `works_count`, `cited_by_count` (aliases supported) |
| Grant funding | `funding_amount` | `amount` (alias supported) |
| Grant text for RAG | `guidelines_text` | `title` (alias; full text preferred when available) |
| Institutions | `world_ranking`, `institution_type` | `researcher_count`, `total_works` from OpenAlex affiliations |

`data_loader.py` accepts **either** column naming scheme and maps string IDs to indices when needed.

## Required Files (`data/raw/`)

**JSON:** `researchers.json`, `grants.json`, `institutions.json`, `topics.json`, `agencies.json`

**CSV edges:**

| File | PDF name | Edge type |
|------|----------|-----------|
| `affiliated.csv` | `affiliated_with.csv` | researcher → institution |
| `researches.csv` | `researches.csv` | researcher → topic |
| `received_past.csv` | `received_past.csv` | researcher → grant (training labels) |
| `funds_topic.csv` | `funds_topic.csv` | grant → topic |
| `provides.csv` | `provides.csv` | agency → grant |

## Node Feature Dimensions (Person A)

| Node | Dims | Source |
|------|------|--------|
| researcher | 9 | bibliometrics + topic distribution from `researches.csv` |
| institution | 5 | affiliation aggregates |
| topic | 5 | edge-derived counts + field encoding |
| grant | 7 | amount, agency, duration, title + topic distribution from `funds_topic.csv` |
| agency | 4 | funding totals and grant counts |

## Common Discrepancies (fixed)

1. **Empty `institutions.json`** — OpenAlex API field was `last_known_institutions` (plural); fetch script updated in `ingestion/fetch_researchers.py`. Re-run ingestion to populate affiliations.
2. **`topics.json` counts all zero** — counts are derived from edge CSVs at load time and when running `build_edges.py`.
3. **PDF field names in CSVs** — supported via `EDGE_COLUMN_ALIASES` in `data_loader.py`.
4. **Hardcoded grant features** — removed; features use amount, agency, title, and topic edges (`graph/features.py`).
5. **Duplicate `LinkPredictor`** — removed from `models/gnn.py` (was overriding the 256-dim head with a 64-dim one).

## Regenerate Data (Person B)

```bash
cd ingestion
pip install -r requirements.txt
python run_pipeline.py
```

## Train (Person A)

```bash
python main.py
```
