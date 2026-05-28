# Data Ingestion Pipeline

This folder contains scripts to populate the GrantMatch ML pipeline with data from free public APIs.

## Setup

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Run in order:**
```bash
python fetch_researchers.py      # ~5-10 min (OpenAlex)
python fetch_grants.py           # ~5 min (NIH + NSF)
python build_files.py            # ~1 min (transforms to JSON)
python build_edges.py            # ~1 min (builds edge CSVs)
```

## What Each Script Does

### 1. `fetch_researchers.py`
- Fetches researchers from OpenAlex API
- Extracts institution affiliations
- Assigns internal integer IDs (0, 1, 2, ...)
- Outputs: `researchers_raw.json`, `institution_mapping.json`

### 2. `fetch_grants.py`
- Fetches grants from NIH Reporter and NSF APIs
- Builds agency mapping
- Outputs: `grants_raw.json`, `agency_mapping.json`

### 3. `build_files.py`
- Transforms raw data into final JSON format
- **Critical**: Ensures integer IDs match array indices
- Derives topics deterministically from OpenAlex concepts and grant text
- Outputs: `researchers.json`, `institutions.json`, `agencies.json`, `grants.json`, `topics.json`

### 4. `build_edges.py`
- Creates all 5 edge relationship types
- Generates CSV files with source/target integer IDs using API-derived name/topic matches
- Outputs: `affiliated.csv`, `researches.csv`, `received_past.csv`, `funds_topic.csv`, `provides.csv`

## Data Files

All files go into `data/raw/`:

**JSON Files:**
- `researchers.json` - [N_r, 9] features
- `institutions.json` - [N_i, 5] features
- `agencies.json` - [N_a, 4] features
- `grants.json` - [N_g, 7] features
- `topics.json` - [N_t, 5] features

**Edge CSVs:**
- `affiliated.csv` - researcher → institution
- `researches.csv` - researcher → topic
- `received_past.csv` - researcher → grant
- `funds_topic.csv` - grant → topic
- `provides.csv` - agency → grant

## Critical: Integer ID Consistency

Every JSON file has an `id` field that **must equal the array index**:
```json
[
  { "id": 0, "name": "Alice" },     // index 0, id: 0 ✓
  { "id": 1, "name": "Bob" },       // index 1, id: 1 ✓
  { "id": 2, "name": "Charlie" }    // index 2, id: 2 ✓
]
```

The edge CSVs reference these same integers:
```csv
source_id,target_id
0,5        # researcher 0 received grant 5
1,2        # researcher 1 researches topic 2
```

The graph loads directly from these integer IDs as array indices.

## Notes

- All APIs are free - no authentication required
- Rate limiting is built in (0.1s between API calls)
- The pipeline is strict: if an API returns no data, it fails instead of fabricating sample records
- Topics and edges are derived deterministically from API fields, not random sampling
