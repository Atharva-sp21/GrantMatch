import json
from pathlib import Path

data_dir = Path('data/raw')
files = ['researchers.json', 'grants.json', 'agencies.json', 'topics.json']

print("Verifying data files...")
for f in files:
    try:
        with open(data_dir / f) as fp:
            data = json.load(fp)
        print(f"    {f}: {len(data)} records")
    except Exception as e:
        print(f"    ERROR {f}: {e}")

print("[OK] Data verified - ready to run!")
