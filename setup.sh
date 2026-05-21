#!/bin/bash
# GrantMatch Setup & Run Script

set -e

echo "=================================================="
echo "GrantMatch ML Pipeline Setup"
echo "=================================================="

# Step 1: Check Python version
echo ""
echo "[1/4] Checking Python installation..."
if ! command -v python &> /dev/null; then
    echo "[ERROR] Python not found. Please install Python 3.8+"
    exit 1
fi

python_version=$(python --version 2>&1 | awk '{print $2}')
echo "[OK] Python $python_version found"

# Step 2: Create virtual environment
echo ""
echo "[2/4] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "[OK] Virtual environment created"
else
    echo "[OK] Virtual environment already exists"
fi

# Activate venv
source venv/bin/activate

# Step 3: Install dependencies
echo ""
echo "[3/4] Installing dependencies..."
echo "(This may take 5-10 minutes...)"
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "[OK] Dependencies installed"

# Step 4: Verify data
echo ""
echo "[4/4] Verifying data files..."
python << 'VERIFY'
import json
from pathlib import Path

data_dir = Path('data/raw')
files = ['researchers.json', 'grants.json', 'agencies.json', 'topics.json']

for f in files:
    with open(data_dir / f) as fp:
        data = json.load(fp)
    print(f"    {f}: {len(data)} records")

print("[OK] Data verified")
VERIFY

# Step 5: Ready to run
echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "To run the training pipeline:"
echo "  python main.py"
echo ""
echo "Expected runtime: 5-10 minutes"
echo ""
