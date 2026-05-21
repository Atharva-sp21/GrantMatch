#!/bin/bash
# Run the entire ingestion pipeline in order

set -e  # Exit on any error

echo "🚀 Starting GrantMatch Data Ingestion Pipeline"
echo ""

echo "Step 1️⃣  Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

echo "Step 2️⃣  Fetching researchers from OpenAlex..."
python fetch_researchers.py
echo ""

echo "Step 3️⃣  Fetching grants from NIH & NSF..."
python fetch_grants.py
echo ""

echo "Step 4️⃣  Building JSON files with integer IDs..."
python build_files.py
echo ""

echo "Step 5️⃣  Building edge CSV files..."
python build_edges.py
echo ""

echo "✅ Pipeline complete!"
echo "📁 Data files ready in data/raw/"
echo ""
echo "Next steps:"
echo "  1. Review data in data/raw/"
echo "  2. Update graph/schema.py to load from these files"
echo "  3. Run python main.py to train the model"
