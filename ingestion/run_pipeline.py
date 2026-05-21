#!/usr/bin/env python3
"""
Simple script to run the entire ingestion pipeline in order.
Works on Windows, macOS, and Linux.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*60}")
    print(f">> {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, shell=True, check=True)
        if result.returncode != 0:
            print(f"[ERROR] {description} failed with code {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e}")
        return False

    return True


def main():
    print("=" * 60)
    print("GrantMatch Data Ingestion Pipeline")
    print("=" * 60)

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    steps = [
        ("pip install -q -r requirements.txt", "Installing dependencies"),
        ("python fetch_researchers.py", "Fetching researchers from OpenAlex"),
        ("python fetch_grants.py", "Fetching grants from NIH & NSF"),
        ("python build_files.py", "Building JSON files with integer IDs"),
        ("python build_edges.py", "Building edge CSV files"),
    ]

    completed = 0
    for cmd, description in steps:
        if run_command(cmd, description):
            completed += 1
        else:
            print(f"\n[ERROR] Pipeline stopped at step {completed + 1}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("[DONE] Pipeline complete!")
    print(f"{'='*60}")
    print("Data files ready in ../data/raw/")
    print("\nGenerated files:")
    print("  - researchers.json")
    print("  - institutions.json")
    print("  - agencies.json")
    print("  - grants.json")
    print("  - topics.json")
    print("  - affiliated.csv")
    print("  - researches.csv")
    print("  - received_past.csv")
    print("  - funds_topic.csv")
    print("  - provides.csv")
    print("\nNext steps:")
    print("  1. Review the data files")
    print("  2. Load them in your graph builder")
    print("  3. Train your model!")


if __name__ == "__main__":
    main()
