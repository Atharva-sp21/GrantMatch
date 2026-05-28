#!/usr/bin/env python3
"""Run the ingestion pipeline in order."""

import os
import subprocess
import sys


def run_command(command: str, description: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f">> {description}")
    print(f"{'=' * 60}")
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] {description} failed: {exc}")
        return False


def main():
    print("=" * 60)
    print("GrantMatch Data Ingestion Pipeline")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    steps = [
        ("python fetch_researchers.py", "Fetching OpenAlex researchers"),
        ("python fetch_grants.py", "Fetching NIH grants"),
        ("python build_files.py", "Building normalized node files"),
        ("python build_edges.py", "Building graph edge files"),
    ]

    for index, (command, description) in enumerate(steps, start=1):
        if not run_command(command, description):
            print(f"\n[ERROR] Pipeline stopped at step {index}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("[DONE] Ingestion pipeline complete")
    print(f"{'=' * 60}")
    print("Data files are ready in data/raw/")


if __name__ == "__main__":
    main()
