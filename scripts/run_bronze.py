"""
scripts/run_bronze.py
─────────────────────────────────────────────────────────────────────────────
Entry point for Part A – Bronze Layer.

Run from your project root:
    python3 scripts/run_bronze.py

What this does:
  1. Starts a local SparkSession
  2. Ingests ratings_1.csv    → bronze/netflix/ratings/yyyy/mm/dd/
  3. Ingests movies.json      → bronze/tmdb/movies/yyyy/mm/dd/
  4. Ingests users.parquet    → bronze/netflix/users/yyyy/mm/dd/
  5. Tries ratings_bad.csv   → quarantined to bronze/quarantine/
  6. Re-runs ratings_1.csv   → SKIPPED (idempotency check)
  7. Prints a summary report
"""

import os
import sys

# ── Make sure project root is on the path ────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.spark_session import get_spark
from scripts.bronze_ingestion import (
    RATINGS_SCHEMA,
    MOVIES_SCHEMA,
    USERS_SCHEMA,
    ingest_to_bronze,
    ingest_directory,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW  = os.path.join(PROJECT_ROOT, "data", "raw")
BASE = PROJECT_ROOT


def print_banner(text: str) -> None:
    print("\n" + "─" * 60)
    print(f"  {text}")
    print("─" * 60)


def main() -> None:
    spark = get_spark("BronzeIngestion")

    print_banner("PART A – BRONZE LAYER INGESTION")

    # ── Step 1: Ratings CSV ───────────────────────────────────────────────────
    print("\n[1/5] Ingesting ratings_1.csv (CSV format)...")
    path_ratings = ingest_to_bronze(
        spark,
        source_file_path=os.path.join(RAW, "ratings_1.csv"),
        source="netflix",
        entity="ratings",
        schema=RATINGS_SCHEMA,
        base_path=BASE,
    )
    if path_ratings:
        print(f"      ✓  Written to: {path_ratings}")

    # ── Step 2: Movies JSON ───────────────────────────────────────────────────
    print("\n[2/5] Ingesting movies.json (JSON format)...")
    path_movies = ingest_to_bronze(
        spark,
        source_file_path=os.path.join(RAW, "movies.json"),
        source="tmdb",
        entity="movies",
        schema=MOVIES_SCHEMA,
        base_path=BASE,
    )
    if path_movies:
        print(f"      ✓  Written to: {path_movies}")

    # ── Step 3: Users Parquet (batch directory ingest) ────────────────────────
    print("\n[3/5] Ingesting users/ directory (Parquet format)...")
    users_dir = os.path.join(RAW, "users")
    paths_users = ingest_directory(
        spark,
        source_dir=users_dir,
        source="netflix",
        entity="users",
        schema=USERS_SCHEMA,
        base_path=BASE,
    )
    for p in paths_users:
        print(f"      ✓  Written to: {p}")

    # ── Step 4: Bad file → quarantine ─────────────────────────────────────────
    print("\n[4/5] Attempting bad file (should be quarantined)...")
    path_bad = ingest_to_bronze(
        spark,
        source_file_path=os.path.join(RAW, "ratings_bad.csv"),
        source="netflix",
        entity="ratings",
        schema=RATINGS_SCHEMA,
        base_path=BASE,
    )
    if path_bad is None:
        print("      ✓  Bad file correctly quarantined or skipped")

    # ── Step 5: Re-run ratings → idempotency check ────────────────────────────
    print("\n[5/5] Re-ingesting ratings_1.csv (should be SKIPPED)...")
    path_repeat = ingest_to_bronze(
        spark,
        source_file_path=os.path.join(RAW, "ratings_1.csv"),
        source="netflix",
        entity="ratings",
        schema=RATINGS_SCHEMA,
        base_path=BASE,
    )
    if path_repeat is None:
        print("      ✓  Duplicate correctly skipped (idempotency works)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_banner("BRONZE INGESTION COMPLETE")
    print("\nFolder structure created:")
    for root, dirs, files in os.walk(os.path.join(BASE, "bronze")):
        level = root.replace(BASE, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            print(f"{'  ' * (level + 1)}{f}")

    print("\nCheck logs/ for:")
    print("  • logs/ingested_file_hashes.json  (idempotency registry)")
    print("  • logs/quarantine_errors.json     (bad file log)")
    print("\nCheck bronze/quarantine/ for the bad file.\n")

    spark.stop()


if __name__ == "__main__":
    main()
