"""
tests/test_bronze.py
Unit tests for Part A - Bronze Layer (Q1-Q5).

Run:  python3 -m pytest tests/test_bronze.py -v
"""

import json
import os
import shutil
import sys
import tempfile

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql import SparkSession
from scripts.bronze_ingestion import (
    HASH_REGISTRY_PATH,
    ERROR_LOG_PATH,
    RATINGS_SCHEMA,
    MOVIES_SCHEMA,
    USERS_SCHEMA,
    compute_file_hash,
    get_bronze_path,
    ingest_to_bronze,
    ingest_directory,
    is_already_ingested,
    register_ingested_file,
)


# ── One SparkSession for the whole test session ───────────────────────────────
@pytest.fixture(scope="session")
def spark():
    s = (
        SparkSession.builder
        .appName("BronzeTests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "1g")
        .master("local[1]")
        .getOrCreate()
    )
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


# ── Fresh temp workspace + clean hash registry per test ──────────────────────
@pytest.fixture(autouse=True)
def clean_registry():
    """Delete the hash registry and error log before every test so tests are isolated."""
    for path in [HASH_REGISTRY_PATH, ERROR_LOG_PATH]:
        if os.path.exists(path):
            os.remove(path)
    yield
    # cleanup after too
    for path in [HASH_REGISTRY_PATH, ERROR_LOG_PATH]:
        if os.path.exists(path):
            os.remove(path)


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_ratings_csv(path: str, rows: int = 3) -> None:
    pd.DataFrame({
        "user_id":  list(range(1, rows + 1)),
        "movie_id": list(range(10, 10 + rows)),
        "rating":   [4.0] * rows,
        "rated_at": ["2024-01-01 10:00:00"] * rows,
    }).to_csv(path, index=False)


def make_movies_json(path: str) -> None:
    data = [
        {"movie_id": 1, "title": "Test Movie",   "genres": "Action",
         "year": 2020, "language": "en", "country": "US"},
        {"movie_id": 2, "title": "Another Film", "genres": "Drama",
         "year": 2021, "language": "fr", "country": "FR"},
    ]
    with open(path, "w") as f:
        json.dump(data, f)


def make_users_parquet(path: str) -> None:
    pd.DataFrame({
        "user_id":     [1, 2],
        "signup_date": ["2023-01-01", "2023-06-15"],
        "country":     ["US", "GB"],
        "plan":        ["premium", "basic"],
    }).to_parquet(path, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Q1 - FOLDER STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def test_q1_folder_structure():
    """Path must follow bronze/{source}/{entity}/{yyyy}/{mm}/{dd}/"""
    from datetime import datetime
    with tempfile.TemporaryDirectory() as tmp:
        dt   = datetime(2024, 3, 15)
        path = get_bronze_path(tmp, "netflix", "ratings", dt)
        expected = os.path.join(tmp, "bronze", "netflix", "ratings", "2024", "03", "15")
        assert path == expected
        assert os.path.isdir(path)


def test_q1_default_date_uses_today():
    """When no date supplied, today's date must appear in the path."""
    from datetime import datetime
    with tempfile.TemporaryDirectory() as tmp:
        path  = get_bronze_path(tmp, "netflix", "ratings")
        today = datetime.utcnow()
        assert today.strftime("%Y") in path
        assert today.strftime("%m") in path
        assert today.strftime("%d") in path


# ══════════════════════════════════════════════════════════════════════════════
# Q2 - MULTI-FORMAT INGESTION
# ══════════════════════════════════════════════════════════════════════════════

def test_q2_csv_ingestion(spark):
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        csv_file = os.path.join(src, "ratings.csv")
        make_ratings_csv(csv_file, rows=3)

        out = ingest_to_bronze(spark, csv_file, "netflix", "ratings",
                               RATINGS_SCHEMA, base_path=dst)
        assert out is not None, f"ingest_to_bronze returned None - check logs"
        df = spark.read.parquet(out)
        assert df.count() == 3
        assert "user_id"  in df.columns
        assert "movie_id" in df.columns
        assert "rating"   in df.columns


def test_q2_json_ingestion(spark):
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        json_file = os.path.join(src, "movies.json")
        make_movies_json(json_file)

        out = ingest_to_bronze(spark, json_file, "tmdb", "movies",
                               MOVIES_SCHEMA, base_path=dst)
        assert out is not None, "ingest_to_bronze returned None for JSON"
        df = spark.read.parquet(out)
        assert df.count() == 2
        assert "movie_id" in df.columns
        assert "title"    in df.columns


def test_q2_parquet_ingestion(spark):
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        parquet_file = os.path.join(src, "users.parquet")
        make_users_parquet(parquet_file)

        out = ingest_to_bronze(spark, parquet_file, "netflix", "users",
                               USERS_SCHEMA, base_path=dst)
        assert out is not None, "ingest_to_bronze returned None for Parquet"
        df = spark.read.parquet(out)
        assert df.count() == 2
        assert "user_id" in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# Q3 - AUDIT COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def test_q3_audit_columns_present(spark):
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        f = os.path.join(src, "r.csv")
        make_ratings_csv(f, rows=1)
        out = ingest_to_bronze(spark, f, "netflix", "ratings",
                               RATINGS_SCHEMA, base_path=dst)
        assert out is not None
        df = spark.read.parquet(out)
        for col in ["ingestion_timestamp", "source_file_path", "job_run_id"]:
            assert col in df.columns, f"Missing audit column: {col}"


def test_q3_audit_columns_not_null(spark):
    from pyspark.sql import functions as F
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        f = os.path.join(src, "r.csv")
        make_ratings_csv(f, rows=2)
        out = ingest_to_bronze(spark, f, "netflix", "ratings",
                               RATINGS_SCHEMA, base_path=dst)
        assert out is not None
        df = spark.read.parquet(out)
        for col in ["ingestion_timestamp", "source_file_path", "job_run_id"]:
            nulls = df.filter(F.col(col).isNull()).count()
            assert nulls == 0, f"{col} has {nulls} null values"


def test_q3_source_file_path_matches(spark):
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        f = os.path.join(src, "ratings_path.csv")
        make_ratings_csv(f, rows=1)
        out = ingest_to_bronze(spark, f, "netflix", "ratings",
                               RATINGS_SCHEMA, base_path=dst)
        assert out is not None
        df   = spark.read.parquet(out)
        row  = df.first()
        assert f in row["source_file_path"], \
            f"Expected {f!r} in source_file_path, got {row['source_file_path']!r}"


# ══════════════════════════════════════════════════════════════════════════════
# Q4 - IDEMPOTENCY
# ══════════════════════════════════════════════════════════════════════════════

def test_q4_same_file_skipped_on_second_run(spark):
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        f = os.path.join(src, "ratings.csv")
        make_ratings_csv(f, rows=2)

        out1 = ingest_to_bronze(spark, f, "netflix", "ratings",
                                RATINGS_SCHEMA, base_path=dst)
        out2 = ingest_to_bronze(spark, f, "netflix", "ratings",
                                RATINGS_SCHEMA, base_path=dst)

        assert out1 is not None, "First ingest must succeed"
        assert out2 is None,     "Second ingest of same file must return None (skipped)"


def test_q4_hash_registry_records_file(tmp_path):
    test_file = tmp_path / "sample.csv"
    test_file.write_text("col1,col2\n1,2\n")
    fhash = compute_file_hash(str(test_file))
    assert not is_already_ingested(fhash), "Hash must not exist before registering"
    register_ingested_file(fhash, str(test_file), "job-abc")
    assert is_already_ingested(fhash),     "Hash must be found after registering"


def test_q4_different_files_both_ingested(spark):
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        f1 = os.path.join(src, "r1.csv")
        f2 = os.path.join(src, "r2.csv")
        # Different content = different hash
        pd.DataFrame({"user_id":[1],"movie_id":[1],"rating":[4.0],"rated_at":["2024-01-01 00:00:00"]}).to_csv(f1, index=False)
        pd.DataFrame({"user_id":[2],"movie_id":[2],"rating":[3.0],"rated_at":["2024-01-02 00:00:00"]}).to_csv(f2, index=False)

        out1 = ingest_to_bronze(spark, f1, "netflix", "ratings", RATINGS_SCHEMA, base_path=dst)
        out2 = ingest_to_bronze(spark, f2, "netflix", "ratings", RATINGS_SCHEMA, base_path=dst)

        assert out1 is not None
        assert out2 is not None
        assert out1 != out2


# ══════════════════════════════════════════════════════════════════════════════
# Q5 - QUARANTINE
# ══════════════════════════════════════════════════════════════════════════════

def test_q5_empty_file_quarantined(spark):
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        f = os.path.join(src, "empty.csv")
        with open(f, "w") as fh:
            fh.write("user_id,movie_id,rating,rated_at\n")  # header only, 0 rows

        out = ingest_to_bronze(spark, f, "netflix", "ratings",
                               RATINGS_SCHEMA, base_path=dst)
        assert out is None, "Empty file must be quarantined (return None)"


def test_q5_missing_file_returns_none(spark):
    with tempfile.TemporaryDirectory() as dst:
        out = ingest_to_bronze(
            spark,
            source_file_path="/tmp/does_not_exist_abc123.csv",
            source="netflix",
            entity="ratings",
            schema=RATINGS_SCHEMA,
            base_path=dst,
        )
        assert out is None, "Missing file must return None without crashing"


def test_q5_quarantine_error_log_written(spark):
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        f = os.path.join(src, "empty2.csv")
        with open(f, "w") as fh:
            fh.write("user_id,movie_id,rating,rated_at\n")

        ingest_to_bronze(spark, f, "netflix", "ratings",
                         RATINGS_SCHEMA, base_path=dst)

        assert os.path.exists(ERROR_LOG_PATH), "quarantine_errors.json must be created"
        with open(ERROR_LOG_PATH) as fh:
            errors = json.load(fh)
        assert len(errors) >= 1
        assert "reason" in errors[0]
