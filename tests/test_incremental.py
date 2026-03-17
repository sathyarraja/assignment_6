"""
tests/test_incremental.py
Unit tests for Part D - Incremental Processing (Q20-Q24).
Run: python3 -m pytest tests/test_incremental.py -v
"""

import json
import os
import sys
import tempfile
import uuid

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, FloatType, StringType
)
from scripts.incremental_processing import (
    WATERMARK_PATH, CHECKPOINT_PATH, CDC_LOG_PATH,
    load_watermarks, save_watermark, get_watermark,
    filter_incremental, update_watermark_from_df,
    detect_cdc_changes,
    PipelineCheckpoint,
    benchmark_full_vs_incremental,
)
from scripts.silver_transforms import standardize_dates


@pytest.fixture(scope="session")
def spark():
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    s = (
        SparkSession.builder
        .appName("IncrementalTests")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "1g")
        .master("local[1]")
        .getOrCreate()
    )
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


@pytest.fixture(autouse=True)
def clean_logs():
    """Remove watermark, checkpoint and CDC logs before each test."""
    for path in [WATERMARK_PATH, CHECKPOINT_PATH, CDC_LOG_PATH]:
        if os.path.exists(path):
            os.remove(path)
    yield
    for path in [WATERMARK_PATH, CHECKPOINT_PATH, CDC_LOG_PATH]:
        if os.path.exists(path):
            os.remove(path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ratings_df(spark, rows):
    """rows: list of (user_id, movie_id, rating, ingestion_timestamp str)"""
    schema = StructType([
        StructField("user_id",              IntegerType(), True),
        StructField("movie_id",             IntegerType(), True),
        StructField("rating",               FloatType(),   True),
        StructField("ingestion_timestamp",  StringType(),  True),
    ])
    df = spark.createDataFrame(rows, schema)
    return standardize_dates(df, ["ingestion_timestamp"])


# ══════════════════════════════════════════════════════════════════════════════
# Q20 - WATERMARK
# ══════════════════════════════════════════════════════════════════════════════

def test_q20_no_watermark_returns_all_rows(spark):
    df = make_ratings_df(spark, [
        (1, 10, 4.0, "2024-01-01 00:00:00"),
        (2, 11, 3.0, "2024-01-02 00:00:00"),
    ])
    new_df, count = filter_incremental(df, "ingestion_timestamp", "ratings")
    assert count == 2


def test_q20_watermark_filters_old_rows(spark):
    # Set watermark to Jan 1
    save_watermark("ratings", "2024-01-01 00:00:00")
    df = make_ratings_df(spark, [
        (1, 10, 4.0, "2024-01-01 00:00:00"),  # at watermark — excluded
        (2, 11, 3.0, "2024-01-02 00:00:00"),  # after watermark — included
        (3, 12, 5.0, "2024-01-03 00:00:00"),  # after watermark — included
    ])
    new_df, count = filter_incremental(df, "ingestion_timestamp", "ratings")
    assert count == 2


def test_q20_watermark_updated_after_load(spark):
    df = make_ratings_df(spark, [
        (1, 10, 4.0, "2024-01-01 00:00:00"),
        (2, 11, 3.0, "2024-03-15 00:00:00"),
    ])
    update_watermark_from_df(df, "ingestion_timestamp", "ratings")
    wm = get_watermark("ratings")
    assert wm is not None
    assert "2024-03-15" in wm


def test_q20_second_run_returns_zero_new_rows(spark):
    df = make_ratings_df(spark, [
        (1, 10, 4.0, "2024-01-01 00:00:00"),
    ])
    # First run
    update_watermark_from_df(df, "ingestion_timestamp", "ratings")
    # Second run — same data, nothing new
    _, count = filter_incremental(df, "ingestion_timestamp", "ratings")
    assert count == 0


def test_q20_watermark_persists_to_disk():
    save_watermark("test_table", "2024-06-01 12:00:00")
    assert os.path.exists(WATERMARK_PATH)
    loaded = load_watermarks()
    assert "test_table" in loaded
    assert loaded["test_table"] == "2024-06-01 12:00:00"


# ══════════════════════════════════════════════════════════════════════════════
# Q21 - CDC
# ══════════════════════════════════════════════════════════════════════════════

def test_q21_detects_inserts(spark):
    old = make_ratings_df(spark, [(1, 10, 4.0, "2024-01-01 00:00:00")])
    new = make_ratings_df(spark, [
        (1, 10, 4.0, "2024-01-01 00:00:00"),
        (2, 11, 3.0, "2024-01-02 00:00:00"),  # new row
    ])
    cdc = detect_cdc_changes(old, new, ["user_id", "movie_id"], ["rating"])
    assert cdc["inserts"].count() == 1
    assert cdc["deletes"].count() == 0


def test_q21_detects_updates(spark):
    old = make_ratings_df(spark, [(1, 10, 3.0, "2024-01-01 00:00:00")])
    new = make_ratings_df(spark, [(1, 10, 5.0, "2024-01-02 00:00:00")])  # rating changed
    cdc = detect_cdc_changes(old, new, ["user_id", "movie_id"], ["rating"])
    assert cdc["updates"].count() == 1
    assert cdc["inserts"].count() == 0


def test_q21_detects_deletes(spark):
    old = make_ratings_df(spark, [
        (1, 10, 4.0, "2024-01-01 00:00:00"),
        (2, 11, 3.0, "2024-01-01 00:00:00"),
    ])
    new = make_ratings_df(spark, [(1, 10, 4.0, "2024-01-01 00:00:00")])  # row 2 deleted
    cdc = detect_cdc_changes(old, new, ["user_id", "movie_id"], ["rating"])
    assert cdc["deletes"].count() == 1


def test_q21_no_changes_detected(spark):
    df = make_ratings_df(spark, [(1, 10, 4.0, "2024-01-01 00:00:00")])
    cdc = detect_cdc_changes(df, df, ["user_id", "movie_id"], ["rating"])
    assert cdc["inserts"].count() == 0
    assert cdc["updates"].count() == 0
    assert cdc["deletes"].count() == 0


def test_q21_cdc_log_written(spark):
    old = make_ratings_df(spark, [(1, 10, 4.0, "2024-01-01 00:00:00")])
    new = make_ratings_df(spark, [(2, 11, 3.0, "2024-01-02 00:00:00")])
    detect_cdc_changes(old, new, ["user_id", "movie_id"], ["rating"])
    assert os.path.exists(CDC_LOG_PATH)
    with open(CDC_LOG_PATH) as f:
        log = json.load(f)
    assert len(log) >= 1
    assert "inserts" in log[0]


def test_q21_cdc_operation_column_present(spark):
    old = make_ratings_df(spark, [(1, 10, 4.0, "2024-01-01 00:00:00")])
    new = make_ratings_df(spark, [
        (1, 10, 5.0, "2024-01-02 00:00:00"),
        (2, 11, 3.0, "2024-01-02 00:00:00"),
    ])
    cdc = detect_cdc_changes(old, new, ["user_id", "movie_id"], ["rating"])
    assert "cdc_operation" in cdc["inserts"].columns
    assert "cdc_operation" in cdc["updates"].columns
    assert "cdc_operation" in cdc["deletes"].columns


# ══════════════════════════════════════════════════════════════════════════════
# Q23 - CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def test_q23_checkpoint_marks_stage_done():
    job_id = str(uuid.uuid4())
    cp = PipelineCheckpoint(job_id)
    assert not cp.is_done("stage_1")
    cp.mark_done("stage_1", rows_processed=100)
    assert cp.is_done("stage_1")


def test_q23_checkpoint_persists_to_disk():
    job_id = str(uuid.uuid4())
    cp = PipelineCheckpoint(job_id)
    cp.mark_done("bronze", rows_processed=50)
    assert os.path.exists(CHECKPOINT_PATH)
    # Reload from disk
    cp2 = PipelineCheckpoint(job_id)
    assert cp2.is_done("bronze")


def test_q23_resume_skips_completed_stages():
    job_id = str(uuid.uuid4())
    cp = PipelineCheckpoint(job_id)
    cp.mark_done("stage_1", 100)
    cp.mark_done("stage_2", 200)
    # New instance simulates re-run after failure
    cp2 = PipelineCheckpoint(job_id)
    assert cp2.is_done("stage_1")
    assert cp2.is_done("stage_2")
    assert not cp2.is_done("stage_3")  # not done yet


def test_q23_mark_failed_sets_status():
    job_id = str(uuid.uuid4())
    cp = PipelineCheckpoint(job_id)
    cp.mark_failed("stage_1", "Connection timeout")
    assert cp.get_status()["status"] == "failed"
    assert cp.get_status()["stages"]["stage_1"]["status"] == "failed"


def test_q23_complete_sets_status():
    job_id = str(uuid.uuid4())
    cp = PipelineCheckpoint(job_id)
    cp.mark_done("stage_1")
    cp.complete()
    assert cp.get_status()["status"] == "completed"
    assert cp.get_status()["completed_at"] is not None


# ══════════════════════════════════════════════════════════════════════════════
# Q24 - BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def test_q24_benchmark_returns_required_keys(spark):
    with tempfile.TemporaryDirectory() as tmp:
        full_df = make_ratings_df(spark, [
            (i, i+10, float(i % 5 + 1), "2024-01-01 00:00:00")
            for i in range(1, 21)
        ])
        incr_df = make_ratings_df(spark, [
            (21, 31, 4.0, "2024-02-01 00:00:00"),
            (22, 32, 3.0, "2024-02-01 00:00:00"),
        ])
        result = benchmark_full_vs_incremental(
            spark, full_df, incr_df,
            silver_path=os.path.join(tmp, "bench"),
            pk_cols=["user_id", "movie_id"],
        )
        assert "full_refresh" in result
        assert "incremental"  in result
        assert "accuracy"     in result
        assert "summary"      in result


def test_q24_incremental_processes_fewer_rows(spark):
    with tempfile.TemporaryDirectory() as tmp:
        full_df = make_ratings_df(spark, [
            (i, i+10, float(i % 5 + 1), "2024-01-01 00:00:00")
            for i in range(1, 11)
        ])
        incr_df = make_ratings_df(spark, [
            (11, 21, 4.0, "2024-02-01 00:00:00"),
        ])
        result = benchmark_full_vs_incremental(
            spark, full_df, incr_df,
            silver_path=os.path.join(tmp, "bench2"),
            pk_cols=["user_id", "movie_id"],
        )
        assert result["incremental"]["rows_processed"] < \
               result["full_refresh"]["rows_processed"]


def test_q24_benchmark_results_saved_to_disk(spark):
    from scripts.incremental_processing import BENCHMARK_PATH
    if os.path.exists(BENCHMARK_PATH):
        os.remove(BENCHMARK_PATH)
    with tempfile.TemporaryDirectory() as tmp:
        full_df = make_ratings_df(spark, [(1, 10, 4.0, "2024-01-01 00:00:00")])
        incr_df = make_ratings_df(spark, [(2, 11, 3.0, "2024-02-01 00:00:00")])
        benchmark_full_vs_incremental(
            spark, full_df, incr_df,
            silver_path=os.path.join(tmp, "bench3"),
            pk_cols=["user_id", "movie_id"],
        )
    assert os.path.exists(BENCHMARK_PATH)
