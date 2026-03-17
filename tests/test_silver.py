"""
tests/test_silver.py
Unit tests for Part B - Silver Layer (Q6-Q13).
Run: python3 -m pytest tests/test_silver.py -v
"""

import json
import os
import sys
import tempfile

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scripts.silver_transforms import (
    standardize_dates,
    standardize_country_codes,
    check_nulls,
    check_rating_range,
    check_duplicates,
    check_referential_integrity,
    deduplicate,
    build_dq_metrics_df,
    build_silver_genres,
    DataQualityResult,
    DQ_LOG_PATH,
)


# ── Shared SparkSession ───────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def spark():
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    s = (
        SparkSession.builder
        .appName("SilverTests")
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
def clean_dq_log():
    if os.path.exists(DQ_LOG_PATH):
        os.remove(DQ_LOG_PATH)
    yield
    if os.path.exists(DQ_LOG_PATH):
        os.remove(DQ_LOG_PATH)


# ── Q6: Star schema structure ─────────────────────────────────────────────────

def test_q6_ratings_has_required_columns(spark):
    df = spark.createDataFrame([
        (1, 10, 4.0, "2024-01-01 10:00:00")
    ], ["user_id", "movie_id", "rating", "rated_at"])
    for col in ["user_id", "movie_id", "rating", "rated_at"]:
        assert col in df.columns


def test_q6_genres_exploded_correctly(spark):
    movies_df = spark.createDataFrame([
        (1, "Movie A", "Action|Comedy", 2020, "en", "US"),
        (2, "Movie B", "Drama",         2021, "fr", "FR"),
    ], ["movie_id", "title", "genres", "year", "language", "country"])
    genres_df = build_silver_genres(movies_df)
    rows = {(r.movie_id, r.genre) for r in genres_df.collect()}
    assert (1, "Action") in rows
    assert (1, "Comedy") in rows
    assert (2, "Drama")  in rows
    assert genres_df.count() == 3


# ── Q7: Data quality checks ───────────────────────────────────────────────────

def test_q7_null_check_detects_nulls(spark):
    df = spark.createDataFrame([
        (1,    10, 4.0),
        (None, 11, 3.0),
        (3,    12, 5.0),
    ], ["user_id", "movie_id", "rating"])
    result = DataQualityResult("test_table")
    check_nulls(df, ["user_id", "movie_id", "rating"], result)
    assert result.null_counts["user_id"] == 1
    assert result.null_counts["movie_id"] == 0
    assert result.passed == False


def test_q7_null_check_passes_when_clean(spark):
    df = spark.createDataFrame([(1, 10, 4.0)], ["user_id", "movie_id", "rating"])
    result = DataQualityResult("test_table")
    check_nulls(df, ["user_id", "movie_id", "rating"], result)
    assert result.passed == True
    assert result.null_counts["user_id"] == 0


def test_q7_rating_range_detects_out_of_range(spark):
    df = spark.createDataFrame([
        (1, 10, 4.0),
        (2, 11, 0.5),   # below 1.0 - invalid
        (3, 12, 6.0),   # above 5.0 - invalid
    ], ["user_id", "movie_id", "rating"])
    result = DataQualityResult("test_ratings")
    check_rating_range(df, result)
    assert result.range_fails["rating"] == 2
    assert result.passed == False


def test_q7_rating_range_passes_valid(spark):
    df = spark.createDataFrame([
        (1, 10, 1.0),
        (2, 11, 3.5),
        (3, 12, 5.0),
    ], ["user_id", "movie_id", "rating"])
    result = DataQualityResult("test_ratings")
    check_rating_range(df, result)
    assert result.range_fails["rating"] == 0
    assert result.passed == True


def test_q7_duplicate_detection(spark):
    df = spark.createDataFrame([
        (1, 10, 4.0, "2024-01-01"),
        (1, 10, 3.0, "2024-01-02"),   # duplicate key
        (2, 11, 5.0, "2024-01-01"),
    ], ["user_id", "movie_id", "rating", "rated_at"])
    result = DataQualityResult("test_ratings")
    check_duplicates(df, ["user_id", "movie_id"], result)
    assert result.dup_count == 1


def test_q7_referential_integrity(spark):
    ratings_df = spark.createDataFrame([
        (1, 10), (2, 10), (99, 10)   # user 99 doesn't exist in users
    ], ["user_id", "movie_id"])
    users_df = spark.createDataFrame([(1,), (2,)], ["user_id"])
    result = DataQualityResult("test_ref")
    check_referential_integrity(ratings_df, "user_id", users_df, "user_id", result)
    assert result.ref_fails["user_id_orphans"] == 1
    assert result.passed == False


# ── Q8: Standardization ───────────────────────────────────────────────────────

def test_q8_date_standardization(spark):
    df = spark.createDataFrame([
        ("2024-01-15 10:30:00",),
        ("2024-03-20",),
    ], ["rated_at"])
    df = standardize_dates(df, ["rated_at"])
    # After standardization the column must be TimestampType, not StringType
    assert df.schema["rated_at"].dataType.typeName() == "timestamp"
    assert df.filter(F.col("rated_at").isNull()).count() == 0


def test_q8_country_code_normalization(spark):
    df = spark.createDataFrame([
        ("USA",), ("UK",), ("FR",), ("GER",)
    ], ["country"])
    df = standardize_country_codes(df, "country")
    results = {r.country for r in df.collect()}
    assert "US" in results
    assert "GB" in results
    assert "FR" in results
    assert "DE" in results
    # originals must be gone
    assert "USA" not in results
    assert "UK"  not in results


# ── Q9: Deduplication ─────────────────────────────────────────────────────────

def test_q9_dedup_keeps_latest(spark):
    df = spark.createDataFrame([
        (1, 10, 3.0, "2024-01-01 00:00:00"),
        (1, 10, 5.0, "2024-03-01 00:00:00"),  # same key, newer - should win
        (2, 11, 4.0, "2024-01-15 00:00:00"),
    ], ["user_id", "movie_id", "rating", "rated_at"])
    df = standardize_dates(df, ["rated_at"])
    result = deduplicate(df, ["user_id", "movie_id"], "rated_at")
    assert result.count() == 2
    row = result.filter(F.col("user_id") == 1).first()
    assert row["rating"] == 5.0


def test_q9_dedup_no_data_loss_when_no_dups(spark):
    df = spark.createDataFrame([
        (1, 10, 4.0, "2024-01-01"),
        (2, 11, 3.5, "2024-01-02"),
        (3, 12, 5.0, "2024-01-03"),
    ], ["user_id", "movie_id", "rating", "rated_at"])
    df = standardize_dates(df, ["rated_at"])
    result = deduplicate(df, ["user_id", "movie_id"], "rated_at")
    assert result.count() == 3


# ── Q12: DQ metrics table ─────────────────────────────────────────────────────

def test_q12_dq_metrics_saved_and_loadable(spark):
    from scripts.silver_transforms import save_dq_metrics
    result = DataQualityResult("test_table")
    result.total_rows = 100
    result.null_counts = {"user_id": 2}
    result.dup_count   = 5
    save_dq_metrics(result)

    assert os.path.exists(DQ_LOG_PATH)
    dq_df = build_dq_metrics_df(spark)
    assert dq_df.count() >= 1
    row = dq_df.filter(F.col("table_name") == "test_table").first()
    assert row is not None
    assert row["total_rows"] == 100
    assert row["duplicate_count"] == 5


def test_q12_dq_metrics_accumulates_multiple_tables(spark):
    from scripts.silver_transforms import save_dq_metrics
    for name in ["table_a", "table_b", "table_c"]:
        r = DataQualityResult(name)
        r.total_rows = 50
        save_dq_metrics(r)

    dq_df = build_dq_metrics_df(spark)
    assert dq_df.count() == 3


# ── Q13: Enriched table ───────────────────────────────────────────────────────

def test_q13_enriched_has_joined_columns(spark):
    from scripts.silver_transforms import build_silver_enriched_ratings
    ratings_df = spark.createDataFrame([
        (1, 10, 4.0, "2024-01-15 10:00:00")
    ], ["user_id", "movie_id", "rating", "rated_at"])
    ratings_df = standardize_dates(ratings_df, ["rated_at"])

    movies_df = spark.createDataFrame([
        (10, "Test Movie", "Action|Drama", 2020, "en", "US")
    ], ["movie_id", "title", "genres", "year", "language", "country"])

    users_df = spark.createDataFrame([
        (1, "2023-01-01", "US", "premium")
    ], ["user_id", "signup_date", "country", "plan"])
    users_df = standardize_dates(users_df, ["signup_date"])

    enriched = build_silver_enriched_ratings(spark, ratings_df, movies_df, users_df)
    assert enriched.count() == 1
    row = enriched.first()
    assert row["title"]          == "Test Movie"
    assert row["plan"]           == "premium"
    assert row["is_high_rating"] == True
    assert row["rating_year"]    == 2024


def test_q13_enriched_business_logic_flags(spark):
    from scripts.silver_transforms import build_silver_enriched_ratings
    ratings_df = spark.createDataFrame([
        (1, 10, 4.5, "2024-06-01 00:00:00"),  # high rating
        (2, 11, 2.0, "2024-06-01 00:00:00"),  # low rating
    ], ["user_id", "movie_id", "rating", "rated_at"])
    ratings_df = standardize_dates(ratings_df, ["rated_at"])

    movies_df = spark.createDataFrame([
        (10, "Film A", "Action", 2020, "en", "US"),
        (11, "Film B", "Drama",  2021, "en", "GB"),
    ], ["movie_id", "title", "genres", "year", "language", "country"])

    users_df = spark.createDataFrame([
        (1, "2024-01-01", "US", "premium"),
        (2, "2024-01-01", "GB", "basic"),
    ], ["user_id", "signup_date", "country", "plan"])
    users_df = standardize_dates(users_df, ["signup_date"])

    enriched = build_silver_enriched_ratings(spark, ratings_df, movies_df, users_df)
    rows = {r.user_id: r for r in enriched.collect()}
    assert rows[1]["is_high_rating"] == True
    assert rows[2]["is_high_rating"] == False
