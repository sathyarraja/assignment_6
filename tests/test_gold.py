"""
tests/test_gold.py
Unit tests for Part C - Gold Layer (Q14-Q19).
Run: python3 -m pytest tests/test_gold.py -v
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, BooleanType, TimestampType
)
from scripts.silver_transforms import standardize_dates
from scripts.gold_transforms import (
    build_dim_dates,
    build_dim_users,
    build_dim_movies,
    build_fact_ratings,
    build_daily_engagement,
    build_content_popularity,
    build_rfm_segmentation,
    build_retention_cohorts,
    build_time_series_kpis,
)


@pytest.fixture(scope="session")
def spark():
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    s = (
        SparkSession.builder
        .appName("GoldTests")
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


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def enriched_df(spark):
    """
    Build enriched_df using explicit schema to avoid column-order bugs.
    Matches exactly what silver_transforms.build_silver_enriched_ratings produces.
    """
    schema = StructType([
        StructField("user_id",          IntegerType(), True),
        StructField("movie_id",         IntegerType(), True),
        StructField("rating",           FloatType(),   True),
        StructField("rated_at",         StringType(),  True),
        StructField("title",            StringType(),  True),
        StructField("genres",           StringType(),  True),
        StructField("year",             IntegerType(), True),
        StructField("plan",             StringType(),  True),
        StructField("user_country",     StringType(),  True),
        StructField("is_high_rating",   BooleanType(), True),
        StructField("days_since_signup",IntegerType(), True),
        StructField("rating_year",      IntegerType(), True),
        StructField("rating_month",     IntegerType(), True),
        StructField("rating_day",       IntegerType(), True),
        StructField("ingestion_timestamp", StringType(), True),
    ])
    data = [
        (1, 10, 4.5, "2024-01-10 10:00:00", "Film A", "Action|Drama", 2020,
         "premium", "US", True,  10, 2024, 1, 10, "2024-01-10 10:00:00"),
        (1, 11, 2.0, "2024-01-20 11:00:00", "Film B", "Comedy",       2021,
         "premium", "US", False, 20, 2024, 1, 20, "2024-01-20 11:00:00"),
        (2, 10, 5.0, "2024-02-05 09:00:00", "Film A", "Action|Drama", 2020,
         "basic",   "GB", True,  35, 2024, 2,  5, "2024-02-05 09:00:00"),
        (3, 12, 3.0, "2024-02-10 14:00:00", "Film C", "Thriller",     2019,
         "basic",   "FR", False, 40, 2024, 2, 10, "2024-02-10 14:00:00"),
        (2, 12, 4.0, "2024-03-01 08:00:00", "Film C", "Thriller",     2019,
         "basic",   "GB", True,  60, 2024, 3,  1, "2024-03-01 08:00:00"),
    ]
    df = spark.createDataFrame(data, schema)
    return standardize_dates(df, ["rated_at", "ingestion_timestamp"])


@pytest.fixture(scope="module")
def fact_df(spark, enriched_df):
    return build_fact_ratings(spark, enriched_df)


@pytest.fixture(scope="module")
def movies_df(spark):
    return spark.createDataFrame([
        (10, "Film A", "Action|Drama", 2020, "en", "US"),
        (11, "Film B", "Comedy",       2021, "en", "US"),
        (12, "Film C", "Thriller",     2019, "fr", "FR"),
    ], ["movie_id", "title", "genres", "year", "language", "country"])


@pytest.fixture(scope="module")
def genres_df(spark):
    return spark.createDataFrame([
        (10, "Action"), (10, "Drama"),
        (11, "Comedy"),
        (12, "Thriller"),
    ], ["movie_id", "genre"])


@pytest.fixture(scope="module")
def users_df(spark):
    """
    Column names match what build_dim_users expects after silver processing.
    Note: country is stored as user_country in enriched but plain country in users.
    build_dim_users renames it to user_country internally.
    """
    schema = StructType([
        StructField("user_id",     IntegerType(), True),
        StructField("signup_date", StringType(),  True),
        StructField("country",     StringType(),  True),
        StructField("plan",        StringType(),  True),
    ])
    data = [
        (1, "2023-12-01", "US", "premium"),
        (2, "2024-01-15", "GB", "basic"),
        (3, "2024-01-20", "FR", "basic"),
    ]
    df = spark.createDataFrame(data, schema)
    return standardize_dates(df, ["signup_date"])


# ── Q14: Star schema ──────────────────────────────────────────────────────────

def test_q14_dim_dates_columns(spark, enriched_df):
    dim = build_dim_dates(spark, enriched_df)
    for col in ["date", "year", "quarter", "month", "week",
                "day", "day_of_week", "is_weekend", "date_key"]:
        assert col in dim.columns, f"Missing column: {col}"


def test_q14_dim_dates_no_duplicates(spark, enriched_df):
    dim = build_dim_dates(spark, enriched_df)
    assert dim.count() == dim.select("date").distinct().count()


def test_q14_dim_users_columns(spark, users_df):
    dim = build_dim_users(spark, users_df)
    for col in ["user_id", "plan", "signup_date"]:
        assert col in dim.columns, f"Missing column: {col}"


def test_q14_dim_users_one_row_per_user(spark, users_df):
    dim = build_dim_users(spark, users_df)
    assert dim.count() == dim.select("user_id").distinct().count()


def test_q14_dim_movies_columns(spark, movies_df):
    dim = build_dim_movies(spark, movies_df)
    for col in ["movie_id", "title", "genres", "year"]:
        assert col in dim.columns, f"Missing column: {col}"


def test_q14_fact_ratings_columns(fact_df):
    for col in ["user_id", "movie_id", "rating_date",
                "rating", "is_high_rating"]:
        assert col in fact_df.columns, f"Missing column: {col}"


def test_q14_fact_no_null_keys(fact_df):
    assert fact_df.filter(F.col("user_id").isNull()).count()  == 0
    assert fact_df.filter(F.col("movie_id").isNull()).count() == 0


# ── Q15: Daily engagement ─────────────────────────────────────────────────────

def test_q15_daily_engagement_columns(fact_df):
    eng = build_daily_engagement(fact_df)
    for col in ["rating_date", "active_users", "total_ratings",
                "avg_rating", "high_rating_pct"]:
        assert col in eng.columns, f"Missing column: {col}"


def test_q15_daily_engagement_row_count(fact_df):
    eng = build_daily_engagement(fact_df)
    # fixture has 5 distinct dates
    assert eng.count() == 5


def test_q15_active_users_correct(fact_df):
    from pyspark.sql.types import DateType
    eng = build_daily_engagement(fact_df)
    jan10 = eng.filter(
        F.col("rating_date") == F.lit("2024-01-10").cast(DateType())
    ).first()
    assert jan10 is not None
    assert jan10["active_users"]  == 1
    assert jan10["total_ratings"] == 1


def test_q15_total_ratings_sum_matches_fact(fact_df):
    eng = build_daily_engagement(fact_df)
    daily_sum = eng.agg(F.sum("total_ratings")).collect()[0][0]
    assert daily_sum == fact_df.count()


# ── Q16: Content popularity ───────────────────────────────────────────────────

def test_q16_top_movies_overall(fact_df, movies_df, genres_df):
    pop  = build_content_popularity(fact_df, movies_df, genres_df, top_n=10)
    top  = pop["top_movies_overall"]
    assert top.count() <= 10
    assert "rank"          in top.columns
    assert "total_ratings" in top.columns
    # Film A (movie_id=10) rated twice - should rank #1
    top1 = top.filter(F.col("rank") == 1).first()
    assert top1["movie_id"] == 10


def test_q16_top_by_genre(fact_df, movies_df, genres_df):
    pop      = build_content_popularity(fact_df, movies_df, genres_df, top_n=10)
    genre_df = pop["top_movies_by_genre"]
    assert "genre"         in genre_df.columns
    assert "rank_in_genre" in genre_df.columns
    assert genre_df.filter(F.col("rank_in_genre") < 1).count() == 0


def test_q16_top_by_month(fact_df, movies_df, genres_df):
    pop      = build_content_popularity(fact_df, movies_df, genres_df, top_n=10)
    month_df = pop["top_movies_by_month"]
    assert "rating_year"   in month_df.columns
    assert "rating_month"  in month_df.columns
    assert "rank_in_month" in month_df.columns


# ── Q17: RFM segmentation ─────────────────────────────────────────────────────

def test_q17_rfm_columns(fact_df):
    rfm = build_rfm_segmentation(fact_df)
    for col in ["user_id", "recency_days", "frequency", "monetary",
                "r_score", "f_score", "m_score", "rfm_score", "segment"]:
        assert col in rfm.columns, f"Missing RFM column: {col}"


def test_q17_rfm_one_row_per_user(fact_df):
    rfm = build_rfm_segmentation(fact_df)
    assert rfm.count() == rfm.select("user_id").distinct().count()


def test_q17_rfm_scores_in_range(fact_df):
    rfm = build_rfm_segmentation(fact_df)
    for score_col in ["r_score", "f_score", "m_score"]:
        bad = rfm.filter(
            (F.col(score_col) < 1) | (F.col(score_col) > 5)
        ).count()
        assert bad == 0, f"{score_col} has values outside 1-5"


def test_q17_rfm_segments_valid(fact_df):
    rfm    = build_rfm_segmentation(fact_df)
    valid  = {"Champions", "Loyal Customers", "At Risk",
              "Lost", "Potential Loyalists"}
    actual = {r.segment for r in rfm.select("segment").collect()}
    assert actual.issubset(valid), f"Unknown segments: {actual - valid}"


# ── Q18: Cohort retention ─────────────────────────────────────────────────────

def test_q18_retention_columns(fact_df, users_df):
    cohorts = build_retention_cohorts(fact_df, users_df)
    for col in ["cohort_month", "cohort_size",
                "months_since_signup", "active_users", "retention_rate"]:
        assert col in cohorts.columns, f"Missing retention column: {col}"


def test_q18_retention_rate_in_bounds(fact_df, users_df):
    cohorts = build_retention_cohorts(fact_df, users_df)
    bad = cohorts.filter(
        (F.col("retention_rate") < 0) | (F.col("retention_rate") > 100)
    ).count()
    assert bad == 0


def test_q18_active_users_lte_cohort_size(fact_df, users_df):
    cohorts = build_retention_cohorts(fact_df, users_df)
    bad = cohorts.filter(
        F.col("active_users") > F.col("cohort_size")
    ).count()
    assert bad == 0


# ── Q19: Time-series KPIs ─────────────────────────────────────────────────────

def test_q19_daily_kpis_columns(fact_df):
    kpis = build_time_series_kpis(fact_df)
    for col in ["rating_date", "total_ratings",
                "active_users", "avg_rating", "high_rating_pct"]:
        assert col in kpis["daily_kpis"].columns


def test_q19_weekly_kpis_columns(fact_df):
    kpis = build_time_series_kpis(fact_df)
    for col in ["year", "week", "week_start", "total_ratings", "active_users"]:
        assert col in kpis["weekly_kpis"].columns


def test_q19_monthly_kpis_columns(fact_df):
    kpis = build_time_series_kpis(fact_df)
    for col in ["year", "month", "month_label",
                "total_ratings", "active_users", "new_raters"]:
        assert col in kpis["monthly_kpis"].columns


def test_q19_daily_totals_match_fact(fact_df):
    kpis        = build_time_series_kpis(fact_df)
    daily_total = kpis["daily_kpis"].agg(F.sum("total_ratings")).collect()[0][0]
    assert daily_total == fact_df.count()


def test_q19_monthly_totals_match_fact(fact_df):
    kpis          = build_time_series_kpis(fact_df)
    monthly_total = kpis["monthly_kpis"].agg(F.sum("total_ratings")).collect()[0][0]
    assert monthly_total == fact_df.count()
