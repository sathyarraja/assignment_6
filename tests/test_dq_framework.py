"""
tests/test_dq_framework.py
Unit tests for Part E - Data Quality Framework (Q25-Q29).
Run: python3 -m pytest tests/test_dq_framework.py -v
"""

import json
import os
import sys
import tempfile

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, IntegerType, FloatType, StringType
)
from scripts.dq_framework import (
    CONFIG_PATH, DQ_RESULTS, EXPECTATIONS,
    write_dq_rules_yaml, load_dq_rules, get_table_rules,
    ExpectationSuite, build_suite_from_yaml,
    build_dq_dashboard, profile_dataframe,
    send_slack_alert, alert_on_failures,
)


@pytest.fixture(scope="session")
def spark():
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    s = (
        SparkSession.builder
        .appName("DQTests")
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
def clean_results():
    for path in [DQ_RESULTS]:
        if os.path.exists(path):
            os.remove(path)
    yield
    for path in [DQ_RESULTS]:
        if os.path.exists(path):
            os.remove(path)


def make_ratings(spark, rows):
    schema = StructType([
        StructField("user_id",   IntegerType(), True),
        StructField("movie_id",  IntegerType(), True),
        StructField("rating",    FloatType(),   True),
        StructField("rated_at",  StringType(),  True),
    ])
    return spark.createDataFrame(rows, schema)


# ══════════════════════════════════════════════════════════════════════════════
# Q25 - YAML rules
# ══════════════════════════════════════════════════════════════════════════════

def test_q25_yaml_file_created():
    write_dq_rules_yaml()
    assert os.path.exists(CONFIG_PATH)


def test_q25_yaml_contains_required_tables():
    write_dq_rules_yaml()
    rules = load_dq_rules()
    tables = rules.get("tables", {})
    for t in ["silver_ratings", "silver_movies", "silver_users"]:
        assert t in tables, f"Missing table in YAML: {t}"


def test_q25_yaml_has_not_null_rules():
    write_dq_rules_yaml()
    rules = get_table_rules("silver_ratings")
    assert "not_null" in rules
    assert "user_id"  in rules["not_null"]
    assert "movie_id" in rules["not_null"]
    assert "rating"   in rules["not_null"]


def test_q25_yaml_has_range_rules():
    write_dq_rules_yaml()
    rules = get_table_rules("silver_ratings")
    assert "range" in rules
    range_cols = [r["column"] for r in rules["range"]]
    assert "rating" in range_cols


def test_q25_yaml_has_thresholds():
    write_dq_rules_yaml()
    rules = get_table_rules("silver_ratings")
    assert "thresholds" in rules
    assert "min_row_count" in rules["thresholds"]


# ══════════════════════════════════════════════════════════════════════════════
# Q26 - Expectation suite
# ══════════════════════════════════════════════════════════════════════════════

def test_q26_not_null_passes(spark):
    df    = make_ratings(spark, [(1, 10, 4.0, "2024-01-01")])
    suite = ExpectationSuite("test", df)
    suite.expect_column_values_to_not_be_null("user_id")
    report = suite.run()
    assert report["passed"] == 1
    assert report["failed"] == 0


def test_q26_not_null_fails_on_nulls(spark):
    df    = make_ratings(spark, [(None, 10, 4.0, "2024-01-01")])
    suite = ExpectationSuite("test", df)
    suite.expect_column_values_to_not_be_null("user_id")
    report = suite.run()
    assert report["failed"] == 1


def test_q26_range_check_passes(spark):
    df    = make_ratings(spark, [(1, 10, 4.0, "2024-01-01")])
    suite = ExpectationSuite("test", df)
    suite.expect_column_values_to_be_between("rating", 1.0, 5.0)
    report = suite.run()
    assert report["passed"] == 1


def test_q26_range_check_fails(spark):
    df    = make_ratings(spark, [(1, 10, 6.0, "2024-01-01")])
    suite = ExpectationSuite("test", df)
    suite.expect_column_values_to_be_between("rating", 1.0, 5.0)
    report = suite.run()
    assert report["failed"] == 1


def test_q26_uniqueness_passes(spark):
    df    = make_ratings(spark, [(1, 10, 4.0, "2024-01-01"),
                                  (2, 11, 3.0, "2024-01-02")])
    suite = ExpectationSuite("test", df)
    suite.expect_column_values_to_be_unique(["user_id", "movie_id"])
    report = suite.run()
    assert report["passed"] == 1


def test_q26_uniqueness_fails_on_dups(spark):
    df    = make_ratings(spark, [(1, 10, 4.0, "2024-01-01"),
                                  (1, 10, 5.0, "2024-01-02")])
    suite = ExpectationSuite("test", df)
    suite.expect_column_values_to_be_unique(["user_id", "movie_id"])
    report = suite.run()
    assert report["failed"] == 1


def test_q26_row_count_check(spark):
    df    = make_ratings(spark, [(i, i, float(i), "2024-01-01") for i in range(1, 11)])
    suite = ExpectationSuite("test", df)
    suite.expect_table_row_count_to_be_between(5, 100)
    report = suite.run()
    assert report["passed"] == 1


def test_q26_row_count_fails_below_min(spark):
    df    = make_ratings(spark, [(1, 10, 4.0, "2024-01-01")])
    suite = ExpectationSuite("test", df)
    suite.expect_table_row_count_to_be_between(100)
    report = suite.run()
    assert report["failed"] == 1


def test_q26_suite_saved_to_file(spark):
    df    = make_ratings(spark, [(1, 10, 4.0, "2024-01-01")])
    suite = ExpectationSuite("my_suite", df)
    suite.expect_column_values_to_not_be_null("user_id")
    suite.run()
    path = os.path.join(EXPECTATIONS, "my_suite.json")
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data["suite_name"] == "my_suite"


def test_q26_build_suite_from_yaml(spark):
    write_dq_rules_yaml()
    df    = make_ratings(spark, [
        (i, i+10, float(i % 5 + 1), "2024-01-01") for i in range(1, 51)
    ])
    suite  = build_suite_from_yaml("silver_ratings", df)
    report = suite.run()
    assert report["total_checks"] >= 3   # at least null+range+unique+rowcount


def test_q26_overall_pass_reflects_all_checks(spark):
    df    = make_ratings(spark, [
        (1, 10, 4.0, "2024-01-01"),
        (2, 10, 0.5, "2024-01-01"),  # out-of-range rating
    ])
    suite = ExpectationSuite("test", df)
    suite.expect_column_values_to_not_be_null("user_id")
    suite.expect_column_values_to_be_between("rating", 1.0, 5.0)
    report = suite.run()
    assert report["overall_pass"] == False
    assert report["passed"] == 1
    assert report["failed"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# Q27 - Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def test_q27_dashboard_generates_html(spark):
    df    = make_ratings(spark, [(1, 10, 4.0, "2024-01-01")])
    suite = ExpectationSuite("test_table", df)
    suite.expect_column_values_to_not_be_null("user_id")
    report = suite.run()
    path   = build_dq_dashboard([report])
    assert os.path.exists(path)
    assert path.endswith(".html")


def test_q27_dashboard_contains_key_elements(spark):
    df    = make_ratings(spark, [(1, 10, 4.0, "2024-01-01")])
    suite = ExpectationSuite("ratings_test", df)
    suite.expect_column_values_to_not_be_null("user_id")
    report = suite.run()
    path   = build_dq_dashboard([report])
    with open(path) as f:
        html = f.read()
    assert "Data Quality Dashboard" in html
    assert "ratings_test"            in html
    assert "Pass rate trend"         in html


# ══════════════════════════════════════════════════════════════════════════════
# Q28 - Alerting
# ══════════════════════════════════════════════════════════════════════════════

def test_q28_no_alert_when_all_pass():
    passing_report = {
        "suite_name": "test", "overall_pass": True,
        "passed": 3, "failed": 0,
        "total_checks": 3, "success_pct": 100.0,
        "results": []
    }
    # Should complete without error and without firing any alert
    alert_on_failures([passing_report])


def test_q28_slack_skipped_without_webhook():
    failing_report = {
        "suite_name": "test", "overall_pass": False,
        "passed": 1, "failed": 1,
        "total_checks": 2, "success_pct": 50.0,
        "results": []
    }
    # Without env var set this should log a skip message, not raise
    os.environ.pop("SLACK_WEBHOOK_URL", None)
    alert_on_failures([failing_report])   # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# Q29 - Data profiling
# ══════════════════════════════════════════════════════════════════════════════

def test_q29_profile_generates_html(spark):
    df   = make_ratings(spark, [
        (1, 10, 4.0, "2024-01-01"),
        (2, 11, 3.0, "2024-01-02"),
        (3, 12, 5.0, "2024-01-03"),
    ])
    path = profile_dataframe(spark, df, "test_profile")
    assert os.path.exists(path)
    assert path.endswith(".html")


def test_q29_profile_contains_all_columns(spark):
    df   = make_ratings(spark, [(1, 10, 4.0, "2024-01-01")])
    path = profile_dataframe(spark, df, "col_check")
    with open(path) as f:
        html = f.read()
    for col in ["user_id", "movie_id", "rating", "rated_at"]:
        assert col in html, f"Column {col} missing from profile report"


def test_q29_profile_shows_row_count(spark):
    df   = make_ratings(spark, [(i, i, float(i), "2024-01-01") for i in range(1, 6)])
    path = profile_dataframe(spark, df, "rowcount_check")
    with open(path) as f:
        html = f.read()
    assert "5" in html   # row count visible in dashboard card


def test_q29_profile_handles_nulls(spark):
    df   = make_ratings(spark, [
        (1, 10, 4.0, "2024-01-01"),
        (None, 11, 3.0, "2024-01-02"),
    ])
    path = profile_dataframe(spark, df, "null_check")
    with open(path) as f:
        html = f.read()
    assert path.endswith(".html")
    assert "null" in html.lower() or "Null" in html
