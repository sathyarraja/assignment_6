"""
tests/test_dag.py
Tests for Part F - Orchestration (Q30-Q35).

These tests validate DAG structure WITHOUT running Airflow.
They import the DAG and assert on its structure, dependencies,
retry config, SLA settings, and task count.

Run: python3 -m pytest tests/test_dag.py -v
"""

import os
import sys
import json
import tempfile
from datetime import timedelta

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ── We test DAG structure without a running Airflow instance ─────────────────
# Import the DAG module directly and inspect its properties

@pytest.fixture(scope="module")
def dag():
    """Import and return the DAG object."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "dags"))
    # Set env var so the DAG knows where the project root is
    os.environ["NETFLIX_PIPELINE_ROOT"] = PROJECT_ROOT
    from netflix_pipeline_dag import dag as _dag
    return _dag


# ══════════════════════════════════════════════════════════════════════════════
# Q30 - DAG exists and is correctly configured
# ══════════════════════════════════════════════════════════════════════════════

def test_q30_dag_exists(dag):
    assert dag is not None
    assert dag.dag_id == "netflix_medallion_pipeline"


def test_q30_dag_has_correct_schedule(dag):
    assert dag.schedule_interval == "0 2 * * *"


def test_q30_dag_has_all_pipeline_tasks(dag):
    task_ids = dag.task_ids
    required = [
        "start",
        "ingest_ratings", "ingest_movies", "ingest_users",
        "bronze_complete",
        "run_silver",
        "run_dq_checks",
        "run_gold",
        "run_incremental",
        "build_monitoring_dashboard",
        "end",
    ]
    for t in required:
        assert t in task_ids, f"Missing task: {t}"


def test_q30_dag_covers_all_layers(dag):
    task_ids = dag.task_ids
    # Bronze
    assert any("ingest" in t for t in task_ids)
    # Silver
    assert any("silver" in t for t in task_ids)
    # Gold
    assert any("gold" in t for t in task_ids)


# ══════════════════════════════════════════════════════════════════════════════
# Q31 - Task dependencies
# ══════════════════════════════════════════════════════════════════════════════

def test_q31_silver_waits_for_bronze_complete(dag):
    silver = dag.get_task("run_silver")
    upstream_ids = {t.task_id for t in silver.upstream_list}
    assert "bronze_complete" in upstream_ids, \
        "run_silver must depend on bronze_complete"


def test_q31_gold_waits_for_dq(dag):
    gold = dag.get_task("run_gold")
    upstream_ids = {t.task_id for t in gold.upstream_list}
    assert "run_dq_checks" in upstream_ids, \
        "run_gold must depend on run_dq_checks"


def test_q31_dq_waits_for_silver(dag):
    dq = dag.get_task("run_dq_checks")
    upstream_ids = {t.task_id for t in dq.upstream_list}
    assert "run_silver" in upstream_ids


def test_q31_bronze_complete_waits_for_all_ingest(dag):
    bc = dag.get_task("bronze_complete")
    upstream_ids = {t.task_id for t in bc.upstream_list}
    assert "ingest_ratings" in upstream_ids
    assert "ingest_movies"  in upstream_ids
    assert "ingest_users"   in upstream_ids


def test_q31_monitoring_runs_after_gold(dag):
    mon = dag.get_task("build_monitoring_dashboard")
    upstream_ids = {t.task_id for t in mon.upstream_list}
    assert "run_gold" in upstream_ids


# ══════════════════════════════════════════════════════════════════════════════
# Q32 - Retry logic
# ══════════════════════════════════════════════════════════════════════════════

def test_q32_retries_set_to_3(dag):
    # Check default_args retries propagate to key tasks
    for task_id in ["ingest_ratings", "run_silver", "run_gold"]:
        task = dag.get_task(task_id)
        assert task.retries == 3, \
            f"{task_id} must have retries=3, got {task.retries}"


def test_q32_exponential_backoff_enabled(dag):
    for task_id in ["ingest_ratings", "run_silver", "run_gold"]:
        task = dag.get_task(task_id)
        assert task.retry_exponential_backoff == True, \
            f"{task_id} must have exponential backoff enabled"


def test_q32_retry_delay_is_5_minutes(dag):
    for task_id in ["ingest_ratings", "run_silver", "run_gold"]:
        task = dag.get_task(task_id)
        assert task.retry_delay == timedelta(minutes=5), \
            f"{task_id} retry_delay must be 5 minutes"


# ══════════════════════════════════════════════════════════════════════════════
# Q33 - SLA monitoring
# ══════════════════════════════════════════════════════════════════════════════

def test_q33_sla_miss_callback_configured(dag):
    assert dag.sla_miss_callback is not None, \
        "DAG must have sla_miss_callback configured"


def test_q33_gold_task_has_7_hour_sla(dag):
    gold = dag.get_task("run_gold")
    assert gold.sla == timedelta(hours=7), \
        f"run_gold SLA must be 7h (9 AM deadline), got {gold.sla}"


def test_q33_bronze_tasks_have_sla(dag):
    for task_id in ["ingest_ratings", "ingest_movies", "ingest_users"]:
        task = dag.get_task(task_id)
        assert task.sla is not None, f"{task_id} must have an SLA set"


def test_q33_silver_has_sla(dag):
    silver = dag.get_task("run_silver")
    assert silver.sla is not None


# ══════════════════════════════════════════════════════════════════════════════
# Q34 - Parallel processing
# ══════════════════════════════════════════════════════════════════════════════

def test_q34_bronze_tasks_are_parallel(dag):
    """All three ingest tasks must share the same upstream (start) and
    have NO dependency on each other — they are parallel."""
    ingest_ratings = dag.get_task("ingest_ratings")
    ingest_movies  = dag.get_task("ingest_movies")
    ingest_users   = dag.get_task("ingest_users")

    # None of the ingest tasks should depend on each other
    ratings_upstream = {t.task_id for t in ingest_ratings.upstream_list}
    movies_upstream  = {t.task_id for t in ingest_movies.upstream_list}
    users_upstream   = {t.task_id for t in ingest_users.upstream_list}

    assert "ingest_movies" not in ratings_upstream
    assert "ingest_users"  not in ratings_upstream
    assert "ingest_ratings" not in movies_upstream
    assert "ingest_users"   not in movies_upstream
    assert "ingest_ratings" not in users_upstream
    assert "ingest_movies"  not in users_upstream


def test_q34_all_bronze_share_same_upstream(dag):
    """All three ingest tasks must have 'start' as their upstream."""
    for task_id in ["ingest_ratings", "ingest_movies", "ingest_users"]:
        task = dag.get_task(task_id)
        upstream = {t.task_id for t in task.upstream_list}
        assert "start" in upstream, f"{task_id} must have 'start' as upstream"


# ══════════════════════════════════════════════════════════════════════════════
# Q35 - Monitoring dashboard (unit test the builder function)
# ══════════════════════════════════════════════════════════════════════════════

def test_q35_monitoring_task_exists(dag):
    assert "build_monitoring_dashboard" in dag.task_ids


def test_q35_monitoring_runs_with_all_done_trigger(dag):
    mon = dag.get_task("build_monitoring_dashboard")
    assert mon.trigger_rule == "all_done", \
        "Monitoring must use trigger_rule='all_done' to run even on failure"


def test_q35_monitoring_dashboard_builds_from_metrics():
    """Test the dashboard builder function directly."""
    from dags.netflix_pipeline_dag import task_build_monitoring, METRICS_PATH

    # Write fake metrics
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    fake_metrics = [
        {"task_id": "ingest_ratings", "status": "success",
         "rows_processed": 200, "elapsed_seconds": 1.5,
         "recorded_at": "2024-01-01T02:00:00"},
        {"task_id": "run_silver", "status": "success",
         "rows_processed": 192, "elapsed_seconds": 3.2,
         "recorded_at": "2024-01-01T02:01:00"},
        {"task_id": "run_gold", "status": "success",
         "rows_processed": 191, "elapsed_seconds": 5.1,
         "recorded_at": "2024-01-01T02:04:00"},
    ]
    with open(METRICS_PATH, "w") as f:
        json.dump(fake_metrics, f)

    # Call the function (no Airflow context needed — no kwargs used)
    task_build_monitoring()

    report_path = os.path.join(
        PROJECT_ROOT, "logs", "reports", "monitoring_dashboard.html"
    )
    assert os.path.exists(report_path)
    with open(report_path) as f:
        html = f.read()
    assert "Pipeline Monitoring Dashboard" in html
    assert "ingest_ratings" in html
    assert "run_silver"     in html
    assert "run_gold"       in html


def test_q35_monitoring_shows_correct_counts():
    from dags.netflix_pipeline_dag import METRICS_PATH
    metrics = [
        {"task_id": f"task_{i}", "status": "success",
         "rows_processed": 100, "elapsed_seconds": 1.0,
         "recorded_at": "2024-01-01T00:00:00"}
        for i in range(5)
    ]
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)

    from dags.netflix_pipeline_dag import task_build_monitoring
    task_build_monitoring()

    report_path = os.path.join(
        PROJECT_ROOT, "logs", "reports", "monitoring_dashboard.html"
    )
    with open(report_path) as f:
        html = f.read()
    # 5 tasks × 100 rows = 500 total
    assert "500" in html
