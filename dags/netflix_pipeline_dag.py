"""
dags/netflix_pipeline_dag.py
=======================================================
Part F - Orchestration (Questions 30-35)
=======================================================

Q30 Airflow DAG: bronze → silver → gold full pipeline
Q31 Task dependencies: silver waits for bronze, gold waits for silver
Q32 Retry logic: 3 retries with exponential backoff
Q33 SLA monitoring: alert if gold not ready by 9 AM
Q34 Parallel processing: multiple bronze tables concurrently
Q35 Monitoring dashboard: job status, duration, data volume
"""

import os
import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from airflow.utils.email import send_email
from airflow.models import TaskInstance

logger = logging.getLogger("netflix_pipeline_dag")

# ── Project root inside the Airflow container ─────────────────────────────────
# Airflow mounts your local project folder as /opt/airflow
# Adjust this if your docker-compose volume path is different
PROJECT_ROOT = os.getenv("NETFLIX_PIPELINE_ROOT", "/opt/airflow")
LOGS_DIR     = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# ── Pipeline metrics log (Q35) ────────────────────────────────────────────────
METRICS_PATH = os.path.join(LOGS_DIR, "pipeline_metrics.json")


def _log_metrics(task_id: str, status: str, rows: int, elapsed: float) -> None:
    """Append a task execution record to the pipeline metrics log."""
    entry = {
        "task_id":      task_id,
        "status":       status,
        "rows_processed": rows,
        "elapsed_seconds": elapsed,
        "recorded_at":  datetime.utcnow().isoformat(),
    }
    metrics = []
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    metrics.append(entry)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Q33 - SLA MISS CALLBACK
# Fires when a task misses its SLA (gold not ready by 9 AM)
# ─────────────────────────────────────────────────────────────────────────────

def sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
    """Called by Airflow when any task misses its SLA."""
    missed = ", ".join([str(s.task_id) for s in slas])
    logger.error(f"SLA MISS — tasks: {missed}")

    # Log the SLA miss
    entry = {
        "type":        "sla_miss",
        "tasks":       missed,
        "occurred_at": datetime.utcnow().isoformat(),
    }
    alerts = []
    alert_path = os.path.join(LOGS_DIR, "sla_alerts.json")
    if os.path.exists(alert_path):
        with open(alert_path) as f:
            alerts = json.load(f)
    alerts.append(entry)
    with open(alert_path, "w") as f:
        json.dump(alerts, f, indent=2)

    # Optionally send email (configure in Airflow's SMTP settings)
    try:
        send_email(
            to=["data-team@netflix.local"],
            subject=f"[SLA MISS] Netflix Pipeline — {missed}",
            html_content=f"<p>SLA missed for tasks: <b>{missed}</b></p>"
                         f"<p>Gold layer may not be ready by 9 AM.</p>",
        )
    except Exception as e:
        logger.warning(f"SLA email failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK FUNCTIONS
# Each function is a thin wrapper that imports and calls the pipeline scripts.
# Keeping business logic in scripts/ means DAGs stay clean and testable.
# ─────────────────────────────────────────────────────────────────────────────

def task_ingest_ratings(**context):
    """Q30/Q34: Ingest Netflix ratings CSV into Bronze."""
    import time, sys
    sys.path.insert(0, PROJECT_ROOT)
    from utils.spark_session import get_spark
    from scripts.bronze_ingestion import (
        RATINGS_SCHEMA, ingest_to_bronze
    )
    t0    = time.time()
    spark = get_spark("Airflow_BronzeRatings")
    try:
        raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "ratings_1.csv")
        out = ingest_to_bronze(
            spark, raw_path, "netflix", "ratings",
            RATINGS_SCHEMA, base_path=PROJECT_ROOT,
        )
        rows    = spark.read.parquet(out).count() if out else 0
        elapsed = round(time.time() - t0, 2)
        _log_metrics("ingest_ratings", "success", rows, elapsed)
        context["ti"].xcom_push(key="bronze_ratings_path", value=out)
        logger.info(f"ingest_ratings: {rows} rows in {elapsed}s")
    finally:
        spark.stop()


def task_ingest_movies(**context):
    """Q30/Q34: Ingest TMDB movies JSON into Bronze (runs in parallel)."""
    import time, sys
    sys.path.insert(0, PROJECT_ROOT)
    from utils.spark_session import get_spark
    from scripts.bronze_ingestion import MOVIES_SCHEMA, ingest_to_bronze
    t0    = time.time()
    spark = get_spark("Airflow_BronzeMovies")
    try:
        raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "movies.json")
        out = ingest_to_bronze(
            spark, raw_path, "tmdb", "movies",
            MOVIES_SCHEMA, base_path=PROJECT_ROOT,
        )
        rows    = spark.read.parquet(out).count() if out else 0
        elapsed = round(time.time() - t0, 2)
        _log_metrics("ingest_movies", "success", rows, elapsed)
        context["ti"].xcom_push(key="bronze_movies_path", value=out)
    finally:
        spark.stop()


def task_ingest_users(**context):
    """Q30/Q34: Ingest Netflix users Parquet into Bronze (runs in parallel)."""
    import time, sys
    sys.path.insert(0, PROJECT_ROOT)
    from utils.spark_session import get_spark
    from scripts.bronze_ingestion import (
        USERS_SCHEMA, ingest_directory
    )
    t0    = time.time()
    spark = get_spark("Airflow_BronzeUsers")
    try:
        users_dir = os.path.join(PROJECT_ROOT, "data", "raw", "users")
        paths = ingest_directory(
            spark, users_dir, "netflix", "users",
            USERS_SCHEMA, base_path=PROJECT_ROOT,
        )
        rows    = sum(spark.read.parquet(p).count() for p in paths)
        elapsed = round(time.time() - t0, 2)
        _log_metrics("ingest_users", "success", rows, elapsed)
    finally:
        spark.stop()


def task_run_silver(**context):
    """Q31: Silver layer — waits for ALL bronze tasks to finish first."""
    import time, sys
    sys.path.insert(0, PROJECT_ROOT)
    from utils.spark_session import get_spark
    from scripts.silver_transforms import run_silver_pipeline
    t0    = time.time()
    spark = get_spark("Airflow_Silver")
    try:
        tables  = run_silver_pipeline(spark)
        rows    = tables["ratings"].count()
        elapsed = round(time.time() - t0, 2)
        _log_metrics("silver_pipeline", "success", rows, elapsed)
        logger.info(f"Silver complete: {rows} ratings rows in {elapsed}s")
    finally:
        spark.stop()


def task_run_dq(**context):
    """Q31: DQ checks — runs after silver, before gold."""
    import time, sys
    sys.path.insert(0, PROJECT_ROOT)
    from utils.spark_session import get_spark
    from scripts.dq_framework import (
        run_all_suites, build_dq_dashboard, alert_on_failures
    )
    t0    = time.time()
    spark = get_spark("Airflow_DQ")
    try:
        reports = run_all_suites(spark)
        build_dq_dashboard(reports)
        alert_on_failures(reports)
        elapsed = round(time.time() - t0, 2)
        _log_metrics("dq_checks", "success", len(reports), elapsed)

        # Fail the task if any table fails DQ — blocks gold from running
        failed = [r for r in reports if not r["overall_pass"]]
        if failed:
            names = ", ".join(r["suite_name"] for r in failed)
            raise ValueError(f"DQ failed for: {names}")
    finally:
        spark.stop()


def task_run_gold(**context):
    """Q31/Q33: Gold layer — waits for silver + DQ. Has 9 AM SLA."""
    import time, sys
    sys.path.insert(0, PROJECT_ROOT)
    from utils.spark_session import get_spark
    from scripts.gold_transforms import run_gold_pipeline
    t0    = time.time()
    spark = get_spark("Airflow_Gold")
    try:
        tables  = run_gold_pipeline(spark)
        rows    = tables["fact_ratings"].count()
        elapsed = round(time.time() - t0, 2)
        _log_metrics("gold_pipeline", "success", rows, elapsed)
        logger.info(f"Gold complete: {rows} fact rows in {elapsed}s")
    finally:
        spark.stop()


def task_run_incremental(**context):
    """Q30: Incremental processing — watermark update after gold."""
    import time, sys
    sys.path.insert(0, PROJECT_ROOT)
    from utils.spark_session import get_spark
    from scripts.incremental_processing import (
        filter_incremental, update_watermark_from_df
    )
    t0    = time.time()
    spark = get_spark("Airflow_Incremental")
    try:
        silver_ratings = os.path.join(PROJECT_ROOT, "silver", "ratings")
        delta_log      = os.path.join(silver_ratings, "_delta_log")
        df = spark.read.format("delta").load(silver_ratings) \
             if os.path.exists(delta_log) \
             else spark.read.option("recursiveFileLookup", "true").parquet(silver_ratings)

        new_df, count = filter_incremental(df, "ingestion_timestamp", "ratings")
        update_watermark_from_df(new_df, "ingestion_timestamp", "ratings")
        elapsed = round(time.time() - t0, 2)
        _log_metrics("incremental", "success", count, elapsed)
        logger.info(f"Incremental: {count} new rows processed in {elapsed}s")
    finally:
        spark.stop()


def task_build_monitoring(**context):
    """Q35: Build monitoring dashboard from pipeline_metrics.json."""
    if not os.path.exists(METRICS_PATH):
        logger.warning("No metrics found — skipping dashboard")
        return

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    # Build HTML monitoring dashboard
    rows_html = ""
    for m in metrics[-50:]:    # show last 50 entries
        status_color = "#1D9E75" if m["status"] == "success" else "#E24B4A"
        rows_html += f"""
        <tr>
          <td>{m['task_id']}</td>
          <td style="color:{status_color};font-weight:500">{m['status']}</td>
          <td>{m['rows_processed']:,}</td>
          <td>{m['elapsed_seconds']}s</td>
          <td>{m['recorded_at'][:19].replace('T', ' ')}</td>
        </tr>"""

    total_tasks   = len(metrics)
    success_tasks = sum(1 for m in metrics if m["status"] == "success")
    total_rows    = sum(m["rows_processed"] for m in metrics)
    avg_duration  = round(
        sum(m["elapsed_seconds"] for m in metrics) / total_tasks, 2
    ) if total_tasks else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pipeline Monitoring Dashboard</title>
<style>
  body  {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
           padding:24px;background:#f5f5f5;color:#333; }}
  h1    {{ font-size:22px;font-weight:600;margin-bottom:4px; }}
  .sub  {{ color:#888;font-size:13px;margin-bottom:20px; }}
  .cards{{ display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap; }}
  .card {{ background:#fff;border-radius:10px;padding:16px 20px;
           min-width:150px;box-shadow:0 1px 3px rgba(0,0,0,0.08); }}
  .card .v {{ font-size:28px;font-weight:700; }}
  .card .l {{ font-size:12px;color:#888;margin-top:2px; }}
  .wrap {{ background:#fff;border-radius:10px;padding:20px;
           box-shadow:0 1px 3px rgba(0,0,0,0.08);overflow-x:auto; }}
  table {{ width:100%;border-collapse:collapse;font-size:14px; }}
  th    {{ background:#f8f8f8;padding:8px 12px;text-align:left;
           font-weight:500;border-bottom:2px solid #e8e8e8; }}
  td    {{ padding:8px 12px;border-bottom:1px solid #f0f0f0; }}
</style>
</head>
<body>
<h1>Pipeline Monitoring Dashboard</h1>
<div class="sub">Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>
<div class="cards">
  <div class="card"><div class="v">{total_tasks}</div>
    <div class="l">Total task runs</div></div>
  <div class="card"><div class="v" style="color:#1D9E75">{success_tasks}</div>
    <div class="l">Successful</div></div>
  <div class="card"><div class="v" style="color:#E24B4A">{total_tasks-success_tasks}</div>
    <div class="l">Failed</div></div>
  <div class="card"><div class="v">{total_rows:,}</div>
    <div class="l">Total rows processed</div></div>
  <div class="card"><div class="v">{avg_duration}s</div>
    <div class="l">Avg task duration</div></div>
</div>
<div class="wrap">
<table>
  <thead>
    <tr><th>Task</th><th>Status</th><th>Rows</th>
        <th>Duration</th><th>Recorded at</th></tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</div>
</body>
</html>"""

    out_path = os.path.join(LOGS_DIR, "reports", "monitoring_dashboard.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    logger.info(f"Monitoring dashboard written to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Q30 - DAG DEFINITION
# Q31 - Task dependencies
# Q32 - Retry logic with exponential backoff
# Q33 - SLA monitoring
# Q34 - Parallel bronze ingestion
# Q35 - Monitoring task at end
# ─────────────────────────────────────────────────────────────────────────────

# Q32: Default args apply retry logic to ALL tasks
default_args = {
    "owner":            "data-engineering",
    "depends_on_past":  False,
    "email_on_failure": True,
    "email_on_retry":   False,
    "email":            ["data-team@netflix.local"],
    "retries":          3,                        # Q32: 3 retries
    "retry_delay":      timedelta(minutes=5),     # Q32: starts at 5 min
    "retry_exponential_backoff": True,            # Q32: 5 → 10 → 20 min
    "max_retry_delay":  timedelta(minutes=30),    # Q32: cap at 30 min
}

with DAG(
    dag_id="netflix_medallion_pipeline",
    description="End-to-end ELT: Bronze → Silver → Gold with DQ and monitoring",
    default_args=default_args,
    schedule_interval="0 2 * * *",       # runs daily at 2 AM
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["netflix", "medallion", "elt"],
    sla_miss_callback=sla_miss_callback, # Q33: fires if any task misses SLA
    doc_md="""
## Netflix Medallion Pipeline

End-to-end ELT pipeline implementing the Bronze/Silver/Gold architecture.

### Schedule
Runs daily at 2 AM. Gold layer must be complete by 9 AM (7-hour SLA).

### Layers
- **Bronze**: Raw ingestion from CSV, JSON, Parquet sources
- **Silver**: Cleaned, validated, enriched data with DQ checks
- **Gold**: Business-ready aggregations, RFM, cohorts, KPIs

### Monitoring
See `logs/reports/monitoring_dashboard.html` after each run.
    """,
) as dag:

    # ── Start sentinel ────────────────────────────────────────────────────────
    start = EmptyOperator(task_id="start")

    # ── Q34: Bronze — three tasks run IN PARALLEL ─────────────────────────────
    ingest_ratings = PythonOperator(
        task_id="ingest_ratings",
        python_callable=task_ingest_ratings,
        sla=timedelta(hours=1),              # Q33: must finish within 1h
    )

    ingest_movies = PythonOperator(
        task_id="ingest_movies",
        python_callable=task_ingest_movies,
        sla=timedelta(hours=1),
    )

    ingest_users = PythonOperator(
        task_id="ingest_users",
        python_callable=task_ingest_users,
        sla=timedelta(hours=1),
    )

    # ── Sync point: all bronze must finish before silver starts ───────────────
    bronze_complete = EmptyOperator(task_id="bronze_complete")

    # ── Q31: Silver waits for bronze_complete ─────────────────────────────────
    run_silver = PythonOperator(
        task_id="run_silver",
        python_callable=task_run_silver,
        sla=timedelta(hours=3),
    )

    # ── DQ checks gate gold ───────────────────────────────────────────────────
    run_dq = PythonOperator(
        task_id="run_dq_checks",
        python_callable=task_run_dq,
        sla=timedelta(hours=4),
    )

    # ── Q31/Q33: Gold waits for silver + DQ, must finish by 9 AM ─────────────
    # DAG starts at 2 AM → 7-hour SLA = 9 AM deadline
    run_gold = PythonOperator(
        task_id="run_gold",
        python_callable=task_run_gold,
        sla=timedelta(hours=7),              # Q33: 9 AM deadline
    )

    # ── Incremental watermark update ──────────────────────────────────────────
    run_incremental = PythonOperator(
        task_id="run_incremental",
        python_callable=task_run_incremental,
    )

    # ── Q35: Monitoring dashboard — always runs last ──────────────────────────
    build_monitoring = PythonOperator(
        task_id="build_monitoring_dashboard",
        python_callable=task_build_monitoring,
        trigger_rule="all_done",   # runs even if upstream tasks fail
    )

    end = EmptyOperator(task_id="end")

    # ── Q31/Q34: Wire up dependencies ─────────────────────────────────────────
    #
    # start
    #   ├── ingest_ratings  ─┐
    #   ├── ingest_movies   ─┼── bronze_complete ── run_silver ── run_dq ── run_gold
    #   └── ingest_users    ─┘                                          └── run_incremental
    #                                                                   └── build_monitoring
    #                                                                   └── end
    #
    start >> [ingest_ratings, ingest_movies, ingest_users]  # Q34: parallel
    [ingest_ratings, ingest_movies, ingest_users] >> bronze_complete
    bronze_complete >> run_silver                           # Q31: silver after bronze
    run_silver      >> run_dq                              # Q31: DQ after silver
    run_dq          >> run_gold                            # Q31: gold after DQ
    run_gold        >> [run_incremental, build_monitoring]
    [run_incremental, build_monitoring] >> end
