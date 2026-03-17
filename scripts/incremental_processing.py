"""
scripts/incremental_processing.py
=======================================================
Part D - Incremental Processing (Questions 20-24)
=======================================================

Q20 Watermark-based incremental loads: track max(timestamp) per table
Q21 Handle CDC: detect inserts, updates, deletes in source data
Q22 Delta MERGE for upserts: merge new data with existing
Q23 Checkpoint mechanism: save state, resume from failure point
Q24 Compare full refresh vs incremental: time, cost, accuracy
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, FloatType, TimestampType
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("incremental_processing")

_PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATERMARK_PATH   = os.path.join(_PROJECT_ROOT, "logs", "watermarks.json")
CHECKPOINT_PATH  = os.path.join(_PROJECT_ROOT, "logs", "checkpoints.json")
CDC_LOG_PATH     = os.path.join(_PROJECT_ROOT, "logs", "cdc_log.json")
SILVER_BASE      = os.path.join(_PROJECT_ROOT, "silver")
BENCHMARK_PATH   = os.path.join(_PROJECT_ROOT, "logs", "benchmark_results.json")


# ═════════════════════════════════════════════════════════════════════════════
# Q20 - WATERMARK-BASED INCREMENTAL LOADS
# Track max(timestamp) per table so each run only processes new records
# ═════════════════════════════════════════════════════════════════════════════

def load_watermarks() -> Dict[str, str]:
    """Load all watermarks from disk. Returns {table_name: last_timestamp}."""
    if os.path.exists(WATERMARK_PATH):
        with open(WATERMARK_PATH, "r") as f:
            return json.load(f)
    return {}


def save_watermark(table_name: str, timestamp: str) -> None:
    """Persist the new high-water mark for a table after successful load."""
    wm = load_watermarks()
    wm[table_name] = timestamp
    os.makedirs(os.path.dirname(WATERMARK_PATH), exist_ok=True)
    with open(WATERMARK_PATH, "w") as f:
        json.dump(wm, f, indent=2)
    logger.info(f"Watermark updated: {table_name} → {timestamp}")


def get_watermark(table_name: str) -> Optional[str]:
    """Return the last watermark for a table, or None for first run."""
    return load_watermarks().get(table_name)


def filter_incremental(
    df: DataFrame,
    timestamp_col: str,
    table_name: str,
) -> Tuple[DataFrame, int]:
    """
    Filter a DataFrame to only rows newer than the last watermark.
    Returns (filtered_df, new_row_count).

    On first run (no watermark) returns all rows.
    After processing, caller must call save_watermark() with the new max.
    """
    watermark = get_watermark(table_name)

    if watermark:
        logger.info(f"Applying watermark filter: {timestamp_col} > {watermark}")
        new_df = df.filter(F.col(timestamp_col) > F.lit(watermark))
    else:
        logger.info(f"No watermark found for {table_name} — full load")
        new_df = df

    count = new_df.count()
    logger.info(f"Incremental rows for {table_name}: {count}")
    return new_df, count


def update_watermark_from_df(
    df: DataFrame,
    timestamp_col: str,
    table_name: str,
) -> Optional[str]:
    """
    Compute max(timestamp_col) from df and save it as the new watermark.
    Returns the new watermark string, or None if df is empty.
    """
    if df.count() == 0:
        logger.info(f"Empty DataFrame — watermark unchanged for {table_name}")
        return get_watermark(table_name)

    max_ts = df.agg(
        F.max(F.col(timestamp_col)).cast(StringType())
    ).collect()[0][0]

    if max_ts:
        save_watermark(table_name, max_ts)
    return max_ts


# ═════════════════════════════════════════════════════════════════════════════
# Q21 - CDC: CHANGE DATA CAPTURE
# Detect inserts, updates, and deletes by comparing source snapshots
# ═════════════════════════════════════════════════════════════════════════════

def detect_cdc_changes(
    old_df: DataFrame,
    new_df: DataFrame,
    pk_cols: list,
    compare_cols: list,
) -> Dict[str, DataFrame]:
    """
    Compare two snapshots of a table to detect CDC changes.

    Strategy:
      INSERTS  - rows in new_df with no matching PK in old_df
      DELETES  - rows in old_df with no matching PK in new_df
      UPDATES  - rows present in both but with changed compare_cols values

    Returns dict with keys: 'inserts', 'updates', 'deletes'
    Each DataFrame has a 'cdc_operation' column added.
    """
    # Build join condition on all PK columns
    join_cond = [old_df[k] == new_df[k] for k in pk_cols]

    # INSERTS: in new but not in old
    inserts = (
        new_df.join(old_df.select(pk_cols), on=pk_cols, how="left_anti")
        .withColumn("cdc_operation", F.lit("INSERT"))
    )

    # DELETES: in old but not in new
    deletes = (
        old_df.join(new_df.select(pk_cols), on=pk_cols, how="left_anti")
        .withColumn("cdc_operation", F.lit("DELETE"))
    )

    # UPDATES: in both but values changed
    # Join old and new on PK, then check if any compare_col differs
    old_prefixed = old_df.select(
        [F.col(c).alias(f"old_{c}") for c in pk_cols + compare_cols]
    )
    new_prefixed = new_df.select(
        pk_cols + compare_cols
    )

    joined = new_prefixed.join(
        old_prefixed,
        on=[new_prefixed[k] == old_prefixed[f"old_{k}"] for k in pk_cols],
        how="inner"
    )

    # Filter to rows where at least one compare_col changed
    change_filter = None
    for col in compare_cols:
        cond = F.col(col) != F.col(f"old_{col}")
        change_filter = cond if change_filter is None else (change_filter | cond)

    updates = (
        joined
        .filter(change_filter)
        .select(new_prefixed.columns)
        .withColumn("cdc_operation", F.lit("UPDATE"))
    )

    # Log CDC summary
    i_count = inserts.count()
    u_count = updates.count()
    d_count = deletes.count()

    logger.info(f"CDC detected — INSERT: {i_count}, UPDATE: {u_count}, DELETE: {d_count}")

    # Persist CDC log entry
    cdc_entry = {
        "timestamp":  datetime.utcnow().isoformat(),
        "inserts":    i_count,
        "updates":    u_count,
        "deletes":    d_count,
        "pk_cols":    pk_cols,
        "compare_cols": compare_cols,
    }
    cdc_log = []
    if os.path.exists(CDC_LOG_PATH):
        with open(CDC_LOG_PATH, "r") as f:
            cdc_log = json.load(f)
    cdc_log.append(cdc_entry)
    os.makedirs(os.path.dirname(CDC_LOG_PATH), exist_ok=True)
    with open(CDC_LOG_PATH, "w") as f:
        json.dump(cdc_log, f, indent=2)

    return {"inserts": inserts, "updates": updates, "deletes": deletes}


# ═════════════════════════════════════════════════════════════════════════════
# Q22 - DELTA MERGE UPSERTS
# Merge new/changed data into existing silver tables without full reload
# ═════════════════════════════════════════════════════════════════════════════

def upsert_to_silver(
    spark: SparkSession,
    new_df: DataFrame,
    silver_path: str,
    pk_cols: list,
    update_cols: list,
    delete_col: Optional[str] = None,
) -> Dict[str, int]:
    """
    Delta MERGE implementing full CDC upsert pattern:
      - MATCHED + not soft-deleted   → UPDATE changed columns
      - MATCHED + soft-deleted        → DELETE (if delete_col provided)
      - NOT MATCHED                   → INSERT new row

    delete_col: optional column name that marks a row as deleted
                (e.g. 'is_deleted' = True means soft delete)

    Returns dict with counts: {inserted, updated, deleted}
    """
    try:
        from delta.tables import DeltaTable

        if not DeltaTable.isDeltaTable(spark, silver_path):
            # First load — write directly as Delta
            new_df.write.format("delta").mode("overwrite").save(silver_path)
            count = new_df.count()
            logger.info(f"First load to Delta table: {silver_path} ({count} rows)")
            return {"inserted": count, "updated": 0, "deleted": 0}

        delta_tbl = DeltaTable.forPath(spark, silver_path)

        merge_condition = " AND ".join(
            [f"target.{k} = source.{k}" for k in pk_cols]
        )
        update_map = {c: f"source.{c}" for c in update_cols}
        insert_map = {c: f"source.{c}" for c in new_df.columns
                      if c != (delete_col or "")}

        merge_builder = (
            delta_tbl.alias("target")
            .merge(new_df.alias("source"), merge_condition)
        )

        if delete_col:
            merge_builder = (
                merge_builder
                .whenMatchedDelete(condition=f"source.{delete_col} = true")
                .whenMatchedUpdate(
                    condition=f"source.{delete_col} = false OR source.{delete_col} IS NULL",
                    set=update_map
                )
            )
        else:
            merge_builder = merge_builder.whenMatchedUpdate(set=update_map)

        merge_builder.whenNotMatchedInsert(values=insert_map).execute()

        logger.info(f"Delta MERGE complete: {silver_path}")
        return {"inserted": -1, "updated": -1, "deleted": -1}  # Delta hides exact counts

    except ImportError:
        logger.warning("Delta not available — falling back to parquet overwrite")
        new_df.write.mode("overwrite").option("compression", "snappy").parquet(silver_path)
        return {"inserted": new_df.count(), "updated": 0, "deleted": 0}


# ═════════════════════════════════════════════════════════════════════════════
# Q23 - CHECKPOINT MECHANISM
# Save pipeline state so failed runs can resume from where they stopped
# ═════════════════════════════════════════════════════════════════════════════

class PipelineCheckpoint:
    """
    Tracks which pipeline stages have completed for a given job_run_id.
    On failure, the next run skips already-completed stages.

    Usage:
        cp = PipelineCheckpoint(job_run_id)
        if not cp.is_done("bronze_ratings"):
            ingest_ratings()
            cp.mark_done("bronze_ratings")
    """

    def __init__(self, job_run_id: str):
        self.job_run_id = job_run_id
        self._data      = self._load()

    def _load(self) -> dict:
        checkpoints = {}
        if os.path.exists(CHECKPOINT_PATH):
            with open(CHECKPOINT_PATH, "r") as f:
                checkpoints = json.load(f)
        return checkpoints.get(self.job_run_id, {
            "job_run_id":   self.job_run_id,
            "started_at":   datetime.utcnow().isoformat(),
            "completed_at": None,
            "stages":       {},
            "status":       "running",
        })

    def _save(self) -> None:
        all_checkpoints = {}
        if os.path.exists(CHECKPOINT_PATH):
            with open(CHECKPOINT_PATH, "r") as f:
                all_checkpoints = json.load(f)
        all_checkpoints[self.job_run_id] = self._data
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(all_checkpoints, f, indent=2)

    def is_done(self, stage: str) -> bool:
        """Return True if this stage already completed successfully."""
        done = self._data["stages"].get(stage, {}).get("status") == "done"
        if done:
            logger.info(f"SKIP (already done): {stage}")
        return done

    def mark_done(self, stage: str, rows_processed: int = 0) -> None:
        """Mark a stage as successfully completed."""
        self._data["stages"][stage] = {
            "status":       "done",
            "completed_at": datetime.utcnow().isoformat(),
            "rows_processed": rows_processed,
        }
        self._save()
        logger.info(f"Checkpoint saved: {stage} ({rows_processed} rows)")

    def mark_failed(self, stage: str, error: str) -> None:
        """Mark a stage as failed with the error message."""
        self._data["stages"][stage] = {
            "status":     "failed",
            "failed_at":  datetime.utcnow().isoformat(),
            "error":      error,
        }
        self._data["status"] = "failed"
        self._save()
        logger.error(f"Checkpoint failed: {stage} — {error}")

    def complete(self) -> None:
        """Mark the entire job as successfully completed."""
        self._data["completed_at"] = datetime.utcnow().isoformat()
        self._data["status"]       = "completed"
        self._save()
        logger.info(f"Job {self.job_run_id[:8]} completed successfully")

    def get_status(self) -> dict:
        return self._data


def run_with_checkpoint(
    spark: SparkSession,
    job_run_id: str,
    source_df: DataFrame,
    silver_path: str,
) -> None:
    """
    Example pipeline run using checkpoints.
    Shows how each stage is guarded and resumes from last good state.
    """
    cp = PipelineCheckpoint(job_run_id)

    try:
        # Stage 1: filter incremental
        if not cp.is_done("filter_incremental"):
            new_df, count = filter_incremental(
                source_df, "ingestion_timestamp", "ratings"
            )
            cp.mark_done("filter_incremental", count)
        else:
            new_df = source_df

        # Stage 2: write to silver
        if not cp.is_done("write_silver"):
            new_df.write.mode("overwrite").option(
                "compression", "snappy"
            ).parquet(silver_path)
            cp.mark_done("write_silver", new_df.count())

        # Stage 3: update watermark
        if not cp.is_done("update_watermark"):
            update_watermark_from_df(new_df, "ingestion_timestamp", "ratings")
            cp.mark_done("update_watermark")

        cp.complete()

    except Exception as e:
        cp.mark_failed("pipeline", str(e))
        raise


# ═════════════════════════════════════════════════════════════════════════════
# Q24 - BENCHMARK: FULL REFRESH vs INCREMENTAL
# Measure time, rows processed, and accuracy of each approach
# ═════════════════════════════════════════════════════════════════════════════

def benchmark_full_vs_incremental(
    spark: SparkSession,
    full_df: DataFrame,
    incremental_df: DataFrame,
    silver_path: str,
    pk_cols: list,
) -> Dict:
    """
    Run both strategies and compare:
      - Execution time (wall clock)
      - Rows processed
      - Output row count
      - Accuracy: do both produce identical results?

    Returns benchmark results dict saved to logs/benchmark_results.json
    """
    logger.info("Starting benchmark: full refresh vs incremental...")
    results = {}

    # ── Full refresh ──────────────────────────────────────────────────────────
    full_path = silver_path + "_benchmark_full"
    t0 = time.time()
    full_df.write.mode("overwrite").option("compression", "snappy").parquet(full_path)
    full_count = full_df.count()
    full_time  = round(time.time() - t0, 3)

    results["full_refresh"] = {
        "rows_processed": full_count,
        "rows_written":   full_count,
        "elapsed_seconds": full_time,
        "cost_proxy":      full_count,   # cost proportional to rows scanned
    }
    logger.info(f"Full refresh: {full_count} rows in {full_time}s")

    # ── Incremental ───────────────────────────────────────────────────────────
    incr_path = silver_path + "_benchmark_incr"
    t0 = time.time()

    # First write existing data (simulates pre-existing silver table)
    full_df.write.mode("overwrite").option("compression", "snappy").parquet(incr_path)

    # Now merge only the incremental slice
    incr_count = incremental_df.count()
    incremental_df.write.mode("append").option(
        "compression", "snappy"
    ).parquet(incr_path)

    # Read result and deduplicate (simulates MERGE behaviour without Delta)
    from pyspark.sql.window import Window
    merged = spark.read.option("recursiveFileLookup", "true").parquet(incr_path)
    window = Window.partitionBy(pk_cols).orderBy(
        F.col("ingestion_timestamp").desc()
    )
    merged_deduped = (
        merged
        .withColumn("_rn", F.row_number().over(window))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )
    merged_count = merged_deduped.count()
    incr_time    = round(time.time() - t0, 3)

    results["incremental"] = {
        "rows_processed":  incr_count,
        "rows_written":    merged_count,
        "elapsed_seconds": incr_time,
        "cost_proxy":      incr_count,
    }
    logger.info(f"Incremental: {incr_count} new rows, {merged_count} total in {incr_time}s")

    # ── Accuracy check ────────────────────────────────────────────────────────
    # Both approaches should produce the same final row count
    full_result  = spark.read.parquet(full_path)
    final_count  = full_result.count()
    accuracy_ok  = (merged_count == final_count)

    results["accuracy"] = {
        "full_output_rows":        final_count,
        "incremental_output_rows": merged_count,
        "row_counts_match":        accuracy_ok,
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    speedup = round(full_time / incr_time, 2) if incr_time > 0 else 0
    cost_savings_pct = round(
        (1 - incr_count / full_count) * 100, 1
    ) if full_count > 0 else 0

    results["summary"] = {
        "speedup_factor":     speedup,
        "cost_savings_pct":   cost_savings_pct,
        "recommendation":     "incremental" if speedup > 1 else "full_refresh",
        "timestamp":          datetime.utcnow().isoformat(),
    }

    logger.info(f"Benchmark complete — speedup: {speedup}x, cost saving: {cost_savings_pct}%")

    # Save results
    os.makedirs(os.path.dirname(BENCHMARK_PATH), exist_ok=True)
    with open(BENCHMARK_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MAIN - Demonstrate all incremental features
# ═════════════════════════════════════════════════════════════════════════════

def run_incremental_demo(spark: SparkSession) -> None:
    logger.info("=" * 60)
    logger.info("PART D - INCREMENTAL PROCESSING DEMO")
    logger.info("=" * 60)

    silver_ratings = os.path.join(SILVER_BASE, "ratings")
    delta_log      = os.path.join(silver_ratings, "_delta_log")

    if os.path.exists(delta_log):
        df = spark.read.format("delta").load(silver_ratings)
    else:
        df = spark.read.option("recursiveFileLookup", "true").parquet(silver_ratings)

    total = df.count()
    logger.info(f"Loaded silver.ratings: {total} rows")

    # Q20: Watermark demo
    logger.info("\n--- Q20: Watermark-based incremental load ---")
    new_df, new_count = filter_incremental(df, "ingestion_timestamp", "ratings")
    update_watermark_from_df(new_df, "ingestion_timestamp", "ratings")
    logger.info(f"After first run watermark set. Re-running...")
    _, second_count = filter_incremental(df, "ingestion_timestamp", "ratings")
    logger.info(f"Second run new rows: {second_count} (should be 0 — all caught by watermark)")

    # Q21: CDC demo — simulate a changed record
    logger.info("\n--- Q21: CDC change detection ---")
    old_snapshot = df.limit(10)
    # Simulate: modify rating of first record to trigger UPDATE detection
    first = old_snapshot.first()
    new_snapshot = old_snapshot.withColumn(
        "rating",
        F.when(
            (F.col("user_id") == first["user_id"]) &
            (F.col("movie_id") == first["movie_id"]),
            F.col("rating") + 0.5
        ).otherwise(F.col("rating"))
    )
    cdc = detect_cdc_changes(
        old_snapshot, new_snapshot,
        pk_cols=["user_id", "movie_id"],
        compare_cols=["rating"],
    )
    logger.info(f"CDC result: {cdc['inserts'].count()} inserts, "
                f"{cdc['updates'].count()} updates, "
                f"{cdc['deletes'].count()} deletes")

    # Q23: Checkpoint demo
    logger.info("\n--- Q23: Checkpoint mechanism ---")
    import uuid
    job_id = str(uuid.uuid4())
    cp = PipelineCheckpoint(job_id)
    cp.mark_done("stage_1_bronze", rows_processed=total)
    cp.mark_done("stage_2_silver", rows_processed=total)
    cp.complete()
    logger.info(f"Checkpoint status: {cp.get_status()['status']}")

    # Q24: Benchmark
    logger.info("\n--- Q24: Full refresh vs incremental benchmark ---")
    incr_slice = df.limit(max(1, total // 5))  # 20% as "new" data
    benchmark  = benchmark_full_vs_incremental(
        spark, df, incr_slice,
        silver_path=os.path.join(SILVER_BASE, "benchmark_test"),
        pk_cols=["user_id", "movie_id"],
    )
    s = benchmark["summary"]
    logger.info(f"Speedup: {s['speedup_factor']}x | "
                f"Cost saving: {s['cost_savings_pct']}% | "
                f"Recommendation: {s['recommendation']}")

    logger.info("=" * 60)
    logger.info("PART D COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    sys.path.insert(0, _PROJECT_ROOT)
    from utils.spark_session import get_spark
    spark = get_spark("IncrementalProcessing")
    run_incremental_demo(spark)
    spark.stop()
