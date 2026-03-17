"""
scripts/bronze_ingestion.py
Part A - Bronze Layer (Questions 1-5)
"""

import hashlib
import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType, TimestampType
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bronze_ingestion")

# Absolute project root - always resolves correctly regardless of cwd
_PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HASH_REGISTRY_PATH = os.path.join(_PROJECT_ROOT, "logs", "ingested_file_hashes.json")
QUARANTINE_BASE    = os.path.join(_PROJECT_ROOT, "bronze", "quarantine")
ERROR_LOG_PATH     = os.path.join(_PROJECT_ROOT, "logs", "quarantine_errors.json")


# ── Schemas ───────────────────────────────────────────────────────────────────

RATINGS_SCHEMA = StructType([
    StructField("user_id",   IntegerType(), nullable=True),
    StructField("movie_id",  IntegerType(), nullable=True),
    StructField("rating",    FloatType(),   nullable=True),
    StructField("rated_at",  StringType(),  nullable=True),
])

MOVIES_SCHEMA = StructType([
    StructField("movie_id",  IntegerType(), nullable=True),
    StructField("title",     StringType(),  nullable=True),
    StructField("genres",    StringType(),  nullable=True),
    StructField("year",      IntegerType(), nullable=True),
    StructField("language",  StringType(),  nullable=True),
    StructField("country",   StringType(),  nullable=True),
])

USERS_SCHEMA = StructType([
    StructField("user_id",     IntegerType(), nullable=True),
    StructField("signup_date", StringType(),  nullable=True),
    StructField("country",     StringType(),  nullable=True),
    StructField("plan",        StringType(),  nullable=True),
])


# ══════════════════════════════════════════════════════════════════════════════
# Q1 - FOLDER STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def get_bronze_path(
    base_path: str,
    source: str,
    entity: str,
    dt: Optional[datetime] = None,
) -> str:
    """Returns bronze/{source}/{entity}/{yyyy}/{mm}/{dd}/ and creates it."""
    dt = dt or datetime.utcnow()
    path = os.path.join(
        base_path, "bronze", source, entity,
        dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"),
    )
    os.makedirs(path, exist_ok=True)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Q4 - IDEMPOTENCY: FILE-HASH REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def _load_hash_registry() -> dict:
    if os.path.exists(HASH_REGISTRY_PATH):
        with open(HASH_REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_hash_registry(registry: dict) -> None:
    os.makedirs(os.path.dirname(HASH_REGISTRY_PATH), exist_ok=True)
    with open(HASH_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def compute_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """MD5 hash of a file - used to detect duplicate uploads."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def is_already_ingested(file_hash: str) -> bool:
    return file_hash in _load_hash_registry()


def register_ingested_file(file_hash: str, file_path: str, job_run_id: str) -> None:
    registry = _load_hash_registry()
    registry[file_hash] = {
        "file_path":    file_path,
        "job_run_id":   job_run_id,
        "ingested_at":  datetime.utcnow().isoformat(),
    }
    _save_hash_registry(registry)
    logger.info(f"Registered hash {file_hash[:8]} for {os.path.basename(file_path)}")


# ══════════════════════════════════════════════════════════════════════════════
# Q5 - QUARANTINE
# ══════════════════════════════════════════════════════════════════════════════

def quarantine_file(file_path: str, reason: str, job_run_id: str) -> None:
    """Move bad file to quarantine folder and log the reason."""
    os.makedirs(QUARANTINE_BASE, exist_ok=True)
    dest = os.path.join(QUARANTINE_BASE, os.path.basename(file_path))
    try:
        if os.path.exists(file_path):
            shutil.copy2(file_path, dest)   # copy instead of move so tests can check
    except Exception as e:
        logger.error(f"Could not copy {file_path} to quarantine: {e}")
        dest = file_path

    error_entry = {
        "file_path":       file_path,
        "quarantine_path": dest,
        "reason":          reason,
        "job_run_id":      job_run_id,
        "timestamp":       datetime.utcnow().isoformat(),
    }
    errors = []
    if os.path.exists(ERROR_LOG_PATH):
        with open(ERROR_LOG_PATH, "r") as f:
            errors = json.load(f)
    errors.append(error_entry)
    os.makedirs(os.path.dirname(ERROR_LOG_PATH), exist_ok=True)
    with open(ERROR_LOG_PATH, "w") as f:
        json.dump(errors, f, indent=2)

    logger.warning(f"Quarantined: {os.path.basename(file_path)} | reason: {reason}")


# ══════════════════════════════════════════════════════════════════════════════
# Q3 - AUDIT COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def add_audit_columns(
    df: DataFrame,
    source_file_path: str,
    job_run_id: str,
) -> DataFrame:
    """
    Adds three mandatory audit columns to every Bronze record:
      ingestion_timestamp - when this record was loaded (Spark server time)
      source_file_path    - absolute path of the originating source file
      job_run_id          - UUID tying all tables from the same pipeline run
    """
    return (
        df
        .withColumn("ingestion_timestamp", F.current_timestamp())
        .withColumn("source_file_path",    F.lit(source_file_path))
        .withColumn("job_run_id",          F.lit(job_run_id))
    )


# ══════════════════════════════════════════════════════════════════════════════
# Q2 - MULTI-FORMAT READER
# ══════════════════════════════════════════════════════════════════════════════

def _infer_format(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    mapping = {".csv": "csv", ".json": "json", ".parquet": "parquet"}
    fmt = mapping.get(ext)
    if not fmt:
        raise ValueError(f"Unsupported file format: {ext!r}")
    return fmt


def read_source_file(
    spark: SparkSession,
    file_path: str,
    schema: StructType,
) -> DataFrame:
    """Read CSV, JSON, or Parquet - preserving raw values, enforcing schema."""
    fmt = _infer_format(file_path)
    logger.info(f"Reading {fmt.upper()}: {os.path.basename(file_path)}")

    if fmt == "csv":
        return (
            spark.read
            .option("header", "true")
            .option("mode", "PERMISSIVE")
            .option("columnNameOfCorruptRecord", "_corrupt_record")
            .schema(schema)
            .csv(file_path)
        )
    elif fmt == "json":
        return (
            spark.read
            .option("multiLine", "true")
            .option("mode", "PERMISSIVE")
            .option("columnNameOfCorruptRecord", "_corrupt_record")
            .schema(schema)
            .json(file_path)
        )
    else:  # parquet
        df = spark.read.parquet(file_path)
        # Validate expected columns are present
        schema_cols = {f.name for f in schema.fields}
        actual_cols = set(df.columns)
        missing = schema_cols - actual_cols
        if missing:
            raise ValueError(f"Parquet missing expected columns: {missing}")
        return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN INGESTION FUNCTION (Q1-Q5 combined)
# ══════════════════════════════════════════════════════════════════════════════

def ingest_to_bronze(
    spark: SparkSession,
    source_file_path: str,
    source: str,
    entity: str,
    schema: StructType,
    base_path: str = None,
    job_run_id: Optional[str] = None,
) -> Optional[str]:
    """
    Full Bronze ingestion for one file. Returns output path or None if
    the file was skipped (duplicate) or quarantined (bad file).

    Steps:
      1. Check file exists
      2. Compute hash - skip if already ingested  (Q4)
      3. Read file in native format               (Q2)
      4. Validate not empty                       (Q5)
      5. Add audit columns                        (Q3)
      6. Write to bronze/{source}/{entity}/date/  (Q1)
      7. Register hash                            (Q4)
    """
    if base_path is None:
        base_path = _PROJECT_ROOT

    job_run_id = job_run_id or str(uuid.uuid4())
    fname = os.path.basename(source_file_path)
    logger.info(f"[{job_run_id[:8]}] Ingesting {source}/{entity}: {fname}")

    # Step 1 - file must exist
    if not os.path.exists(source_file_path):
        logger.error(f"File not found: {source_file_path}")
        return None

    # Step 2 - idempotency check
    try:
        file_hash = compute_file_hash(source_file_path)
    except Exception as e:
        logger.error(f"Could not hash {fname}: {e}")
        return None

    if is_already_ingested(file_hash):
        logger.info(f"SKIP (already ingested): {fname}")
        return None

    # Step 3 - read
    try:
        df = read_source_file(spark, source_file_path, schema)
    except Exception as e:
        logger.error(f"Read failed for {fname}: {e}")
        quarantine_file(source_file_path, reason=f"Read error: {e}", job_run_id=job_run_id)
        return None

    # Step 4 - validate not empty
    try:
        count = df.count()
        if count == 0:
            raise ValueError("File produced 0 rows")
    except ValueError as e:
        quarantine_file(source_file_path, reason=str(e), job_run_id=job_run_id)
        return None

    # Step 5 - add audit columns
    df = add_audit_columns(df, source_file_path, job_run_id)

    # Step 6 - write
    # Use job_run_id subfolder so concurrent runs never overwrite each other
    today = datetime.utcnow()
    output_path = os.path.join(
        get_bronze_path(base_path, source, entity, today),
        job_run_id[:8]       # short prefix keeps paths readable
    )

    try:
        (
            df.write
            .mode("overwrite")
            .option("compression", "snappy")
            .parquet(output_path)
        )
        logger.info(f"Written {count} rows to {output_path}")
    except Exception as e:
        logger.error(f"Write failed for {fname}: {e}")
        quarantine_file(source_file_path, reason=f"Write error: {e}", job_run_id=job_run_id)
        return None

    # Step 7 - register hash (only after successful write)
    register_ingested_file(file_hash, source_file_path, job_run_id)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# BATCH INGESTION
# ══════════════════════════════════════════════════════════════════════════════

def ingest_directory(
    spark: SparkSession,
    source_dir: str,
    source: str,
    entity: str,
    schema: StructType,
    base_path: str = None,
    supported_formats: tuple = (".csv", ".json", ".parquet"),
    job_run_id: Optional[str] = None,
) -> List[str]:
    """Ingest all supported files in a directory. Returns list of output paths."""
    if base_path is None:
        base_path = _PROJECT_ROOT

    job_run_id = job_run_id or str(uuid.uuid4())
    source_files = sorted([
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if Path(f).suffix.lower() in supported_formats
    ])

    if not source_files:
        logger.warning(f"No supported files found in {source_dir}")
        return []

    logger.info(f"Found {len(source_files)} file(s) in {source_dir}")
    output_paths = []
    for file_path in source_files:
        result = ingest_to_bronze(
            spark, file_path, source, entity, schema,
            base_path=base_path, job_run_id=job_run_id,
        )
        if result:
            output_paths.append(result)

    logger.info(f"Batch complete: {len(output_paths)}/{len(source_files)} ingested")
    return output_paths
