"""
scripts/silver_transforms.py
=======================================================
Part B - Silver Layer (Questions 6-13)
=======================================================

Q6  Normalized star schema: users, movies, ratings, genres
Q7  Data quality checks: nulls, ranges, referential integrity, duplicates
Q8  Standardize: dates to ISO, country codes to ISO-3166
Q9  Deduplication: keep latest record per user+movie
Q10 Late-arriving data: Delta MERGE upserts
Q11 SCD Type 2 for user dimension
Q12 DQ metrics table: null_count, duplicate_count, failed_checks per table/day
Q13 Enriched silver table joining bronze sources with business logic
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, TimestampType, BooleanType, DateType
)
from pyspark.sql.window import Window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("silver_transforms")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILVER_BASE   = os.path.join(_PROJECT_ROOT, "silver")
DQ_LOG_PATH   = os.path.join(_PROJECT_ROOT, "logs", "dq_metrics.json")

# ISO-3166 country code normalization map (common variants → standard)
COUNTRY_CODE_MAP = {
    "USA": "US", "UNITED STATES": "US", "U.S.": "US",
    "UK":  "GB", "UNITED KINGDOM": "GB", "BRITAIN": "GB",
    "GER": "DE", "GERMANY": "DE",
    "FRA": "FR", "FRANCE": "FR",
    "JPN": "JP", "JAPAN": "JP",
    "KOR": "KR", "KOREA": "KR",
    "IND": "IN", "INDIA": "IN",
}


# ═════════════════════════════════════════════════════════════════════════════
# Q6 - STAR SCHEMA DESIGN
# Tables: silver_ratings, silver_movies, silver_users, silver_genres
# ═════════════════════════════════════════════════════════════════════════════

def get_silver_path(entity: str) -> str:
    """Returns silver/{entity}/ path, creating it if needed."""
    path = os.path.join(SILVER_BASE, entity)
    os.makedirs(path, exist_ok=True)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# Q8 - STANDARDIZATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def standardize_dates(df: DataFrame, date_cols: list) -> DataFrame:
    """
    Cast date/timestamp string columns to proper TimestampType.
    Uses try_to_timestamp to handle both datetime and date-only strings.
    """
    for col in date_cols:
        if col in df.columns:
            df = df.withColumn(
                col,
                F.coalesce(
                    F.expr(f"try_to_timestamp(`{col}`, 'yyyy-MM-dd HH:mm:ss')"),
                    F.expr(f"try_to_timestamp(`{col}`, 'yyyy-MM-dd')"),
                    F.expr(f"try_to_timestamp(`{col}`, 'MM/dd/yyyy')"),
                )
            )
    return df


def standardize_country_codes(df: DataFrame, country_col: str) -> DataFrame:
    """
    Normalize country codes to ISO-3166 alpha-2.
    Uppercases first, then maps known variants to standard codes.
    """
    if country_col not in df.columns:
        return df

    # Build a Spark map from the Python dict
    mapping_expr = F.create_map(
        *[item for pair in
          [(F.lit(k), F.lit(v)) for k, v in COUNTRY_CODE_MAP.items()]
          for item in pair]
    )
    df = df.withColumn(
        country_col,
        F.coalesce(
            mapping_expr[F.upper(F.col(country_col))],
            F.upper(F.col(country_col))   # already correct → just uppercase
        )
    )
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Q7 - DATA QUALITY CHECKS
# ═════════════════════════════════════════════════════════════════════════════

class DataQualityResult:
    """Holds the outcome of a DQ check run on one table."""
    def __init__(self, table_name: str):
        self.table_name   = table_name
        self.check_date   = datetime.utcnow().strftime("%Y-%m-%d")
        self.null_counts:  Dict[str, int] = {}
        self.range_fails:  Dict[str, int] = {}
        self.ref_fails:    Dict[str, int] = {}
        self.dup_count:    int = 0
        self.total_rows:   int = 0
        self.passed:       bool = True

    def total_failed_checks(self) -> int:
        return (
            sum(self.null_counts.values()) +
            sum(self.range_fails.values()) +
            sum(self.ref_fails.values()) +
            self.dup_count
        )

    def to_dict(self) -> dict:
        return {
            "table_name":        self.table_name,
            "check_date":        self.check_date,
            "total_rows":        self.total_rows,
            "null_counts":       self.null_counts,
            "range_fails":       self.range_fails,
            "referential_fails": self.ref_fails,
            "duplicate_count":   self.dup_count,
            "total_failed":      self.total_failed_checks(),
            "passed":            self.passed,
        }


def check_nulls(df: DataFrame, not_null_cols: list, result: DataQualityResult) -> DataFrame:
    """Q7a: Flag rows where mandatory columns are null."""
    for col in not_null_cols:
        if col not in df.columns:
            continue
        count = df.filter(F.col(col).isNull()).count()
        result.null_counts[col] = count
        if count > 0:
            result.passed = False
            logger.warning(f"  [DQ] {result.table_name}.{col}: {count} null values")
    return df


def check_rating_range(df: DataFrame, result: DataQualityResult) -> DataFrame:
    """Q7b: Rating must be between 1.0 and 5.0."""
    if "rating" not in df.columns:
        return df
    bad = df.filter(
        F.col("rating").isNotNull() &
        ((F.col("rating") < 1.0) | (F.col("rating") > 5.0))
    ).count()
    result.range_fails["rating"] = bad
    if bad > 0:
        result.passed = False
        logger.warning(f"  [DQ] {result.table_name}.rating: {bad} out-of-range values")
    return df


def check_referential_integrity(
    df: DataFrame,
    fk_col: str,
    ref_df: DataFrame,
    ref_col: str,
    result: DataQualityResult,
) -> DataFrame:
    """Q7c: Every FK value must exist in the reference table."""
    ref_vals = ref_df.select(ref_col).distinct()
    orphans  = df.join(ref_vals, df[fk_col] == ref_vals[ref_col], "left_anti").count()
    result.ref_fails[f"{fk_col}_orphans"] = orphans
    if orphans > 0:
        result.passed = False
        logger.warning(f"  [DQ] {result.table_name}.{fk_col}: {orphans} orphan FK values")
    return df


def check_duplicates(
    df: DataFrame,
    key_cols: list,
    result: DataQualityResult,
) -> DataFrame:
    """Q7d: Detect duplicate rows by composite key."""
    dup_count = (
        df.groupBy(key_cols)
        .count()
        .filter(F.col("count") > 1)
        .agg(F.sum(F.col("count") - 1))
        .collect()[0][0] or 0
    )
    result.dup_count = int(dup_count)
    if dup_count > 0:
        logger.warning(f"  [DQ] {result.table_name}: {dup_count} duplicate rows")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Q9 - DEDUPLICATION
# Keep the latest record per (user_id, movie_id) based on rated_at timestamp
# ═════════════════════════════════════════════════════════════════════════════

def deduplicate(df: DataFrame, key_cols: list, order_col: str) -> DataFrame:
    """
    Keep exactly one row per composite key - the one with the latest order_col.
    Uses a Window function (row_number) so it's efficient on large datasets.
    """
    window = Window.partitionBy(key_cols).orderBy(F.col(order_col).desc())
    return (
        df.withColumn("_rn", F.row_number().over(window))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


# ═════════════════════════════════════════════════════════════════════════════
# Q10 - LATE-ARRIVING DATA: Delta MERGE (UPSERT)
# Merge new/late records into existing silver table without full reload
# ═════════════════════════════════════════════════════════════════════════════

def merge_into_silver(
    spark: SparkSession,
    new_df: DataFrame,
    silver_path: str,
    merge_keys: list,
    update_cols: list,
) -> None:
    """
    Delta MERGE: insert new rows, update existing rows if new data is later.
    - merge_keys:  columns that identify a unique record  (e.g. user_id, movie_id)
    - update_cols: columns to update when a match is found (e.g. rating, rated_at)

    Falls back to overwrite if the Delta table doesn't exist yet (first load).
    """
    try:
        from delta.tables import DeltaTable

        if DeltaTable.isDeltaTable(spark, silver_path):
            delta_tbl = DeltaTable.forPath(spark, silver_path)

            # Build merge condition from keys
            merge_condition = " AND ".join(
                [f"target.{k} = source.{k}" for k in merge_keys]
            )

            # Only update if the incoming record is newer
            update_condition = "source.rated_at >= target.rated_at" \
                if "rated_at" in update_cols else "true"

            update_map  = {c: f"source.{c}" for c in update_cols}
            insert_map  = {c: f"source.{c}" for c in new_df.columns}

            (
                delta_tbl.alias("target")
                .merge(new_df.alias("source"), merge_condition)
                .whenMatchedUpdate(condition=update_condition, set=update_map)
                .whenNotMatchedInsert(values=insert_map)
                .execute()
            )
            logger.info(f"Delta MERGE complete into {silver_path}")
        else:
            # First load - write as Delta table
            new_df.write.format("delta").mode("overwrite").save(silver_path)
            logger.info(f"Delta table created at {silver_path}")

    except ImportError:
        # Delta not available - fall back to parquet overwrite
        logger.warning("Delta Lake not available, falling back to parquet overwrite")
        new_df.write.mode("overwrite").option("compression", "snappy").parquet(silver_path)


# ═════════════════════════════════════════════════════════════════════════════
# Q11 - SCD TYPE 2 FOR USER DIMENSION
# Track full history of user attribute changes (plan, country)
# ═════════════════════════════════════════════════════════════════════════════

def apply_scd2(
    spark: SparkSession,
    new_df: DataFrame,
    silver_path: str,
    pk_col: str = "user_id",
    tracked_cols: list = None,
) -> DataFrame:
    """
    SCD Type 2: when a tracked attribute changes, expire the old row and
    insert a new one. Every row gets:
      scd_start_date  - when this version became active
      scd_end_date    - when it was superseded (NULL = current record)
      is_current      - True for the active record

    On first load, all rows are inserted as current with no end date.
    """
    tracked_cols = tracked_cols or ["plan", "country"]
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Add SCD columns to incoming data
    new_df = (
        new_df
        .withColumn("scd_start_date", F.lit(now).cast(TimestampType()))
        .withColumn("scd_end_date",   F.lit(None).cast(TimestampType()))
        .withColumn("is_current",     F.lit(True))
    )

    try:
        from delta.tables import DeltaTable

        if not DeltaTable.isDeltaTable(spark, silver_path):
            # First load - all rows are new and current
            new_df.write.format("delta").mode("overwrite").save(silver_path)
            logger.info(f"SCD2: initial load to {silver_path} ({new_df.count()} rows)")
            return spark.read.format("delta").load(silver_path)

        existing = spark.read.format("delta").load(silver_path)
        delta_tbl = DeltaTable.forPath(spark, silver_path)

        # Find rows where tracked attributes have changed
        join_cond = existing[pk_col] == new_df[pk_col]
        changed_filter = " OR ".join(
            [f"existing.{c} != new_df.{c}" for c in tracked_cols]
        )

        changed = (
            existing.alias("existing")
            .join(new_df.alias("new_df"), join_cond)
            .filter(
                " OR ".join([
                    f"existing.{c} != new_df.{c}" for c in tracked_cols
                ])
            )
            .select("existing.*")
        )

        # Expire old records by setting end_date and is_current=False
        if changed.count() > 0:
            changed_ids = [row[pk_col] for row in changed.select(pk_col).collect()]
            (
                delta_tbl.update(
                    condition=F.col(pk_col).isin(changed_ids) & F.col("is_current"),
                    set={
                        "scd_end_date": F.lit(now).cast(TimestampType()),
                        "is_current":   F.lit(False),
                    }
                )
            )
            logger.info(f"SCD2: expired {len(changed_ids)} changed records")

        # Insert new versions
        new_df.write.format("delta").mode("append").save(silver_path)
        logger.info(f"SCD2: inserted {new_df.count()} new record versions")

    except ImportError:
        # Fallback: parquet with SCD columns, overwrite
        logger.warning("Delta not available - SCD2 written as parquet snapshot")
        new_df.write.mode("overwrite").option("compression", "snappy").parquet(silver_path)

    return spark.read.format("delta").load(silver_path) \
        if os.path.exists(os.path.join(silver_path, "_delta_log")) \
        else spark.read.parquet(silver_path)


# ═════════════════════════════════════════════════════════════════════════════
# Q12 - DQ METRICS TABLE
# Record null_count, duplicate_count, failed_checks per table per day
# ═════════════════════════════════════════════════════════════════════════════

def save_dq_metrics(result: DataQualityResult) -> None:
    """Append DQ result to the running metrics log (logs/dq_metrics.json)."""
    metrics = []
    if os.path.exists(DQ_LOG_PATH):
        with open(DQ_LOG_PATH, "r") as f:
            metrics = json.load(f)
    metrics.append(result.to_dict())
    os.makedirs(os.path.dirname(DQ_LOG_PATH), exist_ok=True)
    with open(DQ_LOG_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"DQ metrics saved for {result.table_name}: "
                f"passed={result.passed}, failed={result.total_failed_checks()}")


def build_dq_metrics_df(spark: SparkSession) -> DataFrame:
    """
    Load the DQ metrics log and return it as a Spark DataFrame.
    This is the silver.dq_metrics table (Q12).
    """
    if not os.path.exists(DQ_LOG_PATH):
        logger.warning("No DQ metrics log found")
        return spark.createDataFrame([], StructType([
            StructField("table_name",      StringType(),  True),
            StructField("check_date",      StringType(),  True),
            StructField("total_rows",      IntegerType(), True),
            StructField("duplicate_count", IntegerType(), True),
            StructField("total_failed",    IntegerType(), True),
            StructField("passed",          BooleanType(), True),
        ]))

    with open(DQ_LOG_PATH, "r") as f:
        metrics = json.load(f)

    rows = []
    for m in metrics:
        rows.append((
            m["table_name"],
            m["check_date"],
            m["total_rows"],
            m["duplicate_count"],
            m["total_failed"],
            m["passed"],
        ))

    schema = StructType([
        StructField("table_name",      StringType(),  True),
        StructField("check_date",      StringType(),  True),
        StructField("total_rows",      IntegerType(), True),
        StructField("duplicate_count", IntegerType(), True),
        StructField("total_failed",    IntegerType(), True),
        StructField("passed",          BooleanType(), True),
    ])
    return spark.createDataFrame(rows, schema)


# ═════════════════════════════════════════════════════════════════════════════
# Q6 + Q13 - TRANSFORM BRONZE → SILVER
# Build each silver table: ratings, movies, users, genres, enriched_ratings
# ═════════════════════════════════════════════════════════════════════════════

def read_latest_bronze(spark: SparkSession, source: str, entity: str) -> DataFrame:
    """Read all parquet files from bronze/{source}/{entity}/ recursively."""
    bronze_path = os.path.join(_PROJECT_ROOT, "bronze", source, entity)
    if not os.path.exists(bronze_path):
        raise FileNotFoundError(f"Bronze path not found: {bronze_path}")
    return spark.read.option("recursiveFileLookup", "true").parquet(bronze_path)


def build_silver_ratings(spark: SparkSession) -> Tuple[DataFrame, DataQualityResult]:
    """
    Q6/Q7/Q8/Q9: Build silver.ratings
    - Cast types, standardize dates
    - DQ: null checks, range checks, duplicate detection
    - Deduplicate by (user_id, movie_id), keep latest
    """
    logger.info("Building silver.ratings...")
    df = read_latest_bronze(spark, "netflix", "ratings")
    result = DataQualityResult("silver_ratings")
    result.total_rows = df.count()

    # Q8: Standardize rated_at to TimestampType
    df = standardize_dates(df, ["rated_at", "ingestion_timestamp"])

    # Q7a: Null checks on mandatory columns
    check_nulls(df, ["user_id", "movie_id", "rating"], result)

    # Q7b: Rating range check (1.0 - 5.0)
    check_rating_range(df, result)

    # Drop audit columns not needed in silver
    df = df.drop("_corrupt_record")

    # Q7d + Q9: Check duplicates then deduplicate
    check_duplicates(df, ["user_id", "movie_id"], result)
    df = deduplicate(df, ["user_id", "movie_id"], "rated_at")

    # Q12: Save DQ metrics
    save_dq_metrics(result)

    logger.info(f"silver.ratings: {df.count()} rows after dedup")
    return df, result


def build_silver_movies(spark: SparkSession) -> Tuple[DataFrame, DataQualityResult]:
    """
    Q6/Q7/Q8: Build silver.movies + silver.genres (exploded)
    - Standardize country codes, cast year
    - DQ: null checks on movie_id, title
    - Explode pipe-separated genres into silver.genres
    """
    logger.info("Building silver.movies...")
    df = read_latest_bronze(spark, "tmdb", "movies")
    result = DataQualityResult("silver_movies")
    result.total_rows = df.count()

    # Q8: Standardize country codes to ISO-3166
    df = standardize_country_codes(df, "country")

    # Q7a: Null checks
    check_nulls(df, ["movie_id", "title"], result)

    # Deduplicate movies by movie_id
    df = deduplicate(df, ["movie_id"], "ingestion_timestamp")

    save_dq_metrics(result)
    logger.info(f"silver.movies: {df.count()} rows")
    return df, result


def build_silver_genres(movies_df: DataFrame) -> DataFrame:
    """
    Q6: Explode pipe-separated genres column into a separate genres dimension.
    Result: silver.genres with (movie_id, genre) rows - one per genre per movie.
    """
    logger.info("Building silver.genres...")
    genres_df = (
        movies_df
        .select("movie_id", F.explode(F.split(F.col("genres"), r"\|")).alias("genre"))
        .filter(F.col("genre").isNotNull() & (F.col("genre") != ""))
        .distinct()
    )
    logger.info(f"silver.genres: {genres_df.count()} rows")
    return genres_df


def build_silver_users(spark: SparkSession) -> Tuple[DataFrame, DataQualityResult]:
    """
    Q6/Q7/Q8/Q11: Build silver.users with SCD Type 2
    - Standardize signup_date, country codes
    - DQ: null checks on user_id
    - Apply SCD Type 2 to track plan/country changes
    """
    logger.info("Building silver.users...")
    df = read_latest_bronze(spark, "netflix", "users")
    result = DataQualityResult("silver_users")
    result.total_rows = df.count()

    # Q8: Standardize dates and country codes
    df = standardize_dates(df, ["signup_date"])
    df = standardize_country_codes(df, "country")

    # Q7a: Null checks
    check_nulls(df, ["user_id"], result)

    # Deduplicate to get one row per user (latest)
    df = deduplicate(df, ["user_id"], "ingestion_timestamp")

    save_dq_metrics(result)

    # Q11: Apply SCD Type 2
    scd2_path = get_silver_path("users_scd2")
    df_scd2 = apply_scd2(spark, df, scd2_path, pk_col="user_id",
                          tracked_cols=["plan", "country"])

    logger.info(f"silver.users: {df.count()} current rows")
    return df, result


def build_silver_enriched_ratings(
    spark: SparkSession,
    ratings_df: DataFrame,
    movies_df: DataFrame,
    users_df: DataFrame,
) -> DataFrame:
    """
    Q13: Join bronze tables to create silver.enriched_ratings.
    Adds movie title, genres, user plan, and user country to each rating.
    This is the primary table used by the Gold layer.
    """
    logger.info("Building silver.enriched_ratings...")

    enriched = (
        ratings_df
        # Join movie metadata
        .join(
            movies_df.select("movie_id", "title", "genres", "year", "country")
                     .withColumnRenamed("country", "movie_country"),
            on="movie_id",
            how="left"
        )
        # Join user metadata
        .join(
            users_df.select("user_id", "plan", "country", "signup_date")
                    .withColumnRenamed("country", "user_country"),
            on="user_id",
            how="left"
        )
        # Business logic columns
        .withColumn("rating_year",  F.year(F.col("rated_at")))
        .withColumn("rating_month", F.month(F.col("rated_at")))
        .withColumn("rating_day",   F.dayofmonth(F.col("rated_at")))
        .withColumn("is_high_rating", F.col("rating") >= 4.0)
        # Days since signup when rating was given
        .withColumn(
            "days_since_signup",
            F.datediff(F.col("rated_at").cast(DateType()),
                       F.col("signup_date").cast(DateType()))
        )
    )

    count = enriched.count()
    logger.info(f"silver.enriched_ratings: {count} rows")
    return enriched


# ═════════════════════════════════════════════════════════════════════════════
# MAIN - Run full Silver pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_silver_pipeline(spark: SparkSession) -> Dict[str, DataFrame]:
    """
    Orchestrates all Silver transforms in the correct dependency order.
    Returns dict of all silver DataFrames for downstream Gold layer use.
    """
    logger.info("=" * 60)
    logger.info("PART B - SILVER LAYER PIPELINE")
    logger.info("=" * 60)

    # Build each silver table
    ratings_df, ratings_dq = build_silver_ratings(spark)
    movies_df,  movies_dq  = build_silver_movies(spark)
    genres_df              = build_silver_genres(movies_df)
    users_df,   users_dq   = build_silver_users(spark)

    # Q13: Enriched table joining all sources
    enriched_df = build_silver_enriched_ratings(
        spark, ratings_df, movies_df, users_df
    )

    # Q7c: Referential integrity - user_id in ratings must exist in users
    ratings_result = DataQualityResult("silver_ratings_ref")
    ratings_result.total_rows = ratings_df.count()
    check_referential_integrity(
        ratings_df, "user_id", users_df, "user_id", ratings_result
    )
    check_referential_integrity(
        ratings_df, "movie_id", movies_df, "movie_id", ratings_result
    )
    save_dq_metrics(ratings_result)

    # Write all silver tables
    tables = {
        "ratings":          ratings_df,
        "movies":           movies_df,
        "genres":           genres_df,
        "enriched_ratings": enriched_df,
    }

    for name, df in tables.items():
        path = get_silver_path(name)
        # Q10: Use Delta MERGE for ratings (handles late-arriving data)
        if name == "ratings":
            merge_into_silver(
                spark, df, path,
                merge_keys=["user_id", "movie_id"],
                update_cols=["rating", "rated_at", "ingestion_timestamp"],
            )
        else:
            df.write.mode("overwrite").option("compression", "snappy").parquet(path)
        logger.info(f"Written silver.{name} → {path}")

    # Q12: Write DQ metrics table
    dq_df = build_dq_metrics_df(spark)
    dq_path = get_silver_path("dq_metrics")
    dq_df.write.mode("overwrite").parquet(dq_path)
    logger.info(f"Written silver.dq_metrics → {dq_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("SILVER PIPELINE COMPLETE")
    logger.info(f"  silver.ratings:          {ratings_df.count()} rows")
    logger.info(f"  silver.movies:           {movies_df.count()} rows")
    logger.info(f"  silver.genres:           {genres_df.count()} rows")
    logger.info(f"  silver.enriched_ratings: {enriched_df.count()} rows")
    logger.info(f"  silver.dq_metrics:       {dq_df.count()} rows")
    logger.info("=" * 60)

    tables["users"]      = users_df
    tables["dq_metrics"] = dq_df
    return tables


if __name__ == "__main__":
    sys.path.insert(0, _PROJECT_ROOT)
    from utils.spark_session import get_spark
    spark = get_spark("SilverPipeline")
    run_silver_pipeline(spark)
    spark.stop()
