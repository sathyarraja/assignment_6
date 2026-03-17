"""
scripts/gold_transforms.py
=======================================================
Part C - Gold Layer (Questions 14-19)
=======================================================

Q14 Star schema: fact_ratings, dim_users, dim_movies, dim_dates
Q15 Daily user engagement: active_users, avg_rating, total_ratings
Q16 Content popularity: top 100 movies by ratings, genre, time period
Q17 Customer segmentation: RFM analysis (Recency, Frequency, Monetary)
Q18 Retention cohort analysis: user retention rate by signup month
Q19 Time-series aggregates: daily, weekly, monthly KPIs
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, FloatType, DateType
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gold_transforms")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILVER_BASE   = os.path.join(_PROJECT_ROOT, "silver")
GOLD_BASE     = os.path.join(_PROJECT_ROOT, "gold")


def get_gold_path(entity: str) -> str:
    path = os.path.join(GOLD_BASE, entity)
    os.makedirs(path, exist_ok=True)
    return path


def read_silver(spark: SparkSession, entity: str) -> DataFrame:
    """Read a silver table - tries Delta first, falls back to parquet."""
    path = os.path.join(SILVER_BASE, entity)
    delta_log = os.path.join(path, "_delta_log")
    if os.path.exists(delta_log):
        return spark.read.format("delta").load(path)
    return spark.read.option("recursiveFileLookup", "true").parquet(path)


def write_gold(df: DataFrame, entity: str) -> str:
    """Write a gold table as parquet, return the path."""
    path = get_gold_path(entity)
    df.write.mode("overwrite").option("compression", "snappy").parquet(path)
    logger.info(f"Written gold.{entity} → {path}  ({df.count()} rows)")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# Q14 - STAR SCHEMA
# fact_ratings  (central fact table)
# dim_users     (user attributes - current snapshot)
# dim_movies    (movie attributes)
# dim_dates     (date dimension with calendar attributes)
# ═════════════════════════════════════════════════════════════════════════════

def build_dim_dates(spark: SparkSession, enriched_df: DataFrame) -> DataFrame:
    """
    Build dim_dates from all distinct dates in the enriched ratings table.
    Includes full calendar attributes for flexible time-based analysis.
    """
    logger.info("Building gold.dim_dates...")
    dim_dates = (
        enriched_df
        .select(F.col("rated_at").cast(DateType()).alias("date"))
        .filter(F.col("date").isNotNull())
        .distinct()
        .withColumn("year",        F.year("date"))
        .withColumn("quarter",     F.quarter("date"))
        .withColumn("month",       F.month("date"))
        .withColumn("month_name",  F.date_format("date", "MMMM"))
        .withColumn("week",        F.weekofyear("date"))
        .withColumn("day",         F.dayofmonth("date"))
        .withColumn("day_of_week", F.dayofweek("date"))
        .withColumn("day_name",    F.date_format("date", "EEEE"))
        .withColumn("is_weekend",
                    F.dayofweek("date").isin([1, 7]))  # 1=Sun, 7=Sat
        .withColumn("date_key",    F.date_format("date", "yyyyMMdd").cast(IntegerType()))
        .orderBy("date")
    )
    return dim_dates


def build_dim_users(spark: SparkSession, users_df: DataFrame) -> DataFrame:
    """
    dim_users: one row per user, current attributes only.
    Filters SCD2 table to is_current = True if that column exists.
    """
    logger.info("Building gold.dim_users...")
    if "is_current" in users_df.columns:
        users_df = users_df.filter(F.col("is_current") == True)

    return (
        users_df
        .select(
            F.col("user_id"),
            F.col("country").alias("user_country"),
            F.col("plan"),
            F.col("signup_date"),
            F.col("signup_date").cast(DateType()).alias("signup_date_key"),
        )
        .distinct()
    )


def build_dim_movies(spark: SparkSession, movies_df: DataFrame) -> DataFrame:
    """dim_movies: one row per movie with all descriptive attributes."""
    logger.info("Building gold.dim_movies...")
    return (
        movies_df
        .select(
            "movie_id", "title", "genres",
            "year", "language", "country"
        )
        .withColumnRenamed("country", "movie_country")
        .distinct()
    )


def build_fact_ratings(
    spark: SparkSession,
    enriched_df: DataFrame,
) -> DataFrame:
    """
    fact_ratings: the central fact table.
    Contains all measurable facts (rating) with FK references to dimensions.
    Degenerate dimensions (year/month) included for query performance.
    """
    logger.info("Building gold.fact_ratings...")
    return (
        enriched_df
        .select(
            "user_id",
            "movie_id",
            F.col("rated_at").cast(DateType()).alias("rating_date"),
            "rating",
            "rating_year",
            "rating_month",
            "is_high_rating",
            "days_since_signup",
            "plan",
            "user_country",
        )
        .filter(F.col("user_id").isNotNull() & F.col("movie_id").isNotNull())
    )


# ═════════════════════════════════════════════════════════════════════════════
# Q15 - DAILY USER ENGAGEMENT METRICS
# active_users, avg_rating, total_ratings per day
# ═════════════════════════════════════════════════════════════════════════════

def build_daily_engagement(fact_df: DataFrame) -> DataFrame:
    """
    Daily engagement KPIs:
      active_users   - distinct users who rated something that day
      total_ratings  - total number of ratings submitted
      avg_rating     - mean rating score
      high_ratings   - count of ratings >= 4.0
      low_ratings    - count of ratings < 3.0
    """
    logger.info("Building gold.daily_engagement...")
    return (
        fact_df
        .groupBy("rating_date")
        .agg(
            F.countDistinct("user_id").alias("active_users"),
            F.count("rating").alias("total_ratings"),
            F.round(F.avg("rating"), 3).alias("avg_rating"),
            F.sum(F.col("is_high_rating").cast(IntegerType())).alias("high_ratings"),
            F.count(F.when(F.col("rating") < 3.0, 1)).alias("low_ratings"),
        )
        .withColumn(
            "high_rating_pct",
            F.round(F.col("high_ratings") / F.col("total_ratings") * 100, 1)
        )
        .orderBy("rating_date")
    )


# ═════════════════════════════════════════════════════════════════════════════
# Q16 - CONTENT POPULARITY
# Top 100 movies by total ratings, by genre, by time period
# ═════════════════════════════════════════════════════════════════════════════

def build_content_popularity(
    fact_df: DataFrame,
    movies_df: DataFrame,
    genres_df: DataFrame,
    top_n: int = 100,
) -> Dict[str, DataFrame]:
    """
    Three content popularity views:
      top_movies_overall  - top N by total rating count
      top_movies_by_genre - top N per genre
      top_movies_by_month - top N per calendar month
    """
    logger.info("Building gold.content_popularity...")

    # Join facts with movie titles
    rated_movies = fact_df.join(
        movies_df.select("movie_id", "title"),
        on="movie_id", how="left"
    )

    # Top N overall
    top_overall = (
        rated_movies
        .groupBy("movie_id", "title")
        .agg(
            F.count("rating").alias("total_ratings"),
            F.round(F.avg("rating"), 3).alias("avg_rating"),
            F.countDistinct("user_id").alias("unique_raters"),
        )
        .withColumn("rank", F.row_number().over(
            Window.orderBy(F.col("total_ratings").desc())
        ))
        .filter(F.col("rank") <= top_n)
        .orderBy("rank")
    )

    # Top N per genre (explode genres first)
    rated_with_genres = fact_df.join(
        genres_df, on="movie_id", how="left"
    ).join(
        movies_df.select("movie_id", "title"), on="movie_id", how="left"
    )

    top_by_genre = (
        rated_with_genres
        .filter(F.col("genre").isNotNull())
        .groupBy("genre", "movie_id", "title")
        .agg(
            F.count("rating").alias("total_ratings"),
            F.round(F.avg("rating"), 3).alias("avg_rating"),
        )
        .withColumn("rank_in_genre", F.row_number().over(
            Window.partitionBy("genre")
                  .orderBy(F.col("total_ratings").desc())
        ))
        .filter(F.col("rank_in_genre") <= top_n)
        .orderBy("genre", "rank_in_genre")
    )

    # Top N per month
    top_by_month = (
        rated_movies
        .groupBy("rating_year", "rating_month", "movie_id", "title")
        .agg(
            F.count("rating").alias("total_ratings"),
            F.round(F.avg("rating"), 3).alias("avg_rating"),
        )
        .withColumn("rank_in_month", F.row_number().over(
            Window.partitionBy("rating_year", "rating_month")
                  .orderBy(F.col("total_ratings").desc())
        ))
        .filter(F.col("rank_in_month") <= top_n)
        .orderBy("rating_year", "rating_month", "rank_in_month")
    )

    return {
        "top_movies_overall":  top_overall,
        "top_movies_by_genre": top_by_genre,
        "top_movies_by_month": top_by_month,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Q17 - RFM CUSTOMER SEGMENTATION
# Recency  = days since last rating
# Frequency = total number of ratings given
# Monetary  = average rating score (proxy for engagement value)
# ═════════════════════════════════════════════════════════════════════════════

def build_rfm_segmentation(fact_df: DataFrame) -> DataFrame:
    """
    RFM Analysis:
      R (Recency)   - how recently did the user rate something?
      F (Frequency) - how many ratings have they given?
      M (Monetary)  - what is their average rating? (engagement proxy)

    Each dimension is scored 1-5 using ntile(5).
    Combined RFM score determines customer segment:
      Champions      - R=5, F=5
      Loyal          - F >= 4
      At Risk        - R <= 2, F >= 3
      Lost           - R=1, F=1
      Potential      - everything else
    """
    logger.info("Building gold.rfm_segmentation...")

    # Reference date = max date in dataset (simulates "today")
    max_date = fact_df.agg(F.max("rating_date")).collect()[0][0]

    rfm_base = (
        fact_df
        .groupBy("user_id")
        .agg(
            F.datediff(F.lit(max_date), F.max("rating_date")).alias("recency_days"),
            F.count("rating").alias("frequency"),
            F.round(F.avg("rating"), 3).alias("monetary"),
        )
    )

    # Score each dimension 1-5 with ntile
    # Note: recency is inverted - lower days = better = higher score
    rfm_scored = (
        rfm_base
        .withColumn("r_score", F.ntile(5).over(
            Window.orderBy(F.col("recency_days").asc())   # fewer days = higher score
        ))
        .withColumn("f_score", F.ntile(5).over(
            Window.orderBy(F.col("frequency").asc())
        ))
        .withColumn("m_score", F.ntile(5).over(
            Window.orderBy(F.col("monetary").asc())
        ))
        .withColumn("rfm_score",
                    F.col("r_score") + F.col("f_score") + F.col("m_score"))
    )

    # Assign segment label based on RFM scores
    rfm_final = rfm_scored.withColumn(
        "segment",
        F.when(
            (F.col("r_score") >= 4) & (F.col("f_score") >= 4), "Champions"
        ).when(
            F.col("f_score") >= 4, "Loyal Customers"
        ).when(
            (F.col("r_score") <= 2) & (F.col("f_score") >= 3), "At Risk"
        ).when(
            (F.col("r_score") == 1) & (F.col("f_score") == 1), "Lost"
        ).otherwise("Potential Loyalists")
    )

    return rfm_final.orderBy(F.col("rfm_score").desc())


# ═════════════════════════════════════════════════════════════════════════════
# Q18 - RETENTION COHORT ANALYSIS
# For each signup cohort (month), what % of users are still active
# in subsequent months?
# ═════════════════════════════════════════════════════════════════════════════

def build_retention_cohorts(
    fact_df: DataFrame,
    users_df: DataFrame,
) -> DataFrame:
    """
    Cohort retention analysis:
      - Cohort = month the user signed up
      - Activity = month the user gave a rating
      - Retention rate = % of cohort users active in month N after signup

    Result columns:
      cohort_month    - signup month (yyyy-MM)
      cohort_size     - total users in cohort
      months_since_signup - 0 = signup month, 1 = one month later, etc.
      active_users    - users from cohort still active that month
      retention_rate  - active_users / cohort_size * 100
    """
    logger.info("Building gold.retention_cohorts...")

    # Get cohort month for each user (month of signup)
    users_cohort = (
        users_df
        .select("user_id", "signup_date")
        .withColumn(
            "cohort_month",
            F.date_format(F.col("signup_date").cast(DateType()), "yyyy-MM")
        )
    )

    # Get activity month for each rating
    activity = (
        fact_df
        .select("user_id", "rating_date")
        .withColumn(
            "activity_month",
            F.date_format("rating_date", "yyyy-MM")
        )
        .distinct()
    )

    # Join to get (user, cohort_month, activity_month)
    cohort_activity = activity.join(users_cohort, on="user_id", how="inner")

    # Count cohort sizes
    cohort_sizes = (
        users_cohort
        .groupBy("cohort_month")
        .agg(F.countDistinct("user_id").alias("cohort_size"))
    )

    # Compute months_since_signup using period difference
    cohort_retention = (
        cohort_activity
        .groupBy("cohort_month", "activity_month")
        .agg(F.countDistinct("user_id").alias("active_users"))
        .join(cohort_sizes, on="cohort_month", how="left")
        .withColumn(
            "months_since_signup",
            (
                F.year(F.to_date(F.col("activity_month"), "yyyy-MM")) * 12 +
                F.month(F.to_date(F.col("activity_month"), "yyyy-MM"))
            ) - (
                F.year(F.to_date(F.col("cohort_month"), "yyyy-MM")) * 12 +
                F.month(F.to_date(F.col("cohort_month"), "yyyy-MM"))
            )
        )
        .withColumn(
            "retention_rate",
            F.round(F.col("active_users") / F.col("cohort_size") * 100, 1)
        )
        .select(
            "cohort_month", "cohort_size",
            "months_since_signup", "active_users", "retention_rate"
        )
        .orderBy("cohort_month", "months_since_signup")
    )

    return cohort_retention


# ═════════════════════════════════════════════════════════════════════════════
# Q19 - TIME-SERIES AGGREGATES
# Daily, weekly, monthly KPIs rolled up from fact_ratings
# ═════════════════════════════════════════════════════════════════════════════

def build_time_series_kpis(fact_df: DataFrame) -> Dict[str, DataFrame]:
    """
    Three time-series aggregation granularities:
      daily_kpis   - per calendar day
      weekly_kpis  - per ISO week
      monthly_kpis - per calendar month

    KPIs computed at each level:
      total_ratings, active_users, avg_rating,
      high_rating_pct, new_raters (first time rating that period)
    """
    logger.info("Building gold.time_series_kpis...")

    # First rating date per user (for new_raters calculation)
    first_rating = (
        fact_df
        .groupBy("user_id")
        .agg(F.min("rating_date").alias("first_rating_date"))
    )

    fact_with_first = fact_df.join(first_rating, on="user_id", how="left")

    # ── Daily ────────────────────────────────────────────────────────────────
    daily = (
        fact_with_first
        .groupBy("rating_date")
        .agg(
            F.count("rating").alias("total_ratings"),
            F.countDistinct("user_id").alias("active_users"),
            F.round(F.avg("rating"), 3).alias("avg_rating"),
            F.round(
                F.avg(F.col("is_high_rating").cast(IntegerType())) * 100, 1
            ).alias("high_rating_pct"),
            F.countDistinct(
                F.when(F.col("rating_date") == F.col("first_rating_date"),
                       F.col("user_id"))
            ).alias("new_raters"),
        )
        .orderBy("rating_date")
    )

    # ── Weekly ───────────────────────────────────────────────────────────────
    weekly = (
        fact_with_first
        .withColumn("year",      F.year("rating_date"))
        .withColumn("week",      F.weekofyear("rating_date"))
        .withColumn("week_start",
                    F.date_trunc("week", F.col("rating_date")).cast(DateType()))
        .groupBy("year", "week", "week_start")
        .agg(
            F.count("rating").alias("total_ratings"),
            F.countDistinct("user_id").alias("active_users"),
            F.round(F.avg("rating"), 3).alias("avg_rating"),
            F.round(
                F.avg(F.col("is_high_rating").cast(IntegerType())) * 100, 1
            ).alias("high_rating_pct"),
        )
        .orderBy("year", "week")
    )

    # ── Monthly ──────────────────────────────────────────────────────────────
    monthly = (
        fact_with_first
        .withColumn("year",  F.year("rating_date"))
        .withColumn("month", F.month("rating_date"))
        .withColumn("month_label",
                    F.date_format("rating_date", "yyyy-MM"))
        .groupBy("year", "month", "month_label")
        .agg(
            F.count("rating").alias("total_ratings"),
            F.countDistinct("user_id").alias("active_users"),
            F.round(F.avg("rating"), 3).alias("avg_rating"),
            F.round(
                F.avg(F.col("is_high_rating").cast(IntegerType())) * 100, 1
            ).alias("high_rating_pct"),
            F.countDistinct(
                F.when(
                    F.date_format(F.col("first_rating_date"), "yyyy-MM") ==
                    F.col("month_label"),
                    F.col("user_id")
                )
            ).alias("new_raters"),
        )
        .orderBy("year", "month")
    )

    return {
        "daily_kpis":   daily,
        "weekly_kpis":  weekly,
        "monthly_kpis": monthly,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN - Run full Gold pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_gold_pipeline(spark: SparkSession) -> Dict[str, DataFrame]:
    logger.info("=" * 60)
    logger.info("PART C - GOLD LAYER PIPELINE")
    logger.info("=" * 60)

    # Load silver tables
    enriched_df = read_silver(spark, "enriched_ratings")
    movies_df   = read_silver(spark, "movies")
    users_df    = read_silver(spark, "users_scd2") \
                  if os.path.exists(os.path.join(SILVER_BASE, "users_scd2")) \
                  else read_silver(spark, "users")
    genres_df   = read_silver(spark, "genres")

    # Q14: Star schema
    dim_dates  = build_dim_dates(spark, enriched_df)
    dim_users  = build_dim_users(spark, users_df)
    dim_movies = build_dim_movies(spark, movies_df)
    fact_df    = build_fact_ratings(spark, enriched_df)

    # Q15: Daily engagement
    daily_eng  = build_daily_engagement(fact_df)

    # Q16: Content popularity
    popularity = build_content_popularity(fact_df, dim_movies, genres_df)

    # Q17: RFM segmentation
    rfm_df     = build_rfm_segmentation(fact_df)

    # Q18: Cohort retention
    cohorts_df = build_retention_cohorts(fact_df, dim_users)

    # Q19: Time-series KPIs
    kpis       = build_time_series_kpis(fact_df)

    # Write all gold tables
    write_gold(dim_dates,  "dim_dates")
    write_gold(dim_users,  "dim_users")
    write_gold(dim_movies, "dim_movies")
    write_gold(fact_df,    "fact_ratings")
    write_gold(daily_eng,  "daily_engagement")
    write_gold(rfm_df,     "rfm_segmentation")
    write_gold(cohorts_df, "retention_cohorts")

    for name, df in popularity.items():
        write_gold(df, name)
    for name, df in kpis.items():
        write_gold(df, name)

    # Summary
    logger.info("=" * 60)
    logger.info("GOLD PIPELINE COMPLETE")
    logger.info(f"  fact_ratings:         {fact_df.count()} rows")
    logger.info(f"  dim_dates:            {dim_dates.count()} rows")
    logger.info(f"  dim_users:            {dim_users.count()} rows")
    logger.info(f"  dim_movies:           {dim_movies.count()} rows")
    logger.info(f"  daily_engagement:     {daily_eng.count()} rows")
    logger.info(f"  rfm_segmentation:     {rfm_df.count()} rows")
    logger.info(f"  retention_cohorts:    {cohorts_df.count()} rows")
    logger.info(f"  time_series (daily):  {kpis['daily_kpis'].count()} rows")
    logger.info(f"  time_series (weekly): {kpis['weekly_kpis'].count()} rows")
    logger.info(f"  time_series (monthly):{kpis['monthly_kpis'].count()} rows")
    logger.info("=" * 60)

    return {
        "fact_ratings":        fact_df,
        "dim_dates":           dim_dates,
        "dim_users":           dim_users,
        "dim_movies":          dim_movies,
        "daily_engagement":    daily_eng,
        "rfm_segmentation":    rfm_df,
        "retention_cohorts":   cohorts_df,
        **popularity,
        **kpis,
    }


if __name__ == "__main__":
    sys.path.insert(0, _PROJECT_ROOT)
    from utils.spark_session import get_spark
    spark = get_spark("GoldPipeline")
    run_gold_pipeline(spark)
    spark.stop()
