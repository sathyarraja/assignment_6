"""
utils/spark_session.py
Centralised SparkSession factory - works on Apple Silicon + local mode.
"""

import os
from pyspark.sql import SparkSession

# Fix for Apple Silicon "Too large frame" error:
# Force Spark driver to bind to localhost only, not the external network interface
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")


def get_spark(app_name: str = "NetflixMedallionPipeline") -> SparkSession:
    """
    Returns a SparkSession configured for Delta Lake.
    Works locally (including Apple Silicon) and on Databricks.
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.jars.packages",
                "io.delta:delta-spark_2.12:3.1.0")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.log.level", "WARN")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
