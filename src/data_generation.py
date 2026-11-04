"""
src/data_generation.py
-----------------------
Deterministic synthetic data generation for the data-quality pipeline.

Schema
------
  id              : LongType   – unique primary key
  feature_1       : DoubleType – N(0, 10), realistic numeric predictor
  feature_2       : DoubleType – Uniform(0, 100), subgroup-B correlated; range 0–100 for CLEAN data
  feature_3       : DoubleType – N(0, 50), realistic numeric predictor
  target          : IntegerType – binary 0/1
  subgroup        : StringType – 'A' or 'B' (50/50)
  event_timestamp : TimestampType – uniform over last 90 days
  batch_id        : StringType – e.g. "batch_20241001"

Generation strategy (Spark-native, no driver collect):
  1. Create a range DataFrame (0..N-1).
  2. Derive all columns from `id` via deterministic math + hash tricks.
  3. Write as Parquet without collecting to driver.

Subgroup-B feature:
  For subgroup B, feature_2 is generated differently so the target label
  is more correlated with feature_2. This means that a 1000x scale shift
  in feature_2 disproportionately degrades subgroup-B model accuracy.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, TimestampType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / clean-data distribution parameters (documented for config bounds)
# ---------------------------------------------------------------------------
FEATURE1_MEAN = 0.0
FEATURE1_STD  = 10.0          # range ≈ -100 to +100 (±10σ)
FEATURE2_MIN  = 0.0
FEATURE2_MAX  = 100.0         # uniform; documented so config bounds match
FEATURE3_MEAN = 0.0
FEATURE3_STD  = 50.0          # range ≈ -100 to +100 (±2σ ≈ ±100)
SEED          = 42


def _find_deequ_jar() -> str | None:
    """
    Locate the Deequ JAR on the local system.
    Priority:
      1. DEEQU_JAR env var (user override / Docker mount)
      2. Ivy2 cache (Spark downloads here on first spark.jars.packages run)
      3. Fall back to Maven package resolution (slow on first run)
    """
    # 1. Explicit override
    explicit = os.getenv("DEEQU_JAR")
    if explicit and os.path.isfile(explicit):
        log.info("Using DEEQU_JAR override: %s", explicit)
        return explicit

    # 2. Ivy2 cache locations (Spark 4.x uses ivy version suffix)
    home = os.path.expanduser("~")
    candidates = [
        # Spark 4.x ivy cache (uses versioned directory)
        os.path.join(home, ".ivy2.5.2", "cache", "com.amazon.deequ", "deequ", "jars",
                     "deequ-2.0.18-spark-4.1.jar"),
        # Standard ivy2 path
        os.path.join(home, ".ivy2", "cache", "com.amazon.deequ", "deequ", "jars",
                     "deequ-2.0.18-spark-4.1.jar"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            log.info("Found cached Deequ JAR: %s", path)
            return path

    log.warning("Deequ JAR not found in cache – will resolve via Maven (may be slow first run)")
    return None


def _build_spark(app_name: str = "DataGeneration") -> SparkSession:
    """Return (or create) a SparkSession compatible with PyDeequ."""
    # Ensure Spark workers use the same Python executable as the driver.
    # On Windows, 'python3' is not available; use the current interpreter.
    python_exec = sys.executable
    os.environ.setdefault("PYSPARK_PYTHON", python_exec)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", python_exec)
    # Bind to loopback to prevent Windows hostname / IPv6 / VPN resolution failures
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    master_url = os.getenv("SPARK_MASTER_URL", "local[*]")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spark_tmp = os.path.join(project_root, ".spark_tmp")
    os.makedirs(spark_tmp, exist_ok=True)

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master_url)
        .config("spark.local.dir", spark_tmp)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "2g"))
        .config("spark.executor.memory", os.getenv("SPARK_EXECUTOR_MEMORY", "2g"))
        .config("spark.pyspark.python", python_exec)
        .config("spark.pyspark.driver.python", python_exec)
    )

    deequ_jar = _find_deequ_jar()
    if deequ_jar:
        # Load JAR in-place on classpath: prevents Spark on Windows from copying JAR
        # to a temp dir that gets locked, which causes ShutdownHookManager delete errors.
        builder = (
            builder
            .config("spark.jars", deequ_jar)
            .config("spark.driver.extraClassPath", deequ_jar)
            .config("spark.executor.extraClassPath", deequ_jar)
        )
    else:
        # Fall back to Maven coordinates (requires network on first run)
        builder = builder.config("spark.jars.packages", "com.amazon.deequ:deequ:2.0.18-spark-4.1")

    return builder.getOrCreate()


def generate_clean_data(
    spark: SparkSession,
    n_rows: int = 100_000,
    seed: int = SEED,
    output_path: str = "data/generated/clean",
    batch_id: str | None = None,
) -> str:
    """
    Generate n_rows of deterministic clean data and write to Parquet.

    Returns the output path.
    """
    if batch_id is None:
        batch_id = "batch_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    log.info("Generating %d rows  seed=%d  batch=%s  output=%s", n_rows, seed, batch_id, output_path)

    # Base: monotonically increasing id column
    df = spark.range(n_rows)  # column: 'id' (LongType)

    # --- subgroup: alternate A/B every row ---
    df = df.withColumn("subgroup", F.when(F.col("id") % 2 == 0, "A").otherwise("B"))

    # --- deterministic pseudo-random helpers using hash + randn seed tricks ---
    # Spark's randn(seed) is row-independent but we need per-row reproducibility;
    # we use `id` as the seed modifier for each call. Because PySpark's
    # randn(seed) is deterministic across the whole column (global seed),
    # we just use different seeds for each logical column.

    # feature_1 ~ N(0, 10)
    df = df.withColumn("feature_1", F.randn(seed + 1) * FEATURE1_STD + FEATURE1_MEAN)

    # feature_2:
    #   subgroup A: Uniform(0,100) derived from rand
    #   subgroup B: Uniform(0,100) but shaped so target is more correlated
    #     → feature_2_B is biased: B gets feature_2 centered around 70 for target=1
    #     We first draw a uniform, then apply a subgroup-conditional shift AFTER target.
    # For simplicity: draw raw feature_2 ~ Uniform(0,100) for all,
    # then post-process for B when computing target.
    df = df.withColumn("_f2_raw", F.rand(seed + 2) * (FEATURE2_MAX - FEATURE2_MIN) + FEATURE2_MIN)

    # feature_3 ~ N(0, 50)
    df = df.withColumn("feature_3", F.randn(seed + 3) * FEATURE3_STD + FEATURE3_MEAN)

    # target: logistic model
    #   P(target=1) = sigmoid(0.1*f1 + 0.5*f2_norm + 0.05*f3)
    #   where f2_norm = (f2 - 50) / 50  → maps [0,100] → [-1,1]
    #   For subgroup B, feature_2 weight is 3x to make it more sensitive.
    df = df.withColumn(
        "_f2_norm", (F.col("_f2_raw") - 50.0) / 50.0
    )
    df = df.withColumn(
        "_logit",
        F.when(
            F.col("subgroup") == "B",
            0.1 * F.col("feature_1") + 1.5 * F.col("_f2_norm") + 0.05 * F.col("feature_3"),
        ).otherwise(
            0.1 * F.col("feature_1") + 0.5 * F.col("_f2_norm") + 0.05 * F.col("feature_3"),
        ),
    )
    df = df.withColumn("_prob", 1.0 / (1.0 + F.exp(-F.col("_logit"))))
    df = df.withColumn("target", (F.rand(seed + 4) < F.col("_prob")).cast(IntegerType()))

    # feature_2: for subgroup B, apply mild bias post-target to increase correlation
    # keep it within [0, 100] by using a mix
    df = df.withColumn(
        "feature_2",
        F.when(
            F.col("subgroup") == "B",
            F.greatest(
                F.lit(FEATURE2_MIN),
                F.least(
                    F.lit(FEATURE2_MAX),
                    F.col("_f2_raw") + F.col("target").cast(DoubleType()) * 15.0
                    - (1 - F.col("target").cast(DoubleType())) * 15.0,
                ),
            ),
        ).otherwise(F.col("_f2_raw")),
    )

    # event_timestamp: uniform over last 90 days
    now_ts = datetime.utcnow().timestamp()
    ninety_days = 90 * 24 * 3600
    df = df.withColumn(
        "event_timestamp",
        F.to_timestamp(
            F.lit(now_ts - ninety_days) + F.rand(seed + 5) * ninety_days
        ).cast(TimestampType()),
    )

    # batch_id column
    df = df.withColumn("batch_id", F.lit(batch_id))

    # Drop helper columns, enforce final schema
    df = df.select(
        F.col("id").cast(LongType()),
        F.col("feature_1").cast(DoubleType()),
        F.col("feature_2").cast(DoubleType()),
        F.col("feature_3").cast(DoubleType()),
        F.col("target").cast(IntegerType()),
        F.col("subgroup").cast(StringType()),
        F.col("event_timestamp"),
        F.col("batch_id").cast(StringType()),
    )

    # Log schema
    log.info("Schema:\n%s", df._jdf.schema().treeString())

    # Write Parquet
    df.write.mode("overwrite").parquet(output_path)
    written = spark.read.parquet(output_path).count()
    log.info("Wrote %d rows to %s", written, output_path)

    # Quick subgroup proportion log
    sg_counts = (
        spark.read.parquet(output_path)
        .groupBy("subgroup")
        .count()
        .collect()
    )
    for row in sg_counts:
        log.info("  subgroup=%s  count=%d  pct=%.2f%%", row["subgroup"], row["count"], 100.0 * row["count"] / written)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate clean synthetic Parquet dataset")
    p.add_argument("--rows", type=int, default=100_000, help="Number of rows (default 100000; use 100000000 for full scale)")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed (default 42)")
    p.add_argument("--output", type=str, default="data/generated/clean", help="Output Parquet path")
    p.add_argument("--batch-id", type=str, default=None, help="batch_id value; auto-generated if not set")
    p.add_argument("--full-scale", action="store_true", help="Shortcut for --rows 100000000")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.full_scale:
        args.rows = 100_000_000

    os.environ.setdefault("SPARK_VERSION", "4.1")

    spark = _build_spark("DataGeneration")
    try:
        out = generate_clean_data(
            spark,
            n_rows=args.rows,
            seed=args.seed,
            output_path=args.output,
            batch_id=args.batch_id,
        )
        log.info("Generation complete: %s", out)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

