"""
src/fault_injection.py
-----------------------
Fault injection module for the data-quality pipeline.

Injections (all independently seeded):
  1. NULL injection  – ~5 % of feature_1 set to null
  2. Unit-scale shift – ~2 % of feature_2 multiplied by 1000
  3. Duplicate rows  – ~3 % of original rows duplicated (duplicate_rate =
     duplicated_output_rows / original_row_count)

No diagnostic flags are added to the output dataset.
Injection metadata is returned as a dict and optionally written to JSON.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

log = logging.getLogger(__name__)


def inject_faults(
    spark: SparkSession,
    df: DataFrame,
    seed: int = 42,
    null_rate: float = 0.05,
    unit_shift_rate: float = 0.02,
    unit_shift_factor: float = 1000.0,
    duplicate_rate: float = 0.03,
) -> tuple[DataFrame, dict]:
    """
    Apply fault injections to df and return (corrupted_df, metadata).

    Parameters
    ----------
    spark            : active SparkSession
    df               : clean input DataFrame
    seed             : master seed; each injection uses seed+offset
    null_rate        : fraction of rows where feature_1 → null
    unit_shift_rate  : fraction of rows where feature_2 × unit_shift_factor
    unit_shift_factor: multiplier for scale-shift injection
    duplicate_rate   : fraction of duplicate rows appended
                       (duplicated_output_rows / original_row_count)

    Returns
    -------
    corrupted_df : DataFrame without any injection-flag columns
    metadata     : dict with injection parameters and actual counts
    """
    input_count = df.count()
    log.info("Fault injection starting – input rows: %d", input_count)

    # ------------------------------------------------------------------
    # 1. NULL injection on feature_1
    # ------------------------------------------------------------------
    null_seed = seed + 10
    df_null = df.withColumn(
        "feature_1",
        F.when(F.rand(null_seed) < null_rate, F.lit(None).cast("double"))
        .otherwise(F.col("feature_1")),
    )

    # Measure actual nulls injected
    actual_null_count = df_null.filter(F.col("feature_1").isNull()).count()
    actual_null_rate = actual_null_count / input_count
    log.info(
        "NULL injection: target=%.2f%% actual=%.2f%% (%d rows)",
        null_rate * 100,
        actual_null_rate * 100,
        actual_null_count,
    )

    # ------------------------------------------------------------------
    # 2. Unit-scale shift on feature_2 (multiply by 1000)
    # ------------------------------------------------------------------
    shift_seed = seed + 20
    df_shifted = df_null.withColumn(
        "feature_2",
        F.when(F.rand(shift_seed) < unit_shift_rate, F.col("feature_2") * unit_shift_factor)
        .otherwise(F.col("feature_2")),
    )

    # Measure actual scale-shifted rows (feature_2 > 100 is a reliable proxy for clean data)
    actual_shift_count = df_shifted.filter(F.col("feature_2") > 100.0).count()
    actual_shift_rate = actual_shift_count / input_count
    log.info(
        "Scale-shift injection: target=%.2f%% actual=%.2f%% (%d rows, feature_2>100)",
        unit_shift_rate * 100,
        actual_shift_rate * 100,
        actual_shift_count,
    )

    # ------------------------------------------------------------------
    # 3. Duplicate rows (same id duplicated)
    # ------------------------------------------------------------------
    dup_seed = seed + 30
    n_duplicates = int(round(input_count * duplicate_rate))
    # Sample n_duplicates rows using pure Spark (no driver collect) and union back.
    # We over-sample slightly then trim to exactly n_duplicates via LIMIT.
    sample_frac = min(1.0, (n_duplicates * 1.2) / max(input_count, 1))
    duplicate_rows = (
        df_shifted
        .sample(withReplacement=False, fraction=sample_frac, seed=dup_seed)
        .limit(n_duplicates)
    )

    actual_dup_count = duplicate_rows.count()
    actual_dup_rate = actual_dup_count / input_count

    df_corrupted = df_shifted.union(duplicate_rows)
    output_count = df_corrupted.count()

    # Duplicate-ID count (IDs that appear more than once)
    dup_id_count = (
        df_corrupted.groupBy("id")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    log.info(
        "Duplicate injection: target=%.2f%% actual=%.2f%% (%d dup rows, %d dup IDs)",
        duplicate_rate * 100,
        actual_dup_rate * 100,
        actual_dup_count,
        dup_id_count,
    )
    log.info("Output row count: %d → %d", input_count, output_count)

    metadata = {
        "seed": seed,
        "input_row_count": input_count,
        "null_injection_rate": null_rate,
        "null_injection_actual_rate": actual_null_rate,
        "null_injection_actual_count": actual_null_count,
        "unit_shift_rate": unit_shift_rate,
        "unit_shift_actual_rate": actual_shift_rate,
        "unit_shift_actual_count": actual_shift_count,
        "unit_shift_factor": unit_shift_factor,
        "duplicate_rate": duplicate_rate,
        "duplicate_actual_rate": actual_dup_rate,
        "duplicate_actual_count": actual_dup_count,
        "duplicate_id_count": dup_id_count,
        "output_row_count": output_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_paths": {},
    }

    return df_corrupted, metadata


def run_injection(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    seed: int = 42,
    null_rate: float = 0.05,
    unit_shift_rate: float = 0.02,
    unit_shift_factor: float = 1000.0,
    duplicate_rate: float = 0.03,
    metadata_path: Optional[str] = None,
) -> dict:
    """
    Load clean data, inject faults, write corrupted Parquet, return metadata.
    """
    log.info("Loading clean data from: %s", input_path)
    df_clean = spark.read.parquet(input_path)

    df_corrupted, meta = inject_faults(
        spark, df_clean,
        seed=seed,
        null_rate=null_rate,
        unit_shift_rate=unit_shift_rate,
        unit_shift_factor=unit_shift_factor,
        duplicate_rate=duplicate_rate,
    )

    log.info("Writing corrupted data to: %s", output_path)
    df_corrupted.write.mode("overwrite").parquet(output_path)

    meta["dataset_paths"] = {"input": input_path, "output": output_path}

    if metadata_path is None:
        metadata_path = str(Path(output_path).parent / "injection_metadata.json")

    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info("Injection metadata written to: %s", metadata_path)

    return meta

