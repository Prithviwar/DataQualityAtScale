#!/usr/bin/env python3
"""
inject_errors.py
-----------------
Top-level CLI script: generate clean data (if needed) then inject faults.

Usage
-----
  python inject_errors.py [--rows 100000] [--seed 42] \
      [--input data/generated/clean] [--output data/generated/corrupted] \
      [--null-rate 0.05] [--shift-rate 0.02] [--shift-factor 1000] \
      [--dup-rate 0.03] [--full-scale]

Environment variables
---------------------
  SPARK_VERSION          : must be '4.1' (auto-set if not present)
  SPARK_DRIVER_MEMORY    : e.g. '4g'  (default '2g')
  SPARK_EXECUTOR_MEMORY  : e.g. '4g'  (default '2g')
  SPARK_SHUFFLE_PARTITIONS : e.g. '200' (default '8' for dev, '200' for full-scale)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("inject_errors")

# Ensure SPARK_VERSION is set before PyDeequ import
os.environ.setdefault("SPARK_VERSION", "4.1")

# Local src modules
sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_generation import generate_clean_data, _build_spark  # noqa: E402
from fault_injection import run_injection  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate clean data and inject faults (nulls, scale shifts, duplicates)."
    )
    p.add_argument("--rows", type=int, default=100_000,
                   help="Number of rows to generate (default 100,000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Master random seed (default 42)")
    p.add_argument("--input", type=str, default="data/generated/clean",
                   help="Path to clean Parquet input (generated if absent)")
    p.add_argument("--output", type=str, default="data/generated/corrupted",
                   help="Output path for corrupted Parquet")
    p.add_argument("--null-rate", type=float, default=0.05,
                   help="Fraction of feature_1 rows set to null (default 0.05)")
    p.add_argument("--shift-rate", type=float, default=0.02,
                   help="Fraction of feature_2 rows scaled by --shift-factor (default 0.02)")
    p.add_argument("--shift-factor", type=float, default=1000.0,
                   help="Multiplier for feature_2 scale-shift (default 1000)")
    p.add_argument("--dup-rate", type=float, default=0.03,
                   help="Fraction of rows to duplicate (default 0.03)")
    p.add_argument("--full-scale", action="store_true",
                   help="Set --rows to 100,000,000 (overrides --rows)")
    p.add_argument("--batch-id", type=str, default=None,
                   help="batch_id for the generated dataset (auto-generated if not set)")
    p.add_argument("--metadata-out", type=str, default=None,
                   help="Where to write injection metadata JSON (default: adjacent to output)")
    p.add_argument("--skip-generate", action="store_true",
                   help="Skip generation step if clean data already exists at --input")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.full_scale:
        args.rows = 100_000_000
        os.environ.setdefault("SPARK_SHUFFLE_PARTITIONS", "200")

    log.info("=== inject_errors.py starting ===")
    log.info("rows=%d  seed=%d  null_rate=%.3f  shift_rate=%.3f  shift_factor=%.0f  dup_rate=%.3f",
             args.rows, args.seed, args.null_rate, args.shift_rate, args.shift_factor, args.dup_rate)

    spark = _build_spark("FaultInjection")

    try:
        # --- Step 1: generate clean data (unless skipped) ---
        clean_path = args.input
        if not args.skip_generate:
            log.info("Generating %d rows of clean data → %s", args.rows, clean_path)
            generate_clean_data(
                spark,
                n_rows=args.rows,
                seed=args.seed,
                output_path=clean_path,
                batch_id=args.batch_id,
            )
        else:
            log.info("Skipping generation; using existing data at %s", clean_path)

        # --- Step 2: inject faults ---
        log.info("Injecting faults → %s", args.output)
        metadata = run_injection(
            spark,
            input_path=clean_path,
            output_path=args.output,
            seed=args.seed,
            null_rate=args.null_rate,
            unit_shift_rate=args.shift_rate,
            unit_shift_factor=args.shift_factor,
            duplicate_rate=args.dup_rate,
            metadata_path=args.metadata_out,
        )

        # Print a summary
        log.info("=== Injection Summary ===")
        log.info("  Input rows          : %d", metadata["input_row_count"])
        log.info("  Output rows         : %d", metadata["output_row_count"])
        log.info("  Null-injected rows  : %d (%.2f%%)",
                 metadata["null_injection_actual_count"],
                 metadata["null_injection_actual_rate"] * 100)
        log.info("  Scale-shifted rows  : %d (%.2f%%)",
                 metadata["unit_shift_actual_count"],
                 metadata["unit_shift_actual_rate"] * 100)
        log.info("  Duplicate rows      : %d (%.2f%%)",
                 metadata["duplicate_actual_count"],
                 metadata["duplicate_actual_rate"] * 100)
        log.info("  Duplicate IDs       : %d", metadata["duplicate_id_count"])
        log.info("  Corrupted path      : %s", args.output)
        log.info("=========================")

    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    code = main()
    os._exit(code)

