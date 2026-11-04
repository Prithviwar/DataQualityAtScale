#!/usr/bin/env python3
"""
validate_pipeline.py
--------------------
Top-level CLI script: run PyDeequ quality checks on a dataset, quarantine
failed batches, and exit with a non-zero code if any blocking check fails.

Usage
-----
  python validate_pipeline.py [--input data/generated/corrupted] \
      [--dataset-label corrupted] [--rules config/quality_rules.yml] \
      [--ref-stats output/reference_stats.json] \
      [--output output/validation_results.json] \
      [--injection-meta data/generated/injection_metadata.json] \
      [--mode collect_all|fail_fast] \
      [--quarantine-base data/quarantined]

Exit codes
----------
  0  – all checks PASSED
  1  – one or more checks FAILED (batch quarantined)
  2  – unexpected error
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("validate_pipeline")

# Ensure SPARK_VERSION before pydeequ import
os.environ.setdefault("SPARK_VERSION", "4.1")

import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_generation import _build_spark  # noqa: E402
from quality_checks import run_checks  # noqa: E402
from quarantine import quarantine_batch  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_rules(rules_path: str) -> dict:
    with open(rules_path) as fh:
        return yaml.safe_load(fh)


def _load_ref_stats(ref_stats_path: Optional[str]) -> Optional[dict]:
    if ref_stats_path and Path(ref_stats_path).exists():
        with open(ref_stats_path) as fh:
            return json.load(fh)
    return None


def _load_injection_meta(meta_path: Optional[str]) -> Optional[dict]:
    if meta_path and Path(meta_path).exists():
        with open(meta_path) as fh:
            return json.load(fh)
    return None


def _detect_fault_categories(records) -> list[str]:
    """
    Inspect failed check names to classify detected fault categories.
    Returns a list of human-readable category strings.
    """
    categories = set()
    for r in records:
        if r.status != "PASSED":
            n = r.name
            if "completeness" in n or "required" in n:
                categories.add("missing_values")
            if "uniqueness" in n:
                categories.add("duplicate_ids")
            if "scale_shift" in n or ("feature_2_max" in n and r.observed and r.observed > 110):
                categories.add("unit_scale_shift")
            if "min" in n or "max" in n:
                categories.add("out_of_range_values")
            if "allowed_values" in n:
                categories.add("invalid_categorical_values")
    return sorted(categories)


def _build_validation_result(
    dataset_label: str,
    batch_id: str,
    row_count: int,
    records,
    runtime_seconds: float,
) -> dict:
    passed = all(r.status == "PASSED" for r in records)
    fault_categories = _detect_fault_categories(records)

    return {
        "dataset": dataset_label,
        "batch_id": batch_id,
        "row_count": row_count,
        "passed": passed,
        "checks": [r.to_dict() for r in records],
        "fault_categories_detected": fault_categories,
        "runtime_seconds": round(runtime_seconds, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run PyDeequ quality validation on a Parquet dataset."
    )
    p.add_argument("--input", type=str, default="data/generated/corrupted",
                   help="Path to Parquet dataset to validate")
    p.add_argument("--dataset-label", type=str, default="corrupted",
                   help="Human-readable dataset name in results (default 'corrupted')")
    p.add_argument("--rules", type=str, default="config/quality_rules.yml",
                   help="Path to quality rules YAML")
    p.add_argument("--ref-stats", type=str, default="output/reference_stats.json",
                   help="Path to reference stats JSON for scale-shift calibration")
    p.add_argument("--output", type=str, default="output/validation_results.json",
                   help="Where to write validation results JSON")
    p.add_argument("--injection-meta", type=str, default=None,
                   help="Path to injection metadata JSON (optional; included in quarantine)")
    p.add_argument("--mode", type=str, default="collect_all",
                   choices=["collect_all", "fail_fast"],
                   help="Validation mode (default 'collect_all')")
    p.add_argument("--quarantine-base", type=str, default="data/quarantined",
                   help="Root directory for quarantined batches")
    p.add_argument("--no-quarantine", action="store_true",
                   help="Skip quarantine write (validation results still written)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    log.info("=== validate_pipeline.py starting ===")
    log.info("input=%s  rules=%s  mode=%s", args.input, args.rules, args.mode)

    spark = _build_spark("QualityValidation")
    exit_code = 0

    try:
        t0 = time.perf_counter()

        # Load dataset
        log.info("Loading dataset from: %s", args.input)
        df = spark.read.parquet(args.input)
        row_count = df.count()
        log.info("Dataset loaded: %d rows, schema: %s", row_count,
                 [(f.name, str(f.dataType)) for f in df.schema.fields])

        # Detect batch_id from data
        try:
            batch_id = df.select("batch_id").first()["batch_id"]
        except Exception:
            batch_id = "unknown"
        log.info("batch_id: %s", batch_id)

        # Load rules & reference stats
        rules = _load_rules(args.rules)
        ref_stats = _load_ref_stats(args.ref_stats)
        injection_meta = _load_injection_meta(args.injection_meta)

        log.info("Loaded %d rule groups from %s", len(rules), args.rules)
        if ref_stats:
            log.info("Reference stats loaded: %s", list(ref_stats.keys()))

        # Run checks
        records = run_checks(spark, df, rules, ref_stats, mode=args.mode)

        runtime = time.perf_counter() - t0

        # Build result dict
        result = _build_validation_result(
            dataset_label=args.dataset_label,
            batch_id=batch_id,
            row_count=row_count,
            records=records,
            runtime_seconds=runtime,
        )

        # Print check summary
        passed = result["passed"]
        log.info("=== Validation Summary ===")
        log.info("  Dataset   : %s", args.dataset_label)
        log.info("  batch_id  : %s", batch_id)
        log.info("  Rows      : %d", row_count)
        log.info("  Passed    : %s", passed)
        log.info("  Runtime   : %.2f s", runtime)
        for r in records:
            status_sym = "✓" if r.status == "PASSED" else "✗"
            log.info("  [%s] %s  observed=%s  expected=%s",
                     status_sym, r.name, r.observed, r.expected)
        if result["fault_categories_detected"]:
            log.warning("  Fault categories detected: %s", result["fault_categories_detected"])
        log.info("==========================")

        # Write validation results JSON
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(result, fh, indent=2)
        log.info("Validation results written to: %s", output_path)

        # Quarantine if failed
        if not passed:
            exit_code = 1
            if not args.no_quarantine:
                q_path = quarantine_batch(
                    spark=spark,
                    df=df,
                    batch_id=batch_id,
                    validation_results=result,
                    injection_metadata=injection_meta,
                    quarantine_base=args.quarantine_base,
                )
                log.warning("Batch quarantined at: %s", q_path)
                log.warning("INFERENCE BLOCKED for batch %r", batch_id)
            else:
                log.warning("Quarantine skipped (--no-quarantine). Batch %r has FAILED checks.", batch_id)
        else:
            log.info("All checks PASSED. Batch %r cleared for inference.", batch_id)

    except RuntimeError as exc:
        log.error("Validation aborted (fail_fast or critical error): %s", exc)
        exit_code = 1
    except Exception as exc:
        log.exception("Unexpected error during validation: %s", exc)
        exit_code = 2
    finally:
        try:
            # Shut down PyDeequ's Py4J callback server so Python doesn't hang waiting for threads
            gw = spark.sparkContext._gateway
            cb = gw.get_callback_server()
            if cb:
                cb.shutdown()
        except Exception:
            pass
        spark.stop()

    log.info("Exiting with code %d", exit_code)
    return exit_code


if __name__ == "__main__":
    code = main()
    # Use os._exit to immediately terminate and release the terminal prompt
    # without hanging on lingering Py4J / JVM background threads
    os._exit(code)

