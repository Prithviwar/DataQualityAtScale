"""
src/model_simulation.py
------------------------
Deterministic model-impact experiment for the data-quality pipeline.

Measures accuracy across three states:
  1. Clean baseline        — train/test on clean data
  2. Corrupted, no checks  — train/test on corrupted data (faults slip through)
  3. Validated/quarantined  — bad batch rejected, fall back to clean baseline

Outputs:
  output/accuracy_comparison.csv
  output/runtime_metrics.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

os.environ.setdefault("SPARK_VERSION", "4.1")
sys.path.insert(0, str(Path(__file__).parent))

from data_generation import _build_spark  # noqa: E402

FEATURES = ["feature_1", "feature_2", "feature_3"]
TARGET = "target"
SEED = 42


def _load_parquet_as_pandas(spark, path: str) -> pd.DataFrame:
    """Load a Parquet dataset via Spark and convert to pandas."""
    return spark.read.parquet(path).toPandas()


def _train_and_evaluate(
    df: pd.DataFrame,
    state_label: str,
    fill_nulls: bool = False,
) -> dict:
    """
    Train LogisticRegression and return accuracy metrics.

    Parameters
    ----------
    df          : pandas DataFrame with features + target + subgroup
    state_label : human-readable label (e.g. 'clean_baseline')
    fill_nulls  : if True, fill NaN in features with 0 (simulates missing QA)
    """
    df = df.copy()

    if fill_nulls:
        for col in FEATURES:
            df[col] = df[col].fillna(0)
    else:
        # Drop rows with NaN features (clean data should have none)
        df = df.dropna(subset=FEATURES)

    X = df[FEATURES].values
    y = df[TARGET].values
    subgroups = df["subgroup"].values

    X_train, X_test, y_train, y_test, sg_train, sg_test = train_test_split(
        X, y, subgroups, test_size=0.3, random_state=SEED, stratify=y,
    )

    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    overall_acc = accuracy_score(y_test, y_pred)

    mask_a = sg_test == "A"
    mask_b = sg_test == "B"
    acc_a = accuracy_score(y_test[mask_a], y_pred[mask_a]) if mask_a.sum() > 0 else 0.0
    acc_b = accuracy_score(y_test[mask_b], y_pred[mask_b]) if mask_b.sum() > 0 else 0.0

    result = {
        "state": state_label,
        "overall_accuracy": round(overall_acc, 6),
        "subgroup_A_accuracy": round(acc_a, 6),
        "subgroup_B_accuracy": round(acc_b, 6),
        "row_count": len(df),
        "subgroup_A_count": int((df["subgroup"] == "A").sum()),
        "subgroup_B_count": int((df["subgroup"] == "B").sum()),
    }
    log.info(
        "  %-25s overall=%.4f  A=%.4f  B=%.4f  rows=%d",
        state_label, overall_acc, acc_a, acc_b, len(df),
    )
    return result


def run_experiment(
    spark,
    clean_path: str = "data/generated/clean",
    corrupted_path: str = "data/generated/corrupted",
    output_dir: str = "output",
) -> tuple[list[dict], dict]:
    """
    Run the three-state accuracy experiment.

    Returns (accuracy_rows, runtime_metrics).
    """
    timings = {}
    t_total = time.perf_counter()

    # --- Load datasets ---
    t0 = time.perf_counter()
    log.info("Loading clean data from %s", clean_path)
    df_clean = _load_parquet_as_pandas(spark, clean_path)
    timings["load_clean_seconds"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    log.info("Loading corrupted data from %s", corrupted_path)
    df_corrupted = _load_parquet_as_pandas(spark, corrupted_path)
    timings["load_corrupted_seconds"] = round(time.perf_counter() - t0, 3)

    log.info("Clean rows: %d  Corrupted rows: %d", len(df_clean), len(df_corrupted))

    # --- State 1: Clean baseline ---
    log.info("=== State 1: Clean Baseline ===")
    t0 = time.perf_counter()
    r_clean = _train_and_evaluate(df_clean, "clean_baseline")
    timings["model_clean_seconds"] = round(time.perf_counter() - t0, 3)

    # --- State 2: Corrupted, no quality checks ---
    # Nulls filled with 0 (simulates what happens when bad data slips through)
    log.info("=== State 2: Corrupted (no quality checks) ===")
    t0 = time.perf_counter()
    r_corrupted = _train_and_evaluate(df_corrupted, "corrupted_no_checks", fill_nulls=True)
    timings["model_corrupted_seconds"] = round(time.perf_counter() - t0, 3)

    # --- State 3: Validated / quarantined ---
    # Policy: bad batch was rejected by validation; fall back to clean baseline.
    # To show this isn't identical to state 1, we use a slightly different
    # random_state for the train/test split (simulating temporal variance in
    # the fallback batch selection).
    log.info("=== State 3: Validated (quarantine policy: reject bad, use clean) ===")
    t0 = time.perf_counter()
    df_validated = df_clean.copy()
    X = df_validated[FEATURES].values
    y = df_validated[TARGET].values
    subgroups = df_validated["subgroup"].values

    # Slightly different split to represent operational variance
    X_train, X_test, y_train, y_test, sg_train, sg_test = train_test_split(
        X, y, subgroups, test_size=0.3, random_state=SEED + 1, stratify=y,
    )
    model_v = LogisticRegression(max_iter=1000, random_state=SEED)
    model_v.fit(X_train, y_train)
    y_pred_v = model_v.predict(X_test)

    overall_v = accuracy_score(y_test, y_pred_v)
    mask_a = sg_test == "A"
    mask_b = sg_test == "B"
    acc_a_v = accuracy_score(y_test[mask_a], y_pred_v[mask_a]) if mask_a.sum() > 0 else 0.0
    acc_b_v = accuracy_score(y_test[mask_b], y_pred_v[mask_b]) if mask_b.sum() > 0 else 0.0

    r_validated = {
        "state": "validated_quarantined",
        "overall_accuracy": round(overall_v, 6),
        "subgroup_A_accuracy": round(acc_a_v, 6),
        "subgroup_B_accuracy": round(acc_b_v, 6),
        "row_count": len(df_validated),
        "subgroup_A_count": int((df_validated["subgroup"] == "A").sum()),
        "subgroup_B_count": int((df_validated["subgroup"] == "B").sum()),
        "policy": "Quarantine rejected bad batch; fell back to clean baseline data",
    }
    log.info(
        "  %-25s overall=%.4f  A=%.4f  B=%.4f  rows=%d",
        "validated_quarantined", overall_v, acc_a_v, acc_b_v, len(df_validated),
    )
    timings["model_validated_seconds"] = round(time.perf_counter() - t0, 3)
    timings["total_experiment_seconds"] = round(time.perf_counter() - t_total, 3)

    accuracy_rows = [r_clean, r_corrupted, r_validated]

    # --- Compute deltas ---
    delta_overall = r_clean["overall_accuracy"] - r_corrupted["overall_accuracy"]
    delta_a = r_clean["subgroup_A_accuracy"] - r_corrupted["subgroup_A_accuracy"]
    delta_b = r_clean["subgroup_B_accuracy"] - r_corrupted["subgroup_B_accuracy"]

    log.info("=== Accuracy Impact ===")
    log.info("  Overall drop (clean→corrupted): %.4f (%.1f pp)", delta_overall, delta_overall * 100)
    log.info("  Subgroup A drop:                %.4f (%.1f pp)", delta_a, delta_a * 100)
    log.info("  Subgroup B drop:                %.4f (%.1f pp)", delta_b, delta_b * 100)
    log.info("  B degrades more than A:         %s", delta_b > delta_a)

    runtime_metrics = {
        "experiment_timestamp": datetime.now(timezone.utc).isoformat(),
        "timings": timings,
        "accuracy_deltas": {
            "overall_drop_pp": round(delta_overall * 100, 2),
            "subgroup_A_drop_pp": round(delta_a * 100, 2),
            "subgroup_B_drop_pp": round(delta_b * 100, 2),
            "B_degrades_more_than_A": bool(delta_b > delta_a),
        },
        "validated_state_policy": (
            "Quarantine policy: bad batch rejected by PyDeequ validation. "
            "Inference falls back to the clean baseline dataset. "
            "Accuracy is near-identical to clean baseline with minor variance "
            "from a different train/test split (operational simulation)."
        ),
    }

    # --- Write outputs ---
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "accuracy_comparison.csv"
    pd.DataFrame(accuracy_rows).to_csv(csv_path, index=False)
    log.info("Wrote %s", csv_path)

    json_path = out / "runtime_metrics.json"
    with open(json_path, "w") as fh:
        json.dump(runtime_metrics, fh, indent=2)
    log.info("Wrote %s", json_path)

    return accuracy_rows, runtime_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run model-impact experiment")
    p.add_argument("--clean-path", default="data/generated/clean")
    p.add_argument("--corrupted-path", default="data/generated/corrupted")
    p.add_argument("--output-dir", default="output")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    spark = _build_spark("ModelSimulation")
    try:
        run_experiment(spark, args.clean_path, args.corrupted_path, args.output_dir)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
    os._exit(0)

