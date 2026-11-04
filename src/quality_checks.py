"""
src/quality_checks.py
----------------------
PyDeequ-based quality check definitions for the data-quality pipeline.

Each check is registered as a named constraint so results carry stable,
machine-readable names. This module exposes:

  build_verification_suite(spark, df, rules, ref_stats) → VerificationSuite builder
  run_checks(spark, df, rules, ref_stats, mode)         → list[CheckRecord]
"""

import logging
from typing import Any

from pyspark.sql import DataFrame, SparkSession

from pydeequ.checks import Check, CheckLevel, CheckStatus
from pydeequ.verification import VerificationResult, VerificationRunBuilder, VerificationSuite

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class for a single check outcome
# ---------------------------------------------------------------------------
class CheckRecord:
    """Structured outcome for a single constraint."""

    __slots__ = ("name", "status", "observed", "expected", "failure_reason", "metric")

    def __init__(
        self,
        name: str,
        status: str,
        observed: float | None,
        expected: float | None,
        failure_reason: str | None,
        metric: Any | None = None,
    ):
        self.name = name
        self.status = status
        self.observed = observed
        self.expected = expected
        self.failure_reason = failure_reason
        self.metric = metric

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "observed": self.observed,
            "expected": self.expected,
            "failure_reason": self.failure_reason,
            "metric": self.metric,
        }

    def __repr__(self):
        return (
            f"CheckRecord(name={self.name!r}, status={self.status!r}, "
            f"observed={self.observed}, expected={self.expected})"
        )


# ---------------------------------------------------------------------------
# Build & run checks
# ---------------------------------------------------------------------------

def build_checks(
    spark: SparkSession,
    rules: dict,
    ref_stats: dict | None = None,
) -> list[Check]:
    """
    Return a list of PyDeequ Check objects derived from the YAML rules dict.

    Parameters
    ----------
    spark     : SparkSession
    rules     : parsed quality_rules.yml as dict
    ref_stats : optional reference statistics dict (from output/reference_stats.json)
    """
    checks: list[Check] = []

    # 1. Completeness checks
    completeness_rules = rules.get("completeness", {})
    for col, threshold in completeness_rules.items():
        thr = float(threshold)
        chk = (
            Check(spark, CheckLevel.Error, f"{col}_completeness")
            .hasCompleteness(col, lambda v, t=thr: v >= t,
                             hint=f"Completeness of {col!r} must be >= {thr}")
        )
        checks.append(chk)
        log.debug("Registered completeness check for %r (threshold=%.2f)", col, thr)

    # 2. Uniqueness checks
    uniqueness_rules = rules.get("uniqueness", {})
    for col, threshold in uniqueness_rules.items():
        thr = float(threshold)
        chk = (
            Check(spark, CheckLevel.Error, f"{col}_uniqueness")
            .hasUniqueness([col], lambda v, t=thr: v >= t,
                           hint=f"Uniqueness of {col!r} must be >= {thr}")
        )
        checks.append(chk)
        log.debug("Registered uniqueness check for %r (threshold=%.2f)", col, thr)

    # 3. Required-column / schema checks (via isComplete on each required column)
    required_cols = rules.get("required_columns", [])
    for col in required_cols:
        # Only add schema check if not already covered by completeness
        if col not in completeness_rules:
            chk = (
                Check(spark, CheckLevel.Error, f"{col}_required")
                .isComplete(col, hint=f"Column {col!r} must exist and be non-null")
            )
            checks.append(chk)
            log.debug("Registered required-column check for %r", col)

    # 4. Numeric range checks (min/max)
    range_rules = rules.get("ranges", {})
    for col, bounds in range_rules.items():
        lo = bounds.get("min")
        hi = bounds.get("max")
        if lo is not None:
            chk_min = (
                Check(spark, CheckLevel.Error, f"{col}_min")
                .hasMin(col, lambda v, lo=lo: v >= lo,
                        hint=f"{col!r} minimum must be >= {lo}")
            )
            checks.append(chk_min)
        if hi is not None:
            chk_max = (
                Check(spark, CheckLevel.Error, f"{col}_max")
                .hasMax(col, lambda v, hi=hi: v <= hi,
                        hint=f"{col!r} maximum must be <= {hi}")
            )
            checks.append(chk_max)
        log.debug("Registered range check for %r [%s, %s]", col, lo, hi)

    # 5. Allowed-value checks (target, subgroup)
    allowed_rules = rules.get("allowed_values", {})
    for col, allowed in allowed_rules.items():
        # Convert to list of strings for isContainedIn
        str_allowed = [str(v) for v in allowed]
        chk = (
            Check(spark, CheckLevel.Error, f"{col}_allowed_values")
            .isContainedIn(col, str_allowed,
                           hint=f"{col!r} must be one of {str_allowed}")
        )
        checks.append(chk)
        log.debug("Registered allowed-values check for %r: %s", col, str_allowed)

    # 6. Explicit feature_2 scale-shift detection
    # Uses configurable alert_max from quality_rules or reference stats.
    scale_cfg = rules.get("scale_shift_detection", {}).get("feature_2", {})
    alert_max = scale_cfg.get("alert_max", 110.0)
    if ref_stats and "feature_2_p99" in ref_stats:
        # Calibrate: use max(ref p99 * 1.1, configured alert_max) for headroom
        alert_max = max(alert_max, ref_stats["feature_2_p99"] * 1.1)
    chk_scale = (
        Check(spark, CheckLevel.Error, "feature_2_scale_shift")
        .hasMax(
            "feature_2",
            lambda v, m=alert_max: v <= m,
            hint=f"feature_2 max must be <= {alert_max} (scale-shift detection: 1000x shift would produce ~100,000)",
        )
    )
    checks.append(chk_scale)
    log.debug("Registered scale-shift detection for feature_2 (alert_max=%.1f)", alert_max)

    return checks


def _map_deequ_results(
    spark: SparkSession,
    vr: VerificationResult,
    checks: list[Check],
    rules: dict,
    ref_stats: dict | None,
) -> list[CheckRecord]:
    """
    Convert PyDeequ VerificationResult → list of CheckRecord.
    """
    # Pull raw JSON from PyDeequ
    raw_check_results = VerificationResult.checkResultsAsJson(spark, vr)

    # Build a lookup by check description
    result_map: dict[str, dict] = {}
    for entry in raw_check_results:
        desc = entry.get("check", "")
        result_map[desc] = entry

    records: list[CheckRecord] = []

    # Pull completeness thresholds for expected values
    completeness_rules = rules.get("completeness", {})
    uniqueness_rules = rules.get("uniqueness", {})
    range_rules = rules.get("ranges", {})
    allowed_rules = rules.get("allowed_values", {})
    scale_cfg = rules.get("scale_shift_detection", {}).get("feature_2", {})
    alert_max = scale_cfg.get("alert_max", 110.0)

    # Try to get per-metric observed values via success metrics
    try:
        metrics_df = VerificationResult.successMetricsAsDataFrame(spark, vr, pandas=True)
        metric_lookup: dict[str, float] = {}
        for _, row in metrics_df.iterrows():
            key = f"{row.get('instance', '')}_{row.get('name', '')}"
            metric_lookup[key] = row.get("value")
    except Exception:
        metric_lookup = {}

    def _get_observed(col: str, metric: str) -> float | None:
        return metric_lookup.get(f"{col}_{metric}")

    def _status_from_entry(entry: dict) -> str:
        status = entry.get("constraint_status", "")
        if status == "Success":
            return "PASSED"
        return "FAILED"

    def _get_constraint_entries(desc: str) -> list[dict]:
        """Filter raw_check_results for entries matching this check description."""
        return [e for e in raw_check_results if e.get("check") == desc]

    # Completeness
    for col, threshold in completeness_rules.items():
        desc = f"{col}_completeness"
        entries = _get_constraint_entries(desc)
        observed = _get_observed(col, "Completeness")
        for entry in entries:
            status = _status_from_entry(entry)
            reason = None if status == "PASSED" else f"Null rate exceeds threshold (expected >= {threshold})"
            records.append(CheckRecord(desc, status, observed, float(threshold), reason))
        if not entries:
            records.append(CheckRecord(desc, "UNKNOWN", None, float(threshold), "No result returned by PyDeequ"))

    # Uniqueness
    for col, threshold in uniqueness_rules.items():
        desc = f"{col}_uniqueness"
        entries = _get_constraint_entries(desc)
        observed = _get_observed(col, "Uniqueness")
        for entry in entries:
            status = _status_from_entry(entry)
            reason = None if status == "PASSED" else f"Duplicate IDs found (uniqueness < {threshold})"
            records.append(CheckRecord(desc, status, observed, float(threshold), reason))
        if not entries:
            records.append(CheckRecord(desc, "UNKNOWN", None, float(threshold), "No result returned by PyDeequ"))

    # Required columns (completeness == 1.0 for non-completeness-listed cols)
    required_cols = rules.get("required_columns", [])
    for col in required_cols:
        if col not in completeness_rules:
            desc = f"{col}_required"
            entries = _get_constraint_entries(desc)
            observed = _get_observed(col, "Completeness")
            for entry in entries:
                status = _status_from_entry(entry)
                reason = None if status == "PASSED" else f"Column {col!r} is null or missing"
                records.append(CheckRecord(desc, status, observed, 1.0, reason))
            if not entries:
                records.append(CheckRecord(desc, "UNKNOWN", None, 1.0, "No result returned by PyDeequ"))

    # Ranges
    for col, bounds in range_rules.items():
        lo, hi = bounds.get("min"), bounds.get("max")
        if lo is not None:
            desc = f"{col}_min"
            entries = _get_constraint_entries(desc)
            observed = _get_observed(col, "Minimum")
            for entry in entries:
                status = _status_from_entry(entry)
                reason = None if status == "PASSED" else f"{col} minimum {observed} < threshold {lo}"
                records.append(CheckRecord(desc, status, observed, lo, reason))
            if not entries:
                records.append(CheckRecord(desc, "UNKNOWN", None, lo, "No result returned by PyDeequ"))
        if hi is not None:
            desc = f"{col}_max"
            entries = _get_constraint_entries(desc)
            observed = _get_observed(col, "Maximum")
            for entry in entries:
                status = _status_from_entry(entry)
                reason = None if status == "PASSED" else f"{col} maximum {observed} > threshold {hi}"
                records.append(CheckRecord(desc, status, observed, hi, reason))
            if not entries:
                records.append(CheckRecord(desc, "UNKNOWN", None, hi, "No result returned by PyDeequ"))

    # Allowed values
    for col, allowed in allowed_rules.items():
        desc = f"{col}_allowed_values"
        entries = _get_constraint_entries(desc)
        observed = _get_observed(col, "Compliance")
        for entry in entries:
            status = _status_from_entry(entry)
            reason = None if status == "PASSED" else f"{col!r} contains values not in {allowed}"
            records.append(CheckRecord(desc, status, observed, 1.0, reason))
        if not entries:
            records.append(CheckRecord(desc, "UNKNOWN", None, 1.0, "No result returned by PyDeequ"))

    # Scale-shift detection
    desc = "feature_2_scale_shift"
    entries = _get_constraint_entries(desc)
    observed = _get_observed("feature_2", "Maximum")
    for entry in entries:
        status = _status_from_entry(entry)
        reason = (
            None if status == "PASSED"
            else f"feature_2 max ({observed}) > alert_max ({alert_max}): scale-shift detected"
        )
        records.append(CheckRecord(desc, status, observed, alert_max, reason))
    if not entries:
        records.append(CheckRecord(desc, "UNKNOWN", None, alert_max, "No result returned by PyDeequ"))

    return records


def run_checks(
    spark: SparkSession,
    df: DataFrame,
    rules: dict,
    ref_stats: dict | None = None,
    mode: str = "collect_all",
) -> list[CheckRecord]:
    """
    Run all quality checks against df.

    Parameters
    ----------
    spark    : SparkSession
    df       : DataFrame to validate
    rules    : parsed quality_rules.yml
    ref_stats: reference statistics for scale-shift calibration
    mode     : 'collect_all' (default) or 'fail_fast'
    """
    checks = build_checks(spark, rules, ref_stats)

    suite = VerificationSuite(spark).onData(df)
    for chk in checks:
        suite = suite.addCheck(chk)

    log.info("Running %d check groups (mode=%s)...", len(checks), mode)
    vr: VerificationResult = suite.run()

    records = _map_deequ_results(spark, vr, checks, rules, ref_stats)

    if mode == "fail_fast":
        failed = [r for r in records if r.status == "FAILED"]
        if failed:
            log.error("FAIL_FAST: first failure → %r", failed[0])
            raise RuntimeError(f"Quality check failed in fail_fast mode: {failed[0].name}")

    return records

