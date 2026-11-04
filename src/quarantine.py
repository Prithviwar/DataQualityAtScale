"""
src/quarantine.py
-----------------
Quarantine logic for the data-quality pipeline.

When any blocking check fails:
  1. Mark the batch invalid.
  2. Prevent inference from running on it.
  3. Write failed records (or the entire failed batch) under
       data/quarantined/batch_id=<id>/records/
  4. Write validation_results.json and injection metadata beside the records.
  5. Preserve enough metadata for remediation and replay.
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession

log = logging.getLogger(__name__)

QUARANTINE_BASE = "data/quarantined"


def quarantine_batch(
    spark: SparkSession,
    df: DataFrame,
    batch_id: str,
    validation_results: dict,
    injection_metadata: Optional[dict] = None,
    quarantine_base: str = QUARANTINE_BASE,
) -> str:
    """
    Write the corrupted/failed batch to quarantine storage.

    Parameters
    ----------
    spark              : SparkSession
    df                 : the full dataset that failed validation
    batch_id           : identifier for this batch
    validation_results : the validation output dict (from validate_pipeline)
    injection_metadata : optional dict from fault_injection (for replay/remediation)
    quarantine_base    : root quarantine directory

    Returns
    -------
    quarantine_path : str – the path where records were written
    """
    quarantine_path = str(Path(quarantine_base) / f"batch_id={batch_id}")
    records_path = str(Path(quarantine_path) / "records")

    log.info("Quarantining batch %r → %s", batch_id, quarantine_path)

    # Write full failed batch as Parquet
    df.write.mode("overwrite").parquet(records_path)
    written = spark.read.parquet(records_path).count()
    log.info("Quarantine: wrote %d rows to %s", written, records_path)

    # Write validation results JSON
    vr_path = Path(quarantine_path) / "validation_results.json"
    vr_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vr_path, "w") as fh:
        json.dump(validation_results, fh, indent=2)
    log.info("Quarantine: validation results → %s", vr_path)

    # Write injection metadata JSON (if provided)
    if injection_metadata is not None:
        meta_path = Path(quarantine_path) / "injection_metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(injection_metadata, fh, indent=2)
        log.info("Quarantine: injection metadata → %s", meta_path)

    # Write a compact quarantine manifest
    manifest = {
        "batch_id": batch_id,
        "quarantined_at": datetime.now(timezone.utc).isoformat(),
        "records_path": records_path,
        "row_count": written,
        "validation_summary": {
            "passed": validation_results.get("passed"),
            "fault_categories_detected": validation_results.get("fault_categories_detected", []),
        },
        "remediation_notes": (
            "1. Inspect validation_results.json for specific failures.\n"
            "2. Use injection_metadata.json to understand injected faults.\n"
            "3. Re-run the pipeline after fixing the data source.\n"
            "4. Remove quarantine dir before re-ingesting corrected batch."
        ),
    }
    manifest_path = Path(quarantine_path) / "quarantine_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Quarantine: manifest → %s", manifest_path)

    return str(quarantine_path)


def is_quarantined(batch_id: str, quarantine_base: str = QUARANTINE_BASE) -> bool:
    """Return True if the given batch_id is in quarantine."""
    path = Path(quarantine_base) / f"batch_id={batch_id}"
    return path.exists()


def block_inference(batch_id: str, quarantine_base: str = QUARANTINE_BASE) -> None:
    """
    Raise RuntimeError if the batch is quarantined (inference guard).
    Call this before any model inference step.
    """
    if is_quarantined(batch_id, quarantine_base):
        raise RuntimeError(
            f"Batch {batch_id!r} is quarantined – inference blocked. "
            f"Inspect: {quarantine_base}/batch_id={batch_id}/validation_results.json"
        )

