# Part 1: Build the Pipeline

## Goal

Build a reproducible Spark/PyDeequ data-quality demonstration:

1. Generate deterministic clean synthetic data.
2. Inject nulls, unit-scale corruption, and duplicate IDs.
3. Validate before inference or feature-store ingestion.
4. Quarantine invalid batches and persist machine-readable diagnostics.

The implementation must support both a small development run (default
100,000 rows) and a configurable 100-million-row run without collecting the
dataset to the driver.

## Deliverables

Populate or add:

```text
inject_errors.py
validate_pipeline.py
requirements.txt
docker-compose.yml
config/quality_rules.yml
src/
  data_generation.py
  fault_injection.py
  quality_checks.py
  quarantine.py
```

Keep the two top-level scripts usable directly. Modules may be merged into
those scripts for a small academic submission, but retain the same logical
boundaries and CLI behavior.

Dependencies: Python 3.x, PySpark, PyDeequ, Java compatible with the selected
Spark/PyDeequ versions, PyYAML, pandas, NumPy, scikit-learn, matplotlib,
seaborn, and pytest. Pin or bound versions and document Java/Spark
compatibility.

## Data generation

Generate Parquet with a fixed seed and these columns:

| Column | Type | Requirement |
|---|---|---|
| `id` | long | unique primary key |
| `feature_1`, `feature_2`, `feature_3` | double | realistic numeric predictors |
| `target` | integer | binary class, 0 or 1 |
| `subgroup` | string | only `A` or `B` |
| `event_timestamp` | timestamp | temporal/batch analysis |
| `batch_id` | string | validation/quarantine unit |

Make subgroup B more dependent on `feature_2`, so its model accuracy degrades
more under scale corruption. The clean data must have no required-field nulls,
unique IDs, valid ranges, and documented subgroup proportions.

CLI requirements:

- configurable input/output paths and row count;
- fixed seed, overridable from the CLI;
- development default of 100,000 rows;
- full-scale mode for 100,000,000 rows;
- schema and row-count logging.

## Fault injection

Implement `inject_errors.py` (or `src/fault_injection.py`) with independent,
seeded rates:

- Set approximately 5% of `feature_1` to null.
- Multiply approximately 2% of `feature_2` by 1,000.
- Add approximately 3% duplicate rows with duplicate `id` values.

Define duplicate rate explicitly as duplicated output rows divided by the
original row count. Preserve all non-targeted columns. Do not replace invalid
values with zero. The scale shift must remain structurally numeric but violate
business/reference expectations.

Write injection metadata, including:

```json
{
  "seed": 42,
  "input_row_count": 100000,
  "null_injection_rate": 0.05,
  "unit_shift_rate": 0.02,
  "unit_shift_factor": 1000,
  "duplicate_rate": 0.03,
  "created_at": "...",
  "dataset_paths": {}
}
```

Also report actual affected counts/rates, before/after row counts, and
duplicate-ID counts. Do not include diagnostic injection flags in the dataset
passed to validation.

## Configuration

Create `config/quality_rules.yml` with required columns, completeness
thresholds, ID uniqueness, numeric bounds, allowed values, and failure policy.
Example shape:

```yaml
required_columns: [id, feature_1, feature_2, feature_3, target, subgroup, batch_id]
completeness: {feature_1: 1.0, feature_2: 1.0, id: 1.0, target: 1.0}
uniqueness: {id: 1.0}
ranges:
  feature_1: {min: -100.0, max: 100.0}
  feature_2: {min: 0.0, max: 10000.0}
  feature_3: {min: -100.0, max: 100.0}
allowed_values: {target: [0, 1], subgroup: [A, B]}
failure_policy: quarantine
```

Bounds must be derived from the generator's clean distribution, not chosen
arbitrarily. Keep configuration separate from code.

## Validation

Implement `validate_pipeline.py` using PyDeequ. Run validation before model
inference. Required checks:

- completeness for `feature_1`, `feature_2`, `id`, and `target`;
- uniqueness of `id`;
- required columns and schema;
- finite numeric values and configured ranges for all features;
- `target` in `{0, 1}` and `subgroup` in `{A, B}`;
- explicit `feature_2` scale-shift detection.

For scale detection, calculate clean-reference statistics (bounds and/or
quantiles/distribution metrics), persist the reference, and calibrate rules so
clean data passes while 1,000x values fail. A non-negative check alone is not
acceptable.

Support `collect_all` (default for the demonstration) and `fail_fast` modes.
Use clear CLI exit codes: invalid batches must exit non-zero.

Persist results such as `output/validation_results.json`:

```json
{
  "dataset": "corrupted",
  "batch_id": "...",
  "row_count": 100000,
  "passed": false,
  "checks": [{
    "name": "feature_1_completeness",
    "status": "FAILED",
    "observed": 0.95,
    "expected": 1.0,
    "failure_reason": "Null rate exceeds threshold",
    "metric": null
  }],
  "fault_categories_detected": ["missing_values", "unit_scale_shift", "duplicate_ids"],
  "runtime_seconds": 0.0,
  "timestamp": "..."
}
```

Every check needs a stable name, status, observed value, expected threshold,
failure reason, and dataset/batch identity. Failed validation must not produce
a success-shaped result.

## Quarantine

When any blocking check fails:

1. Mark the batch invalid.
2. Prevent inference from running on it.
3. Write failed records or the failed batch under
   `data/quarantined/batch_id=.../records/`.
4. Write `validation_results.json` and injection metadata beside the records.
5. Preserve enough metadata for remediation and replay.

Log input/output paths, counts, check outcomes, runtimes, quarantine path, and
exit status. Do not suppress Spark/PyDeequ errors or silently fall back to
empty data.

## Docker

Provide a reproducible `docker-compose.yml` with Spark master, worker, and
pipeline services; Jupyter is optional. The pipeline service must wait for
Spark, run generation/injection/validation, write outputs, and return a
meaningful status.

Document commands equivalent to:

```powershell
docker compose up --build
docker compose run --rm quality-pipeline python inject_errors.py --rows 100000
docker compose run --rm quality-pipeline python validate_pipeline.py --input data/generated/corrupted
```

Make Spark resources and shuffle partitions configurable.
