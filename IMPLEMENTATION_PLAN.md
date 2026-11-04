# Implementation Plan: Data Quality at Scale

## 1. Project Objective

Build a reproducible Spark-based data-quality pipeline that:

1. Generates a clean synthetic dataset representing a 100-million-row enterprise pipeline.
2. Injects realistic silent data-corruption patterns:
   - 5% nulls in `feature_1`
   - 2% ×1000 unit-scale shifts in `feature_2`
   - 3% duplicate rows
3. Validates data using PyDeequ before feature-store/model ingestion.
4. Detects and quarantines invalid batches.
5. Measures validation runtime and detection rates.
6. Demonstrates the effect of corruption on:
   - Overall model accuracy
   - Subgroup A accuracy
   - Subgroup B accuracy
7. Produces reproducible reports and visualizations.

The implementation must support the claims in the technical report and presentation context.

---

## 2. Target Results

The implementation should reproduce or clearly approximate these reported values:

| Pipeline state | Overall accuracy | Subgroup A | Subgroup B | Validation overhead |
|---|---:|---:|---:|---:|
| Clean baseline | 88.4% | 89.1% | 87.6% | 0.0 min |
| Corrupted, no checks | 71.2% | 76.4% | 61.8% | 0.0 min |
| Validated and quarantined | 87.9% | 88.5% | 87.1% | +4.2 min |

The implementation should report whether values are exact, approximate, or generated from a controlled simulation.

Important distinction:

- The overall degradation is **17.2 percentage points**, from 88.4% to 71.2%.
- Subgroup A loses **12.7 percentage points**.
- Subgroup B loses **25.8 percentage points**.

---

## 3. Recommended Repository Structure

Populate the existing scaffold files:

- `inject_errors.py`
- `validate_pipeline.py`
- `results_analysis.ipynb`
- `docker-compose.yml`

Add the following supporting files:

```text
DataQualityProject/
├── inject_errors.py
├── validate_pipeline.py
├── results_analysis.ipynb
├── docker-compose.yml
├── requirements.txt
├── README.md
├── config/
│   └── quality_rules.yml
├── src/
│   ├── data_generation.py
│   ├── fault_injection.py
│   ├── quality_checks.py
│   ├── quarantine.py
│   ├── model_simulation.py
│   └── reporting.py
├── tests/
│   ├── test_fault_injection.py
│   ├── test_quality_checks.py
│   └── test_metrics.py
├── data/
│   ├── generated/
│   ├── quarantined/
│   └── reports/
└── output/
    ├── validation_results.json
    ├── accuracy_comparison.csv
    └── runtime_metrics.json
```

If a simpler academic submission is preferred, the `src/` modules may be merged into the two main Python scripts, but the logical separation should remain.

---

## 4. Environment and Dependency Setup

### Primary technology stack

- Python 3.x
- Apache Spark
- PySpark
- PyDeequ
- Java runtime compatible with the selected Spark/PyDeequ versions
- Docker and Docker Compose
- Pandas and Matplotlib/Seaborn for analysis
- Scikit-learn or a deterministic custom model simulator

### Dependency requirements

Create `requirements.txt` containing pinned or bounded versions for:

- `pyspark`
- `pydeequ`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pyyaml`
- `pytest`

Document the required Java version and Spark/PyDeequ compatibility in `README.md`.

---

## 5. Data Model

Generate a deterministic synthetic dataset with at least these columns:

| Column | Type | Purpose |
|---|---|---|
| `id` | Long | Primary key and duplicate detection |
| `feature_1` | Double | Primary predictor affected by null injection |
| `feature_2` | Double | Numeric predictor affected by unit shifts |
| `feature_3` | Double | Additional clean predictor |
| `target` | Integer | Binary classification target |
| `subgroup` | String | Values `A` and `B` |
| `event_timestamp` | Timestamp | Batch and temporal analysis |
| `batch_id` | String | Validation and quarantine unit |

### Dataset generation requirements

- Default size: 100,000,000 rows.
- Add a configurable smaller development size, such as 100,000 rows.
- Use a fixed random seed.
- Generate realistic feature distributions rather than arbitrary uniform values.
- Make Subgroup B more dependent on `feature_2`, so unit corruption disproportionately harms it.
- Ensure the clean dataset contains:
  - No nulls in required fields
  - Unique IDs
  - Values inside expected feature ranges
  - Balanced or intentionally documented subgroup proportions

The generator must not collect the entire dataset to the driver.

---

## 6. Fault Injection Design

Implement fault injection in `inject_errors.py` or a dedicated `fault_injection.py` module.

### 6.1 Null injection

Inject nulls into `feature_1` for approximately 5% of records.

Requirements:

- Use a deterministic random seed.
- Preserve all other columns.
- Report:
  - Number of affected rows
  - Actual null percentage
  - Target null percentage
- Do not silently replace invalid values with zero.

### 6.2 Unit-scale shift injection

Multiply approximately 2% of `feature_2` values by `1000`.

Requirements:

- Preserve the original value or injection indicator only in an internal diagnostic dataset, not as a validation input.
- Ensure shifted values remain valid numeric values.
- Make the corruption semantically invalid but structurally valid.
- Report the number and percentage of shifted records.

A non-negative check alone is insufficient because multiplying positive values by 1000 still produces non-negative values. The implementation must use one or more of:

- Upper and lower business bounds
- Baseline quantile bounds
- Distribution statistics
- Mean, standard deviation, or quantile drift checks
- Reference-data comparison

### 6.3 Duplicate injection

Duplicate approximately 3% of rows.

Requirements:

- Duplicate complete records or selected entity records consistently.
- Ensure duplicate IDs are present so `hasUniqueness("id")` fails.
- Report:
  - Total row count before and after injection
  - Number of duplicate IDs
  - Duplicate rate
- Define whether the 3% rate refers to duplicated rows or duplicate groups.

### 6.4 Injection metadata

Write a metadata file containing:

```json
{
  "seed": 42,
  "input_row_count": 100000000,
  "null_injection_rate": 0.05,
  "unit_shift_rate": 0.02,
  "unit_shift_factor": 1000,
  "duplicate_rate": 0.03,
  "created_at": "...",
  "dataset_paths": {}
}
```

---

## 7. Quality Validation Architecture

Implement the validation workflow in `validate_pipeline.py`.

The pipeline should execute validation before model inference or feature-store ingestion.

### 7.1 Required PyDeequ checks

#### Completeness

```text
isComplete("feature_1")
isComplete("feature_2")
isComplete("id")
isComplete("target")
```

`feature_1` must have zero null values for a valid batch.

#### ID uniqueness

```text
hasUniqueness("id", lambda value: value == 1.0)
```

The uniqueness check must fail when duplicate IDs exist.

#### Numeric validity

Check that:

- `feature_1` is within expected bounds.
- `feature_2` is within expected bounds.
- `feature_3` is within expected bounds.
- Values are not NaN or infinite.
- `target` contains only valid class values.
- `subgroup` contains only `A` or `B`.

#### Unit-shift detection

Implement explicit range or distribution checks for `feature_2`.

Recommended approach:

1. Calculate reference statistics on the clean baseline.
2. Store expected lower and upper bounds in configuration.
3. Validate corrupted data against those bounds.
4. Supplement static bounds with distribution metrics such as:
   - Mean
   - Standard deviation
   - Quantiles
   - Approximate histogram
   - Relative distribution drift

The rules must be calibrated so that the 1000× shift is detected without incorrectly rejecting clean data.

### 7.2 Optional Great Expectations comparison

The report abstract references both PyDeequ and Great Expectations. The primary implementation should use PyDeequ, but the plan may include an optional comparator:

- Reimplement completeness, uniqueness, and range rules in Great Expectations.
- Run both frameworks on the development dataset.
- Compare:
  - Detection rate
  - Runtime
  - Rule expressiveness
  - Report format
  - Operational complexity

This comparison should not delay the core PyDeequ implementation.

---

## 8. Validation Result Contract

Persist validation results in a structured format such as JSON:

```json
{
  "dataset": "corrupted",
  "row_count": 100000000,
  "passed": false,
  "checks": [
    {
      "name": "feature_1_completeness",
      "status": "FAILED",
      "observed": 0.95,
      "expected": 1.0,
      "failure_reason": "Null rate exceeds zero-tolerance threshold"
    }
  ],
  "fault_categories_detected": [
    "missing_values",
    "unit_scale_shift",
    "duplicate_ids"
  ],
  "runtime_seconds": 252.0
}
```

Each check must have:

- Stable name
- Status
- Observed value
- Expected threshold
- Failure reason
- Optional metric value
- Timestamp
- Dataset or batch identifier

Do not return a success-shaped result when validation fails.

---

## 9. Quarantine Workflow

When any blocking quality check fails:

1. Mark the batch as invalid.
2. Prevent it from reaching model inference.
3. Write the failed batch or invalid records to a quarantine location.
4. Write validation diagnostics alongside the quarantined data.
5. Return a non-zero process exit code for production-style execution.
6. Preserve enough metadata for remediation and replay.

Recommended quarantine partitioning:

```text
data/quarantined/
└── batch_id=...
    ├── records/
    ├── validation_results.json
    └── injection_metadata.json
```

The implementation should support two modes:

- `fail_fast`: stop after the first blocking failure.
- `collect_all`: execute all checks and report every failure before quarantining.

Use `collect_all` for the demonstration and reporting workflow.

---

## 10. Model Impact Simulation

Implement a deterministic model-impact experiment.

### Clean baseline

Run the model on the clean dataset and measure:

- Overall accuracy: target approximately 88.4%
- Subgroup A: approximately 89.1%
- Subgroup B: approximately 87.6%

### Corrupted, unchecked pipeline

Run inference on corrupted data without validation and measure:

- Overall accuracy: approximately 71.2%
- Subgroup A: approximately 76.4%
- Subgroup B: approximately 61.8%

The model or data-generation process must make Subgroup B more sensitive to `feature_2`.

### Validated and quarantined pipeline

Exclude quarantined corrupted records or use the clean valid portion according to the chosen batch policy.

Measure:

- Overall accuracy: approximately 87.9%
- Subgroup A: approximately 88.5%
- Subgroup B: approximately 87.1%

Clearly document whether the validated result represents:

- Clean records only,
- Reprocessed data after remediation, or
- The original clean baseline after bad-batch rejection.

---

## 11. Results Analysis Notebook

Implement `results_analysis.ipynb` with these sections:

1. Load validation and model result files.
2. Display injected fault rates.
3. Display PyDeequ check outcomes.
4. Compare clean, corrupted, and validated accuracy.
5. Calculate absolute and relative accuracy changes.
6. Analyze subgroup disparity.
7. Visualize:
   - Accuracy by pipeline state
   - Accuracy by subgroup
   - Runtime overhead
   - Detection rate by fault category
   - Feature distributions before and after unit shifts
8. Generate a final summary table.
9. Export figures and CSV results to `data/reports/`.

Required charts:

- Grouped bar chart for overall/Subgroup A/Subgroup B accuracy.
- Distribution comparison for `feature_2`.
- Null-rate and duplicate-rate summary.
- Validation runtime chart.
- Fault detection matrix.

---

## 12. Docker Compose Execution

Implement `docker-compose.yml` with a reproducible Spark environment.

Recommended services:

- `spark-master`
- `spark-worker`
- `quality-pipeline`
- Optional `jupyter`

The pipeline service should:

1. Wait for Spark availability.
2. Run data generation or load existing data.
3. Execute fault injection.
4. Run PyDeequ validation.
5. Write results and quarantine outputs.
6. Exit with a meaningful status.

Example documented commands:

```powershell
docker compose up --build
docker compose run --rm quality-pipeline python inject_errors.py --rows 100000
docker compose run --rm quality-pipeline python validate_pipeline.py --input data/generated/corrupted
```

The 100-million-row run should be configurable separately from development runs.

---

## 13. Configuration

Create `config/quality_rules.yml`:

```yaml
required_columns:
  - id
  - feature_1
  - feature_2
  - feature_3
  - target
  - subgroup
  - batch_id

completeness:
  feature_1: 1.0
  feature_2: 1.0
  id: 1.0

uniqueness:
  id: 1.0

ranges:
  feature_1:
    min: -100.0
    max: 100.0
  feature_2:
    min: 0.0
    max: 10000.0
  feature_3:
    min: -100.0
    max: 100.0

allowed_values:
  target: [0, 1]
  subgroup: ["A", "B"]

failure_policy: quarantine
```

Bounds must be derived from the generated data distribution and documented. They should not be arbitrary values that accidentally fail the clean baseline.

---

## 14. Testing Strategy

### Unit tests

Test:

- Correct null injection percentage.
- Correct unit-shift percentage and factor.
- Correct duplicate generation.
- Fixed-seed reproducibility.
- Completeness check failure.
- Uniqueness check failure.
- Range/distribution check failure.
- Clean data passing all checks.
- Invalid data being quarantined.
- Correct accuracy-difference calculations.

### Integration tests

Run a small dataset, such as 10,000 rows, through the complete workflow:

1. Generate clean data.
2. Inject faults.
3. Validate.
4. Quarantine.
5. Produce model metrics.
6. Generate analysis outputs.

### Acceptance thresholds

For the small-scale test, use tolerances rather than requiring exact percentages:

- Null rate: within ±0.1 percentage points of 5%.
- Unit-shift rate: within ±0.1 percentage points of 2%.
- Duplicate rate: within ±0.1 percentage points of 3%.
- Detection rate: 100% for all three injected fault categories.
- Clean dataset: all blocking checks pass.
- Corrupted dataset: at least one blocking check fails.
- Subgroup B degradation must exceed Subgroup A degradation.
- Validation must not send a failed batch to inference.

---

## 15. Runtime and Scalability Requirements

Measure separately:

- Data generation time
- Fault injection time
- Validation time
- Model evaluation time
- Total pipeline time
- Driver memory usage
- Executor memory usage

For the 100-million-row run:

- Avoid driver-side collection.
- Use Spark aggregations and approximate statistics where appropriate.
- Use partitioned Parquet output.
- Make shuffle partitions configurable.
- Record the machine and Docker resource configuration.
- Report whether the approximately 4.2-minute validation overhead is reproduced.

If hardware prevents reproducing 4.2 minutes exactly, report the measured value and explain the environment difference rather than fabricating the target.

---

## 16. Logging and Error Handling

The pipeline must provide explicit logs for:

- Input and output paths
- Row counts
- Fault injection counts
- Quality-check results
- Runtime per stage
- Quarantine location
- Exit status

Use structured logging where practical.

Avoid:

- Broad exception suppression
- Silent fallback to empty datasets
- Treating failed validation as success
- Replacing invalid numeric values without reporting them
- Hiding Spark or PyDeequ errors

---

## 17. Documentation Requirements

Create `README.md` covering:

1. Project motivation.
2. Architecture diagram or textual flow.
3. Environment prerequisites.
4. Local setup.
5. Docker setup.
6. Development-size execution.
7. Full 100-million-row execution.
8. Quality rules.
9. Quarantine behavior.
10. Results interpretation.
11. Known limitations.
12. Reproducibility details.
13. Mapping between implementation outputs and report claims.

The documentation should explicitly explain why overall accuracy alone is insufficient and why Subgroup B requires separate monitoring.

---

## 18. Implementation Sequence

### Phase 1: Environment

- Add dependencies.
- Define Docker services.
- Verify Spark starts successfully.
- Add configuration loading.

### Phase 2: Data generation

- Implement deterministic synthetic data generation.
- Add development and full-scale row-count options.
- Write clean Parquet output.
- Add row-count and schema checks.

### Phase 3: Fault injection

- Implement nulls, scale shifts, and duplicates.
- Produce injection metadata.
- Add unit tests.

### Phase 4: Validation

- Implement PyDeequ analyzers and constraints.
- Add completeness, uniqueness, range, and distribution checks.
- Persist structured results.
- Add quarantine routing.

### Phase 5: Model impact

- Implement deterministic model or model simulator.
- Calculate global and subgroup metrics.
- Produce the three pipeline-state comparison.

### Phase 6: Analysis

- Complete the notebook.
- Export tables and figures.
- Compare measured results with target report values.

### Phase 7: Verification

- Run unit tests.
- Run the small integration test.
- Run the Docker workflow.
- Run the full-scale test if resources permit.
- Document actual timings and deviations.

---

## 19. Final Acceptance Criteria

The project is complete when:

- The clean and corrupted datasets can be generated reproducibly.
- All three fault vectors are injected at configurable rates.
- PyDeequ detects missing values, duplicates, and unit-scale corruption.
- Failed data is quarantined before model inference.
- Validation results are persisted in a machine-readable format.
- Accuracy is reported globally and by subgroup.
- Subgroup B shows materially greater degradation under corruption.
- The notebook produces the required comparison tables and charts.
- Docker Compose runs the workflow from a clean environment.
- Tests verify the core detection and quarantine behavior.
- Documentation explains execution, assumptions, thresholds, and measured results.
- Any difference from the report’s target values is explicitly reported rather than concealed.
