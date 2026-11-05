# Data Quality Pipeline for ML Feature Stores

A reproducible Spark/PyDeequ data-quality demonstration that generates deterministic
synthetic data, injects configurable faults, validates before inference, and quarantines
invalid batches with machine-readable diagnostics.

## Motivation

Data quality failures silently degrade ML model accuracy — especially for underrepresented
subgroups. **Overall accuracy alone is insufficient**: a 1000× scale-shift in `feature_2`
drops subgroup B accuracy far more than subgroup A because B's target label is more
correlated with that feature. Without per-subgroup monitoring, this disparity goes
undetected while aggregate metrics look acceptable.

This project demonstrates:
1. **Detection** — PyDeequ catches nulls, duplicates, and distribution shifts before inference.
2. **Prevention** — quarantine blocks bad batches from reaching the model.
3. **Fairness** — subgroup-level accuracy monitoring reveals disproportionate impact.

## Architecture

```
┌─────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│  Data Generation │────▶│  Fault Injection   │────▶│  Quality Validation│
│  (Spark Parquet) │     │  (nulls, scale,    │     │  (PyDeequ checks)  │
│  seed=42, N rows │     │   duplicates)      │     │  config/rules.yml  │
└─────────────────┘     └───────────────────┘     └────────┬───────────┘
                                                           │
                                              ┌────────────┴────────────┐
                                              │                         │
                                         ALL PASSED               ANY FAILED
                                              │                         │
                                    ┌─────────▼─────────┐   ┌──────────▼──────────┐
                                    │  Model Inference   │   │   Quarantine Batch   │
                                    │  (proceed safely)  │   │   (block inference,  │
                                    └───────────────────┘   │    write diagnostics) │
                                                            └─────────────────────┘
```

## Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10 – 3.12 | Tested with 3.11 |
| Java | 17 | Oracle or OpenJDK |
| PySpark | 4.3.x | Bundled Spark 4.x |
| PyDeequ | 1.7.x | Deequ JAR auto-downloaded |

## Setup

### Local (Windows/Linux/Mac)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Set required env var
export SPARK_VERSION=4.1
# $env:SPARK_VERSION="4.1"  # Windows PowerShell
```

### Docker

```bash
docker compose up --build
```

## Usage

### Development Run (100K rows, default)

```bash
# Step 1: Generate clean data + inject faults
python inject_errors.py --rows 100000

# Step 2: Validate corrupted dataset
python validate_pipeline.py 
    --input data/generated/corrupted 
    --injection-meta data/generated/injection_metadata.json

# Step 3: Model impact experiment
python src/model_simulation.py

# Step 4: Run tests
pytest tests/ -v

# Step 5: Open analysis notebook
jupyter lab results_analysis.ipynb
```

### Full-Scale Run (100M rows)

```bash
# Increase resources
export SPARK_DRIVER_MEMORY=8g
export SPARK_EXECUTOR_MEMORY=8g
export SPARK_SHUFFLE_PARTITIONS=200

python inject_errors.py --full-scale
python validate_pipeline.py --input data/generated/corrupted
```

### Docker Commands

```bash
docker compose run --rm quality-pipeline python inject_errors.py --rows 100000
docker compose run --rm quality-pipeline python validate_pipeline.py --input data/generated/corrupted
docker compose run --rm quality-pipeline python src/model_simulation.py
```

## Quality Rules

Rules are defined in [`config/quality_rules.yml`](config/quality_rules.yml):

| Rule | Column | Threshold | Derived From |
|------|--------|-----------|-------------|
| Completeness | feature_1, feature_2, id, target | 1.0 (no nulls) | Schema requirement |
| Uniqueness | id | 1.0 (all unique) | Primary key constraint |
| Range (min/max) | feature_1 | [-100, 100] | N(0,10) → ±10σ |
| Range (min/max) | feature_2 | [0, 110] | U(0,100) + 10% headroom |
| Range (min/max) | feature_3 | [-300, 300] | N(0,50) → ±6σ |
| Allowed values | target | {0, 1} | Binary classification |
| Allowed values | subgroup | {A, B} | Design spec |
| Scale-shift | feature_2 | max ≤ 110 | Clean p99 ≈ 100; 1000× → ~100K |

### Reference-Bound Calibration

Bounds are **derived from the generator's clean distribution**, not chosen arbitrarily:
- `feature_2` clean max ≈ 100.0, so alert threshold = 110 (10% headroom)
- A 1000× scale shift produces values ~100,000 → **immediately detected**
- A simple non-negative check would miss this — the shifted values are still valid doubles

## Quarantine Behavior

When any blocking check fails:
1. Batch is marked **invalid**
2. **Inference is blocked** (`block_inference()` raises `RuntimeError`)
3. Failed batch written to `data/quarantined/batch_id=<id>/records/`
4. Diagnostic files saved alongside records:
   - `validation_results.json` — per-check outcomes
   - `injection_metadata.json` — fault parameters
   - `quarantine_manifest.json` — remediation guide

## Output Files

| File | Description |
|------|-------------|
| `data/generated/clean/` | Clean Parquet dataset |
| `data/generated/corrupted/` | Corrupted Parquet dataset |
| `data/generated/injection_metadata.json` | Fault injection parameters and actual counts |
| `output/validation_results.json` | PyDeequ check outcomes |
| `output/accuracy_comparison.csv` | 3-state accuracy table |
| `output/runtime_metrics.json` | Timing and accuracy deltas |
| `data/quarantined/batch_id=.../` | Quarantined batch with diagnostics |
| `data/reports/*.png` | Analysis charts (from notebook) |
| `data/reports/*.csv` | Analysis tables (from notebook) |

## Why Subgroup B Needs Separate Monitoring

In the synthetic data, subgroup B's target label is **3× more correlated** with
`feature_2` than subgroup A. When `feature_2` gets a 1000× scale shift:

| State | Overall | Subgroup A | Subgroup B |
|-------|---------|-----------|-----------|
| Clean Baseline | 85.30% | 80.43% | **90.39%** |
| Corrupted (No QA) | 80.97% | 81.08% | **80.86%** |
| **Drop** | **-4.33 pp** | **+0.65 pp** | **-9.53 pp** |
| Validated (Quarantine Active) | 84.80% | 80.16% | **89.39%** |

Subgroup B loses **9.53 percentage points of accuracy**, while Subgroup A is essentially unaffected. Monitoring only overall accuracy would show a modest ~4.3 pp drop—which might easily slip past standard alerts. The real problem is that Subgroup B suffers the entire localized brunt of the failure, demonstrating why per-subgroup data quality gates are essential for AI fairness.

## Reproducibility

- All random operations use `seed=42` (overridable via `--seed`)
- Data generation is deterministic (Spark `randn(seed)` and `rand(seed)`)
- Same seed → identical Parquet files → identical validation results
- The `validated_quarantined` state uses `random_state=43` for train/test split
  to simulate minor operational variance (not identical to clean baseline)

## Resource Assumptions

| Scenario | Rows | Driver Memory | Executor Memory | Shuffle Partitions | Expected Time |
|----------|------|---------------|-----------------|-------------------|---------------|
| Dev | 100K | 2g | 2g | 8 | ~30s |
| Full-scale | 100M | 8g | 8g | 200 | ~10–15 min |

Validation overhead target: ~4.2 minutes for 100M rows. Actual times depend on
hardware (CPU cores, SSD vs HDD, available memory). Measured dev-scale validation
takes ~10s for 10K rows.

## Known Limitations

1. **PySpark 4.3 + pandas ≥ 3.0**: FutureWarning about incomplete support; no functional impact.
2. **Windows `python3`**: Spark workers look for `python3`; fixed by setting `PYSPARK_PYTHON` automatically.
3. **Deequ JAR first-run**: First `spark.jars.packages` invocation downloads the JAR (~2MB) via Maven, which can take several minutes. Subsequent runs use the cached JAR.
4. **Docker Spark version**: `docker-compose.yml` uses `bitnami/spark:3.5` for the cluster; the pipeline container uses PySpark 4.3 in local mode by default.
5. **Accuracy targets**: The plan specifies target accuracy values. Actual values depend on the exact logistic model and data distribution; measured values may differ slightly.

## Project Structure

```
DataQualityProject/
├── inject_errors.py          # CLI: generate + inject faults
├── validate_pipeline.py      # CLI: PyDeequ validation + quarantine
├── review_guide.md           # Reviewer guide: layman summary, graph guide & Q&A
├── src/
│   ├── data_generation.py    # Spark-native Parquet generation
│   ├── fault_injection.py    # NULL, scale-shift, duplicate injection
│   ├── quality_checks.py     # PyDeequ check builder + result mapper
│   ├── quarantine.py         # Quarantine logic + inference guard
│   └── model_simulation.py   # 3-state accuracy experiment
├── config/
│   └── quality_rules.yml     # Quality rules (bounds from clean distribution)
├── tests/
│   ├── conftest.py           # Shared SparkSession fixture + clean exit hooks
│   └── test_pipeline.py      # Unit + integration tests
├── results_analysis.ipynb    # Analysis notebook with visualizations
├── requirements.txt          # Pinned dependencies
├── docker-compose.yml        # Multi-container Spark setup
├── Dockerfile                # Pipeline container
├── .gitignore                # Git ignore rules for Spark, venv, and datasets
└── README.md                 # Project documentation
```

