# Data Quality & ML Feature Store Pipeline: Comprehensive Review Guide

> **Quick Summary**: This document is a complete walkthrough for the academic/technical review of the Data Quality Pipeline project. It covers the layman's explanation of the core problem, our automated firewall solution, visual graph interpretations, technical architecture, and answers to common reviewer questions.  
> 📄 **Formal PDF Technical Report**: The complete 4-page technical report with formal derivations, architectural schematics, and embedded figures is available at [`data/reports/technical_report.pdf`](data/reports/technical_report.pdf). Rebuild using `python generate_technical_report.py`.

---

## 1. Layman's Explanation: The Problem & The Solution

### The Real-World Problem: "Silent Poisoning" of AI Systems
In modern software, machine learning (ML) models make critical decisions every second—such as approving credit cards, detecting financial fraud, or diagnosing patients. However, production data pipelines are messy:
- Sensors drop Wi-Fi connections $\rightarrow$ **Missing values (nulls)**.
- Systems change measurement units (e.g., dollars vs. cents, grams vs. kilograms) $\rightarrow$ **1000× scale shifts**.
- Network retry storms submit the same record multiple times $\rightarrow$ **Duplicate primary keys**.

#### Why is this dangerous?
1. **The Model Won't Crash**: The model receives the corrupted data and happily outputs predictions. There is no error message, yet the predictions are completely wrong (**silent failure**).
2. **The Damage Is Unequal (Algorithmic Inequity)**: If an engineering team only monitors the *average* accuracy across all users, the problem appears minor (average accuracy drops by ~4%). But behind that average, **Subgroup B** bore 100% of the damage (plummeting by **~10%**), while **Subgroup A** was unaffected. In loan approvals or hiring, this inadvertently creates severe demographic bias.

---

### Our Solution: An Automated Pre-Inference Quality Firewall
Instead of blindly passing incoming batches to the ML model, our system acts as an automated security checkpoint:

```
Incoming Raw Batch
       │
       ▼
┌──────────────────────────────────────┐
│  PyDeequ Data Quality Firewall       │
│  (17 statistical checks on Spark)    │
│  - No nulls allowed on feature_1     │
│  - No duplicate IDs                  │
│  - Distributions within bounds       │
└──────────────────┬───────────────────┘
                   │
         Did all 17 checks pass?
         ├── YES ──▶ Send to ML Model (Safe Inference)
         └── NO  ──▶ QUARANTINE & BLOCK INFERENCE (Circuit Breaker)
                     ├─ Save bad batch to isolated storage
                     ├─ Write forensic JSON audit logs
                     └─ Fall back to last known good baseline
```

---

## 2. Walkthrough of the Resultant Graphs

All visual artifacts and tables are persisted in [`data/reports/`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/data/reports/).

### Graph 1: Model Accuracy & Subgroup Disparity
**File:** [`data/reports/accuracy_comparison.png`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/data/reports/accuracy_comparison.png)  
*(Derived from [`data/reports/accuracy_changes.csv`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/data/reports/accuracy_changes.csv))*

| Pipeline State | Overall Accuracy | Subgroup A Accuracy | Subgroup B Accuracy |
| :--- | :---: | :---: | :---: |
| **1. Clean Baseline** | 85.30% | 80.43% | **90.39%** |
| **2. Corrupted (No QA)** | 80.97% *(-4.33 pp)* | 81.08% *(+0.65 pp, stable)* | **80.86% (-9.53 pp drop!)** |
| **3. Validated (Quarantine Active)** | 84.80% *(recovered)* | 80.16% *(stable)* | **89.39% (recovered +8.53 pp)** |

- **What it shows**: The bar chart contrasts model accuracy across the three states. While Subgroup A is immune to the injected faults, Subgroup B suffers a massive **9.53 percentage point crash**.
- **The Takeaway**: High-level aggregate metrics mask localized demographic failures. When our quarantine intercepts the corrupted batch, Subgroup B recovers immediately to **89.39%**.

---

### Graph 2: Unit Scale Shift Anomaly
**File:** [`data/reports/feature2_distribution.png`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/data/reports/feature2_distribution.png)

- **What it shows**: Under normal conditions, `feature_2` strictly ranges between $0.0$ and $100.0$. However, just a 2% injection of values scaled by $1000\times$ pushes values past $+3,000$ (and up to $100,000$).
- **The Takeaway**: Traditional statistical models (like Logistic Regression) cannot withstand multi-thousand magnitude outliers. PyDeequ's calibrated threshold check (`feature_2_scale_shift <= 110.0`) flags this distribution anomaly in milliseconds.

---

### Graph 3: Nulls & Duplicate Keys Summary
**File:** [`data/reports/null_duplicate_summary.png`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/data/reports/null_duplicate_summary.png)

- **What it shows**:
  - `feature_1`: Completeness dropped from 100% to **95.28%** (475 missing values injected).
  - `id`: Uniqueness dropped from 100% to **94.17%** (300 duplicate IDs injected).
- **The Takeaway**: Identifies schema and identity corruptions that would otherwise break database joins and feature encoders downstream.

---

### Graph 4: System Runtime & Overhead
**File:** [`data/reports/runtime_breakdown.png`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/data/reports/runtime_breakdown.png)

- **What it shows**: Execution latency for each phase of the pipeline.
- **The Takeaway**: Running deep PyDeequ verification over 10,000+ records in Apache Spark takes **~3.5 to 8 seconds**. The safety firewall adds virtually zero overhead compared to the cost of deploying a poisoned model to production.

---

## 3. Technical Architecture & Code Links

### A. Scalable Data Generation (No Driver Bottlenecks)
- **Source File:** [`src/data_generation.py`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/src/data_generation.py#L132-L220)
- **Design Pattern**: Many naive data generators use Pandas or Python loops, which run out of memory when scaling to millions of rows. Our pipeline uses `spark.range(n_rows)` and native Spark SQL columnar operations (`F.randn()`, `F.rand()`, `F.when()`). No data is ever collected back to the driver during generation.
- **Deterministic Seeding**: Driven by `SEED = 42`. Every partition and random draw produces identical Parquet datasets across runs.

### B. Mathematical Basis for Subgroup Sensitivity
- **Source File:** [`src/data_generation.py#L183-L191`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/src/data_generation.py#L183-L191)
- **The Formula**: The ground-truth classification label is generated from a logistic probability:
  $$\text{logit} = 0.1 \cdot f_1 + w_2 \cdot f_{2\text{norm}} + 0.05 \cdot f_3$$
  - For **Subgroup A**: $w_2 = 0.5$
  - For **Subgroup B**: $w_2 = 1.5$ *(3× greater dependence on `feature_2`!)*
- **Reviewer Insight**: When `feature_2` is corrupted with a $1000\times$ multiplier, Subgroup B's decision boundary is devastated because the model relies heavily on that feature to classify Subgroup B. Subgroup A barely uses `feature_2`, explaining why Subgroup A experienced zero degradation.

### C. Declarative PyDeequ Verification Suite
- **Source File:** [`src/quality_checks.py`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/src/quality_checks.py#L85-L173)
- **Configuration:** [`config/quality_rules.yml`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/config/quality_rules.yml)
- **Single-Pass Execution**: Instead of running 17 separate Spark SQL scans, PyDeequ aggregates metric computation states in a **single distributed pass** across the cluster.
- **Rule Categories**:
  - `hasCompleteness(col, >= 1.0)`: Catches null injections.
  - `hasUniqueness(['id'], == 1.0)`: Catches duplicate records.
  - `hasMin` / `hasMax`: Range bounds calibrated to clean distribution limits ($\pm 10\sigma$).
  - `feature_2_scale_shift` (`hasMax("feature_2", <= 110.0)`): Identifies unit shifts.

### D. Quarantine & Circuit Breaker Guard
- **Source File:** [`src/quarantine.py`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/src/quarantine.py#L29-L117)
- **Inference Guard**: When any rule fails, the pipeline:
  1. Writes the invalid batch to `data/quarantined/batch_id=<id>/records/`.
  2. Saves full forensic logs: `validation_results.json` and `quarantine_manifest.json`.
  3. Invokes [`block_inference()`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/src/quarantine.py#L108-L117), which raises `RuntimeError` and triggers a non-zero exit code (`exit 1`) to halt downstream workflows.

### E. Automated Test Suite
- **Source Files:** [`tests/test_pipeline.py`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/tests/test_pipeline.py) and [`tests/conftest.py`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/tests/conftest.py)
- 10 automated unit and integration tests verifying:
  - Seeded reproducibility.
  - Null, scale-shift, and duplicate injection accuracy within tight bounds.
  - 100% detection rate across all 4 fault categories.
  - Quarantine inference blocking.
  - Subgroup B degradation disparity assertion (`acc_drop_B > acc_drop_A`).

---

## 4. Reviewer Questions & Model Answers

### Q1: "Why use Spark/Deequ instead of just writing Pandas or simple Python assertions?"
> **Answer**: Pandas requires loading the entire dataset into the memory of a single machine (the driver). For production ML pipelines handling gigabytes or terabytes (100M+ rows), Pandas crashes with an `OutOfMemory` exception. Apache Spark distributes computation across a cluster, and AWS Deequ compiles all quality rules into a single execution plan, scanning the data once rather than running dozens of separate queries.

### Q2: "Why did you quarantine the entire batch instead of simply dropping invalid rows?"
> **Answer**: In mission-critical environments (financial transactions, patient records), silently dropping invalid rows introduces severe selection bias and corrupts aggregate totals. For instance, dropping duplicate IDs without investigating can obscure replay attack vulnerabilities or upstream ingestion retries. Quarantining the batch isolates the issue, generates an audit trail, and lets the system safely serve a verified baseline model.

### Q3: "What is 'Disparate Impact' and how did your experiment demonstrate it?"
> **Answer**: Disparate impact is when an engineering failure or model degradation disproportionately harms a specific sub-population. In our experiment, overall model accuracy dropped by only 4.33 percentage points. However, Subgroup B suffered a 9.53 percentage point crash, while Subgroup A was untouched. Aggregate monitoring alone creates a false sense of security; per-subgroup monitoring is vital for AI fairness.

### Q4: "How does the pipeline handle Windows environment challenges?"
> **Answer**: PySpark on Windows often encounters issues with multi-adapter hostname resolution and Py4J thread lifecycles. We resolved this by:
> 1. Pinning Spark networking to loopback (`127.0.0.1` via `SPARK_LOCAL_IP`).
> 2. Direct JAR classpath loading (`spark.driver.extraClassPath`) to avoid Windows temporary file-locking errors.
> 3. Explicit Py4J gateway callback shutdown hooks in [`validate_pipeline.py`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/validate_pipeline.py) and [`tests/conftest.py`](file:///d:/Prithvi/Academics/College/Sem5/BDSRAE/DataQualityProject/tests/conftest.py).

---

## 5. Live Demo Script (Step-by-Step)

To run the live demo in front of a reviewer, activate the virtual environment and execute:

```powershell
# 1. Generate clean data and inject deterministic faults
python inject_errors.py --rows 10000

# 2. Run PyDeequ quality checks & quarantine corrupted batch (exits cleanly with code 1)
python validate_pipeline.py --input data/generated/corrupted --injection-meta data/generated/injection_metadata.json

# 3. Run 3-state ML fairness experiment
python src/model_simulation.py

# 4. Run automated test suite (all 10 tests pass)
pytest tests/ -v
```

