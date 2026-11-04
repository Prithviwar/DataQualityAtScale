"""
tests/test_pipeline.py
-----------------------
Focused tests for the data-quality pipeline.

Run with:  pytest tests/ -v --tb=short
"""

import os
import sys

os.environ["SPARK_VERSION"] = "4.1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_generation import generate_clean_data
from fault_injection import inject_faults, run_injection
from quality_checks import run_checks
from quarantine import quarantine_batch, block_inference, is_quarantined

SMALL_ROWS = 1000
SEED = 42


# ========================================================================
# 1. Seeded reproducibility
# ========================================================================
class TestReproducibility:
    def test_seeded_reproducibility(self, spark, tmp_dir):
        """Generate data twice with the same seed → identical outputs."""
        path1 = os.path.join(tmp_dir, "run1")
        path2 = os.path.join(tmp_dir, "run2")

        generate_clean_data(spark, n_rows=SMALL_ROWS, seed=SEED, output_path=path1, batch_id="repro1")
        generate_clean_data(spark, n_rows=SMALL_ROWS, seed=SEED, output_path=path2, batch_id="repro2")

        df1 = spark.read.parquet(path1).orderBy("id")
        df2 = spark.read.parquet(path2).orderBy("id")

        assert df1.count() == df2.count() == SMALL_ROWS

        ids1 = [r.id for r in df1.select("id").limit(10).collect()]
        ids2 = [r.id for r in df2.select("id").limit(10).collect()]
        assert ids1 == ids2

        # Feature values should be identical
        f1_vals = [r.feature_1 for r in df1.select("feature_1").limit(10).collect()]
        f2_vals = [r.feature_1 for r in df2.select("feature_1").limit(10).collect()]
        assert f1_vals == f2_vals


# ========================================================================
# 2. Null injection rate
# ========================================================================
class TestNullInjection:
    def test_null_injection_rate(self, spark, tmp_dir):
        """feature_1 null rate should be within ±1% of target 5%."""
        path = os.path.join(tmp_dir, "clean")
        generate_clean_data(spark, n_rows=SMALL_ROWS, seed=SEED, output_path=path, batch_id="null_test")
        df = spark.read.parquet(path)

        corrupted, meta = inject_faults(spark, df, seed=SEED, null_rate=0.05,
                                        unit_shift_rate=0.0, duplicate_rate=0.0)

        null_count = corrupted.filter(corrupted.feature_1.isNull()).count()
        null_rate = null_count / SMALL_ROWS
        # Within ±1 percentage point of 5%
        assert 0.04 <= null_rate <= 0.06, f"Null rate {null_rate:.3f} out of [0.04, 0.06]"
        assert meta["null_injection_actual_count"] == null_count


# ========================================================================
# 3. Scale-shift injection
# ========================================================================
class TestScaleShift:
    def test_scale_shift_injection(self, spark, tmp_dir):
        """feature_2 should have ~2% of values scaled by 1000x."""
        path = os.path.join(tmp_dir, "clean")
        generate_clean_data(spark, n_rows=SMALL_ROWS, seed=SEED, output_path=path, batch_id="shift_test")
        df = spark.read.parquet(path)

        corrupted, meta = inject_faults(spark, df, seed=SEED, null_rate=0.0,
                                        unit_shift_rate=0.02, duplicate_rate=0.0)

        from pyspark.sql import functions as F
        shifted = corrupted.filter(F.col("feature_2") > 100.0).count()
        shift_rate = shifted / SMALL_ROWS
        # Within ±1 percentage point of 2%
        assert 0.01 <= shift_rate <= 0.03, f"Shift rate {shift_rate:.3f} out of [0.01, 0.03]"

        # Shifted values should be in the thousands (100 * 1000 = 100,000)
        max_val = corrupted.agg(F.max("feature_2")).collect()[0][0]
        assert max_val > 1000, f"Max feature_2 {max_val} should exceed 1000"


# ========================================================================
# 4. Duplicate injection
# ========================================================================
class TestDuplicateInjection:
    def test_duplicate_injection(self, spark, tmp_dir):
        """Output should have ~3% more rows, with duplicate IDs."""
        path = os.path.join(tmp_dir, "clean")
        generate_clean_data(spark, n_rows=SMALL_ROWS, seed=SEED, output_path=path, batch_id="dup_test")
        df = spark.read.parquet(path)

        corrupted, meta = inject_faults(spark, df, seed=SEED, null_rate=0.0,
                                        unit_shift_rate=0.0, duplicate_rate=0.03)

        output_count = corrupted.count()
        expected_dups = int(round(SMALL_ROWS * 0.03))
        assert output_count == SMALL_ROWS + expected_dups

        from pyspark.sql import functions as F
        dup_ids = corrupted.groupBy("id").count().filter(F.col("count") > 1).count()
        assert dup_ids > 0, "Should have at least some duplicate IDs"


# ========================================================================
# 5. Clean data passes all checks
# ========================================================================
class TestCleanValidation:
    def test_clean_data_passes_validation(self, spark, tmp_dir, quality_rules):
        """Clean data should pass ALL blocking checks."""
        path = os.path.join(tmp_dir, "clean")
        generate_clean_data(spark, n_rows=SMALL_ROWS, seed=SEED, output_path=path, batch_id="clean_val")
        df = spark.read.parquet(path)

        records = run_checks(spark, df, quality_rules, mode="collect_all")

        failed = [r for r in records if r.status == "FAILED"]
        assert len(failed) == 0, f"Clean data should pass all checks; failed: {[r.name for r in failed]}"


# ========================================================================
# 6. Corrupted data fails at least one check
# ========================================================================
class TestCorruptedValidation:
    def test_corrupted_data_fails_validation(self, spark, tmp_dir, quality_rules):
        """Corrupted data should fail at least one blocking check."""
        path = os.path.join(tmp_dir, "clean")
        generate_clean_data(spark, n_rows=SMALL_ROWS, seed=SEED, output_path=path, batch_id="corr_val")
        df = spark.read.parquet(path)

        corrupted, _ = inject_faults(spark, df, seed=SEED)

        records = run_checks(spark, corrupted, quality_rules, mode="collect_all")

        failed = [r for r in records if r.status == "FAILED"]
        assert len(failed) >= 1, "Corrupted data should fail at least one check"


# ========================================================================
# 7. All fault categories detected
# ========================================================================
class TestFaultCategoryDetection:
    def test_all_fault_categories_detected(self, spark, tmp_dir, quality_rules):
        """Validation should detect missing_values, duplicate_ids, and unit_scale_shift."""
        path = os.path.join(tmp_dir, "clean")
        generate_clean_data(spark, n_rows=SMALL_ROWS, seed=SEED, output_path=path, batch_id="cat_test")
        df = spark.read.parquet(path)

        corrupted, _ = inject_faults(spark, df, seed=SEED)
        records = run_checks(spark, corrupted, quality_rules, mode="collect_all")

        # Detect fault categories from failed check names
        categories = set()
        for r in records:
            if r.status != "PASSED":
                if "completeness" in r.name:
                    categories.add("missing_values")
                if "uniqueness" in r.name:
                    categories.add("duplicate_ids")
                if "scale_shift" in r.name or ("feature_2_max" in r.name):
                    categories.add("unit_scale_shift")

        assert "missing_values" in categories, "Should detect missing values"
        assert "duplicate_ids" in categories, "Should detect duplicate IDs"
        assert "unit_scale_shift" in categories, "Should detect unit scale shift"


# ========================================================================
# 8. Quarantine blocks inference
# ========================================================================
class TestQuarantineBlocksInference:
    def test_quarantine_blocks_inference(self, spark, tmp_dir):
        """block_inference() should raise RuntimeError for quarantined batch."""
        q_base = os.path.join(tmp_dir, "quarantine")
        batch_id = "blocked_batch"

        # Generate minimal data for quarantine
        path = os.path.join(tmp_dir, "clean")
        generate_clean_data(spark, n_rows=100, seed=SEED, output_path=path, batch_id=batch_id)
        df = spark.read.parquet(path)

        # Quarantine it
        quarantine_batch(
            spark, df, batch_id,
            validation_results={"passed": False, "fault_categories_detected": ["test"]},
            quarantine_base=q_base,
        )

        assert is_quarantined(batch_id, quarantine_base=q_base)

        with pytest.raises(RuntimeError, match="quarantined"):
            block_inference(batch_id, quarantine_base=q_base)


# ========================================================================
# 9. Subgroup B degrades more than A
# ========================================================================
class TestSubgroupDegradation:
    def test_subgroup_b_degrades_more(self, spark, tmp_dir):
        """Under corruption, subgroup B accuracy should drop more than A."""
        path_clean = os.path.join(tmp_dir, "clean")
        generate_clean_data(spark, n_rows=SMALL_ROWS, seed=SEED,
                            output_path=path_clean, batch_id="sg_test")
        df_clean_spark = spark.read.parquet(path_clean)
        corrupted, _ = inject_faults(spark, df_clean_spark, seed=SEED)

        df_clean = df_clean_spark.toPandas()
        df_corr = corrupted.toPandas()

        features = ["feature_1", "feature_2", "feature_3"]

        def eval_accuracy(df, fill_nulls=False):
            df = df.copy()
            if fill_nulls:
                for c in features:
                    df[c] = df[c].fillna(0)
            else:
                df = df.dropna(subset=features)
            X, y = df[features].values, df["target"].values
            sg = df["subgroup"].values
            X_tr, X_te, y_tr, y_te, _, sg_te = train_test_split(
                X, y, sg, test_size=0.3, random_state=SEED, stratify=y)
            model = LogisticRegression(max_iter=1000, random_state=SEED)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            acc_a = accuracy_score(y_te[sg_te == "A"], pred[sg_te == "A"])
            acc_b = accuracy_score(y_te[sg_te == "B"], pred[sg_te == "B"])
            return acc_a, acc_b

        clean_a, clean_b = eval_accuracy(df_clean)
        corr_a, corr_b = eval_accuracy(df_corr, fill_nulls=True)

        drop_a = clean_a - corr_a
        drop_b = clean_b - corr_b

        assert drop_b > drop_a, (
            f"Subgroup B drop ({drop_b:.4f}) should exceed A ({drop_a:.4f})"
        )


# ========================================================================
# 10. Integration workflow
# ========================================================================
class TestIntegrationWorkflow:
    def test_full_integration(self, spark, tmp_dir, quality_rules):
        """End-to-end: generate → inject → validate → quarantine → block."""
        clean_path = os.path.join(tmp_dir, "clean")
        corr_path = os.path.join(tmp_dir, "corrupted")
        q_base = os.path.join(tmp_dir, "quarantine")
        batch_id = "integration_batch"

        # Step 1: Generate clean data
        generate_clean_data(spark, n_rows=SMALL_ROWS, seed=SEED,
                            output_path=clean_path, batch_id=batch_id)
        df_clean = spark.read.parquet(clean_path)
        assert df_clean.count() == SMALL_ROWS

        # Step 2: Inject faults
        corrupted, meta = inject_faults(spark, df_clean, seed=SEED)
        corrupted.write.mode("overwrite").parquet(corr_path)
        df_corrupted = spark.read.parquet(corr_path)
        assert df_corrupted.count() > SMALL_ROWS  # duplicates added

        # Step 3: Validate with collect_all
        records = run_checks(spark, df_corrupted, quality_rules, mode="collect_all")
        passed = all(r.status == "PASSED" for r in records)
        assert not passed, "Corrupted data should not pass validation"

        # Step 4: Quarantine
        val_result = {
            "passed": False,
            "fault_categories_detected": ["missing_values", "duplicate_ids", "unit_scale_shift"],
        }
        q_path = quarantine_batch(
            spark, df_corrupted, batch_id, val_result,
            injection_metadata=meta, quarantine_base=q_base,
        )
        assert os.path.isdir(q_path)

        # Step 5: Inference blocked
        with pytest.raises(RuntimeError, match="quarantined"):
            block_inference(batch_id, quarantine_base=q_base)

        # Step 6: Verify clean passes
        clean_records = run_checks(spark, df_clean, quality_rules, mode="collect_all")
        clean_passed = all(r.status == "PASSED" for r in clean_records)
        assert clean_passed, f"Clean data should pass; failed: {[r.name for r in clean_records if r.status == 'FAILED']}"

