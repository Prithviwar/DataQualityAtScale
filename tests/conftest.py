"""
tests/conftest.py
-----------------
Shared pytest fixtures for the data-quality pipeline test suite.
"""

import os
import sys
import shutil
import tempfile

import pytest

# Must set SPARK_VERSION before any pydeequ import
os.environ["SPARK_VERSION"] = "4.1"

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_generation import _build_spark  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    """Session-scoped SparkSession for all tests."""
    session = _build_spark("PipelineTests")
    yield session
    try:
        gw = session.sparkContext._gateway
        cb = gw.get_callback_server()
        if cb:
            cb.shutdown()
    except Exception:
        pass
    session.stop()


# ---------------------------------------------------------------------------
# Prevent PySpark / PyDeequ background threads from hanging pytest on Windows
# ---------------------------------------------------------------------------
_pytest_exit_code = 0


def pytest_sessionfinish(session, exitstatus):
    global _pytest_exit_code
    _pytest_exit_code = exitstatus


def pytest_unconfigure(config):
    """Exit immediately after pytest finishes printing so lingering Py4J threads don't block terminal."""
    global _pytest_exit_code
    import os
    os._exit(_pytest_exit_code)


@pytest.fixture
def tmp_dir():
    """Per-test temporary directory, cleaned up after test."""
    d = tempfile.mkdtemp(prefix="dq_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def quality_rules():
    """Load the project quality rules."""
    import yaml
    rules_path = os.path.join(os.path.dirname(__file__), "..", "config", "quality_rules.yml")
    with open(rules_path) as fh:
        return yaml.safe_load(fh)

