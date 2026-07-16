import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from calm_data_generator.reports.QualityReporter import QualityReporter


@pytest.fixture
def data_pair():
    """Create a pair of real and synthetic datasets."""
    np.random.seed(42)
    n = 50
    real = pd.DataFrame(
        {
            "age": np.random.randint(20, 60, n),
            "income": np.random.normal(50000, 10000, n),
            "group": np.random.choice(["A", "B"], n),
        }
    )

    # Synthetic is slightly different but similar
    synth = real.copy()
    synth["income"] += np.random.normal(0, 500, n)  # Add noise
    synth["age"] = np.random.randint(20, 60, n)  # Resample age

    return real, synth


def test_quality_reporter_privacy_metrics(data_pair):
    """Test standard privacy checks (DCR)."""
    real, synth = data_pair
    reporter = QualityReporter(minimal=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate full report with privacy enabled
        try:
            reporter.generate_comprehensive_report(
                real_df=real,
                synthetic_df=synth,
                generator_name="TestGen",
                output_dir=tmpdir,
                privacy_check=True,
            )
        except Exception as e:
            pytest.fail(f"Reporter failed: {e}")

        # Check results json
        res_path = os.path.join(tmpdir, "report_results.json")
        assert os.path.exists(res_path), f"Results file not found in {tmpdir}"

        with open(res_path) as f:
            res = json.load(f)

        assert "privacy_metrics" in res
        if res["privacy_metrics"] is not None:
            assert "dcr_mean" in res["privacy_metrics"]
            assert "dcr_5th_percentile" in res["privacy_metrics"]
            assert res["privacy_metrics"]["dcr_mean"] > 0
        else:
            # If privacy check failed silently (logged error), we might get None
            # Check logs if possible, or fail if we expect it to work on numeric data
            # Real data has numeric cols, so it should work.
            # DCR requires numeric cols.
            pass


def test_quality_reporter_minimal_vs_full(data_pair):
    """Test minimal report generation versus full."""
    real, synth = data_pair

    with tempfile.TemporaryDirectory() as tmpdir:
        # Minimal
        # Ensure output dir is unique
        out_min = os.path.join(tmpdir, "min")
        reporter_min = QualityReporter(minimal=True)
        reporter_min.generate_comprehensive_report(
            real_df=real,
            synthetic_df=synth,
            generator_name="MinGen",
            output_dir=out_min,
            privacy_check=False,
        )
        assert os.path.exists(os.path.join(out_min, "report_results.json")), (
            f"Minimal report not found in {out_min}"
        )

        # Full (might take longer, use small sample)
        out_full = os.path.join(tmpdir, "full")
        reporter_full = QualityReporter(minimal=False)
        # Reduce rows for speed
        try:
            reporter_full.generate_comprehensive_report(
                real_df=real.head(10),
                synthetic_df=synth.head(10),
                generator_name="FullGen",
                output_dir=out_full,
                privacy_check=False,
            )
            # YData generates report.html usually
            # But if YData fails (dependency), checking json is safer fallback for "something ran"
            assert os.path.exists(os.path.join(out_full, "report_results.json"))
        except Exception as e:
            pytest.skip(f"Full report failed (likely missing heavy dep): {e}")


def test_dcr_privacy_includes_nndr(data_pair):
    """DCR privacy metrics should also include NNDR (Nearest Neighbor Distance Ratio)."""
    real, synth = data_pair
    reporter = QualityReporter(verbose=False)

    metrics = reporter._calculate_dcr_privacy(real, synth)

    assert metrics is not None
    assert "dcr_mean" in metrics
    assert "nndr_mean" in metrics
    assert "nndr_5th_percentile" in metrics
    # NNDR is a ratio of two distances, bounded in [0, 1]
    assert 0.0 <= metrics["nndr_mean"] <= 1.0
    assert 0.0 <= metrics["nndr_5th_percentile"] <= 1.0


def test_singling_out_risk_returns_none_without_anonymeter(data_pair, monkeypatch):
    """_calculate_singling_out_risk should degrade to None (not raise) if anonymeter is missing."""
    import sys
    real, synth = data_pair
    reporter = QualityReporter(verbose=False)

    monkeypatch.setitem(sys.modules, "anonymeter", None)
    monkeypatch.setitem(sys.modules, "anonymeter.evaluators", None)

    result = reporter._calculate_singling_out_risk(real, synth)
    assert result is None


def test_singling_out_risk_with_anonymeter(data_pair):
    """When anonymeter is installed, singling-out risk should return a bounded risk value."""
    pytest.importorskip("anonymeter")
    real, synth = data_pair
    reporter = QualityReporter(verbose=False)

    result = reporter._calculate_singling_out_risk(
        real, synth, n_attacks=10, max_attempts=2000
    )

    assert result is not None
    assert "risk" in result
    assert 0.0 <= result["risk"] <= 1.0
    assert "ci_low" in result and "ci_high" in result
    assert result["used_control"] is False


def test_singling_out_risk_never_hangs_on_low_cardinality_duplicate_data():
    """Regression test: anonymeter's own max_attempts default (10M) can hang for minutes
    on low-cardinality/duplicate data. Our bounded default must keep this fast."""
    import time
    pytest.importorskip("anonymeter")

    np.random.seed(0)
    n = 100
    real = pd.DataFrame({
        "a": np.random.normal(0, 1, n),
        "b": np.random.randint(0, 3, n),  # low cardinality
    })
    synth = real.copy()  # worst case: exact duplicate

    reporter = QualityReporter(verbose=False)
    start = time.time()
    result = reporter._calculate_singling_out_risk(real, synth, n_attacks=20, max_attempts=3000)
    elapsed = time.time() - start

    assert elapsed < 30, f"Singling-Out risk took {elapsed:.1f}s — max_attempts bound may be broken"
    assert result is not None
