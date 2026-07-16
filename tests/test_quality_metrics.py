import numpy as np
import pandas as pd
import pytest

from calm_data_generator.generators.configs import ReportConfig
from calm_data_generator.reports.QualityReporter import QualityReporter


@pytest.fixture
def sample_data():
    """Create sample real and synthetic dataframes."""
    np.random.seed(42)

    # Real data
    real_data = {
        "age": np.random.randint(20, 60, 100),
        "salary": np.random.normal(50000, 10000, 100),
        "department": np.random.choice(["A", "B", "C"], 100),
    }
    real_df = pd.DataFrame(real_data)

    # Synthetic data (slightly different distribution)
    synth_data = {
        "age": np.random.randint(20, 60, 100),
        "salary": np.random.normal(52000, 11000, 100),
        "department": np.random.choice(["A", "B", "C"], 100),
    }
    synthetic_df = pd.DataFrame(synth_data)

    return real_df, synthetic_df


def test_calculate_quality_metrics(sample_data):
    """Test the calculate_quality_metrics method returns expected keys and types."""
    real_df, synthetic_df = sample_data

    reporter = QualityReporter(verbose=False)
    metrics = reporter.calculate_quality_metrics(real_df, synthetic_df)

    # Check if metrics are returned
    assert isinstance(metrics, dict)

    # Check if we got an error (e.g. if sdmetrics not installed) or actual metrics
    if "error" in metrics:
        pytest.skip(f"SDMetrics not available or failed: {metrics['error']}")

    # Check for expected keys
    assert "overall_quality_score" in metrics
    assert "weighted_quality_score" in metrics

    # Check types
    assert isinstance(metrics["overall_quality_score"], (float, int))
    assert isinstance(metrics["weighted_quality_score"], (float, int))

    # Check range (scores should be between 0 and 1)
    assert 0 <= metrics["overall_quality_score"] <= 1
    assert 0 <= metrics["weighted_quality_score"] <= 1

    print(f"\nQuality Metrics Retrieved: {metrics}")


def test_calculate_quality_metrics_empty(sample_data):
    """Test behavior with empty dataframe."""
    real_df, _ = sample_data
    empty_synth = pd.DataFrame(columns=real_df.columns)

    reporter = QualityReporter(verbose=False)
    metrics = reporter.calculate_quality_metrics(real_df, empty_synth)

    # Existing logic for empty synth usually results in low score or error handling works
    # We mainly want to ensure no crash
    assert isinstance(metrics, dict)


def test_evaluate_returns_in_memory_metrics(sample_data, tmp_path):
    """evaluate() should compute the same kind of metrics as the full report, in memory only."""
    real_df, synthetic_df = sample_data

    reporter = QualityReporter(verbose=False)
    result = reporter.evaluate(real_df, synthetic_df)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"quality_scores", "statistical_metrics", "tstr_metrics"}

    # No target_column given -> TSTR is not run
    assert result["tstr_metrics"] is None

    # Statistical tests should include Wasserstein distance per numeric column
    stats = result["statistical_metrics"]
    assert stats is not None
    assert "mmd" in stats
    assert "age" in stats["per_column"]
    assert "wasserstein_distance" in stats["per_column"]["age"]
    assert "ks_statistic" in stats["per_column"]["age"]

    # evaluate() must not write anything to disk
    assert list(tmp_path.iterdir()) == []


def test_evaluate_with_target_column_runs_tstr(sample_data):
    """Passing target_column should trigger TSTR without writing an HTML report."""
    real_df, synthetic_df = sample_data
    real_df = real_df.copy()
    synthetic_df = synthetic_df.copy()
    real_df["target"] = np.random.choice([0, 1], len(real_df))
    synthetic_df["target"] = np.random.choice([0, 1], len(synthetic_df))

    reporter = QualityReporter(verbose=False)
    result = reporter.evaluate(real_df, synthetic_df, target_column="target")

    assert result["tstr_metrics"] is not None
    assert result["tstr_metrics"]["task"] == "classification"
    assert "roc_auc" in result["tstr_metrics"]


def test_generate_comprehensive_report_returns_results_dict(sample_data, tmp_path):
    """generate_comprehensive_report should return the results dict, not None."""
    real_df, synthetic_df = sample_data

    reporter = QualityReporter(verbose=False)
    result = reporter.generate_comprehensive_report(
        real_df, synthetic_df, "TestGen", str(tmp_path), minimal=True
    )

    assert isinstance(result, dict)
    assert result["generator_name"] == "TestGen"
    assert "statistical_metrics" in result
    assert (tmp_path / "report_results.json").exists()

    # Regression test: statistical_tests.html used to load Plotly from a CDN
    # (include_plotlyjs="cdn") and would render blank without internet access.
    # It must now be fully self-contained.
    stats_html = tmp_path / "statistical_tests.html"
    assert stats_html.exists()
    html = stats_html.read_text(encoding="utf-8")
    assert '<script src="https://cdn' not in html
    assert "Plotly.newPlot" in html  # inline bundle actually present


def _make_target_data():
    """Real/synthetic frames with a binary target, for TSTR-triggering tests."""
    np.random.seed(7)
    real_df = pd.DataFrame({
        "a": np.random.normal(0, 1, 60),
        "target": np.random.choice([0, 1], 60),
    })
    synthetic_df = real_df.copy()
    return real_df, synthetic_df


def test_report_config_explicit_arg_overrides_false_default(tmp_path):
    """An explicit tstr=True arg must win even if report_config.tstr is left at its False default."""
    real_df, synthetic_df = _make_target_data()
    reporter = QualityReporter(verbose=False)

    cfg = ReportConfig(output_dir=str(tmp_path), target_column="target")
    result = reporter.generate_comprehensive_report(
        real_df, synthetic_df, "TestGen", str(tmp_path),
        report_config=cfg, tstr=True, target_column="target", minimal=True,
    )

    assert result["tstr_metrics"] is not None


def test_report_config_field_alone_triggers_tstr(tmp_path):
    """tstr=True set directly on ReportConfig must run TSTR even with no explicit method arg."""
    real_df, synthetic_df = _make_target_data()
    reporter = QualityReporter(verbose=False)

    cfg = ReportConfig(output_dir=str(tmp_path), target_column="target", tstr=True)
    result = reporter.generate_comprehensive_report(
        real_df, synthetic_df, "TestGen", str(tmp_path), report_config=cfg, minimal=True,
    )

    assert result["tstr_metrics"] is not None


def test_report_config_defaults_no_tstr(tmp_path):
    """Neither report_config.tstr nor the tstr arg set -> TSTR should not run."""
    real_df, synthetic_df = _make_target_data()
    reporter = QualityReporter(verbose=False)

    cfg = ReportConfig(output_dir=str(tmp_path), target_column="target")
    result = reporter.generate_comprehensive_report(
        real_df, synthetic_df, "TestGen", str(tmp_path), report_config=cfg, minimal=True,
    )

    assert result["tstr_metrics"] is None


def test_report_config_explicit_privacy_check_overrides_false_default(tmp_path):
    """Same OR-merge fix as tstr, applied to privacy_check: an explicit arg must win
    even when report_config.privacy_check is left at its False default."""
    real_df, synthetic_df = _make_target_data()
    reporter = QualityReporter(verbose=False)

    cfg = ReportConfig(output_dir=str(tmp_path), target_column="target")
    result = reporter.generate_comprehensive_report(
        real_df, synthetic_df, "TestGen", str(tmp_path),
        report_config=cfg, privacy_check=True, minimal=True,
    )

    assert result["privacy_metrics"] is not None
    assert "dcr_mean" in result["privacy_metrics"]
