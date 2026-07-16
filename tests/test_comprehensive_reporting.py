import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris


@pytest.fixture
def real_df():
    """Load Iris dataset for testing."""
    iris = load_iris()
    # Use standard feature names as they appear in the dataset
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df


@pytest.fixture
def output_base():
    """Create a temporary output base directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_real_generator_drift_reporting(real_df, output_base):
    """Test RealGenerator with drift_injection_config report generation."""
    from calm_data_generator.generators.configs import DriftConfig
    from calm_data_generator.generators.tabular.RealGenerator import RealGenerator

    output_dir = os.path.join(output_base, "1_real_generator_drift")
    gen = RealGenerator(auto_report=True)

    column_to_drift = real_df.columns[0]

    synthetic_df = gen.generate(
        data=real_df,
        n_samples=50,
        method="cart",
        output_dir=output_dir,
        drift_injection_config=[
            DriftConfig(
                method="inject_drift",
                params={
                    "columns": [column_to_drift],
                    "drift_type": "shift",
                    "magnitude": 0.3,
                    "mode": "abrupt",
                },
            )
        ],
    )

    assert synthetic_df is not None
    assert os.path.exists(os.path.join(output_dir, "index.html"))
    files = os.listdir(output_dir)
    assert "drift_stats.html" in files
    assert "plot_comparison.html" in files
    assert "report_results.json" in files


def test_drift_injector_standalone_reporting(real_df, output_base):
    """Test DriftInjector standalone report generation."""
    from calm_data_generator.generators.drift.DriftInjector import DriftInjector
    from calm_data_generator.reports.QualityReporter import QualityReporter

    output_dir = os.path.join(output_base, "2_drift_injector")
    os.makedirs(output_dir, exist_ok=True)

    # Generate synthetic-like data
    synthetic_base = real_df.copy()
    for col in synthetic_base.select_dtypes(include=[np.number]).columns:
        if col != "target":
            synthetic_base[col] += np.random.randn(len(synthetic_base)) * 0.1

    injector = DriftInjector()
    column_to_drift = real_df.columns[0]

    drifted_df = injector.inject_drift(
        df=synthetic_base,
        columns=column_to_drift,
        drift_type="shift",
        magnitude=0.5,
        auto_report=False,  # the QualityReporter call below is the report under test
    )

    reporter = QualityReporter()
    # FIXED: reporters return None. Check the output_dir instead.
    reporter.generate_report(
        real_df=real_df,
        synthetic_df=drifted_df,
        generator_name="DriftTest",
        output_dir=output_dir,
        drift_config={
            "columns": [column_to_drift],
            "method": "manual",
            "type": "shift",
        },
    )

    assert os.path.exists(os.path.join(output_dir, "index.html"))
    assert "drift_stats.html" in os.listdir(output_dir)


def test_scenario_injector_evolution_report(real_df, output_base):
    """Test ScenarioInjector evolution report generation."""
    from calm_data_generator.generators.dynamics.ScenarioInjector import (
        ScenarioInjector,
    )
    from calm_data_generator.reports.QualityReporter import QualityReporter

    output_dir = os.path.join(output_base, "3_scenario_injector")
    os.makedirs(output_dir, exist_ok=True)

    # Add timestamp
    real_df_ts = real_df.copy()
    real_df_ts["timestamp"] = pd.date_range(
        "2024-01-01", periods=len(real_df_ts), freq="h"
    )

    scenario = ScenarioInjector()
    column_to_evolve = real_df.columns[0]

    evolved_df = scenario.evolve_features(
        df=real_df_ts,
        evolution_config={column_to_evolve: {"type": "trend", "rate": 0.01}},
        time_col="timestamp",
    )

    # Generate report via QualityReporter
    reporter = QualityReporter()
    # FIXED: reporters return None. Check the output_dir instead.
    reporter.generate_report(
        real_df=real_df_ts,
        synthetic_df=evolved_df,
        generator_name="ScenarioTest",
        output_dir=output_dir,
        drift_config={
            "columns": [column_to_evolve],
            "method": "trend",
            "type": "evolution",
        },
    )

    assert os.path.exists(os.path.join(output_dir, "index.html"))
    assert "drift_stats.html" in os.listdir(output_dir)
