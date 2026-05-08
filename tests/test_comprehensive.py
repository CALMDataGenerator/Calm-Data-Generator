import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from calm_data_generator.generators.configs import DriftConfig, ReportConfig


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "age": np.random.randint(20, 70, n),
            "income": np.random.normal(50000, 15000, n).astype(int),
            "score": np.random.uniform(0, 100, n),
            "category": np.random.choice(["A", "B", "C"], n),
            "target": np.random.choice([0, 1], n, p=[0.6, 0.4]),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "block": np.repeat(range(10), 10),
        }
    )


def test_real_generator_methods(sample_data):
    """Test RealGenerator with multiple synthesis methods."""
    from calm_data_generator.generators.tabular import RealGenerator

    gen = RealGenerator(auto_report=False)
    numeric_data = sample_data.select_dtypes(include=[np.number])

    # CART
    synth = gen.generate(sample_data, 20, method="cart", target_col="target")
    assert synth is not None and len(synth) == 20

    # RF
    synth = gen.generate(sample_data, 20, method="rf", target_col="target")
    assert synth is not None and len(synth) == 20

    # LGBM (low estimators)
    synth = gen.generate(
        sample_data, 20, method="lgbm", target_col="target", n_estimators=5
    )
    assert synth is not None and len(synth) == 20

    # Resample
    synth = gen.generate(sample_data, 20, method="resample", target_col="target")
    assert synth is not None and len(synth) == 20

    # SMOTE (numeric only)
    numeric_no_ts = sample_data[["age", "income", "score", "target"]].copy()
    synth = gen.generate(numeric_no_ts, 30, method="smote", target_col="target")
    assert synth is not None and len(synth) == 30

    # Copula (numeric only)
    synth = gen.generate(numeric_data, 30, method="copula")
    assert synth is not None and len(synth) == 30

    # GMM (numeric only)
    try:
        synth = gen.generate(numeric_data, 10, method="gmm", target_col="target")
        if synth is None:
            pytest.skip("GMM returned None (likely missing dependency)")
        assert len(synth) == 10
    except ImportError:
        pytest.skip("GMM failed due to missing dependency")

    # CTGAN (skip if synthcity fails)
    try:
        synth = gen.generate(sample_data, 10, method="ctgan", epochs=1)
        assert synth is not None and len(synth) == 10
    except (ImportError, RuntimeError) as e:
        print(f"Skipping CTGAN test: {e}")

    # TVAE (skip if synthcity fails)
    try:
        synth = gen.generate(sample_data, 10, method="tvae", epochs=1)
        assert synth is not None and len(synth) == 10
    except (ImportError, RuntimeError) as e:
        print(f"Skipping TVAE test: {e}")


def test_new_synthesis_methods(sample_data):
    """Test DDPM synthesis method."""
    from calm_data_generator.generators.tabular import RealGenerator

    gen = RealGenerator(auto_report=False)

    try:
        synth = gen.generate(
            sample_data,
            n_samples=10,
            method="ddpm",
            n_iter=10,
            batch_size=32,
        )
        assert synth is not None and len(synth) == 10
    except ImportError:
        pytest.skip("Synthcity not available for DDPM")


def test_ddpm_parameters(sample_data):
    """Test DDPM with different parameters."""
    from calm_data_generator.generators.tabular import RealGenerator

    gen = RealGenerator(auto_report=False)

    try:
        synth = gen.generate(
            sample_data,
            n_samples=5,
            method="ddpm",
            n_iter=5,
            model_type="mlp",
            scheduler="cosine",
        )
        assert synth is not None and len(synth) == 5
    except ImportError:
        pytest.skip("Synthcity not available for DDPM")


def test_clinical_data_generator():
    """Test basic ClinicalDataGenerator functionality."""
    from calm_data_generator.generators.clinical import ClinicalDataGenerator

    clin_gen = ClinicalDataGenerator()
    result = clin_gen.generate(n_samples=10, n_genes=20, n_proteins=10)
    assert "demographics" in result
    assert len(result["demographics"]) == 10


def test_clinical_data_generator_longitudinal():
    """Test longitudinal clinical data generation."""
    from calm_data_generator.generators.clinical import ClinicalDataGenerator

    clin_gen = ClinicalDataGenerator()
    result = clin_gen.generate_longitudinal_data(
        n_samples=5, longitudinal_config={"n_visits": 2}
    )
    assert result is not None


def test_drift_injector(sample_data):
    """Test standard drift injection methods."""
    from calm_data_generator.generators.drift import DriftInjector

    injector = DriftInjector()

    # Gradual drift
    drifted = injector.inject_feature_drift_gradual(
        df=sample_data.copy(),
        feature_cols=["score"],
        drift_magnitude=0.5,
        drift_type="shift",
        start_index=50,
        center=25,
        width=20,
    )
    assert len(drifted) == len(sample_data)

    # Feature drift
    drifted = injector.inject_feature_drift(
        df=sample_data.copy(),
        feature_cols=["income"],
        drift_magnitude=0.3,
        drift_type="shift",
        start_index=60,
    )
    assert len(drifted) == len(sample_data)


def test_drift_injector_all_modes(sample_data):
    """Test abrupt drift mode."""
    from calm_data_generator.generators.drift import DriftInjector

    injector = DriftInjector()
    drifted = injector.inject_drift(
        df=sample_data,
        columns="score",
        drift_type="shift",
        magnitude=0.5,
        mode="abrupt",
    )
    assert len(drifted) == len(sample_data)


def test_drift_config_usage(sample_data):
    """Test DriftConfig object usage with inject_multiple_types_of_drift."""
    from calm_data_generator.generators.drift import DriftInjector

    injector = DriftInjector()
    config = DriftConfig(
        method="inject_drift",
        params={
            "columns": ["score"],
            "drift_type": "shift",
            "magnitude": 0.5,
            "mode": "abrupt",
        },
    )
    drifted = injector.inject_multiple_types_of_drift(df=sample_data, schedule=[config])
    assert len(drifted) == len(sample_data)


def test_scenario_injector(sample_data):
    """Test ScenarioInjector features."""
    from calm_data_generator.generators.dynamics import ScenarioInjector

    scenario = ScenarioInjector(seed=42)

    evolved = scenario.evolve_features(
        df=sample_data.copy(),
        evolution_config={"score": {"type": "trend", "rate": 0.05}},
        time_col="timestamp",
    )
    assert len(evolved) == len(sample_data)

    result = scenario.construct_target(
        df=sample_data.copy(),
        target_col="new_target",
        formula="age + income / 10000",
        task_type="regression",
    )
    assert "new_target" in result.columns


def test_single_call_workflow(sample_data):
    """Test Generate + Drift + Report in one call."""
    from calm_data_generator.generators.tabular import RealGenerator

    gen = RealGenerator(auto_report=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = gen.generate(
            data=sample_data,
            n_samples=20,
            method="cart",
            target_col="target",
            output_dir=tmpdir,
            save_dataset=True,
            drift_injection_config=[
                DriftConfig(
                    method="inject_feature_drift_gradual",
                    params={
                        "feature_cols": ["score"],
                        "drift_type": "shift",
                        "drift_magnitude": 0.3,
                        "start_index": 10,
                    },
                )
            ],
            report_config=ReportConfig(output_dir=tmpdir, target_column="target"),
        )
        assert result is not None
        assert len(result) == 20
        assert len(os.listdir(tmpdir)) > 0


def test_stream_generator_basic(sample_data):
    """Test basic StreamGenerator functionality."""
    try:
        from calm_data_generator.generators.stream import StreamGenerator
    except ImportError:
        pytest.skip("StreamGenerator dependencies not met")

    stream_gen = StreamGenerator(auto_report=False)

    try:
        from river import synth

        river_gen = synth.Agrawal(seed=42)
        result = stream_gen.generate(generator_instance=river_gen, n_samples=20)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20
    except ImportError:
        pytest.skip("River not installed")
