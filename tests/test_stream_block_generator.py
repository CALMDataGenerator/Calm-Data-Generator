import pandas as pd
import pytest

try:
    from river import synth

    RIVER_AVAILABLE = True
except ImportError:
    try:
        from river.datasets import synth

        RIVER_AVAILABLE = True
    except ImportError:
        RIVER_AVAILABLE = False
        synth = None

try:
    from calm_data_generator.generators.stream.StreamBlockGenerator import (
        SyntheticBlockGenerator,
    )
except ImportError:
    SyntheticBlockGenerator = None

from calm_data_generator.generators.configs import DriftConfig, ReportConfig

pytestmark = pytest.mark.skipif(not RIVER_AVAILABLE, reason="River not installed")


def test_initialization():
    gen = SyntheticBlockGenerator()
    assert isinstance(gen, SyntheticBlockGenerator)


def test_generate_simple_interface(tmp_path):
    gen = SyntheticBlockGenerator()
    path = gen.generate_blocks_simple(
        output_dir=str(tmp_path),
        filename="test_stream_blocks.csv",
        n_blocks=2,
        total_samples=100,
        methods=["sea"],
        method_params=[{"seed": 42}, {"seed": 43}],
        generate_report=False,
    )

    assert path.endswith(".csv")
    df = pd.read_csv(path)
    assert len(df) == 100
    assert "block" in df.columns
    assert set(df["block"].unique()) == {1, 2}


def test_generate_manual_interface(tmp_path):
    gen = SyntheticBlockGenerator()

    gen1 = synth.Agrawal(seed=42, classification_function=0)
    gen2 = synth.Agrawal(seed=42, classification_function=1)

    path = gen.generate(
        output_dir=str(tmp_path),
        filename="test_stream_blocks.csv",
        n_blocks=2,
        total_samples=50,
        n_samples_block=[25, 25],
        generators=[gen1, gen2],
        generate_report=False,
    )

    df = pd.read_csv(path)
    assert len(df) == 50
    assert df["block"].value_counts()[1] == 25
    assert df["block"].value_counts()[2] == 25


def test_generate_with_config_objects(tmp_path):
    gen = SyntheticBlockGenerator()

    drift_conf = DriftConfig(
        method="inject_feature_drift",
        params={
            "feature_cols": ["feature1"],
            "drift_magnitude": 0.5,
            "drift_type": "shift",
        },
    )
    report_conf = ReportConfig(output_dir=str(tmp_path))

    path = gen.generate_blocks_simple(
        output_dir=str(tmp_path),
        filename="test_config_blocks.csv",
        n_blocks=2,
        total_samples=20,
        methods=["sea"],
        drift_config=[drift_conf],
        report_config=report_conf,
        generate_report=False,
    )

    assert path.endswith(".csv")
