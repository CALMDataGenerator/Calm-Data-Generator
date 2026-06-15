import numpy as np
import pandas as pd
import pytest

from calm_data_generator.generators.configs import DriftConfig, ReportConfig
from calm_data_generator.generators.tabular.RealBlockGenerator import RealBlockGenerator


def _make_data():
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, n_samples),
            "feature2": np.random.choice(["A", "B"], n_samples),
            "block_col": np.repeat(["Block1", "Block2"], n_samples // 2),
            "target": np.random.randint(0, 2, n_samples),
        }
    )


def test_initialization():
    gen = RealBlockGenerator()
    assert isinstance(gen, RealBlockGenerator)


def test_generate_with_existing_block_column(tmp_path):
    data = _make_data()
    gen = RealBlockGenerator(auto_report=False)
    synthetic_data = gen.generate(
        data=data,
        output_dir=str(tmp_path),
        method="cart",
        block_column="block_col",
        target_col="target",
    )

    assert isinstance(synthetic_data, pd.DataFrame)
    assert "block_col" in synthetic_data.columns
    assert set(synthetic_data["block_col"].unique()) == {"Block1", "Block2"}
    assert not synthetic_data.empty


def test_generate_with_chunk_size(tmp_path):
    data = _make_data()
    gen = RealBlockGenerator(auto_report=False)
    data_no_block = data.drop(columns=["block_col"])

    synthetic_data = gen.generate(
        data=data_no_block,
        output_dir=str(tmp_path),
        method="cart",
        chunk_size=20,
        target_col="target",
    )

    assert "chunk" in synthetic_data.columns
    assert len(synthetic_data["chunk"].unique()) == 5


def test_generate_with_n_samples_block_dict(tmp_path):
    data = _make_data()
    gen = RealBlockGenerator(auto_report=False)

    synthetic_data = gen.generate(
        data=data,
        output_dir=str(tmp_path),
        method="cart",
        block_column="block_col",
        n_samples_block={"Block1": 10, "Block2": 20},
        target_col="target",
    )

    counts = synthetic_data["block_col"].value_counts()
    assert counts["Block1"] == 10
    assert counts["Block2"] == 20


def test_generate_with_config_objects(tmp_path):
    data = _make_data()
    gen = RealBlockGenerator(auto_report=False)

    drift_conf = DriftConfig(
        method="inject_feature_drift",
        params={"feature_cols": ["feature1"], "drift_magnitude": 0.5},
    )
    report_conf = ReportConfig(output_dir=str(tmp_path), target_column="target")

    synthetic_data = gen.generate(
        data=data,
        output_dir=str(tmp_path),
        method="cart",
        block_column="block_col",
        drift_config=[drift_conf],
        report_config=report_conf,
        target_col="target",
    )

    assert not synthetic_data.empty
