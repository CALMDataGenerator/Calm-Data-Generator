import numpy as np
import pandas as pd
import pytest

from calm_data_generator.generators.tabular import RealGenerator


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature1": np.random.randn(80),
            "feature2": np.random.randint(0, 10, 80),
            "target": np.random.choice([0, 1], 80),
        }
    )


def test_fit_returns_self_for_chaining(sample_data):
    gen = RealGenerator(auto_report=False, random_state=42)
    result = gen.fit(sample_data, method="cart", target_col="target")
    assert result is gen


def test_fit_then_sample_multiple_times_without_retraining(sample_data):
    gen = RealGenerator(auto_report=False, random_state=42)
    gen.fit(sample_data, method="cart", target_col="target")

    s1 = gen.sample(20)
    s2 = gen.sample(150)

    assert isinstance(s1, pd.DataFrame)
    assert isinstance(s2, pd.DataFrame)
    assert len(s1) == 20
    assert len(s2) == 150
    assert list(s1.columns) == list(sample_data.columns)


def test_chained_fit_sample(sample_data):
    result = RealGenerator(auto_report=False, random_state=1).fit(
        sample_data, method="cart"
    ).sample(15)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 15


def test_sample_without_fit_raises(sample_data):
    gen = RealGenerator(auto_report=False)
    with pytest.raises(ValueError, match="No fitted model found"):
        gen.sample(10)


def test_fit_does_not_write_files_even_with_default_auto_report(sample_data, tmp_path, monkeypatch):
    """fit() must not write report files to disk even when the generator was created with
    the default auto_report=True — fit() passes no output_dir, so nothing should be written
    to the current working directory."""
    monkeypatch.chdir(tmp_path)
    gen = RealGenerator(random_state=42)  # default auto_report=True
    gen.fit(sample_data, method="cart", target_col="target")
    assert list(tmp_path.iterdir()) == []


def test_generate_normal_usage_does_not_warn_about_legacy_params(sample_data, caplog):
    """Using generate() without any legacy alias should not log a legacy warning."""
    gen = RealGenerator(auto_report=False, random_state=1)
    with caplog.at_level("WARNING", logger="RealGenerator"):
        gen.generate(sample_data, n_samples=15, method="cart")
    assert not any("legacy" in r.message for r in caplog.records)


def test_generate_custom_distribution_singular_warns(sample_data, caplog):
    """The legacy singular `custom_distribution` alias should log a deprecation-style warning."""
    gen = RealGenerator(auto_report=False, random_state=1)
    with caplog.at_level("WARNING", logger="RealGenerator"):
        # {category: proportion} is the schema custom_distributions/custom_distribution
        # expect — proportions for the existing values of a discrete column, summing to 1.0.
        gen.generate(
            sample_data, n_samples=15, method="cart",
            custom_distribution={"target": {0: 0.5, 1: 0.5}},
        )
    assert any(
        "custom_distribution" in r.message and "legacy" in r.message
        for r in caplog.records
    )


def test_generate_date_start_legacy_warns(sample_data, caplog):
    """The legacy date_start/date_every/date_step shorthand should log a warning."""
    gen = RealGenerator(auto_report=False, random_state=1)
    with caplog.at_level("WARNING", logger="RealGenerator"):
        gen.generate(sample_data, n_samples=15, method="cart", date_start="2024-01-01")
    assert any(
        "date_start" in r.message and "legacy" in r.message for r in caplog.records
    )
