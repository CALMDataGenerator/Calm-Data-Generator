import numpy as np
import pandas as pd
import pytest

from calm_data_generator.generators.drift import DriftInjector


@pytest.fixture
def base_df():
    return pd.DataFrame(
        {
            "A": [10.0, 20.0, 30.0, 40.0, 50.0],
            "B": [1.0, 1.0, 1.0, 1.0, 1.0],
            "C": [100, 200, 300, 400, 500],
        }
    )


def test_drift_math_operations(base_df):
    """Test standard mathematical drift types."""
    injector = DriftInjector(auto_report=False)

    # Add Value (add 5 to all rows)
    res_add = injector.inject_feature_drift(
        df=base_df, feature_cols=["A"], drift_type="add_value", drift_value=5.0
    )
    # Drift applied with default sigmoid, so center transition.
    # To test pure math, let's use rows selection that covers all or check max impact
    # Default is gradual over entire range probably if not specified?
    # No, default is just 'feature_drift' without gradual params,
    # but wait, inject_feature_drift uses _get_target_rows which defaults to ALL rows if no time/block specified.
    # And then uses w=ones! So it should be full impact.
    assert np.allclose(res_add["A"], base_df["A"] + 5.0)

    # Multiply Value
    res_mul = injector.inject_feature_drift(
        df=base_df, feature_cols=["B"], drift_type="multiply_value", drift_value=2.0
    )
    assert np.allclose(res_mul["B"], base_df["B"] * 2.0)

    # Divide
    res_div = injector.inject_feature_drift(
        df=base_df, feature_cols=["C"], drift_type="divide_value", drift_value=10.0
    )
    assert np.allclose(res_div["C"], base_df["C"] / 10.0)


def test_drift_math_validation(base_df):
    """Test error conditions."""
    injector = DriftInjector(auto_report=False)

    # Divide by zero
    with pytest.raises(ValueError, match="drift_value cannot be zero"):
        injector.inject_feature_drift(
            df=base_df, feature_cols=["A"], drift_type="divide_value", drift_value=0.0
        )

    # Magnitude < 0 for noise
    with pytest.raises(ValueError, match="magnitude"):
        injector.inject_feature_drift(
            df=base_df,
            feature_cols=["A"],
            drift_type="gaussian_noise",
            drift_magnitude=-1.0,
        )


def test_drift_profiles(base_df):
    """Test window functions generation."""
    injector = DriftInjector(auto_report=False)
    rows = base_df.index

    # Linear profile
    w_linear = injector._calculate_drift_probabilities(
        rows=rows, center=2, width=4, profile="linear", speed_k=1.0
    )
    # Check shape
    assert len(w_linear) == 5
    # Center should be around 0.5?
    # The logic maps index to weight.
    # Center index 2 -> weight ~0.5
    assert 0.4 < w_linear[2] < 0.6

    # Cosine profile
    w_cosine = injector._calculate_drift_probabilities(
        rows=rows, center=2, width=4, profile="cosine"
    )
    assert len(w_cosine) == 5
