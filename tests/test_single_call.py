"""
CALM-Data-Generator - Single Call Test
=======================================

Test that demonstrates generating data, injecting drift, and generating
a report all in ONE method call using drift_injection_config parameter.
"""

import os
import tempfile

import numpy as np
import pandas as pd


def test_single_call_with_drift():
    """
    Test: Generate + Drift + Report in a SINGLE call to RealGenerator.generate()
    """

    print("=" * 60)
    print("CALM-Data-Generator - Single Call Test")
    print("Generation + Drift + Report in ONE call")
    print("=" * 60)

    # 1. Create sample data
    print("\n[1/3] Creating sample data...")
    np.random.seed(42)

    real_data = pd.DataFrame(
        {
            "age": np.random.randint(20, 70, 100),
            "income": np.random.normal(50000, 15000, 100).astype(int),
            "score": np.random.uniform(0, 100, 100),
            "target": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        }
    )

    print(f"   ✓ Real data shape: {real_data.shape}")

    # 2. Import and configure
    print("\n[2/3] Configuring single-call generation with drift...")

    from calm_data_generator.generators.configs import DriftConfig
    from calm_data_generator.generators.tabular import RealGenerator

    gen = RealGenerator()

    # CORRECT FORMAT: List of DriftConfig objects
    drift_config = [
        DriftConfig(
            method="inject_feature_drift_gradual",
            params={
                "feature_cols": ["score"],
                "drift_type": "shift",
                "drift_magnitude": 0.5,
                "start_index": 20,
                "center": 35,
                "width": 20,
            },
        )
    ]

    # Create temp output dir
    with tempfile.TemporaryDirectory() as tmpdir:
        # 3. SINGLE CALL: Generate + Drift + Report
        print("\n[3/3] Executing single-call generation with drift...")

        result = gen.generate(
            data=real_data,
            n_samples=50,
            method="cart",
            target_col="target",
            output_dir=tmpdir,
            save_dataset=True,
            drift_injection_config=drift_config,  # <-- Correct format!
            constraints=[
                {"col": "age", "op": ">=", "val": 18},
                {"col": "income", "op": ">", "val": 0},
            ],
        )

        assert result is not None, "generate() returned None"
        assert len(result) == 50
        assert "age" in result.columns
        assert "score" in result.columns
        assert len(os.listdir(tmpdir)) > 0


if __name__ == "__main__":
    test_single_call_with_drift()
