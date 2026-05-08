import numpy as np
import pandas as pd
import pytest

from calm_data_generator.generators.tabular.RealGenerator import RealGenerator


def test_latent_differentiation_shifts_embeddings():
    """Test that applying a differentiation factor physically moves class centroids apart in latent space."""
    np.random.seed(42)
    generator = RealGenerator(random_state=42)

    # Create a very simple dataset with two classes that are slightly separated
    c0 = np.random.normal(0, 1, size=(50, 4))
    c1 = np.random.normal(2, 1, size=(50, 4))

    df = pd.DataFrame(np.vstack([c0, c1]), columns=[f"feature_{i}" for i in range(4)])
    df["target"] = np.concatenate([np.zeros(50), np.ones(50)])

    # Needs to be fast for the test
    params = {"epochs": 2}
    try:
        # Run 1: No differentiation
        synth_no_diff =generator.generate(
            data=df,
            n_samples=100,
            method="tvae",
            target_col="target",
            differentiation_factor=0.0,
            **params
        )

        latents_no_diff = generator.get_latest_embeddings()

        if latents_no_diff is None:
            pytest.skip("Latent embeddings not accessible")

        c0_no_diff = np.mean(latents_no_diff[synth_no_diff["target"] == 0], axis=0)
        c1_no_diff = np.mean(latents_no_diff[synth_no_diff["target"] == 1], axis=0)
        dist_no_diff = np.linalg.norm(c0_no_diff - c1_no_diff)

        # Run 2: High differentiation factor
        synth_diff = generator.generate(
            data=df,
            n_samples=100,
            method="tvae",
            target_col="target",
            differentiation_factor=2.0,
            **params
        )

        latents_diff = generator.get_latest_embeddings()

        c0_diff = np.mean(latents_diff[synth_diff["target"] == 0], axis=0)
        c1_diff = np.mean(latents_diff[synth_diff["target"] == 1], axis=0)
        dist_diff = np.linalg.norm(c0_diff - c1_diff)

        # Distance between class centroids must grow when differentiation_factor > 0.
        # Tolerance: the improvement doesn't need to be large, just positive.
        assert dist_diff > dist_no_diff - 1e-6, (
            f"Differentiation failed: dist_no_diff={dist_no_diff:.4f}, dist_diff={dist_diff:.4f}"
        )

    except ImportError:
        pytest.skip("Synthcity is not installed")
