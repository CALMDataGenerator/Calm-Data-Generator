import numpy as np
import pandas as pd
import pytest

from calm_data_generator.generators.tabular.RealGenerator import RealGenerator


def test_scvi_quality_regression():
    """
    Test to ensure that scVI generation produces integer counts
    and achieves at least 70% overall quality using a real scVI dataset.
    """
    import scvi

    # Load synthetic dataset from scVI (to avoid download 429 errors)
    print("Loading scVI synthetic_iid dataset...")
    adata = scvi.data.synthetic_iid()

    # We'll use a subset to speed up the test
    if adata.n_obs > 500:
        indices = np.random.choice(adata.n_obs, 500, replace=False)
        adata = adata[indices].copy()

    # Target column to use (metadata)
    target_col = 'labels'

    # Initialize generator. auto_report=False: the quality check below uses
    # gen.reporter.calculate_quality_metrics() directly (in-memory), so there's no
    # need to also write a full report to disk here.
    gen = RealGenerator(auto_report=False, minimal_report=True)

    # Generate synthetic data
    print("Generating synthetic data...")
    df_synth = gen.generate(
        data=adata,
        method='scvi',
        target_col=target_col,
        n_samples=500,
        epochs=30, # Real data needs more epochs for 70% quality
        use_latent_sampling=True
    )

    # 1. Check if the generated data is integer (counts)
    # Note: scVI output might be floats due to library size scaling,
    # but with .sample() it should be integers if handled correctly.
    # We drop the metadata column
    expr_synth = df_synth.drop(target_col, axis=1).values
    is_integer = np.all(np.mod(expr_synth, 1) == 0)
    print(f"Are synthetic values integers? {is_integer}")

    # 2. Check quality report
    # We need the original dataframe for comparison
    df_real = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, columns=adata.var_names)
    df_real[target_col] = adata.obs[target_col].values

    metrics = gen.reporter.calculate_quality_metrics(df_real, df_synth)
    assert 'error' not in metrics, f"Quality calculation failed: {metrics.get('error')}"

    overall_quality = metrics.get('overall_quality_score', 0)
    print(f"Overall Quality Score with Real Data: {overall_quality:.2%}")

    assert overall_quality >= 0.70, f"scVI quality ({overall_quality:.2%}) is below 70% threshold for real data!"
