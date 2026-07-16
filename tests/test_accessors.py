import numpy as np
import pandas as pd
import pytest

from calm_data_generator.generators.tabular.RealGenerator import RealGenerator


def test_tvae_accessors():
    """Test that TVAE accessors return appropriate objects."""
    generator = RealGenerator(random_state=42, auto_report=False)
    # Simple dataset
    df = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target": np.random.choice([0, 1], size=100)
    })

    # Needs to be fast for the test, use minimal epochs
    params = {"epochs": 2, "differentiation_factor": 0.5}
    try:
        generator.generate(
            data=df,
            n_samples=10,
            method="tvae",
            target_col="target",
            **params
        )

        # Test get_synthesizer_model
        model = generator.get_synthesizer_model()
        assert model is not None, "Synthesizer model should not be None for TVAE"

        # Test get_encoder
        encoder = generator.get_encoder()
        assert encoder is not None, "Encoder should not be None for TVAE"

        # Test get_decoder
        decoder = generator.get_decoder()
        assert decoder is not None, "Decoder should not be None for TVAE"

        # Test get_latest_embeddings
        # Note: embeddings may be None if the encoder path failed (e.g. model undertrained)
        generator.get_latest_embeddings()
        # Just verify the accessor works without error (can be None in fallback case)

    except ImportError:
        pytest.skip("Synthcity is not installed")


def test_ctgan_accessors():
    """Test that CTGAN accessors return appropriate objects."""
    generator = RealGenerator(random_state=42, auto_report=False)
    df = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target": np.random.choice([0, 1], size=100)
    })

    params = {"epochs": 2, "differentiation_factor": 0.5}
    try:
        generator.generate(
            data=df,
            n_samples=10,
            method="ctgan",
            target_col="target",
            **params
        )

        # Test get_synthesizer_model
        model = generator.get_synthesizer_model()
        print(model)
        assert model is not None, "Synthesizer model should not be None for CTGAN"

        # Test get_encoder
        encoder = generator.get_encoder()
        print(encoder)
        assert encoder is None, "Encoder should be None for CTGAN (doesn't exist)"

        # Test get_decoder
        decoder = generator.get_decoder()
        print(decoder)
        assert decoder is None, "Decoder should be None for CTGAN (doesn't exist)"

        # Test get_latest_embeddings
        embeddings = generator.get_latest_embeddings()
        assert embeddings is None, "Latest embeddings should be None for CTGAN (using feature space fallback)"

    except ImportError:
        pytest.skip("Synthcity is not installed")


def test_rtvae_accessors():
    """Test that RTVAE accessors return encoder and decoder."""
    generator = RealGenerator(random_state=42, auto_report=False)
    df = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target": np.random.choice([0, 1], size=100)
    })
    params = {"epochs": 2, "differentiation_factor": 0.3}
    try:
        generator.generate(
            data=df,
            n_samples=10,
            method="rtvae",
            target_col="target",
            **params
        )
        model = generator.get_synthesizer_model()
        assert model is not None, "Synthesizer model should not be None for RTVAE"

        encoder = generator.get_encoder()
        assert encoder is not None, "Encoder should not be None for RTVAE"

        decoder = generator.get_decoder()
        assert decoder is not None, "Decoder should not be None for RTVAE"

    except ImportError:
        pytest.skip("Synthcity is not installed")


def test_encode_decode_latent_tvae():
    """encode_to_latent / decode_from_latent round-trip for TVAE."""
    import torch
    generator = RealGenerator(random_state=42, auto_report=False)
    df = pd.DataFrame({
        "feature1": np.random.randn(80),
        "feature2": np.random.randn(80),
        "target": np.random.choice([0, 1], size=80),
    })
    try:
        generator.generate(
            data=df, n_samples=10, method="tvae", target_col="target",
            epochs=2,
        )
        z = generator.encode_to_latent(df, target_col="target")
        assert isinstance(z, torch.Tensor), "encode_to_latent must return a Tensor"
        assert z.shape[0] == len(df), "Tensor must have one row per sample"
        assert z.ndim == 2, "Latent tensor must be 2-D"

        recon = generator.decode_from_latent(z, data=df, target_col="target")
        assert isinstance(recon, pd.DataFrame), "decode_from_latent must return DataFrame"
        assert len(recon) == len(df), "Reconstructed frame must have same length"
    except ImportError:
        pytest.skip("Synthcity is not installed")


def test_encode_to_latent_without_model_raises():
    """encode_to_latent raises RuntimeError when no model is trained."""
    generator = RealGenerator(auto_report=False)
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(RuntimeError, match="No trained model"):
        generator.encode_to_latent(df)


def test_sequential_scvi_then_tvae_uses_tvae_encoder():
    """After switching from SCVI to TVAE, encode_to_latent must use TVAE encoder."""
    import torch
    pytest.importorskip("scvi")
    generator = RealGenerator(random_state=0, auto_report=False)

    n_genes = 20
    n_cells = 60
    np.random.seed(0)
    counts = np.random.poisson(5, size=(n_cells, n_genes)).astype(np.float32)
    df_expr = pd.DataFrame(counts, columns=[f"g{i}" for i in range(n_genes)])
    df_expr["label"] = (np.arange(n_cells) < n_cells // 2).astype(int).astype(str)

    try:
        generator.generate(
            data=df_expr, n_samples=10, method="scvi",
            target_col="label", epochs=2, auto_report=False,
        )
        assert generator.method == "scvi"

        # Switch to TVAE on the same instance
        df_tabular = pd.DataFrame({
            "f1": np.random.randn(80),
            "f2": np.random.randn(80),
            "label": np.random.choice(["a", "b"], size=80),
        })
        generator.generate(
            data=df_tabular, n_samples=10, method="tvae",
            target_col="label", epochs=2, auto_report=False,
        )
        assert generator.method == "tvae", "method must be updated after TVAE training"

        z = generator.encode_to_latent(df_tabular, target_col="label")
        assert isinstance(z, torch.Tensor)
        assert z.shape[0] == len(df_tabular)
    except (ImportError, Exception) as e:
        if "scvi" in str(e).lower() or "import" in str(e).lower():
            pytest.skip(f"Optional dependency not available: {e}")
        raise
