import numpy as np
import pytest

from calm_data_generator.generators.clinical.Clinic import ClinicalDataGenerator


def test_clinical_generation_flow():
    """Test full Clinical Generator flow."""
    np.random.seed(42)
    try:
        generator = ClinicalDataGenerator(seed=42, auto_report=False)
        results = generator.generate(
            n_samples=50, n_genes=100, n_proteins=50, save_dataset=False
        )

        assert results is not None
        assert "demographics" in results
        assert "genes" in results
        assert "proteins" in results

        demo_df = results["demographics"]
        assert len(demo_df) == 50
        assert "Group" in demo_df.columns

    except ImportError as e:
        pytest.skip(f"Missing dependency: {e}")
    except Exception as e:
        pytest.fail(f"Clinical generator failed: {e}")
