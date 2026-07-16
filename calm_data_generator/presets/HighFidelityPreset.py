import logging

import pandas as pd

from calm_data_generator.generators.tabular import RealGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class HighFidelityPreset(GeneratorPreset):
    """
    Preset optimized for maximum data quality and fidelity.

    Uses CTGAN with a high number of epochs and optimized batch size,
    prioritizing quality over speed. Forces adversarial validation.
    """

    def generate(
        self,
        data: pd.DataFrame,
        n_samples: int,
        auto_report: bool = True,
    ) -> pd.DataFrame:
        """Generate high-fidelity synthetic data with adversarial-validation quality checks.

        Args:
            data: Original dataset to learn from.
            n_samples: Number of synthetic samples to generate.
            auto_report (bool): If True, generate a quality report after synthesis (default True).

        Returns:
            pd.DataFrame: The high-fidelity synthetic dataset.
        """
        gen = RealGenerator(auto_report=auto_report, random_state=self.random_state)

        # Default high-fidelity configuration
        # If fast_dev_run is True, use minimal settings for testing
        epochs = 1 if self.fast_dev_run else 1000

        config = {
            "method": "ctgan",
            "epochs": epochs,
            "batch_size": 250,
            "adversarial_validation": True,  # Ensure quality check
        }

        if self.verbose:
            logger.info(
                f"[HighFidelityPreset] Generating data with high-fidelity settings (epochs={epochs})..."
            )

        return gen.generate(data=data, n_samples=n_samples, **config)
