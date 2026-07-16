import logging

import pandas as pd

from calm_data_generator.generators.tabular import RealGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class ImbalancedGeneratorPreset(GeneratorPreset):
    """
    Preset designed to generate synthetic data with a specific imbalanced distribution.

    Useful for creating test datasets for drift detection or bias analysis.
    Uses 'resample' or generative methods with forced custom distributions on the target.
    """

    def generate(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: str,
        imbalance_ratio: float = 0.1,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate class-imbalanced data via CTGAN with a target minority-class ratio.

        Args:
            data: Original dataset to learn from.
            n_samples: Number of synthetic samples to generate.
            target_col: Target column whose class balance is controlled.
            imbalance_ratio (float): Desired minority-class proportion (default 0.1).
            **kwargs: Overrides for configuration parameters.

        Returns:
            pd.DataFrame: The class-imbalanced synthetic dataset.
        """
        gen = RealGenerator(
            auto_report=kwargs.pop("auto_report", False), random_state=self.random_state
        )

        unique_vals = data[target_col].unique()
        if len(unique_vals) != 2:
            raise ValueError(
                f"ImbalancedGeneratorPreset requires a binary target column, "
                f"but '{target_col}' has {len(unique_vals)} unique values."
            )

        minority_class = unique_vals[1]
        majority_class = unique_vals[0]

        custom_dist = {
            target_col: {
                minority_class: imbalance_ratio,
                majority_class: 1.0 - imbalance_ratio,
            }
        }

        # Merge with user provided custom_distributions if any
        user_dists = kwargs.pop("custom_distributions", {})
        custom_dist.update(user_dists)

        if self.verbose:
            logger.info(
                "[ImbalancedGeneratorPreset] Generating imbalanced data (ratio %s) for '%s'...",
                imbalance_ratio,
                target_col,
            )

        # Enforce CTGAN for this preset
        # Use minimal epochs if fast_dev_run is True
        epochs = 1 if self.fast_dev_run else 300

        config = {"method": "ctgan", "epochs": epochs}

        return gen.generate(
            data=data,
            n_samples=n_samples,
            target_col=target_col,
            custom_distributions=custom_dist,
            **config,
        )
