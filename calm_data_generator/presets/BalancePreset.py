import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from calm_data_generator.generators.tabular import RealGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class BalancedDataGeneratorPreset(GeneratorPreset):
    """
    Preset designed to balance an originally imbalanced dataset.

    Uses SMOTE (or ADASYN) to oversample minority classes to achieve a balanced distribution.
    """

    def generate(
        self, data: pd.DataFrame, n_samples: int, target_col: str, **kwargs
    ) -> pd.DataFrame:
        """Balance the target classes using SMOTE oversampling.

        Args:
            data: Original (imbalanced) dataset to learn from.
            n_samples: Number of synthetic samples to generate.
            target_col: Target column whose classes are balanced.
            **kwargs: Overrides for configuration parameters.

        Returns:
            pd.DataFrame: The class-balanced synthetic dataset.
        """
        gen = RealGenerator(
            auto_report=kwargs.pop("auto_report", False), random_state=self.random_state
        )

        if self.verbose:
            logger.info(
                f"[BalancedDataGeneratorPreset] Balancing data based on '{target_col}' using SMOTE..."
            )

        # Enforce SMOTE
        # SMOTE is fast, so fast_dev_run might not need to change much,
        # but we ensure parameters are fixed.

        return gen.generate(
            data=data,
            n_samples=n_samples,
            target_col=target_col,
            method="smote",
            # No kwargs passed to generate to prevent override of method
        )
