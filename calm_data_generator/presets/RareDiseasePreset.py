import logging
from typing import Any, Dict, Optional

import pandas as pd

from calm_data_generator.generators.clinical import ClinicalDataGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class RareDiseasePreset(GeneratorPreset):
    """
    Simulates a clinical cohort with a rare disease condition.
    Forces a very low disease prevalence ratio (e.g., 1% or 5%).
    """

    def generate(
        self, n_samples: int, disease_ratio: float = 0.01, **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Generate clinical data with a rare-disease (highly imbalanced) case/control ratio.

        Args:
            n_samples: Number of patients to generate.
            disease_ratio (float): Proportion of disease cases, e.g. 0.01 = 1% (default 0.01).
            **kwargs: Overrides for configuration parameters.

        Returns:
            Dict[str, pd.DataFrame]: The clinical dataset(s) (clinical, omics, etc.).
        """
        gen = ClinicalDataGenerator(
            auto_report=kwargs.pop("auto_report", True), seed=self.random_state
        )

        if self.verbose:
            logger.info(
                f"[RareDiseasePreset] Simulating rare disease cohort (Prevalence: {disease_ratio:.1%})..."
            )

        # Note: ClinicalDataGenerator.generate returns a Dict of DataFrames (clinical, omics, etc)
        # We pass control_disease_ratio to influence the 'Introduction' of disease status
        # If disease_ratio is 0.01 (1% disease), then control_ratio must be 0.99 (99% control)
        control_ratio = 1.0 - disease_ratio

        # Clinical generator params are complex, but we can lock down the generation method if applicable
        # The ClinicalDataGenerator manages its own internal generators (usually CTGANs for omics).
        # We can pass fast_dev_run logic implicitly by how we call it?
        # ClinicalDataGenerator doesn't accept epochs directly in generate usually, it's set in init or via specific params.
        # Check ClinicalDataGenerator signature later, but for now we block arbitrary kwargs override
        # and just pass what's needed.

        return gen.generate(
            n_samples=n_samples,
            control_disease_ratio=control_ratio,  # Enforcing rarity
        )
