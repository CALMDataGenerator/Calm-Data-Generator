import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from calm_data_generator.generators.configs import DriftConfig
from calm_data_generator.generators.tabular import RealGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class DriftScenarioPreset(GeneratorPreset):
    """
    Preset designed to generate data with specific injected drift characteristics.
    Used for stress-testing ML models and drift detection systems.
    """

    def generate(
        self,
        data: pd.DataFrame,
        n_samples: int,
        drift_scenarios: List[Dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate data with one or more injected drift scenarios via CTGAN.

        Args:
            data: Original dataset to learn from.
            n_samples: Number of synthetic samples to generate.
            drift_scenarios (List[Dict]): List of drift scenario configurations to inject.
            **kwargs: Overrides for configuration parameters.

        Returns:
            pd.DataFrame: The synthetic dataset with injected drift.
        """
        gen = RealGenerator(
            auto_report=kwargs.pop("auto_report", True), random_state=self.random_state
        )

        if self.verbose:
            logger.info(
                "[DriftScenarioPreset] Generating data with injected drift scenarios..."
            )

        # Enforce CTGAN for drift scenarios to ensure good distribution capturing
        # Use minimal epochs if fast_dev_run is True
        epochs = 1 if self.fast_dev_run else 300

        config = {"method": "ctgan", "epochs": epochs}

        if drift_scenarios:
            # Wrap in DriftConfig for validation
            drift_config = DriftConfig(drift_injection_config=drift_scenarios)
            return gen.generate(
                data=data,
                n_samples=n_samples,
                drift_injection_config=drift_config,
                **config,
            )

        return gen.generate(data=data, n_samples=n_samples, **config)
