import logging

from calm_data_generator.generators.configs import DriftConfig
from calm_data_generator.generators.tabular import RealGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class GradualDriftPreset(GeneratorPreset):
    """
    Simulates gradual drift over time or index.
    """

    def generate(self, data, n_samples, drift_cols, slope=0.01, **kwargs):
        """Generate data with gradual (linear-trend) drift on selected columns via CTGAN.

        Args:
            data: Original dataset to learn from.
            n_samples: Number of synthetic samples to generate.
            drift_cols: Columns on which to apply the gradual linear drift.
            slope (float): Per-step slope of the linear drift trend (default 0.01).
            **kwargs: Overrides for configuration parameters.

        Returns:
            pd.DataFrame: The synthetic dataset with gradual drift.
        """
        drift_conf = []
        for col in drift_cols:
            drift_conf.append(
                {
                    "column": col,
                    "type": "linear_drift",  # or shift_mean with trend
                    "slope": slope,
                }
            )

        gen = RealGenerator(
            auto_report=kwargs.pop("auto_report", True), random_state=self.random_state
        )

        if self.verbose:
            logger.info(
                f"[GradualDriftPreset] Injecting gradual drift (slope={slope}) into {drift_cols}..."
            )

        # Enforce CTGAN for gradual drift
        epochs = 1 if self.fast_dev_run else 300

        config = {"method": "ctgan", "epochs": epochs}

        # Leveraging RealGenerator's drift injection which supports linear trends via DriftInjector/ScenarioInjector linkage
        return gen.generate(
            data=data,
            n_samples=n_samples,
            drift_injection_config=DriftConfig(drift_injection_config=drift_conf),
            **config,
        )
