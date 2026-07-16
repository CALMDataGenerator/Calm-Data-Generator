import logging

from calm_data_generator.generators.tabular import RealGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class FastPrototypePreset(GeneratorPreset):
    """
    Preset optimized for rapid data generation.
    Ideal for integration testing, CI/CD pipelines, or quick prototyping.
    """

    def generate(self, data, n_samples, **kwargs):
        """Generate synthetic data quickly (LightGBM, minimal reporting) for prototyping.

        See ``GeneratorPreset.generate`` for the shared signature (data, n_samples, **kwargs).
        """
        # Uses LightGBM for speed, minimal reporting
        gen = RealGenerator(
            auto_report=kwargs.pop("auto_report", False),
            minimal_report=kwargs.pop("minimal_report", True),
            random_state=self.random_state,
        )

        if self.verbose:
            logger.info("[FastPrototypePreset] Generating data quickly using LightGBM...")

        # 10 iterations is very fast but provides decent enough structure
        # Use 1 iteration if fast_dev_run is True
        iterations = 1 if self.fast_dev_run else 10

        return gen.generate(
            data=data,
            n_samples=n_samples,
            method="lgbm",
            iterations=iterations,
            # No kwargs passed
        )
