import logging

from calm_data_generator.generators.tabular import RealGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class CopulaPreset(GeneratorPreset):
    """
    Uses Gaussian Copula to model dependencies.
    Very fast and statistically robust baseline, though supports privacy less than GANs.
    """

    def generate(self, data, n_samples, **kwargs):
        """Generate synthetic data with a fast Gaussian Copula synthesizer.

        See ``GeneratorPreset.generate`` for the shared signature (data, n_samples, **kwargs).
        """
        gen = RealGenerator(
            auto_report=kwargs.pop("auto_report", True), random_state=self.random_state
        )

        if self.verbose:
            logger.info("[CopulaPreset] Generating data using Gaussian Copula...")

        return gen.generate(
            data=data,
            n_samples=n_samples,
            method="copula",
            # Copula is fast, minimal params needed usually.
        )
