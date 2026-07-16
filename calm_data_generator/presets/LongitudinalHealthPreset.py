import logging

from calm_data_generator.generators.clinical import ClinicalDataGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class LongitudinalHealthPreset(GeneratorPreset):
    """
    Generates longitudinal clinical data (multi-visit patients).
    """

    def generate(self, n_samples, n_visits=5, **kwargs):
        """Generate longitudinal clinical data across multiple patient visits.

        Args:
            n_samples: Number of patients to generate.
            n_visits (int): Number of visits (time points) per patient (default 5).
            **kwargs: Overrides for configuration parameters.

        Returns:
            pd.DataFrame or Dict: The longitudinal clinical dataset(s).
        """
        gen = ClinicalDataGenerator(
            auto_report=kwargs.pop("auto_report", False), seed=self.random_state
        )

        if self.verbose:
            logger.info(
                f"[LongitudinalHealthPreset] Simulating {n_samples} patients with ~{n_visits} visits each..."
            )

        # Clinical generator supports longitudinal generation natively
        # We lock method but ClinicalGenerator usually handles multiple internal models.
        # fast_dev_run doesn't explicitly map to ClinicalGenerator epochs easily without checking implementation,
        # but we definitely block kwargs.

        return gen.generate(n_samples=n_samples, longitudinal=True, avg_visits=n_visits)
