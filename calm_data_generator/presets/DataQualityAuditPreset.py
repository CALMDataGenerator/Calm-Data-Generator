import logging

from calm_data_generator.generators.tabular import RealGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class DataQualityAuditPreset(GeneratorPreset):
    """
    Focused on high-integrity generation with comprehensive automated reporting.
    Uses TVAE (often better than CTGAN for structure) and enables full reporting.
    """

    def generate(self, data, n_samples, **kwargs):
        """Generate data with TVAE and force a full quality report for auditing.

        See ``GeneratorPreset.generate`` for the shared signature (data, n_samples, **kwargs).
        """
        gen = RealGenerator(
            auto_report=True,  # Force True
            minimal_report=False,  # Force Full Report
            random_state=self.random_state,
        )

        if self.verbose:
            logger.info(
                "[DataQualityAuditPreset] Generating data with TVAE and full quality audit..."
            )

        # TVAE is strict
        epochs = 1 if self.fast_dev_run else 300

        return gen.generate(
            data=data,
            n_samples=n_samples,
            method="tvae",
            epochs=epochs,
            # No kwargs
        )
