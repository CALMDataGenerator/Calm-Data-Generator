import logging

from calm_data_generator.generators.tabular import RealGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class TimeSeriesPreset(GeneratorPreset):
    """
    Preset optimized for generating sequential or time-series data.
    Uses TimeGAN, TimeVAE, or FourierFlows (fflows) to capture temporal dynamics.

    TimeGAN: best for complex temporal patterns.
    TimeVAE: faster, good for regular series.
    fflows: most stable, best for periodic/seasonal series.
    """

    def generate(
        self, data, n_samples, sequence_key, time_key=None, method="timegan", **kwargs
    ):
        """Generate time-series data preserving temporal correlations.

        Args:
            data: Original dataset to learn from.
            n_samples: Number of synthetic samples to generate.
            sequence_key: Column identifying each sequence/entity.
            time_key: Optional column marking time ordering within a sequence.
            method (str): Time-series synthesis method (default "timegan").
            **kwargs: Overrides for configuration parameters.

        Returns:
            pd.DataFrame: The synthetic time-series dataset.
        """
        gen = RealGenerator(
            auto_report=kwargs.pop("auto_report", True), random_state=self.random_state
        )

        if self.verbose:
            logger.info(
                f"[TimeSeriesPreset] Generating time-series data using {method} "
                f"(seq_key='{sequence_key}')..."
            )

        epochs = 1 if self.fast_dev_run else 500

        return gen.generate(
            data=data,
            n_samples=n_samples,
            method=method,
            sequence_key=sequence_key,
            time_key=time_key,
            n_iter=epochs,
            batch_size=100,
        )
