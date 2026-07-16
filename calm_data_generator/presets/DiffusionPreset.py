import logging

import pandas as pd

from calm_data_generator.generators.tabular import RealGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class DiffusionPreset(GeneratorPreset):
    """
    Uses Tabular Denoising Diffusion Probabilistic Models (TabDDPM) for synthesis.
    Diffusion models often capture complex multi-modal distributions better than GANs.
    """

    def generate(self, data: pd.DataFrame, n_samples: int, **kwargs) -> pd.DataFrame:
        """Generate high-quality synthetic data with a diffusion model (TabDDPM).

        See ``GeneratorPreset.generate`` for the shared signature (data, n_samples, **kwargs).
        """
        gen = RealGenerator(
            auto_report=kwargs.pop("auto_report", True), random_state=self.random_state
        )

        if self.verbose:
            logger.info(
                "[DiffusionPreset] Generating data using TabDDPM (Diffusion Model)..."
            )

        # Configuration for TabDDPM
        steps = 2 if self.fast_dev_run else 1000

        config = {
            "method": "ddpm",  # Uses SynthCity's TabDDPM plugin
            "n_steps": steps,  # Diffusion steps (higher = better quality, slower)
            "batch_size": 256,
        }
        # No kwargs update allow.

        return gen.generate(data=data, n_samples=n_samples, **config)
