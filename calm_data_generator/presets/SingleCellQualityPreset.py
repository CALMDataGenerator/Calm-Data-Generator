from typing import Any, Dict, Optional, Union

import pandas as pd

from calm_data_generator.generators.tabular import RealGenerator

from .base import GeneratorPreset


class SingleCellQualityPreset(GeneratorPreset):
    def generate(self, data: Any, n_samples: int, **kwargs) -> pd.DataFrame:
        """Generate high-quality single-cell RNA-seq data with scVI.

        See ``GeneratorPreset.generate`` for the shared signature (data, n_samples, **kwargs).
        """
        gen = RealGenerator(
            auto_report=kwargs.pop("auto_report", True), random_state=self.random_state
        )

        # Configuration optimized for single-cell
        config = {
            "method": "scvi",
            "epochs": 400,  # scVI converges well, 400 is often sufficient/good
            "n_latent": 10,  # Typical latent dimension for scVI
        }
        config.update(kwargs)

        return gen.generate(data=data, n_samples=n_samples, **config)
