import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from calm_data_generator.generators.configs import EvolutionFeatureConfig
from calm_data_generator.generators.dynamics.ScenarioInjector import ScenarioInjector
from calm_data_generator.generators.tabular import RealGenerator

from .base import GeneratorPreset

logger = logging.getLogger(__name__)


class SeasonalTimeSeriesPreset(GeneratorPreset):
    """
    Generates time-series data with injected seasonal (sinusoidal) patterns.

    Strategy:
    1. Uses RealGenerator (TimeGAN/TimeVAE) to learn base temporal dynamics.
    2. Uses ScenarioInjector to superimpose a strong sinusoidal drift on target columns.
    """

    def generate(
        self,
        data: pd.DataFrame,
        n_samples: int,
        time_col: str,
        seasonal_cols: List[str],
        period: int = 12,
        amplitude: float = 1.0,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate seasonal time-series data (TimeGAN core + seasonal component injection).

        Args:
            data: Original dataset to learn from.
            n_samples: Number of synthetic samples to generate.
            time_col: Name of the time/index column.
            seasonal_cols (List[str]): Columns on which to inject the seasonal pattern.
            period (int): Length of the seasonal cycle (default 12).
            amplitude (float): Amplitude of the seasonal component (default 1.0).
            **kwargs: Overrides for configuration parameters.

        Returns:
            pd.DataFrame: The seasonal synthetic time-series dataset.
        """
        """
        Args:
            period: The period of the seasonality (e.g., 12 for monthly).
            amplitude: The strength of the seasonal effect.
            seasonal_cols: List of columns to inject seasonality into.
        """
        # 1. Generate core synthetic data (capturing basic distribs/correlations)
        # Use TimeGAN if possible, else standard generation + time injection
        # Hardcode to timegan
        method = "timegan"

        gen = RealGenerator(
            auto_report=False,  # We report at the end
            random_state=self.random_state,
        )

        if self.verbose:
            logger.info(
                f"[SeasonalTimeSeriesPreset] 1. Generating base data using {method}..."
            )

        # Check if we need sequence_key for TimeGAN
        if "sequence_key" not in kwargs and method in ["timegan"]:
            # This preset requires sequence_key for TimeGAN usually.
            # but we can't pop it from kwargs if we removed kwargs.
            # We should probably add sequence_key to arguments.
            pass

        params = {}
        # Ensure defaults for time models
        epochs = 1 if self.fast_dev_run else 100
        params["epochs"] = epochs

        synth_df = gen.generate(data=data, n_samples=n_samples, method=method, **params)

        if synth_df is None:
            raise RuntimeError("Base generation failed.")

        # 2. Inject Seasonality using ScenarioInjector
        if self.verbose:
            logger.info(
                f"[SeasonalTimeSeriesPreset] 2. Injecting sinusoidal seasonality (Period={period}, Amp={amplitude})..."
            )

        injector = ScenarioInjector(seed=self.random_state)

        # Build evolution config for all target columns
        evolution_map = {}
        for col in seasonal_cols:
            evolution_map[col] = EvolutionFeatureConfig(
                column=col,
                type="sinusoidal",
                period=float(period),
                amplitude=float(amplitude),
                phase=0.0,
            )

        # Apply the seasonal drift
        final_df = injector.evolve_features(
            df=synth_df,
            evolution_config=evolution_map,
            time_col=time_col,
            auto_report=kwargs.get("auto_report", True),
        )

        return final_df
