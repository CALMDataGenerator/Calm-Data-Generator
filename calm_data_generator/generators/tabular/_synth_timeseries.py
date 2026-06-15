"""
Mixin providing time-series synthesis methods for RealGenerator.

Methods: _synthesize_timegan, _synthesize_timevae, _synthesize_fflows.
"""

from typing import Optional

import pandas as pd


class _TimeSeriesSynthMixin:

    def _synthesize_timegan(
        self, data: pd.DataFrame, n_samples: int, **kwargs
    ) -> pd.DataFrame:
        """
        Synthesizes time series data using Synthcity's TimeGAN.

        TimeGAN is designed for sequential/temporal data with multiple entities.
        It learns both temporal dynamics and feature distributions.

        Args:
            data: Input DataFrame with temporal structure
            n_samples: Number of sequences to generate
            **kwargs: Additional parameters for TimeGAN:
                - n_iter: int = 1000 - Training epochs
                - n_units_hidden: int = 100 - Hidden units
                - batch_size: int = 128 - Batch size
                - lr: float = 0.001 - Learning rate

        Returns:
            Synthetic DataFrame with temporal structure
        """
        self.logger.info("Starting TimeGAN synthesis (Synthcity)...")

        try:
            from synthcity.plugins import Plugins
            from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        except ImportError:
            self.logger.error("Synthcity not available for TimeGAN.")
            raise ImportError(
                "TimeGAN requires synthcity. Install with: pip install synthcity"
            )

        # Extract TimeGAN-specific parameters
        n_iter = kwargs.get("n_iter", kwargs.get("epochs", 1000))
        n_units_hidden = kwargs.get("n_units_hidden", 100)
        batch_size = kwargs.get("batch_size", 128)
        lr = kwargs.get("lr", 0.001)

        # Load plugin
        plugin = Plugins().get(
            "timegan",
            n_iter=n_iter,
            n_units_hidden=n_units_hidden,
            batch_size=batch_size,
            lr=lr,
        )

        # Prepare time series data for TimeSeriesDataLoader.
        # synthcity expects:
        #   temporal_data: list of DataFrames, one per sequence (numeric features only)
        #   observation_times: list of numeric arrays, one per sequence
        #   outcome: optional per-sequence outcome Series
        sequence_key = kwargs.get("sequence_key", None)
        time_key = kwargs.get("time_key", None)
        target_col = kwargs.get("target_col", None)

        def _to_numeric_times(times_array):
            """Convert datetime arrays to numeric seconds from sequence start."""
            s = pd.Series(times_array)
            if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_object_dtype(s):
                try:
                    s = pd.to_datetime(s)
                    s = (s - s.iloc[0]).dt.total_seconds().astype(float)
                except Exception as e:
                    self.logger.debug(f"Could not parse datetime column; falling back to integer index. Reason: {e}")
                    s = pd.Series(range(len(times_array)), dtype=float)
            return s.tolist()

        if sequence_key and sequence_key in data.columns:
            exclude = {sequence_key}
            if time_key:
                exclude.add(time_key)
            # Separate target (static per sequence) from temporal features
            if target_col and target_col in data.columns:
                exclude.add(target_col)
                # Outcome: one value per sequence (first value, since it's constant)
                outcome_series = (
                    data.groupby(sequence_key)[target_col]
                    .first()
                    .reset_index(drop=True)
                )
            else:
                outcome_series = None

            feature_cols = [c for c in data.columns if c not in exclude]
            temporal_data = []
            observation_times = []
            for _, grp in data.groupby(sequence_key, sort=False):
                grp = grp.reset_index(drop=True)
                if time_key and time_key in grp.columns:
                    obs_times = _to_numeric_times(grp[time_key].values)
                else:
                    obs_times = list(range(len(grp)))
                temporal_data.append(grp[feature_cols].reset_index(drop=True))
                observation_times.append(obs_times)

            # synthcity requires outcome as DataFrame, not Series
            _outcome = (
                outcome_series.to_frame()
                if outcome_series is not None
                else pd.DataFrame({"outcome": [0] * len(temporal_data)})
            )
            loader = TimeSeriesDataLoader(
                temporal_data=temporal_data,
                observation_times=observation_times,
                outcome=_outcome,
            )
        else:
            # Fallback: treat the entire DataFrame as ONE multi-step sequence.
            # This is the correct approach for a flat time series with no sequence_key.
            exclude = set()
            if target_col and target_col in data.columns:
                exclude.add(target_col)
                outcome_series = (
                    data[target_col].iloc[[0]].reset_index(drop=True)
                )  # one outcome per sequence
            else:
                outcome_series = None
            feature_cols = [c for c in data.columns if c not in exclude]
            temporal_data = [data[feature_cols].reset_index(drop=True)]
            observation_times = [list(range(len(data)))]
            # synthcity requires outcome as DataFrame, not Series
            _outcome = (
                outcome_series.to_frame()
                if outcome_series is not None
                else pd.DataFrame({"outcome": [0]})
            )
            loader = TimeSeriesDataLoader(
                temporal_data=temporal_data,
                observation_times=observation_times,
                outcome=_outcome,
            )

        # Train
        self.logger.info(f"Training TimeGAN for {n_iter} epochs...")
        plugin.fit(loader)
        self.synthesizer = plugin
        self.method = "timegan"

        # Generate
        self.logger.info(f"Generating {n_samples} synthetic sequences...")
        synth = plugin.generate(count=n_samples, random_state=self.random_state)
        synth_df = synth.dataframe()

        self.logger.info(
            f"TimeGAN synthesis complete. Generated {len(synth_df)} samples."
        )
        return synth_df

    def _synthesize_timevae(
        self, data: pd.DataFrame, n_samples: int, **kwargs
    ) -> pd.DataFrame:
        """
        Synthesizes time series data using Synthcity's TimeVAE.

        TimeVAE is a variational autoencoder designed for temporal data.
        It's generally faster than TimeGAN and works well for regular time series.

        Args:
            data: Input DataFrame with temporal structure
            n_samples: Number of sequences to generate
            **kwargs: Additional parameters for TimeVAE:
                - n_iter: int = 1000 - Training epochs
                - decoder_n_layers_hidden: int = 2 - Decoder layers
                - decoder_n_units_hidden: int = 100 - Decoder units
                - batch_size: int = 128 - Batch size
                - lr: float = 0.001 - Learning rate

        Returns:
            Synthetic DataFrame with temporal structure
        """
        self.logger.info("Starting TimeVAE synthesis (Synthcity)...")

        try:
            from synthcity.plugins import Plugins
            from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        except ImportError:
            self.logger.error("Synthcity not available for TimeVAE.")
            raise ImportError(
                "TimeVAE requires synthcity. Install with: pip install synthcity"
            )

        # Extract TimeVAE-specific parameters
        n_iter = kwargs.get("n_iter", kwargs.get("epochs", 1000))
        decoder_n_layers_hidden = kwargs.get("decoder_n_layers_hidden", 2)
        decoder_n_units_hidden = kwargs.get("decoder_n_units_hidden", 100)
        batch_size = kwargs.get("batch_size", 128)
        lr = kwargs.get("lr", 0.001)

        # Load plugin
        plugin = Plugins().get(
            "timevae",
            n_iter=n_iter,
            decoder_n_layers_hidden=decoder_n_layers_hidden,
            decoder_n_units_hidden=decoder_n_units_hidden,
            batch_size=batch_size,
            lr=lr,
        )

        # Prepare time series data for TimeSeriesDataLoader — same logic as _synthesize_timegan.
        # synthcity requires temporal_data as list of DataFrames, observation_times as list of
        # numeric arrays, and outcome as a DataFrame (not a flat input DataFrame).
        sequence_key = kwargs.get("sequence_key", None)
        time_key = kwargs.get("time_key", None)
        target_col = kwargs.get("target_col", None)

        def _to_numeric_times(times_array):
            s = pd.Series(times_array)
            if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_object_dtype(s):
                try:
                    s = pd.to_datetime(s)
                    s = (s - s.iloc[0]).dt.total_seconds().astype(float)
                except Exception as e:
                    self.logger.debug(f"Could not parse datetime column; falling back to integer index. Reason: {e}")
                    s = pd.Series(range(len(times_array)), dtype=float)
            return s.tolist()

        if sequence_key and sequence_key in data.columns:
            exclude = {sequence_key}
            if time_key:
                exclude.add(time_key)
            if target_col and target_col in data.columns:
                exclude.add(target_col)
                outcome_series = (
                    data.groupby(sequence_key)[target_col]
                    .first()
                    .reset_index(drop=True)
                )
            else:
                outcome_series = None
            feature_cols = [c for c in data.columns if c not in exclude]
            temporal_data = []
            observation_times = []
            for _, grp in data.groupby(sequence_key, sort=False):
                grp = grp.reset_index(drop=True)
                obs_times = (
                    _to_numeric_times(grp[time_key].values)
                    if time_key and time_key in grp.columns
                    else list(range(len(grp)))
                )
                temporal_data.append(grp[feature_cols].reset_index(drop=True))
                observation_times.append(obs_times)
            _outcome = (
                outcome_series.to_frame()
                if outcome_series is not None
                else pd.DataFrame({"outcome": [0] * len(temporal_data)})
            )
            loader = TimeSeriesDataLoader(
                temporal_data=temporal_data,
                observation_times=observation_times,
                outcome=_outcome,
            )
        else:
            # Fallback: treat the entire DataFrame as ONE multi-step sequence.
            exclude = set()
            if target_col and target_col in data.columns:
                exclude.add(target_col)
                outcome_series = data[target_col].iloc[[0]].reset_index(drop=True)
            else:
                outcome_series = None
            feature_cols = [c for c in data.columns if c not in exclude]
            temporal_data = [data[feature_cols].reset_index(drop=True)]
            observation_times = [list(range(len(data)))]
            _outcome = (
                outcome_series.to_frame()
                if outcome_series is not None
                else pd.DataFrame({"outcome": [0]})
            )
            loader = TimeSeriesDataLoader(
                temporal_data=temporal_data,
                observation_times=observation_times,
                outcome=_outcome,
            )

        # Train
        self.logger.info(f"Training TimeVAE for {n_iter} epochs...")
        plugin.fit(loader)
        self.synthesizer = plugin
        self.method = "timevae"

        # Generate
        self.logger.info(f"Generating {n_samples} synthetic sequences...")
        synth = plugin.generate(count=n_samples, random_state=self.random_state)
        synth_df = synth.dataframe()

        self.logger.info(
            f"TimeVAE synthesis complete. Generated {len(synth_df)} samples."
        )
        return synth_df

    def _synthesize_fflows(
        self,
        data: pd.DataFrame,
        n_samples: int,
        **kwargs,
    ) -> pd.DataFrame:
        """Synthesizes time series data using Synthcity's FourierFlows (fflows).

        FourierFlows uses normalizing flows in the frequency domain to generate
        realistic temporal sequences. Generally more stable than TimeGAN and
        particularly effective for periodic or quasi-periodic time series.

        Args:
            data: Input DataFrame with temporal structure.
            n_samples: Number of synthetic sequences to generate.
            **kwargs:
                - sequence_key: str - Column identifying each sequence (required for multi-sequence data).
                - time_key: str - Column with timestamps/time indices.
                - target_col: str - Target column to use as outcome (separated from features).
                - n_iter: int = 1000 - Training epochs.
                - batch_size: int = 128 - Batch size.
                - lr: float = 0.001 - Learning rate.

        Returns:
            Synthetic DataFrame with temporal structure.
        """
        self.logger.info("Starting FourierFlows (fflows) synthesis via Synthcity...")

        try:
            from synthcity.plugins import Plugins
            from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        except ImportError:
            raise ImportError(
                "FourierFlows requires synthcity. Install with: pip install synthcity"
            )

        # Extract fflows-specific parameters
        n_iter = kwargs.get("n_iter", kwargs.get("epochs", 1000))
        batch_size = kwargs.get("batch_size", 128)
        lr = kwargs.get("lr", 0.001)

        plugin = Plugins().get("fflows", n_iter=n_iter, batch_size=batch_size, lr=lr)

        # Reuse the same TimeSeriesDataLoader preparation logic as timegan/timevae
        sequence_key = kwargs.get("sequence_key", None)
        time_key = kwargs.get("time_key", None)
        target_col = kwargs.get("target_col", None)

        def _to_numeric_times(times_array):
            s = pd.Series(times_array)
            if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_object_dtype(s):
                try:
                    s = pd.to_datetime(s)
                    s = (s - s.iloc[0]).dt.total_seconds().astype(float)
                except Exception as e:
                    self.logger.debug(f"Could not parse datetime column; falling back to integer index. Reason: {e}")
                    s = pd.Series(range(len(times_array)), dtype=float)
            return s.tolist()

        if sequence_key and sequence_key in data.columns:
            exclude = {sequence_key}
            if time_key:
                exclude.add(time_key)
            if target_col and target_col in data.columns:
                exclude.add(target_col)
                outcome_series = (
                    data.groupby(sequence_key)[target_col]
                    .first()
                    .reset_index(drop=True)
                )
            else:
                outcome_series = None
            feature_cols = [c for c in data.columns if c not in exclude]
            temporal_data, observation_times = [], []
            for _, grp in data.groupby(sequence_key, sort=False):
                grp = grp.reset_index(drop=True)
                obs_times = (
                    _to_numeric_times(grp[time_key].values)
                    if time_key and time_key in grp.columns
                    else list(range(len(grp)))
                )
                temporal_data.append(grp[feature_cols].reset_index(drop=True))
                observation_times.append(obs_times)
            _outcome = (
                outcome_series.to_frame()
                if outcome_series is not None
                else pd.DataFrame({"outcome": [0] * len(temporal_data)})
            )
            loader = TimeSeriesDataLoader(
                temporal_data=temporal_data,
                observation_times=observation_times,
                outcome=_outcome,
            )
        else:
            # Fallback: treat the entire DataFrame as one multi-step sequence
            exclude = set()
            if target_col and target_col in data.columns:
                exclude.add(target_col)
                outcome_series = data[target_col].iloc[[0]].reset_index(drop=True)
            else:
                outcome_series = None
            feature_cols = [c for c in data.columns if c not in exclude]
            temporal_data = [data[feature_cols].reset_index(drop=True)]
            _outcome = (
                outcome_series.to_frame()
                if outcome_series is not None
                else pd.DataFrame({"outcome": [0]})
            )
            loader = TimeSeriesDataLoader(
                temporal_data=temporal_data,
                observation_times=[list(range(len(data)))],
                outcome=_outcome,
            )

        self.logger.info(f"Training FourierFlows for {n_iter} epochs...")
        plugin.fit(loader)
        self.synthesizer = plugin
        self.method = "fflows"
        self.logger.info(f"Generating {n_samples} synthetic sequences...")
        synth = plugin.generate(count=n_samples, random_state=self.random_state)
        synth_df = synth.dataframe()
        self.logger.info(
            f"FourierFlows synthesis complete. Generated {len(synth_df)} samples."
        )
        return synth_df
