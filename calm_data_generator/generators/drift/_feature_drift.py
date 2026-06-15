"""
Feature drift mixin for DriftInjector.
Contains methods for injecting various forms of feature drift.
"""

import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from calm_data_generator.generators.configs import DriftConfig


class _FeatureDriftMixin:
    """Mixin providing feature drift injection methods for DriftInjector."""

    def inject_feature_drift(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        drift_type: str = "gaussian_noise",
        drift_magnitude: float = 0.2,
        drift_value: Optional[float] = None,
        drift_values: Optional[Dict[str, float]] = None,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
        correlations: Optional[Union[pd.DataFrame, Dict, bool]] = None,
        drift_config: Optional[Union[DriftConfig, Dict]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Applies drift at once based on various selection criteria.
        Can use a DriftConfig object/dict or individual arguments.
        """
        # Resolve config
        config_args = {
            "feature_cols": feature_cols,
            "drift_type": drift_type,
            "magnitude": drift_magnitude,  # Map drift_magnitude to magnitude
            "drift_value": drift_value,
            "drift_values": drift_values,
            "start_index": start_index,
            "block_index": block_index,
            "block_column": block_column,
            "time_start": time_start,
            "time_end": time_end,
            # Pass extra params that might be in Config (via extra='allow') or just kwargs
            "time_ranges": time_ranges,
            "specific_times": specific_times,
            "time_col": time_col,
        }
        # Filter None from config_args to avoid overwriting defaults/config values with None
        config_args = {k: v for k, v in config_args.items() if v is not None}

        config = self._resolve_drift_config(drift_config, **config_args)

        # Fallbacks for required fields if not in config
        if not config.feature_cols:
            # Should not happen if passed as arg, but if config passed w/o it and arg is None
            # We can't do much. Raise error?
            # For now, assume user knows what they are doing or it enters loop with empty list
            config.feature_cols = []

        # Override magnitude if it was passed as drift_magnitude
        # (Already handled by mapping above)

        self._validate_feature_op(config.drift_type, config.magnitude)
        df_drift = df.copy()

        # Use config properties for get_target_rows
        # We need to extract them or pass config as kwargs?
        # _get_target_rows takes specific args.

        # Extract args for get_target_rows from config (including extras)
        row_selector_args = config.model_dump(
            exclude={
                "feature_cols",
                "drift_type",
                "magnitude",
                "drift_value",
                "drift_values",
            }
        )
        # Map magnitude back? No, get_target_rows doesn't need it.

        # Filter kwargs to avoid passing duplicate parameters
        # Remove all parameters that _get_target_rows accepts explicitly
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in {
                "start_index",
                "end_index",
                "index_step",
                "block_index",
                "block_column",
                "blocks",
                "block_start",
                "n_blocks",
                "block_step",
                "time_col",
                "time_start",
                "time_end",
                "time_ranges",
                "specific_times",
                "time_step",
            }
        }

        rows = self._get_target_rows(
            df,
            **row_selector_args,
            **filtered_kwargs,  # Use filtered kwargs
        )

        w = np.ones(len(rows), dtype=float)

        # Pre-calculate correlations if needed to avoid re-calc per column
        # If correlations is implicitly True, we can either calculate here or inside propagation
        # Let's clean it up slightly if it's just a boolean flag to start with

        for col in config.feature_cols:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found")
                continue

            column_drift_value = None
            if drift_type in {
                "add_value",
                "subtract_value",
                "multiply_value",
                "divide_value",
            }:
                column_drift_value = (
                    config.drift_values.get(col)
                    if config.drift_values
                    else config.drift_value
                )
                if column_drift_value is None:
                    raise ValueError(
                        f"For '{config.drift_type}', provide drift_value or drift_values['{col}']"
                    )

            if pd.api.types.is_numeric_dtype(df[col]):
                x_original = df_drift.loc[rows, col].to_numpy(copy=True)

                # Pre-calculate std for propagation (before drift applied)
                driver_std = float(df_drift[col].std())

                x2 = self._apply_numeric_op_with_weights(
                    x_original,
                    config.drift_type,
                    config.magnitude,
                    w,
                    self.rng,
                    column_drift_value,
                )

                # Calculate Delta for propagation
                delta = x2 - x_original

                # Apply change
                df_drift.loc[rows, col] = x2

                # Propagate if correlations provided
                if correlations is not None and correlations is not False:
                    self._propagate_numeric_drift(
                        df_drift, rows, col, delta, correlations, driver_std=driver_std
                    )
            else:
                s = df_drift.loc[rows, col]
                s2 = self._apply_categorical_with_weights(
                    s, w, config.magnitude, self.rng
                )
                df_drift.loc[rows, col] = s2

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir

            drift_config_dict = {
                "drift_method": "inject_feature_drift",
                "feature_cols": config.feature_cols,
                "drift_type": config.drift_type,
                "drift_magnitude": config.magnitude,
                "start_index": config.start_index,
                "block_index": config.block_index,
                "time_start": config.time_start,
                "generator_name": f"{gen_name}_feature_drift",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config_dict['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(
                    df,
                    df_drift,
                    drift_config_dict,
                    time_col=time_col,
                    resample_rule=kwargs.get("resample_rule"),
                )
        return df_drift

    # -------------------------
    # Feature drift "windowed": gradual, abrupt, incremental, recurrent
    # -------------------------
    def inject_feature_drift_gradual(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        drift_type: str = "gaussian_noise",
        drift_magnitude: float = 0.2,
        drift_value: Optional[float] = None,
        drift_values: Optional[Dict[str, float]] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        center: Optional[int] = None,
        width: Optional[int] = None,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        inconsistency: float = 0.0,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
        resample_rule: Optional[Union[str, int]] = None,
        correlations: Optional[Union[pd.DataFrame, Dict, bool]] = None,
        drift_config: Optional[Union[DriftConfig, Dict]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Injects gradual drift on selected rows using a transition window.
        Can use DriftConfig or individual args.
        """
        # Resolve config
        config_args = {
            "feature_cols": feature_cols,
            "drift_type": drift_type,
            "magnitude": drift_magnitude,
            "drift_value": drift_value,
            "drift_values": drift_values,
            "start_index": start_index,
            "end_index": end_index,
            "block_index": block_index,
            "block_column": block_column,
            "time_start": time_start,
            "time_end": time_end,
            "center": center,
            "width": width,
            "profile": profile,
            "speed_k": speed_k,
            "direction": direction,
            "inconsistency": inconsistency,
            # Extras
            "time_ranges": time_ranges,
            "specific_times": specific_times,
            "blocks": blocks,
            "block_start": block_start,
            "n_blocks": n_blocks,
            "time_col": time_col,
        }
        config_args = {k: v for k, v in config_args.items() if v is not None}
        config = self._resolve_drift_config(drift_config, **config_args)

        if not config.feature_cols:
            config.feature_cols = []

        self._validate_feature_op(config.drift_type, config.magnitude)
        df_drift = df.copy()

        # Extract target rows args from config
        row_selector_args = config.model_dump(
            exclude={
                "feature_cols",
                "drift_type",
                "magnitude",
                "drift_value",
                "drift_values",
            }
        )

        rows = self._get_target_rows(df, **row_selector_args, **kwargs)

        n = len(rows)
        if n == 0:
            return df_drift

        c = (
            int(n // 2)
            if config.center is None
            else int(np.clip(config.center, 0, n - 1))
        )
        w_width = max(
            1, int(config.width if config.width is not None else max(1, n // 5))
        )
        w = self._window_weights(
            n,
            center=c,
            width=w_width,
            profile=config.profile,
            k=float(config.speed_k),
            direction=config.direction,
        )

        if config.inconsistency > 0:
            # Simplified inconsistency logic for brevity
            noise = self.rng.normal(0, 0.1 * config.inconsistency, n)
            w = np.clip(w + noise, 0.0, 1.0)

        for col in config.feature_cols:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found")
                continue

            column_drift_value = (
                config.drift_values.get(col)
                if config.drift_values
                else config.drift_value
            )
            if pd.api.types.is_numeric_dtype(df[col]):
                x_original = df_drift.loc[rows, col].to_numpy(copy=True)

                # Pre-calculate std for propagation
                driver_std = float(df_drift[col].std())

                x2 = self._apply_numeric_op_with_weights(
                    x_original,
                    config.drift_type,
                    config.magnitude,
                    w,
                    self.rng,
                    column_drift_value,
                )

                delta = x2 - x_original
                df_drift.loc[rows, col] = x2

                if correlations is not None and correlations is not False:
                    self._propagate_numeric_drift(
                        df_drift, rows, col, delta, correlations, driver_std=driver_std
                    )
            else:
                s = df_drift.loc[rows, col]
                s2 = self._apply_categorical_with_weights(
                    s, w, config.magnitude, self.rng
                )
                df_drift.loc[rows, col] = s2

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config_dict = {
                "drift_method": "inject_feature_drift_gradual",
                "feature_cols": config.feature_cols,
                "drift_type": config.drift_type,
                "drift_magnitude": config.magnitude,
                "profile": config.profile,
                "center": config.center,
                "width": config.width,
                "inconsistency": config.inconsistency,
                "generator_name": f"{gen_name}_feature_gradual",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config_dict['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(
                    df,
                    df_drift,
                    drift_config_dict,
                    time_col=time_col,
                    resample_rule=resample_rule,
                )
        return df_drift

    def inject_feature_drift_incremental(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_type: str,
        drift_magnitude: float,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Injects a constant and smooth drift using a single wide sigmoid transition.
        """
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            blocks=blocks,
            block_start=block_start,
            n_blocks=n_blocks,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )
        n = len(rows)
        if n == 0:
            return df.copy()

        center = n / 2
        width = n

        return self.inject_feature_drift_gradual(
            df=df,
            feature_cols=feature_cols,
            drift_type=drift_type,
            drift_magnitude=drift_magnitude,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            blocks=blocks,
            block_start=block_start,
            n_blocks=n_blocks,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
            center=int(round(center)),
            width=width,
            profile="sigmoid",
            speed_k=1.0,
            direction="up",
            **kwargs,
        )

    def inject_feature_drift_recurrent(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_type: str,
        drift_magnitude: float,
        drift_value: Optional[float] = None,
        drift_values: Optional[Dict[str, float]] = None,
        windows: Optional[Sequence[Tuple[int, int]]] = None,
        block_column: Optional[str] = None,
        cycle_blocks: Optional[Sequence] = None,
        repeats: int = 1,
        random_repeat_order: bool = False,
        center: Optional[int] = None,
        width: Optional[int] = None,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Injects recurrent drift by applying several drift windows.
        """
        df_out = df.copy()

        # This method's logic gets complex with time selection.
        # For now, we assume 'windows' applies to the selected rows from time/block criteria.
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            blocks=blocks,
            block_start=block_start,
            n_blocks=n_blocks,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        # The logic for 'cycle_blocks' and 'windows' needs careful integration with the new time selection.
        # This is a simplified version.
        if not rows.empty:
            # Apply drift to the selected rows
            n_total = len(rows)
            chunk_size = n_total // max(1, repeats)

            for i in range(repeats):
                # Determine start/end of this recurrence window
                c_start = i * chunk_size
                c_end = (i + 1) * chunk_size if i < repeats - 1 else n_total

                if c_end <= c_start:
                    continue

                chunk_indices = rows[c_start:c_end]
                n_chunk = len(chunk_indices)

                # Determine local center/width for this chunk
                local_center = n_chunk // 2 if center is None else center
                local_width = n_chunk if width is None else width

                w = self._window_weights(
                    n_chunk,
                    float(local_center),
                    int(local_width),
                    profile,
                    speed_k,
                    direction,
                )

                # Apply to each feature
                for col in feature_cols:
                    if col not in df_out.columns:
                        continue

                    column_drift_value = (
                        drift_values.get(col) if drift_values else drift_value
                    )

                    if pd.api.types.is_numeric_dtype(df_out[col]):
                        x = df_out.loc[chunk_indices, col].to_numpy(copy=True)
                        x2 = self._apply_numeric_op_with_weights(
                            x,
                            drift_type,
                            drift_magnitude,
                            w,
                            self.rng,
                            column_drift_value,
                        )
                        df_out.loc[chunk_indices, col] = x2
                    else:
                        s = df_out.loc[chunk_indices, col]
                        s2 = self._apply_categorical_with_weights(
                            s, w, drift_magnitude, self.rng
                        )
                        df_out.loc[chunk_indices, col] = s2

        return df_out

    def inject_feature_drift_abrupt(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_type: str = "gaussian_noise",
        drift_magnitude: float = 0.2,
        change_index: int = 0,
        direction: str = "up",  # direction is unused in simple shift but kept for API compat
        time_col: Optional[str] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Alias/Wrapper for injecting abrupt drift (step change) using inject_feature_drift.
        This corresponds to drifting all rows starting from `change_index`.
        """
        # "Abrupt" drift typically means a permanent change from a point onwards.
        # We can simulate this by setting start_index=change_index in inject_feature_drift.
        return self.inject_feature_drift(
            df=df,
            feature_cols=feature_cols,
            drift_type=drift_type,
            drift_magnitude=drift_magnitude,
            start_index=change_index,
            time_col=time_col,
            output_dir=output_dir,
            generator_name=generator_name,
        )

    # -------------------------
    # Binary Probabilistic Drift
    # -------------------------
    def inject_binary_probabilistic_drift(
        self,
        df: pd.DataFrame,
        target_col: str,
        probability: float = 0.4,
        noise_range: Tuple[float, float] = (0.1, 0.7),
        threshold: float = 0.5,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        center: Optional[int] = None,
        width: Optional[int] = None,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Injects probabilistic drift into a binary/boolean variable.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
            time_col=time_col,
        )

        df_drift = df.copy()

        # Simple probabilistic flip based on 'probability'
        # Or more complex logic if 'center/width' provided for gradual drift
        # For now, let's stick to the simpler version or use the gradual parameters if provided.

        # If gradual parameters are provided, calculate probabilities
        if center is not None and width is not None:
            # Sort rows for gradual progression
            rows = sorted(rows)
            if not rows:
                return df_drift

            probs = self._calculate_drift_probabilities(
                rows=rows,
                center=center,
                width=width,
                profile=profile,
                speed_k=speed_k,
                direction=direction,
                index_min=rows[0],
                index_max=rows[-1],
            )
            # Adjust probabilities: base prob * gradual factor?
            # Or replace base prob? Let's say probs is the probability of flip.
            # But the user also provided 'probability'.
            # Let's assume 'probability' is the max probability of flip.
            probs = probs * probability
        else:
            probs = np.full(len(rows), probability)

        for i, idx in enumerate(rows):
            p = probs[i]
            if self.rng.random() < p:
                # Add noise if float, or flip if binary
                curr_val = df_drift.at[idx, target_col]
                # Assuming binary 0/1 for simplification as per original intent
                if pd.api.types.is_numeric_dtype(df[target_col]) and set(
                    df[target_col].unique()
                ).issubset({0, 1}):
                    df_drift.at[idx, target_col] = 1 - curr_val
                else:
                    # If float (probabilistic output), add noise?
                    # The method name suggests binary drift.
                    pass
                    warnings.warn(
                        f"Skipping binary drift on non-binary column '{target_col}'"
                    )

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_binary_probabilistic_drift",
                "target_col": target_col,
                "probability": probability,
                "start_index": start_index,
                "block_index": block_index,
                "time_start": time_start,
                "generator_name": f"{gen_name}_binary_probabilistic_drift",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config, time_col=time_col)
        return df_drift
