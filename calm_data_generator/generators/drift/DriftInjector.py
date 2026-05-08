#!/usr/bin/env python3
"""
Drift Injector for Real Data - Injects various types of drift into real datasets.

This module provides the DriftInjector class, which is designed to introduce a wide range of controlled
drifts into a pandas DataFrame. It supports various drift types, including feature drift, label drift,
and more complex patterns like gradual, abrupt, and recurrent drifts.

Key Features:
- **Multiple Drift Types**: Inject gaussian_noise, shift, scale, and other transformations.
- **Flexible Targeting**: Apply drift to the entire dataset, specific blocks, or row indices.
- **Advanced Drift Profiles**: Simulate gradual, abrupt, incremental, and recurrent drifts using window functions (sigmoid, linear, cosine).
- **Label and Concept Drift**: Includes methods for label flipping (label_drift), changing target distribution (label_shift), and introducing new categories (new_category_drift).
- **Covariate and Virtual Drift**: Modify correlation structures (correlation_matrix_drift) and introduce missing values (missing_values_drift).
- **Integrated Reporting**: Automatically generates detailed reports and visualizations comparing the original and drifted datasets.
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from calm_data_generator.generators.configs import DriftConfig
from calm_data_generator.generators.utils.propagation import apply_func, propagate_numeric_drift
from calm_data_generator.reports.QualityReporter import QualityReporter

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class DriftInjector:
    """
    A class to inject various types of drift into a pandas DataFrame.
    """

    def _resolve_drift_config(
        self, config: Optional[Union[DriftConfig, Dict]] = None, **kwargs
    ) -> DriftConfig:
        """
        Resolves the drift configuration from a DriftConfig object, a dictionary, or kwargs.
        Kwargs override config values.
        """
        if isinstance(config, DriftConfig):
            # Pydantic copy with update
            if kwargs:
                # filter kwargs that are valid fields
                valid_keys = config.model_fields.keys()
                updates = {
                    k: v for k, v in kwargs.items() if k in valid_keys and v is not None
                }
                return config.copy(update=updates)
            return config

        elif isinstance(config, dict):
            # Merge kwargs into dict
            merged = {**config, **{k: v for k, v in kwargs.items() if v is not None}}
            return DriftConfig(**merged)

        else:
            # Create from kwargs
            # Filter out None to let default values take precedence if not provided
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return DriftConfig(**filtered_kwargs)

    # -------------------------
    # Init & utils
    # -------------------------
    def __init__(
        self,
        output_dir: str = "drift_output",
        generator_name: str = "DriftInjector",
        random_state: Optional[int] = None,
        time_col: Optional[str] = None,
        block_column: Optional[str] = None,
        target_column: Optional[str] = None,
        original_df: Optional[pd.DataFrame] = None,
        minimal_report: bool = False,
        auto_report: bool = True,
    ):
        """
        Initializes the DriftInjector.

        Args:
            output_dir (str): Default directory to save reports and drifted datasets.
            generator_name (str): Default name for the generator, used in output file names.
            random_state (Optional[int]): Seed for the random number generator for reproducibility.
            auto_report (bool): If True, automatically generates reports after drift injection.
            minimal_report (bool): If True, generates minimal reports (faster, no correlations/PCA).
        """
        self.rng = np.random.default_rng(random_state)
        self.output_dir = output_dir
        self.generator_name = generator_name
        self.random_state = random_state
        self.time_col = time_col
        self.block_column = block_column
        self.target_column = target_column
        self.auto_report = auto_report
        self.minimal_report = minimal_report

        self.reporter = QualityReporter(minimal=minimal_report)

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _check_column_exists(df: pd.DataFrame, col: str) -> bool:
        """Returns False and is falsy if col is missing from df."""
        return col in df.columns

    @staticmethod
    def _frac(x: float) -> float:
        """Clips a float to the [0.0, 1.0] range."""
        return float(np.clip(x, 0.0, 1.0))

    def _generate_reports(
        self,
        original_df,
        drifted_df,
        drift_config,
        time_col: Optional[str] = None,
        resample_rule: Optional[Union[str, int]] = None,
    ):
        """Helper to generate the standard report."""
        # Generate the primary report in the main output directory
        self.reporter.update_report_after_drift(
            original_df=original_df,
            drifted_df=drifted_df,
            output_dir=self.output_dir,
            drift_config=drift_config,
            time_col=time_col,
            resample_rule=resample_rule,
        )

    _psd_cache: dict = {}

    @classmethod
    def _ensure_psd_matrix(cls, matrix: np.ndarray) -> np.ndarray:
        """Ensures a matrix is positive semi-definite (PSD) by adjusting its eigenvalues."""
        key = matrix.tobytes()
        if key in cls._psd_cache:
            return cls._psd_cache[key]
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues[eigenvalues < 1e-6] = 1e-6  # Clamp small eigenvalues
        psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Renormalize to have 1s on the diagonal
        d = np.sqrt(np.diag(psd_matrix))
        d_inv = np.where(d > 1e-9, 1.0 / d, 0)
        psd_matrix = np.diag(d_inv) @ psd_matrix @ np.diag(d_inv)
        cls._psd_cache[key] = psd_matrix
        return psd_matrix

    def _get_target_rows(
        self,
        df: pd.DataFrame,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        index_step: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        block_step: Optional[int] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        time_step: Optional[Any] = None,
        **kwargs,
    ) -> pd.Index:
        """
        Selects rows for drift injection based on a hierarchy of criteria.
        """
        # Fallback to instance attributes if not provided
        block_column = block_column or self.block_column
        time_col = time_col or self.time_col

        if time_start or time_end or time_ranges or specific_times:
            return self._select_rows_by_time(
                df,
                time_col=time_col,
                time_start=time_start,
                time_end=time_end,
                time_ranges=time_ranges,
                specific_times=specific_times,
                time_step=time_step,
            )
        if blocks is not None or block_start is not None:
            return self._select_rows_by_blocks(
                df,
                block_column=block_column,
                blocks=blocks,
                block_start=block_start,
                n_blocks=n_blocks,
                block_step=block_step,
            )
        if block_index is not None:
            used_block_column = block_column
            if used_block_column not in df.columns:
                raise ValueError(f"Block column '{used_block_column}' not found")
            return df.index[df[used_block_column] == block_index]
        if start_index is not None or end_index is not None:
            return self._select_rows_by_index(df, start_index, end_index, index_step)

        return df.index

    def _select_rows_by_index(
        self,
        df: pd.DataFrame,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: Optional[int] = None,
    ) -> pd.Index:
        """
        Selects rows by index range and step.
        """
        start = start if start is not None else 0
        end = end if end is not None else len(df)
        step = step if step is not None else 1
        return df.iloc[start:end:step].index

    # -------------------------
    # Advanced time selection
    # -------------------------
    def _select_rows_by_time(
        self,
        df: pd.DataFrame,
        time_col: Optional[str],
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        time_step: Optional[Any] = None,
    ) -> pd.Index:
        """
        Selects rows based on time criteria.
        """
        if not time_col or time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' is required")

        time_series = pd.to_datetime(df[time_col])
        mask = pd.Series(False, index=df.index)

        if specific_times:
            mask |= time_series.isin(pd.to_datetime(specific_times))

        if time_ranges:
            for start, end in time_ranges:
                mask |= (time_series >= pd.to_datetime(start)) & (
                    time_series <= pd.to_datetime(end)
                )

        if time_start or time_end:
            start_dt = pd.to_datetime(time_start) if time_start else pd.Timestamp.min
            end_dt = pd.to_datetime(time_end) if time_end else pd.Timestamp.max
            mask |= (time_series >= start_dt) & (time_series <= end_dt)

        if time_step:
            if not (time_start and time_end):
                raise ValueError(
                    "time_start and time_end are required when using time_step"
                )
            step_range = pd.date_range(start=time_start, end=time_end, freq=time_step)
            mask &= time_series.isin(step_range)

        return df.index[mask]

    # -------------------------
    # Advanced block selection
    # -------------------------
    def _select_rows_by_blocks(
        self,
        df: pd.DataFrame,
        block_column: str,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        block_step: Optional[int] = None,
    ) -> pd.Index:
        """
        Selects rows based on block identifiers.
        """
        if not block_column or block_column not in df.columns:
            raise ValueError(f"Block column '{block_column}' is required")

        if blocks:
            return df.index[df[block_column].isin(blocks)]

        if block_start is not None:
            uniq = sorted(df[block_column].dropna().unique())
            if block_start not in uniq:
                warnings.warn(f"block_start '{block_start}' not in '{block_column}'")
                return df.iloc[0:0].index

            i0 = uniq.index(block_start)
            n_blocks = n_blocks if n_blocks is not None else len(uniq) - i0
            block_step = block_step if block_step is not None else 1

            selected_blocks = uniq[i0 : i0 + n_blocks * block_step : block_step]
            return df.index[df[block_column].isin(selected_blocks)]

        return df.iloc[0:0].index

    # -------------------------
    # Windows (profiles + speed)
    # -------------------------
    @staticmethod
    def _sigmoid_weights(n: int, center: float, width: int) -> np.ndarray:
        """
        Creates weights w in [0,1] over n positions with a sigmoid transition.

        Args:
            n (int): Number of positions (rows).
            center (float): The center of the transition (in coordinates 0..n-1).
            width (int): Controls how many rows it takes to go from ~10% to ~90%.

        Returns:
            np.ndarray: An array of weights.
        """
        if n <= 0:
            return np.zeros(0, dtype=float)
        i = np.arange(n, dtype=float)
        width = max(1, int(width))
        # Map width -> sigmoid scale. Approximately 4*scale ~ width (10%->90%)
        scale = width / 4.0
        z = (i - float(center)) / max(1e-9, scale)
        w = 1.0 / (1.0 + np.exp(-z))
        return w

    @staticmethod
    def _window_weights(
        n: int,
        center: float,
        width: int,
        profile: str = "sigmoid",
        k: float = 1.0,
        direction: str = "up",
    ) -> np.ndarray:
        """
        Returns weights w in [0,1] of size n with a transition centered at `center` and of `width`.

        Args:
            n (int): Number of positions.
            center (float): Center of the transition.
            width (int): Width of the transition.
            profile (str): The shape of the transition window ("sigmoid", "linear", "cosine").
            k (float): Controls the "speed" (slope) of the transition.
            direction (str): "up" (0->1) or "down" (1->0).

        Returns:
            np.ndarray: An array of weights.
        """
        if n <= 0:
            return np.zeros(0, dtype=float)

        i = np.arange(n, dtype=float)
        width = max(1, int(width))
        center = float(center)

        if profile == "sigmoid":
            base_scale = width / 4.0
            scale = max(1e-9, base_scale / max(1e-9, float(k)))  # high k -> faster
            z = (i - center) / scale
            w = 1.0 / (1.0 + np.exp(-z))
        elif profile == "linear":
            left = center - width / 2.0
            right = center + width / 2.0
            w = (i - left) / max(1e-9, (right - left))
            w = np.clip(w, 0.0, 1.0)
            if k != 1.0:
                w = np.clip((w - 0.5) * k + 0.5, 0.0, 1.0)
        elif profile == "cosine":
            left = center - width / 2.0
            right = center + width / 2.0
            t = (i - left) / max(1e-9, (right - left))
            t = np.clip(t, 0.0, 1.0)
            w = 0.5 - 0.5 * np.cos(np.pi * t)
            if k != 1.0:
                w = np.clip((w - 0.5) * k + 0.5, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown profile: {profile}")

        if direction == "down":
            w = 1.0 - w

        return w

    def _calculate_drift_probabilities(
        self,
        rows: pd.Index,
        center: int,
        width: int,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        index_min: Optional[int] = None,
        index_max: Optional[int] = None,
    ) -> np.ndarray:
        """
        Calculates drift probabilities for a given set of rows based on window parameters.

        Args:
            rows: The row indices to calculate probabilities for.
            center: Center of the transition window.
            width: Width of the transition window.
            profile: Window profile ("sigmoid", "linear", "cosine").
            speed_k: Speed factor for the transition.
            direction: "up" (0->1) or "down" (1->0).
            index_min: Minimum index for normalization (default: min(rows)).
            index_max: Maximum index for normalization (default: max(rows)).

        Returns:
            np.ndarray: Array of probabilities for each row.
        """
        n = len(rows)
        if n == 0:
            return np.zeros(0, dtype=float)

        # Validate row indices
        if len(rows) == 0:
            return np.zeros(0, dtype=float)

        # Compute window weights based on positions
        return self._window_weights(
            n,
            center=float(center),
            width=int(width),
            profile=profile,
            k=speed_k,
            direction=direction,
        )

    # -------------------------
    # Common engine for features
    # -------------------------
    def _apply_numeric_op_with_weights(
        self,
        values: np.ndarray,
        drift_type: str,
        drift_magnitude: float,
        w: np.ndarray,
        rng: np.random.Generator,
        column_drift_value: Optional[float],
    ) -> np.ndarray:
        """
        Applies a numeric drift operation, scaled by weights `w` row by row.
        """
        x = values.astype(float, copy=True)
        n = len(x)
        if n == 0:
            return x

        mean = float(np.mean(x)) if n > 0 else 0.0
        std = float(np.std(x)) if n > 0 else 0.0
        w = np.asarray(w, dtype=float)

        # Fix for broadcasting error when w is shorter than x
        if len(w) < n:
            w = np.pad(w, (0, n - len(w)), "edge")

        w = np.clip(w, 0.0, 1.0)

        if drift_type == "gaussian_noise":
            if std == 0:
                return x
            noise = rng.normal(0.0, drift_magnitude * std, size=n)
            x = x + noise * w

        elif drift_type == "shift":
            shift_amt = drift_magnitude * mean
            x = x + shift_amt * w

        elif drift_type == "scale":
            # row-wise factor: 1 + w*m
            factor = 1.0 + (w * drift_magnitude)
            x = mean + (x - mean) * factor

        elif drift_type == "add_value":
            if column_drift_value is None:
                raise ValueError("add_value requires drift_value/drift_values[col]")
            x = x + (w * column_drift_value)

        elif drift_type == "subtract_value":
            if column_drift_value is None:
                raise ValueError(
                    "subtract_value requires drift_value/drift_values[col]"
                )
            x = x - (w * column_drift_value)

        elif drift_type == "multiply_value":
            if column_drift_value is None:
                raise ValueError(
                    "multiply_value requires drift_value/drift_values[col]"
                )
            # mix towards the indicated factor: x * (1 + w*(f-1))
            factor = 1.0 + w * (float(column_drift_value) - 1.0)
            x = x * factor

        elif drift_type == "divide_value":
            if column_drift_value is None:
                raise ValueError("divide_value requires drift_value/drift_values[col]")
            if float(column_drift_value) == 0.0:
                raise ValueError("drift_value cannot be zero for 'divide_value'")
            # dividing is equivalent to multiplying by (1/val); we mix towards that factor
            target = 1.0 / float(column_drift_value)
            factor = 1.0 + w * (target - 1.0)
            x = x * factor

        elif drift_type == "uniform_noise":
            # Uniform noise in range [-magnitude*std, +magnitude*std]
            if std == 0:
                return x
            noise = rng.uniform(-drift_magnitude * std, drift_magnitude * std, size=n)
            x = x + noise * w

        else:
            raise ValueError(f"Unknown drift_type: {drift_type}")

        # Preserve original dtype to avoid FutureWarnings
        original_dtype = values.dtype
        if pd.api.types.is_integer_dtype(original_dtype):
            x = np.round(x).astype(original_dtype)

        return x

    def _propagate_numeric_drift(
        self,
        df: pd.DataFrame,
        rows: pd.Index,
        driver_col: str,
        delta_driver: np.ndarray,
        correlations: Union[pd.DataFrame, Dict, bool],
        driver_std: Optional[float] = None,
    ) -> pd.DataFrame:
        """Delegates to the shared propagate_numeric_drift utility."""
        return propagate_numeric_drift(df, rows, driver_col, delta_driver, correlations, driver_std)

    def _apply_categorical_with_weights(
        self,
        original_vals: pd.Series,
        w: np.ndarray,
        drift_magnitude: float,
        rng: np.random.Generator,
    ) -> pd.Series:
        """
        Changes categorical values with a probability per row p = clamp(w * drift_magnitude).
        Replaces the value with a random category different from the current one.
        """
        s = original_vals.copy()
        uniques = s.dropna().unique()
        if len(uniques) <= 1:
            return s

        w = np.clip(np.asarray(w, dtype=float), 0.0, 1.0)
        p = np.clip(w * self._frac(drift_magnitude), 0.0, 1.0)

        # flip a coin for each row
        mask = rng.random(len(s)) < p
        idxs = s.index[mask]
        if len(idxs) == 0:
            return s

        # for each row to change, choose a different category (vectorized by value group)
        current = s.loc[idxs]
        result_vals = current.copy()
        for uval in uniques:
            sub_idx = current.index[current == uval]
            if len(sub_idx) == 0:
                continue
            choices = [u for u in uniques if u != uval]
            if choices:
                result_vals.loc[sub_idx] = rng.choice(choices, size=len(sub_idx))
        s.loc[idxs] = result_vals
        return s

    def _validate_feature_op(self, drift_type: str, drift_magnitude: float):
        """Validates the feature drift operation and its magnitude."""
        if drift_type in {"gaussian_noise", "shift", "scale"} and drift_magnitude < 0:
            raise ValueError(
                "drift_magnitude must be >= 0 for gaussian_noise/shift/scale"
            )
        valid = {
            "gaussian_noise",
            "shift",
            "scale",
            "add_value",
            "subtract_value",
            "multiply_value",
            "divide_value",
            "uniform_noise",
        }
        if drift_type not in valid:
            raise ValueError(f"Unknown drift_type: {drift_type}")

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
    # Feature drift “windowed”: gradual, abrupt, incremental, recurrent
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

    def inject_conditional_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        conditions: List[Dict[str, Any]],
        drift_type: str,
        drift_magnitude: float,
        drift_method: str = "abrupt",  # New parameter: abrupt, gradual, incremental
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        index_step: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        block_step: Optional[int] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        time_step: Optional[Any] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Injects drift (abrupt, gradual, etc.) on a data subset defined by conditions.

        Args:
            df (pd.DataFrame): The input DataFrame.
            feature_cols (List[str]): Columns to apply drift to.
            conditions (List[Dict[str, Any]]): A list of dicts defining filters (e.g. [{"column": "age", "operator": ">", "value": 50}]).
            drift_type (str): Type of numeric/categorical drift.
            drift_magnitude (float): Magnitude of the drift.
            drift_method (str): Method of injection: "abrupt" (default), "gradual", "incremental".
            **kwargs: Additional args passed to the underlying injection method (e.g. center, width, profile for gradual).
        """
        df_drift = df.copy()

        # 1. Select initial candidate rows based on index/block/time
        base_rows = self._get_target_rows(
            df,
            start_index=start_index,
            end_index=end_index,
            index_step=index_step,
            block_index=block_index,
            block_column=block_column,
            blocks=blocks,
            block_start=block_start,
            n_blocks=n_blocks,
            block_step=block_step,
            time_col=time_col,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
            time_step=time_step,
        )

        # 2. Apply filtering conditions
        final_mask = pd.Series(True, index=base_rows)
        for condition in conditions:
            col = condition["column"]
            op = condition["operator"]
            val = condition["value"]

            if col not in df.columns:
                raise ValueError(f"Condition column '{col}' not found in dataframe")

            # Safe access to column series
            series = df.loc[base_rows, col]

            if op == ">":
                final_mask &= series > val
            elif op == ">=":
                final_mask &= series >= val
            elif op == "<":
                final_mask &= series < val
            elif op == "<=":
                final_mask &= series <= val
            elif op == "==":
                final_mask &= series == val
            elif op == "!=":
                final_mask &= series != val
            elif op == "in":
                final_mask &= series.isin(val)
            else:
                raise ValueError(f"Unsupported operator: {op}")

        target_rows_idx = base_rows[final_mask]

        if target_rows_idx.empty:
            warnings.warn("No rows matched the conditions. No drift injected.")
            return df

        # 3. Dynamic Dispatch based on drift_method
        subset_df = df.loc[target_rows_idx].copy()

        # Prepare common kwargs
        method_kwargs = {
            "feature_cols": feature_cols,
            "drift_type": drift_type,
            "drift_magnitude": drift_magnitude,
            **kwargs,
        }

        if drift_method == "abrupt":
            # Uses standard feature drift (supports auto_report)
            method_kwargs["auto_report"] = False
            drifted_subset = self.inject_feature_drift(df=subset_df, **method_kwargs)

        elif drift_method == "gradual":
            # Ensure center/width relate to the subset size if not provided
            if "width" not in kwargs:
                method_kwargs["width"] = len(subset_df)

            drifted_subset = self.inject_feature_drift_gradual(
                df=subset_df, **method_kwargs
            )

        elif drift_method == "incremental":
            drifted_subset = self.inject_feature_drift_incremental(
                df=subset_df, **method_kwargs
            )

        elif drift_method == "recurrent":
            # Recurrent requires 'repeats' parameter
            if "repeats" not in method_kwargs:
                method_kwargs["repeats"] = 2  # Default to 2 cycles
            drifted_subset = self.inject_feature_drift_recurrent(
                df=subset_df, **method_kwargs
            )

        else:
            raise ValueError(
                f"Unknown drift_method: {drift_method}. Use 'abrupt', 'gradual', 'incremental', or 'recurrent'."
            )

        # 4. Update and Report
        df_drift.update(drifted_subset)

        if self.auto_report:
            drift_config = {
                "drift_method": "inject_conditional_drift",
                "sub_method": drift_method,
                "feature_cols": feature_cols,
                "conditions": conditions,
                "drift_type": drift_type,
                "drift_magnitude": drift_magnitude,
                "generator_name": f"{self.generator_name}_conditional_{drift_method}",
                **kwargs,
            }
            self._generate_reports(df, df_drift, drift_config, time_col=self.time_col)

        return df_drift

    # -------------------------
    # Label drift
    # -------------------------
    def inject_label_drift(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        drift_magnitude: float = 0.1,
        drift_magnitudes: Optional[Dict[str, float]] = None,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Injects random label flips for a specified section.

        Args:
            df: Input DataFrame.
            target_cols: List of target columns to apply label drift to.
            drift_magnitude: Fraction of labels to flip (0.0 to 1.0).
            drift_magnitudes: Per-column magnitudes (overrides drift_magnitude).
            start_index, block_index, block_column: Selection parameters.
            time_start, time_end, time_ranges, specific_times: Time-based selection.
            auto_report: Whether to generate a report.
            output_dir: Directory for reports.
            generator_name: Name for reports.

        Returns:
            pd.DataFrame: DataFrame with label drift applied.
        """
        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        if len(rows) == 0:
            return df_drift

        for col in target_cols:
            if col not in df.columns:
                warnings.warn(f"Target column '{col}' not found")
                continue

            col_magnitude = (
                drift_magnitudes.get(col, drift_magnitude)
                if drift_magnitudes
                else drift_magnitude
            )
            col_magnitude = self._frac(col_magnitude)

            unique_labels = df[col].dropna().unique()
            if len(unique_labels) <= 1:
                warnings.warn(f"Column '{col}' has only one unique value, skipping")
                continue

            # Determine number of rows to flip
            n_to_flip = int(len(rows) * col_magnitude)
            if n_to_flip == 0:
                continue

            # Select random rows to flip
            flip_indices = self.rng.choice(rows, size=n_to_flip, replace=False)

            # Flip labels
            for idx in flip_indices:
                current_val = df_drift.at[idx, col]
                # Choose a different label
                possible_vals = [v for v in unique_labels if v != current_val]
                if possible_vals:
                    df_drift.at[idx, col] = self.rng.choice(possible_vals)

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_label_drift",
                "target_cols": target_cols,
                "drift_magnitude": drift_magnitude,
                "start_index": start_index,
                "block_index": block_index,
                "time_start": time_start,
                "generator_name": f"{gen_name}_label_drift",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config)

        return df_drift

    def inject_label_drift_gradual(
        self,
        df: pd.DataFrame,
        target_col: str,
        drift_magnitude: float = 0.3,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
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
    ) -> pd.DataFrame:
        """
        Injects gradual label drift using a transition window.

        The probability of flipping a label increases gradually based on the
        window profile (sigmoid, linear, cosine).

        Args:
            df: Input DataFrame.
            target_col: Target column to apply drift to.
            drift_magnitude: Maximum fraction of labels to flip (0.0 to 1.0).
            start_index, end_index, block_index, block_column: Selection parameters.
            time_start, time_end, time_ranges, specific_times: Time-based selection.
            center: Center of the transition window (default: middle of selected rows).
            width: Width of the transition window (default: n/5).
            profile: Window profile ("sigmoid", "linear", "cosine").
            speed_k: Speed of transition (higher = faster).
            direction: "up" (increasing flip probability) or "down" (decreasing).
            inconsistency: Random noise to add to probabilities.
            auto_report: Whether to generate a report.

        Returns:
            pd.DataFrame: DataFrame with gradual label drift applied.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            end_index=end_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        n = len(rows)
        if n == 0:
            return df_drift

        unique_labels = df[target_col].dropna().unique()
        if len(unique_labels) <= 1:
            warnings.warn(f"Column '{target_col}' has only one unique value")
            return df_drift

        # Calculate transition weights
        c = int(n // 2) if center is None else int(np.clip(center, 0, n - 1))
        w_width = max(1, int(width if width is not None else max(1, n // 5)))
        w = self._window_weights(
            n,
            center=c,
            width=w_width,
            profile=profile,
            k=float(speed_k),
            direction=direction,
        )

        # Add inconsistency noise
        if inconsistency > 0:
            noise = self.rng.normal(0, 0.1 * inconsistency, n)
            w = np.clip(w + noise, 0.0, 1.0)

        # Calculate flip probability for each row
        flip_probs = w * self._frac(drift_magnitude)

        # Apply gradual flips
        for i, idx in enumerate(rows):
            if self.rng.random() < flip_probs[i]:
                current_val = df_drift.at[idx, target_col]
                possible_vals = [v for v in unique_labels if v != current_val]
                if possible_vals:
                    df_drift.at[idx, target_col] = self.rng.choice(possible_vals)

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_label_drift_gradual",
                "target_col": target_col,
                "drift_magnitude": drift_magnitude,
                "profile": profile,
                "center": center,
                "width": width,
                "generator_name": f"{gen_name}_label_gradual",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config)

        return df_drift

    def inject_label_drift_abrupt(
        self,
        df: pd.DataFrame,
        target_col: str,
        drift_magnitude: float,
        change_index: int,
        **kwargs,
    ) -> pd.DataFrame:
        """Wrapper for a very fast gradual drift to simulate an abrupt change."""
        return self.inject_label_drift_gradual(
            df=df,
            target_col=target_col,
            drift_magnitude=drift_magnitude,
            center=change_index,
            width=3,
            speed_k=5.0,
            **kwargs,
        )

    def inject_label_drift_incremental(
        self, df: pd.DataFrame, target_col: str, drift_magnitude: float, **kwargs
    ) -> pd.DataFrame:
        """Applies a constant and smooth label drift over the selected rows."""
        # Extract auto_report before passing kwargs to avoid duplicate
        auto_report = kwargs.pop("auto_report", True)

        rows = self._get_target_rows(df, **kwargs)
        n = len(rows)
        if n == 0:
            return df.copy()

        center = n / 2
        width = n

        return self.inject_label_drift_gradual(
            df=df,
            target_col=target_col,
            drift_magnitude=drift_magnitude,
            center=int(round(center)),
            width=width,
            auto_report=auto_report,
            **kwargs,
        )

    def inject_label_drift_recurrent(
        self,
        df: pd.DataFrame,
        target_col: str,
        drift_magnitude: float,
        windows: List[Tuple[int, int]],
        **kwargs,
    ) -> pd.DataFrame:
        """Applies label drift over a series of explicit windows."""
        df_out = df.copy()
        for center, width in windows:
            df_out = self.inject_label_drift_gradual(
                df=df_out,
                target_col=target_col,
                drift_magnitude=drift_magnitude,
                center=center,
                width=width,
                auto_report=False,
                **kwargs,
            )
        # Final reporting logic
        return df_out

    # -------------------------
    # Target distribution drift
    # -------------------------
    def inject_outliers_global(
        self,
        df: pd.DataFrame,
        cols: List[str],
        outlier_prob: float = 0.05,
        factor: float = 3.0,
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
    ) -> pd.DataFrame:
        """
        Injects global outliers by scaling values by a factor.
        """
        df_drift = df.copy()
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

        for col in cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Randomly select rows to outlier
                mask = self.rng.random(len(rows)) < outlier_prob
                outlier_rows = rows[mask]

                # Apply factor (random sign)
                signs = self.rng.choice([-1, 1], size=len(outlier_rows))
                df_drift.loc[outlier_rows, col] += (
                    factor * df_drift.loc[outlier_rows, col].std() * signs
                )
            else:
                warnings.warn(f"Global outliers failed for column {col}.")

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_outliers_global",
                "cols": cols,
                "outlier_prob": outlier_prob,
                "factor": factor,
                "start_index": start_index,
                "block_index": block_index,
                "time_start": time_start,
                "generator_name": f"{gen_name}_global_outliers",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )

        return df_drift

    # -------------------------
    def inject_new_value(
        self,
        df: pd.DataFrame,
        cols: List[str],
        new_value: Any,  # Or a distribution function
        prob: float = 1.0,
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
        **kwargs,
    ) -> pd.DataFrame:
        """
        Injects a completely new value into a column (Categorical shift).
        """
        df_drift = df.copy()
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
            **kwargs,
        )

        for col in cols:
            if col in df.columns:
                mask = self.rng.random(len(rows)) < prob
                rows_mod = rows[mask]
                df_drift.loc[rows_mod, col] = new_value
            else:
                warnings.warn(f"New value injection failed: {col} not found")

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_new_value",
                "cols": cols,
                "new_value": str(new_value),
                "prob": prob,
                "start_index": start_index,
                "block_index": block_index,
                "time_start": time_start,
                "generator_name": f"{gen_name}_new_value",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config, time_col=time_col)
        return df_drift

    def inject_data_quality_issues(
        self,
        df: pd.DataFrame,
        issues: List[Dict],
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Orchestrates multiple data quality issues (drifts).
        """
        df_drift = df.copy()

        for issue in issues:
            method_name = issue.get("method")
            params = issue.get("params", {})

            # Inject df and context if not present
            if "df" not in params:
                params["df"] = df_drift
            if "time_col" not in params:
                params["time_col"] = time_col
            if "block_column" not in params and block_column:
                params["block_column"] = block_column
            if "auto_report" not in params:
                params["auto_report"] = (
                    False  # Don't report individual steps if orchestrating?
                )
                # Or maybe set to False and report at the end?

            if hasattr(self, method_name):
                method = getattr(self, method_name)
                try:
                    df_drift = method(**params)
                except Exception as e:
                    warnings.warn(f"Failed to apply {method_name}: {e}")
            else:
                print(f"Method {method_name} not found")

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_data_quality_issues",
                "issues": issues,
                "generator_name": f"{gen_name}_data_quality",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config, time_col=time_col)
        return df_drift

    # -------------------------
    def inject_nulls(
        self,
        df: pd.DataFrame,
        cols: List[str],
        prob: float = 0.1,
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
    ) -> pd.DataFrame:
        """
        Injects Nulls/NaNs completely at random (MCAR).
        """
        return self.inject_missing_values_drift(
            df=df,
            feature_cols=cols,
            missing_fraction=prob,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

    # -------------------------
    def inject_label_shift(
        self,
        df: pd.DataFrame,
        target_col: str,
        target_distribution: dict,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Injects label shift by resampling the target column to match a target distribution.

        Args:
            df: Input DataFrame.
            target_col: Target column to modify.
            target_distribution: Dict mapping label values to target proportions.
                                 Example: {0: 0.3, 1: 0.7} for 30% class 0, 70% class 1.
            start_index, block_index, block_column: Selection parameters.
            time_start, time_end, time_ranges, specific_times: Time-based selection.
            auto_report: Whether to generate a report.

        Returns:
            pd.DataFrame: DataFrame with shifted label distribution.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        n = len(rows)
        if n == 0:
            return df_drift

        # Normalize target distribution
        total = sum(target_distribution.values())
        if total <= 0:
            raise ValueError("Target distribution values must sum to > 0")
        normalized_dist = {k: v / total for k, v in target_distribution.items()}

        # Calculate target counts for each label
        target_counts = {k: int(n * v) for k, v in normalized_dist.items()}

        # Adjust for rounding errors - add remainder to largest class
        remainder = n - sum(target_counts.values())
        if remainder > 0:
            max_label = max(normalized_dist, key=normalized_dist.get)
            target_counts[max_label] += remainder

        # Resample: assign labels to rows based on target distribution
        new_labels = []
        for label, count in target_counts.items():
            new_labels.extend([label] * count)

        # Shuffle and assign
        self.rng.shuffle(new_labels)
        df_drift.loc[rows, target_col] = new_labels[:n]

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_label_shift",
                "target_col": target_col,
                "target_distribution": target_distribution,
                "start_index": start_index,
                "block_index": block_index,
                "time_start": time_start,
                "generator_name": f"{gen_name}_label_shift",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config)

        return df_drift

    # -------------------------
    # Target distribution drift
    # -------------------------
    def inject_concept_drift(
        self,
        df: pd.DataFrame,
        concept_drift_type: str = "label_flip",
        concept_drift_magnitude: float = 0.2,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        target_column: Optional[str] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Injects concept drift locally on selected rows.
        """
        if not target_column:
            raise ValueError("target_column must be provided for concept drift.")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_col=time_col,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        df_drift = df.copy()

        # Logic for drift types
        if concept_drift_type == "label_flip":
            # Flip a fraction of labels
            rows_to_flip = self.rng.choice(
                rows, size=int(len(rows) * concept_drift_magnitude), replace=False
            )

            # Assuming binary or categorical. If binary 0/1, just 1-x.
            # If categorical, pick another random category.
            unique_labels = df[target_column].unique()
            if len(unique_labels) == 2 and {0, 1}.issubset(
                unique_labels
            ):  # Binary numeric
                df_drift.loc[rows_to_flip, target_column] = (
                    1 - df_drift.loc[rows_to_flip, target_column]
                )
            else:
                # General categorical flip
                for r in rows_to_flip:
                    current_val = df_drift.at[r, target_column]
                    possible_vals = [v for v in unique_labels if v != current_val]
                    if possible_vals:
                        df_drift.at[r, target_column] = self.rng.choice(possible_vals)

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir

            drift_config = {
                "drift_method": "inject_concept_drift",
                "concept_drift_type": concept_drift_type,
                "magnitude": concept_drift_magnitude,
                "start_index": start_index,
                "block_index": block_index,
                "time_start": time_start,
                "generator_name": f"{gen_name}_concept_drift",
            }

            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config, time_col=time_col)

        return df_drift

    # -------------------------
    # Orchestration / Legacy Support
    # -------------------------
    def inject_multiple_types_of_drift(
        self,
        df: pd.DataFrame,
        schedule: List[Union[Dict[str, Any], DriftConfig]],
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
        time_col: Optional[str] = None,
        block_column: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Applies a sequence of drift injections defined in a schedule.

        Args:
            df (pd.DataFrame): The dataframe to apply drift to.
            schedule (List[Union[Dict, Any], DriftConfig]): A list of drift configurations.
                Each config must have a 'method' key (or be a DriftConfig with a method).
            output_dir (Optional[str]): Override output directory.
            generator_name (Optional[str]): Override generator name.

            # Global Context Overrides (optional)
            time_col (Optional[str]): Timestamp column name.
            block_column (Optional[str]): Block column name.
            target_column (Optional[str]): Target column name.

        Returns:
            pd.DataFrame: The drifted dataframe.
        """
        current_df = df

        for i, config in enumerate(schedule):
            method_name = "inject_feature_drift"
            params = {}
            drift_obj = None

            if isinstance(config, DriftConfig):
                method_name = config.method
                drift_obj = config
                params = config.params.copy() if config.params else {}
            elif isinstance(config, dict):
                method_name = config.get("method")
                params = config.get("params", {}).copy()

            if not method_name or not hasattr(self, method_name):
                warnings.warn(
                    f"Unknown drift method '{method_name}' in schedule at index {i}. Skipping."
                )
                continue

            # Inject global overrides if not present in params
            if output_dir and "output_dir" not in params:
                params["output_dir"] = output_dir
            if generator_name and "generator_name" not in params:
                params["generator_name"] = f"{generator_name}_step_{i}"

            # Inject column context if not present
            if time_col and "time_col" not in params:
                params["time_col"] = time_col
            if block_column and "block_column" not in params:
                params["block_column"] = block_column
            if target_column and "target_column" not in params:
                # Some methods call use 'target_col' instead of 'target_column', handle both
                if "target_col" not in params:
                    params["target_col"] = target_column
                if "target_column" not in params:
                    params["target_column"] = target_column

            # Pass 'df' as the first argument (or keyword arg)
            # Most methods signature: method(df, ...)
            try:
                drift_method = getattr(self, method_name)
                # We pass current_df as the first argument 'df'
                # If params contains 'df', we remove it to avoid double argument
                if "df" in params:
                    del params["df"]

                if drift_obj:
                    # Pass config object if available
                    current_df = drift_method(
                        current_df, drift_config=drift_obj, **params
                    )
                else:
                    # Pass params as kwargs
                    current_df = drift_method(current_df, **params)
            except Exception as e:
                warnings.warn(f"Failed to apply drift '{method_name}': {e}")

        return current_df

    # ------------------------------------------------------------------
    # Pilar 5: Functional Drift & Causal Cascades
    # ------------------------------------------------------------------

    def inject_functional_drift(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        driver_col: str,
        magnitude_func: Union[str, Any],
        magnitude_params: Optional[Dict] = None,
        drift_type: str = "additive",
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        conditions: Optional[List[Dict]] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Injects drift whose magnitude varies per row as a function of driver_col's current value.

        The magnitude of the perturbation applied to each row is computed as:
            magnitude_i = f(driver_col_i)
        where f is specified by magnitude_func and magnitude_params.

        Use case: sensor noise that scales exponentially with temperature:
            inject_functional_drift(
                df, target_cols=["sensor_reading"],
                driver_col="temperature",
                magnitude_func="exponential",
                magnitude_params={"scale": 0.001, "rate": 0.1},
            )

        Args:
            df: Input DataFrame.
            target_cols: Columns to perturb.
            driver_col: Column whose values determine the per-row drift magnitude.
            magnitude_func: Function name ("linear", "exponential", "power", "polynomial")
                            or a callable that takes an ndarray and returns an ndarray.
            magnitude_params: Parameters for magnitude_func (e.g. {"scale": 0.01, "rate": 0.2}).
            drift_type: "additive" (x += magnitude) or "multiplicative" (x *= magnitude).
            start_index, end_index: Row index range (optional).
            time_col, time_start, time_end: Time-based row selection (optional).
            conditions: List of condition dicts for row filtering (optional).
            output_dir: If provided, generate a quality report.
            generator_name: Label for the report.

        Returns:
            Modified DataFrame.
        """
        df_result = df.copy()

        if driver_col not in df_result.columns:
            raise ValueError(f"driver_col '{driver_col}' not found in DataFrame.")

        missing = [c for c in target_cols if c not in df_result.columns]
        if missing:
            raise ValueError(f"target_cols not found in DataFrame: {missing}")

        rows = self._get_target_rows(
            df_result,
            start_index=start_index,
            end_index=end_index,
            time_col=time_col,
            time_start=time_start,
            time_end=time_end,
            **{k: v for k, v in kwargs.items()
               if k in {"block_index", "block_column", "blocks", "time_ranges",
                        "specific_times", "index_step"}},
        )

        # Apply additional conditions filter
        if conditions:
            mask = pd.Series(True, index=rows)
            for cond in conditions:
                col, op, val = cond["column"], cond["operator"], cond["value"]
                series = df_result.loc[rows, col]
                if op == ">":
                    mask &= series > val
                elif op == ">=":
                    mask &= series >= val
                elif op == "<":
                    mask &= series < val
                elif op == "<=":
                    mask &= series <= val
                elif op == "==":
                    mask &= series == val
                elif op == "!=":
                    mask &= series != val
                elif op == "in":
                    mask &= series.isin(val)
            rows = rows[mask]

        driver_values = df_result.loc[rows, driver_col].values
        magnitude = apply_func(magnitude_func, magnitude_params or {}, driver_values)

        if drift_type == "additive":
            df_result.loc[rows, target_cols] = (
                df_result.loc[rows, target_cols].values + magnitude[:, np.newaxis]
            )
        elif drift_type == "multiplicative":
            df_result.loc[rows, target_cols] = (
                df_result.loc[rows, target_cols].values * magnitude[:, np.newaxis]
            )
        else:
            raise ValueError(f"drift_type must be 'additive' or 'multiplicative', got '{drift_type}'.")

        if (output_dir or self.output_dir) and (self.auto_report if output_dir is None else bool(output_dir)):
            _out = output_dir or self.output_dir
            _name = generator_name or self.generator_name or "DriftInjector"
            try:
                reporter = QualityReporter(verbose=True, minimal=self.minimal_report)
                reporter.update_report_after_drift(
                    original_df=df,
                    drifted_df=df_result,
                    output_dir=_out,
                    drift_config={
                        "generator_name": _name,
                        "feature_cols": target_cols,
                        "drift_type": "functional_drift",
                        "driver_col": driver_col,
                        "magnitude_func": str(magnitude_func),
                    },
                    time_col=time_col,
                )
            except Exception as e:
                warnings.warn(f"inject_functional_drift: report failed: {e}")

        return df_result

    def inject_causal_cascade(
        self,
        df: pd.DataFrame,
        dag_config: Dict,
        trigger_col: str,
        trigger_delta: Union[float, Any],
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        conditions: Optional[List[Dict]] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Propagates a perturbation from trigger_col through a DAG of causal dependencies.

        Each node in the DAG can have one or more parent edges with a transfer function
        (linear, exponential, power, polynomial, or callable). The propagation is
        differential: delta_child = f(v_parent + delta_parent) - f(v_parent).

        DAG format:
            {
                "temperature": [],
                "pressure": [
                    {"parent": "temperature", "func": "linear", "params": {"slope": 1.2}}
                ],
                "sensor_fail": [
                    {"parent": "pressure", "func": "exponential",
                     "params": {"scale": 0.001, "rate": 0.3}}
                ],
            }

        Args:
            df: Input DataFrame.
            dag_config: Causal DAG specification (see format above).
            trigger_col: Column that receives the initial perturbation. Must be a DAG node.
            trigger_delta: Scalar or array of perturbation values for trigger_col.
            start_index, end_index: Row index range (optional).
            time_col, time_start, time_end: Time-based row selection (optional).
            conditions: List of condition dicts for row filtering (optional).
            output_dir: If provided, generate a quality report.
            generator_name: Label for the report.

        Returns:
            Modified DataFrame.
        """
        from calm_data_generator.generators.dynamics.CausalEngine import CausalEngine

        df_result = df.copy()

        rows = self._get_target_rows(
            df_result,
            start_index=start_index,
            end_index=end_index,
            time_col=time_col,
            time_start=time_start,
            time_end=time_end,
            **{k: v for k, v in kwargs.items()
               if k in {"block_index", "block_column", "blocks", "time_ranges",
                        "specific_times", "index_step"}},
        )

        if conditions:
            mask = pd.Series(True, index=rows)
            for cond in conditions:
                col, op, val = cond["column"], cond["operator"], cond["value"]
                series = df_result.loc[rows, col]
                if op == ">":
                    mask &= series > val
                elif op == ">=":
                    mask &= series >= val
                elif op == "<":
                    mask &= series < val
                elif op == "<=":
                    mask &= series <= val
                elif op == "==":
                    mask &= series == val
                elif op == "!=":
                    mask &= series != val
                elif op == "in":
                    mask &= series.isin(val)
            rows = rows[mask]

        engine = CausalEngine(dag_config)
        delta = (
            np.full(len(rows), float(trigger_delta))
            if np.isscalar(trigger_delta)
            else np.asarray(trigger_delta, dtype=float)
        )
        engine.apply_cascade(df_result, trigger_col, delta, rows)

        if (output_dir or self.output_dir) and (self.auto_report if output_dir is None else bool(output_dir)):
            _out = output_dir or self.output_dir
            _name = generator_name or self.generator_name or "DriftInjector"
            try:
                affected = [c for c in dag_config if c in df_result.columns]
                reporter = QualityReporter(verbose=True, minimal=self.minimal_report)
                reporter.update_report_after_drift(
                    original_df=df,
                    drifted_df=df_result,
                    output_dir=_out,
                    drift_config={
                        "generator_name": _name,
                        "feature_cols": affected,
                        "drift_type": "causal_cascade",
                        "trigger_col": trigger_col,
                        "trigger_delta": str(trigger_delta),
                    },
                    time_col=time_col,
                )
            except Exception as e:
                warnings.warn(f"inject_causal_cascade: report failed: {e}")

        return df_result

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

    # -------------------------
    # Virtual Drift (Missing Values)
    # -------------------------
    def inject_missing_values_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        missing_fraction: float = 0.1,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Injects missing values (NaN) into specified columns.
        """
        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )
        if len(rows) == 0:
            return df_drift

        for col in feature_cols:
            if col not in df.columns:
                continue

            # Simple random injection for now, can be upgraded to windowed later if needed
            mask = self.rng.random(len(rows)) < missing_fraction
            target_indices = rows[mask]

            df_drift.loc[target_indices, col] = np.nan

        return df_drift

    # -------------------------
    # Covariate Shift
    # -------------------------
    def inject_correlation_matrix_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_correlation_matrix: np.ndarray,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Injects covariate drift by transforming numeric features to match a target correlation matrix.

        Uses Cholesky decomposition to transform standardized features to achieve the
        desired correlation structure.

        Args:
            df: Input DataFrame.
            feature_cols: List of numeric columns to transform.
            target_correlation_matrix: Target correlation matrix (n_features x n_features).
            start_index, block_index, block_column: Selection parameters.
            time_start, time_end, time_ranges, specific_times: Time-based selection.
            auto_report: Whether to generate a report.

        Returns:
            pd.DataFrame: DataFrame with modified correlation structure.
        """
        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        if len(rows) == 0:
            return df_drift

        # Validate feature columns
        valid_cols = [
            c
            for c in feature_cols
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(valid_cols) == 0:
            warnings.warn("No valid numeric columns found for correlation drift")
            return df_drift

        # Validate target matrix shape
        n_features = len(valid_cols)
        if target_correlation_matrix.shape != (n_features, n_features):
            raise ValueError(
                f"Target matrix shape {target_correlation_matrix.shape} doesn't match "
                f"number of features ({n_features})"
            )

        # Ensure target matrix is positive semi-definite
        target_corr = self._ensure_psd_matrix(target_correlation_matrix)

        # Extract and standardize data
        X = df_drift.loc[rows, valid_cols].values.astype(float)
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1.0  # Avoid division by zero
        X_std = (X - means) / stds

        # Cholesky decomposition of target correlation
        try:
            L_target = np.linalg.cholesky(target_corr)
        except np.linalg.LinAlgError:
            warnings.warn("Target matrix not positive definite, using adjusted version")
            target_corr = self._ensure_psd_matrix(
                target_corr + 0.01 * np.eye(n_features)
            )
            L_target = np.linalg.cholesky(target_corr)

        # Current correlation and its Cholesky
        current_corr = np.corrcoef(X_std, rowvar=False)
        if np.isnan(current_corr).any():
            current_corr = np.eye(n_features)
        current_corr = self._ensure_psd_matrix(current_corr)

        try:
            L_current = np.linalg.cholesky(current_corr)
            L_current_inv = np.linalg.inv(L_current)
        except np.linalg.LinAlgError:
            L_current_inv = np.eye(n_features)

        # Transform: X_new = X_std @ L_current_inv.T @ L_target.T
        X_transformed = X_std @ L_current_inv.T @ L_target.T

        # De-standardize
        X_final = X_transformed * stds + means

        df_drift.loc[rows, valid_cols] = X_final

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_correlation_matrix_drift",
                "feature_cols": valid_cols,
                "generator_name": f"{gen_name}_correlation_drift",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config)

        return df_drift

    # -------------------------
    # New Category Drift
    # -------------------------
    def inject_new_category_drift(
        self,
        df: pd.DataFrame,
        feature_col: str,
        new_category: object,
        probability: float = 0.1,
        replace_categories: Optional[List] = None,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
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
        Injects a new category into a feature column.

        This drift simulates the emergence of a new category value that wasn't
        present in the original data (e.g., a new product type, region, etc.).

        Args:
            df: Input DataFrame.
            feature_col: Categorical column to modify.
            new_category: The new category value to introduce.
            probability: Probability of replacing existing values (0.0 to 1.0).
            replace_categories: Optional list of existing categories to replace.
                               If None, any existing category can be replaced.
            start_index, block_index, block_column: Selection parameters.
            time_start, time_end, time_ranges, specific_times: Time-based selection.
            center, width, profile, speed_k, direction: Gradual transition parameters.
            auto_report: Whether to generate a report.

        Returns:
            pd.DataFrame: DataFrame with new category injected.
        """
        if feature_col not in df.columns:
            raise ValueError(f"Feature column '{feature_col}' not found")

        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        n = len(rows)
        if n == 0:
            return df_drift

        # Calculate transition weights if gradual parameters provided
        if center is not None or width is not None:
            c = int(n // 2) if center is None else int(np.clip(center, 0, n - 1))
            w_width = max(1, int(width if width is not None else max(1, n // 5)))
            w = self._window_weights(
                n,
                center=c,
                width=w_width,
                profile=profile,
                k=float(speed_k),
                direction=direction,
            )
            replace_probs = w * self._frac(probability)
        else:
            replace_probs = np.full(n, self._frac(probability))

        # Determine which rows to replace
        for i, idx in enumerate(rows):
            if self.rng.random() < replace_probs[i]:
                current_val = df_drift.at[idx, feature_col]

                # Only replace if current value is in replace_categories (if specified)
                if replace_categories is not None:
                    if current_val not in replace_categories:
                        continue

                df_drift.at[idx, feature_col] = new_category

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_new_category_drift",
                "feature_col": feature_col,
                "new_category": str(new_category),
                "probability": probability,
                "generator_name": f"{gen_name}_new_category_drift",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config)

        return df_drift

    # -------------------------
    # Binary Probabilistic Drift
    # -------------------------
    def inject_concept_drift_gradual(
        self,
        df: pd.DataFrame,
        concept_drift_type: str = "label_flip",
        concept_drift_magnitude: float = 0.2,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        target_col: Optional[str] = None,
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
    ) -> pd.DataFrame:
        """
        Injects probabilistic drift into a binary/boolean variable.

        Logic:
        1. Calculates a temporal weight 'w' (0 to 1) based on the window parameters (sigmoid, linear, etc.).
        2. For each eligible row, with probability p = w * probability:
           - Adds or subtracts a random noise value (from noise_range) to the current binary value (0 or 1).
           - e.g. NewValue = OldValue +/- Noise
        3. Re-binarizes the result: 1 if NewValue > threshold, else 0.

        Args:
            df: Input DataFrame.
            target_col: The binary column to modify.
            probability: The maximum probability that a modification occurs (when temporal weight w=1).
            noise_range: Tuple (min_noise, max_noise) to add/subtract.
            threshold: Threshold to decide the final 0 or 1.
            ... standard selection and window params ...
        """
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found.")

        df_drift = df.copy()

        # 1. Select Target Rows
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
            return df_drift

        # 2. Compute Temporal Weights (w)
        c = int(n // 2) if center is None else int(np.clip(center, 0, n - 1))
        w_width = max(1, int(width if width is not None else max(1, n // 5)))

        w = self._window_weights(
            n,
            center=c,
            width=w_width,
            profile=profile,
            k=float(speed_k),
            direction=direction,
        )

        # 3. Apply Drift
        current_vals = df_drift.loc[rows, target_col].astype(float).values

        # Decide which rows are modified based on probability * w
        # random_draw < w * concept_drift_magnitude
        modification_mask = self.rng.random(n) < (w * concept_drift_magnitude)

        if np.any(modification_mask):
            # For label flipping/binary drift, we simulate it via noise + threshold
            # Default defaults for label flipping equivalent
            noise_range = (-1.0, 1.0)
            threshold = 0.5

            # Generate noise for all, but only use it where modification_mask is True
            noise = self.rng.uniform(noise_range[0], noise_range[1], size=n)

            # Decide sign: + or - (50% chance)
            signs = self.rng.choice([-1, 1], size=n)

            # Apply modifications
            deltas = signs * noise
            # Zero out deltas where we shouldn't modify
            deltas[~modification_mask] = 0.0

            new_vals_numeric = current_vals + deltas

            # Thresholding
            final_vals = (new_vals_numeric > threshold).astype(int)

            df_drift.loc[rows, target_col] = final_vals

    # -------------------------
    # Categorical & Boolean Drift
    # -------------------------
    def inject_categorical_frequency_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_magnitude: float = 0.3,
        perturbation: str = "uniform",  # 'uniform', 'invert', 'random'
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Changes the frequency distribution of categories in a column.
        e.g. Makes rare categories more frequent, or inverts the distribution.
        """
        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_col=time_col,
            **kwargs,
        )

        for col in feature_cols:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found")
                continue

            original_series = df.loc[rows, col]
            if original_series.empty:
                continue

            unique_vals = original_series.unique()
            if len(unique_vals) < 2:
                continue

            # Calculate target probabilities
            n_rows = len(rows)

            if perturbation == "uniform":
                # Tends towards uniform distribution (max entropy)
                probs = np.ones(len(unique_vals)) / len(unique_vals)
            elif perturbation == "random":
                probs = self.rng.random(len(unique_vals))
                probs /= probs.sum()
            elif perturbation == "invert":
                # Inverts current frequencies
                counts = original_series.value_counts(normalize=True).reindex(
                    unique_vals, fill_value=0
                )
                inv_counts = 1.0 - counts
                if inv_counts.sum() == 0:
                    probs = np.ones(len(unique_vals)) / len(unique_vals)
                else:
                    probs = inv_counts / inv_counts.sum()
            else:
                probs = np.ones(len(unique_vals)) / len(unique_vals)

            # Apply drift based on magnitude: mix original distribution with target distribution
            # Implementation: For 'magnitude' % of rows, resample from NEW distribution.

            mask = self.rng.random(n_rows) < drift_magnitude
            n_drift = mask.sum()

            if n_drift > 0:
                new_values = self.rng.choice(unique_vals, size=n_drift, p=probs)
                # Map back to dataframe indices
                drift_indices = rows[mask]
                df_drift.loc[drift_indices, col] = new_values

        if self.auto_report:
            drift_config = {
                "drift_method": "inject_categorical_frequency_drift",
                "feature_cols": feature_cols,
                "drift_magnitude": drift_magnitude,
                "perturbation": perturbation,
                "generator_name": f"{self.generator_name}_freq_drift",
            }
            self._generate_reports(df, df_drift, drift_config, time_col=self.time_col)

        return df_drift

    def inject_typos_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_magnitude: float = 0.1,  # Probability of a row having a typo
        typo_density: int = 1,  # Number of typos per string
        typo_type: str = "random",  # 'swap', 'delete', 'duplicate', 'random'
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Injects typos into string columns.
        """

        def apply_typo(text, n_errors, method):
            if not isinstance(text, str) or len(text) < 2:
                return text

            chars = list(text)
            for _ in range(n_errors):
                if len(chars) < 2:
                    break
                idx = self.rng.integers(0, len(chars))

                op = method
                if op == "random":
                    op = self.rng.choice(["swap", "delete", "duplicate"])

                if op == "swap" and len(chars) > 1:
                    idx2 = (idx + 1) % len(chars)
                    chars[idx], chars[idx2] = chars[idx2], chars[idx]
                elif op == "delete":
                    chars.pop(idx)
                elif op == "duplicate":
                    chars.insert(idx, chars[idx])

            return "".join(chars)

        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_col=time_col,
            **kwargs,
        )

        for col in feature_cols:
            if col not in df.columns:
                continue

            # Filter for rows to affect
            mask = self.rng.random(len(rows)) < drift_magnitude
            target_indices = rows[mask]

            if not target_indices.empty:
                # Vectorized apply is hard for random typos, using list comprehension
                original_vals = df_drift.loc[target_indices, col].astype(str).tolist()
                drifted_vals = [
                    apply_typo(x, typo_density, typo_type) for x in original_vals
                ]
                df_drift.loc[target_indices, col] = drifted_vals

        if self.auto_report:
            drift_config = {
                "drift_method": "inject_typos_drift",
                "feature_cols": feature_cols,
                "drift_magnitude": drift_magnitude,
                "typo_type": typo_type,
                "generator_name": f"{self.generator_name}_typos",
            }
            self._generate_reports(df, df_drift, drift_config, time_col=self.time_col)

        return df_drift

    def inject_category_merge_drift(
        self,
        df: pd.DataFrame,
        col: str,
        categories_to_merge: List[Any],
        new_category_name: Any,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Merges specific categories into one (e.g. 'Cat' + 'Dog' -> 'Pet').
        """
        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_col=time_col,
            **kwargs,
        )

        if col in df.columns:
            mask = df_drift.loc[rows, col].isin(categories_to_merge)
            drift_indices = rows[mask]
            df_drift.loc[drift_indices, col] = new_category_name

        if self.auto_report:
            drift_config = {
                "drift_method": "inject_category_merge_drift",
                "col": col,
                "merged": str(categories_to_merge),
                "new_name": str(new_category_name),
                "generator_name": f"{self.generator_name}_merge",
            }
            self._generate_reports(df, df_drift, drift_config, time_col=self.time_col)

        return df_drift

    def inject_boolean_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_magnitude: float = 0.3,  # Probability of flipping
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Specific drift for boolean columns: flips True <-> False.
        """
        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_col=time_col,
            **kwargs,
        )

        for col in feature_cols:
            if col not in df.columns:
                continue

            # Check if boolean-like
            if not pd.api.types.is_bool_dtype(df[col]) and not set(
                df[col].dropna().unique()
            ).issubset({0, 1, 0.0, 1.0, True, False}):
                warnings.warn(
                    f"Column '{col}' does not appear to be boolean. Skipping."
                )
                continue

            mask = self.rng.random(len(rows)) < drift_magnitude
            flip_indices = rows[mask]

            if not flip_indices.empty:
                # Logical NOT for booleans
                if pd.api.types.is_bool_dtype(df[col]):
                    df_drift.loc[flip_indices, col] = ~df_drift.loc[flip_indices, col]
                else:
                    # For 0/1 integers
                    df_drift.loc[flip_indices, col] = (
                        1 - df_drift.loc[flip_indices, col]
                    )

        if self.auto_report:
            drift_config = {
                "drift_method": "inject_boolean_drift",
                "feature_cols": feature_cols,
                "drift_magnitude": drift_magnitude,
                "generator_name": f"{self.generator_name}_boolean",
            }
            self._generate_reports(df, df_drift, drift_config, time_col=self.time_col)

        return df_drift

    # -------------------------
    # Unified Drift Injection API
    # -------------------------
    def inject_drift(
        self,
        df: pd.DataFrame,
        columns: List[str],
        drift_magnitude: float = 0.3,
        drift_mode: str = "abrupt",
        # Operations per column type (auto-detect if None)
        numeric_operation: Optional[str] = None,
        categorical_operation: Optional[str] = None,
        boolean_operation: Optional[str] = None,
        # Gradual/incremental mode parameters
        center: Optional[int] = None,
        width: Optional[int] = None,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        # Recurrent mode parameters
        repeats: int = 3,
        windows: Optional[List[Tuple[int, int]]] = None,
        # Row selection parameters
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        blocks: Optional[Sequence] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        # Conditional drift
        conditions: Optional[List[Dict]] = None,
        # Output
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
        auto_report: Optional[bool] = None,
        resample_rule: Optional[Union[str, int]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Unified drift injection method that auto-detects column types and applies
        appropriate drift operations.

        Args:
            df: Input DataFrame.
            columns: List of columns to apply drift to (any type).
            drift_magnitude: Intensity of the drift (0.0 to 1.0).
            drift_mode: Type of drift pattern:
                - 'abrupt': Immediate change from start_index
                - 'gradual': Smooth transition using window function
                - 'incremental': Constant smooth drift over entire range
                - 'recurrent': Multiple drift windows
            numeric_operation: Operation for numeric columns. Default: 'shift'.
                Options: 'shift', 'scale', 'gaussian_noise', 'uniform_noise',
                         'add_value', 'subtract_value', 'multiply_value'
            categorical_operation: Operation for categorical columns. Default: 'frequency'.
                Options: 'frequency', 'typos', 'new_category'
            boolean_operation: Operation for boolean columns. Default: 'flip'.
                Options: 'flip'
            center: Center of transition window (for gradual mode).
            width: Width of transition window (for gradual mode).
            profile: Transition profile: 'sigmoid', 'linear', 'cosine'.
            speed_k: Speed factor for transition.
            direction: 'up' (0->1) or 'down' (1->0) for transition.
            repeats: Number of drift windows (for recurrent mode).
            windows: Explicit windows as [(start, end), ...] (for recurrent mode).
            start_index, end_index: Row index range for drift.
            block_index, block_column, blocks: Block-based selection.
            time_col, time_start, time_end, time_ranges, specific_times: Time-based selection.
            conditions: List of conditions for conditional drift.
            output_dir: Output directory for reports.
            generator_name: Name for generated files.
            auto_report: Override instance auto_report setting.

        Returns:
            DataFrame with drift applied.

        Example:
            >>> drifted = injector.inject_drift(
            ...     df=data,
            ...     columns=['age', 'income', 'is_active', 'gender'],
            ...     drift_mode='gradual',
            ...     drift_magnitude=0.3,
            ...     center=500,
            ...     width=100,
            ... )
        """
        valid_modes = {"abrupt", "gradual", "incremental", "recurrent"}
        if drift_mode not in valid_modes:
            raise ValueError(f"Invalid drift_mode '{drift_mode}'. Valid: {valid_modes}")

        # Auto-detect column types
        numeric_cols = []
        categorical_cols = []
        boolean_cols = []

        for col in columns:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found in DataFrame, skipping.")
                continue

            dtype = df[col].dtype
            n_unique = df[col].nunique()

            # Boolean detection: bool dtype or exactly 2 unique values
            if pd.api.types.is_bool_dtype(dtype) or (
                n_unique == 2 and dtype in [np.int64, np.int32, object]
            ):
                boolean_cols.append(col)
            # Categorical detection: object or category dtype
            elif pd.api.types.is_object_dtype(dtype) or isinstance(
                dtype, pd.CategoricalDtype
            ):
                categorical_cols.append(col)
            # Numeric detection: numeric dtypes
            elif pd.api.types.is_numeric_dtype(dtype):
                numeric_cols.append(col)
            else:
                warnings.warn(
                    f"Column '{col}' has unknown dtype {dtype}, treating as categorical."
                )
                categorical_cols.append(col)

        # Set default operations if not specified
        numeric_op = numeric_operation or "shift"
        categorical_op = categorical_operation or "frequency"
        boolean_op = boolean_operation or "flip"

        df_drift = df.copy()
        should_report = auto_report if auto_report is not None else self.auto_report
        original_auto_report = self.auto_report
        self.auto_report = (
            False  # Disable per-method reports, we'll generate one at the end
        )

        try:
            # Filter kwargs to avoid duplicate parameter errors
            # Remove parameters that are already explicitly passed to _apply_drift_by_mode
            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                not in {
                    "drift_type",
                    "drift_magnitude",
                    "drift_mode",
                    "center",
                    "width",
                    "profile",
                    "speed_k",
                    "direction",
                    "repeats",
                    "windows",
                    "start_index",
                    "end_index",
                    "block_index",
                    "block_column",
                    "blocks",
                    "time_col",
                    "time_start",
                    "time_end",
                    "time_ranges",
                    "specific_times",
                    "conditions",
                    "feature_cols",
                }
            }

            # Apply drift to numeric columns
            if numeric_cols:
                df_drift = self._apply_drift_by_mode(
                    df=df_drift,
                    feature_cols=numeric_cols,
                    drift_type=numeric_op,
                    drift_magnitude=drift_magnitude,
                    drift_mode=drift_mode,
                    center=center,
                    width=width,
                    profile=profile,
                    speed_k=speed_k,
                    direction=direction,
                    repeats=repeats,
                    windows=windows,
                    start_index=start_index,
                    end_index=end_index,
                    block_index=block_index,
                    block_column=block_column,
                    blocks=blocks,
                    time_col=time_col,
                    time_start=time_start,
                    time_end=time_end,
                    time_ranges=time_ranges,
                    specific_times=specific_times,
                    conditions=conditions,
                    **filtered_kwargs,
                )

            # Apply drift to categorical columns
            if categorical_cols:
                if categorical_op == "frequency":
                    for col in categorical_cols:
                        df_drift = self.inject_categorical_frequency_drift(
                            df=df_drift,
                            feature_cols=[col],
                            drift_magnitude=drift_magnitude,
                            start_index=start_index,
                            block_index=block_index,
                            block_column=block_column,
                            time_col=time_col,
                            **kwargs,
                        )
                elif categorical_op == "new_category":
                    for col in categorical_cols:
                        new_cat = kwargs.get("new_category", f"NEW_{col.upper()}")
                        df_drift = self.inject_new_category_drift(
                            df=df_drift,
                            feature_col=col,
                            new_category=new_cat,
                            probability=drift_magnitude,
                            start_index=start_index,
                            block_column=block_column,
                            center=center,
                            width=width,
                            profile=profile,
                            **kwargs,
                        )
                elif categorical_op == "typos":
                    # Use the apply_categorical_with_weights for typo-like changes
                    rows = self._get_target_rows(
                        df_drift,
                        start_index=start_index,
                        end_index=end_index,
                        block_index=block_index,
                        block_column=block_column,
                        time_col=time_col,
                        time_start=time_start,
                        time_end=time_end,
                        **kwargs,
                    )
                    w = np.ones(len(rows), dtype=float)
                    for col in categorical_cols:
                        s = df_drift.loc[rows, col]
                        s2 = self._apply_categorical_with_weights(
                            s, w, drift_magnitude, self.rng
                        )
                        df_drift.loc[rows, col] = s2

            # Apply drift to boolean columns
            if boolean_cols:
                if boolean_op == "flip":
                    df_drift = self._apply_drift_by_mode(
                        df=df_drift,
                        feature_cols=boolean_cols,
                        drift_type="flip",
                        drift_magnitude=drift_magnitude,
                        drift_mode=drift_mode,
                        center=center,
                        width=width,
                        profile=profile,
                        speed_k=speed_k,
                        direction=direction,
                        repeats=repeats,
                        windows=windows,
                        start_index=start_index,
                        end_index=end_index,
                        block_index=block_index,
                        block_column=block_column,
                        blocks=blocks,
                        time_col=time_col,
                        time_start=time_start,
                        time_end=time_end,
                        time_ranges=time_ranges,
                        specific_times=specific_times,
                        conditions=conditions,
                        is_boolean=True,
                        **filtered_kwargs,
                    )

        finally:
            self.auto_report = original_auto_report

        # Generate unified report
        if should_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir

            drift_config = {
                "drift_method": "inject_drift",
                "drift_mode": drift_mode,
                "columns": columns,
                "column_types": {
                    "numeric": numeric_cols,
                    "categorical": categorical_cols,
                    "boolean": boolean_cols,
                },
                "drift_magnitude": drift_magnitude,
                "operations": {
                    "numeric": numeric_op,
                    "categorical": categorical_op,
                    "boolean": boolean_op,
                },
                "start_index": start_index,
                "center": center,
                "width": width,
                "profile": profile,
                "generator_name": f"{gen_name}_unified_drift",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(
                    df,
                    df_drift,
                    drift_config,
                    time_col=time_col,
                    resample_rule=resample_rule,
                )

        return df_drift

    def _apply_drift_by_mode(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_type: str,
        drift_magnitude: float,
        drift_mode: str,
        center: Optional[int] = None,
        width: Optional[int] = None,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        repeats: int = 3,
        windows: Optional[List[Tuple[int, int]]] = None,
        is_boolean: bool = False,
        conditions: Optional[List[Dict]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Internal method to apply drift based on drift_mode.
        Routes to the appropriate specialized method.
        """
        # Handle boolean columns specially
        if is_boolean:
            if drift_mode == "abrupt":
                start_idx = kwargs.get("start_index")
                if start_idx is None:
                    start_idx = 0
                # Filter kwargs to remove arguments not supported by inject_label_drift_abrupt/gradual
                # Specifically block/time args that are handled elsewhere or not supported for label drift yet
                clean_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in [
                        "start_index",
                        "blocks",
                        "block_index",
                        "block_column",
                        "time_start",
                        "time_end",
                        "time_ranges",
                        "specific_times",
                        "conditions",
                        "time_col",
                        "resample_rule",
                    ]
                }
                for col in feature_cols:
                    df = self.inject_label_drift_abrupt(
                        df=df,
                        target_col=col,
                        drift_magnitude=drift_magnitude,
                        change_index=start_idx,
                        **clean_kwargs,
                    )
            elif drift_mode == "gradual":
                for col in feature_cols:
                    df = self.inject_binary_probabilistic_drift(
                        df=df,
                        target_col=col,
                        probability=drift_magnitude,
                        mode="gradual",
                        center=center,
                        width=width,
                        profile=profile,
                        speed_k=speed_k,
                        direction=direction,
                        **kwargs,
                    )
            elif drift_mode == "incremental":
                for col in feature_cols:
                    df = self.inject_label_drift_incremental(
                        df=df,
                        target_col=col,
                        drift_magnitude=drift_magnitude,
                        **kwargs,
                    )
            elif drift_mode == "recurrent":
                win = windows or self._generate_recurrent_windows(len(df), repeats)
                for col in feature_cols:
                    df = self.inject_label_drift_recurrent(
                        df=df,
                        target_col=col,
                        drift_magnitude=drift_magnitude,
                        windows=win,
                        **kwargs,
                    )
            return df

        # Handle conditional drift
        if conditions:
            return self.inject_conditional_drift(
                df=df,
                feature_cols=feature_cols,
                conditions=conditions,
                drift_type=drift_type,
                drift_magnitude=drift_magnitude,
                mode=drift_mode,
                center=center,
                width=width,
                profile=profile,
                **kwargs,
            )

        # Handle numeric/categorical columns
        if drift_mode == "abrupt":
            return self.inject_feature_drift(
                df=df,
                feature_cols=feature_cols,
                drift_type=drift_type,
                drift_magnitude=drift_magnitude,
                **kwargs,
            )
        elif drift_mode == "gradual":
            # Calculate center and width if not provided
            start_idx = kwargs.get("start_index")
            if start_idx is None:
                start_idx = 0
            n_rows = len(df) - start_idx
            calc_center = center if center is not None else start_idx + n_rows // 2
            calc_width = width if width is not None else n_rows // 4
            return self.inject_feature_drift_gradual(
                df=df,
                feature_cols=feature_cols,
                drift_type=drift_type,
                drift_magnitude=drift_magnitude,
                center=calc_center,
                width=calc_width,
                profile=profile,
                speed_k=speed_k,
                direction=direction,
                **kwargs,
            )
        elif drift_mode == "incremental":
            return self.inject_feature_drift_incremental(
                df=df,
                feature_cols=feature_cols,
                drift_type=drift_type,
                drift_magnitude=drift_magnitude,
                **kwargs,
            )
        elif drift_mode == "recurrent":
            win = windows or self._generate_recurrent_windows(len(df), repeats)
            return self.inject_feature_drift_recurrent(
                df=df,
                feature_cols=feature_cols,
                drift_type=drift_type,
                drift_magnitude=drift_magnitude,
                windows=win,
                **kwargs,
            )

        return df

    def _generate_recurrent_windows(
        self, n_rows: int, repeats: int, duty_cycle: float = 0.3
    ) -> List[Tuple[int, int]]:
        """Generates evenly spaced windows for recurrent drift."""
        windows = []
        segment_size = n_rows // repeats
        window_size = int(segment_size * duty_cycle)

        for i in range(repeats):
            start = i * segment_size + segment_size // 4
            end = min(start + window_size, n_rows - 1)
            if start < end:
                windows.append((start, end))

        return windows
