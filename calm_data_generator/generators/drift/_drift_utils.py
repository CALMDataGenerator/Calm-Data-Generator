"""
Private utility mixin for DriftInjector.
Contains helper methods shared across all drift injection strategies.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from calm_data_generator.generators.configs import DriftConfig
from calm_data_generator.generators.utils.propagation import propagate_numeric_drift


class _DriftUtilsMixin:
    """Mixin providing internal helper utilities for DriftInjector."""

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
                valid_keys = type(config).model_fields.keys()
                updates = {
                    k: v for k, v in kwargs.items() if k in valid_keys and v is not None
                }
                return config.model_copy(update=updates)
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
