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

import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from calm_data_generator.generators.configs import DriftConfig
from calm_data_generator.generators.utils.propagation import apply_func, propagate_numeric_drift

from ._data_quality_drift import _DataQualityDriftMixin
from ._drift_utils import _DriftUtilsMixin
from ._feature_drift import _FeatureDriftMixin
from ._label_drift import _LabelDriftMixin
from ._structural_drift import _StructuralDriftMixin

logger = logging.getLogger(__name__)

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class DriftInjector(
    _DriftUtilsMixin,
    _FeatureDriftMixin,
    _LabelDriftMixin,
    _DataQualityDriftMixin,
    _StructuralDriftMixin,
):
    """
    A class to inject various types of drift into a pandas DataFrame.
    """

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

        from calm_data_generator.reports.QualityReporter import QualityReporter
        self.reporter = QualityReporter(minimal=minimal_report)

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

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
