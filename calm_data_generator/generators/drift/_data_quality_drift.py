"""
Data quality drift mixin for DriftInjector.
Contains methods for injecting data quality issues, nulls, outliers, and conditional drift.
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class _DataQualityDriftMixin:
    """Mixin providing data quality drift injection methods for DriftInjector."""

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
        import logging
        logger = logging.getLogger(__name__)

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
                logger.warning("Method %s not found", method_name)

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
