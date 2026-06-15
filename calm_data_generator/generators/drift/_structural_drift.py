"""
Structural drift mixin for DriftInjector.
Contains methods for injecting structural, categorical, functional, and causal drifts.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from calm_data_generator.generators.utils.propagation import apply_func


class _StructuralDriftMixin:
    """Mixin providing structural drift injection methods for DriftInjector."""

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
        import os
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
        means = np.nanmean(X, axis=0)
        stds = np.nanstd(X, axis=0)
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
        import os
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
                from calm_data_generator.reports.QualityReporter import QualityReporter
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
                from calm_data_generator.reports.QualityReporter import QualityReporter
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
