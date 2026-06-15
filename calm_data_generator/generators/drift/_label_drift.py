"""
Label drift mixin for DriftInjector.
Contains methods for injecting label drift, label shift, and concept drift.
"""

import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class _LabelDriftMixin:
    """Mixin providing label drift injection methods for DriftInjector."""

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
