import os
from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd

from calm_data_generator.generators.configs import (
    EvolutionFeatureConfig,
    ScenarioConfig,
)
from calm_data_generator.generators.utils.propagation import apply_func, propagate_numeric_drift


class ScenarioInjector:
    """
    A standalone module to modify scenarios by evolving features and constructing target variables.
    """

    def __init__(self, seed: Optional[int] = None, minimal_report: bool = False):
        self.rng = np.random.default_rng(seed)
        self.minimal_report = minimal_report

    def _resolve_evolution_feature_configs(
        self,
        evolution_config: Optional[Dict] = None,
        scenario_config: Optional[Union[ScenarioConfig, Dict]] = None,
    ) -> Dict[str, EvolutionFeatureConfig]:
        """
        Resolves evolution configuration into a dictionary of EvolutionFeatureConfig objects.
        """
        configs: Dict[str, EvolutionFeatureConfig] = {}

        # 1. Load from ScenarioConfig
        if scenario_config:
            if isinstance(scenario_config, ScenarioConfig):
                # Copy dict to avoid mutation of original if needed
                for col, conf in scenario_config.evolve_features.items():
                    # Handle if conf is dict inside ScenarioConfig (if deserialized partially)
                    if isinstance(conf, dict):
                        configs[col] = EvolutionFeatureConfig(**conf)
                    else:
                        configs[col] = conf
            elif isinstance(scenario_config, dict):
                # Extract evolve_features dict
                raw_evolve = scenario_config.get("evolve_features", {})
                for col, conf in raw_evolve.items():
                    if isinstance(conf, dict):
                        configs[col] = EvolutionFeatureConfig(**conf)
                    elif isinstance(conf, EvolutionFeatureConfig):
                        configs[col] = conf

        # 2. explicit evolution_config (overrides)
        if evolution_config:
            for col, conf in evolution_config.items():
                if isinstance(conf, EvolutionFeatureConfig):
                    configs[col] = conf
                elif isinstance(conf, dict):
                    configs[col] = EvolutionFeatureConfig(**conf)

        return configs

    def evolve_features(
        self,
        df: pd.DataFrame,
        evolution_config: Optional[
            Union[Dict, Dict[str, EvolutionFeatureConfig]]
        ] = None,
        time_col: Optional[str] = None,
        output_dir: Optional[str] = None,
        auto_report: bool = True,
        generator_name: str = "ScenarioInjector",
        resample_rule: Optional[Union[str, int]] = None,
        correlations: Optional[Union[pd.DataFrame, Dict, bool]] = None,
        scenario_config: Optional[Union[ScenarioConfig, Dict]] = None,
    ) -> pd.DataFrame:
        """
        Evolves features in the DataFrame based on the provided configuration.
        """
        df_evolved = df.copy()

        if time_col:
            if time_col not in df.columns:
                raise ValueError(f"Time column '{time_col}' not found in DataFrame.")
            t = df[time_col].values
            # Ensure t is numeric
            if not np.issubdtype(t.dtype, np.number):
                # Try to convert to numeric if it's datetime
                if np.issubdtype(t.dtype, np.datetime64):
                    t = t.astype(np.int64) // 10**9  # Seconds
                else:
                    # Fallback to range
                    t = np.arange(len(df))
        else:
            t = np.arange(len(df))

        evolution_configs = self._resolve_evolution_feature_configs(
            evolution_config, scenario_config
        )

        for col, config in evolution_configs.items():
            if col not in df_evolved.columns:
                continue  # Or raise warning

            drift_type = config.type

            delta = np.zeros_like(t, dtype=float)

            if drift_type in ["linear", "trend"]:
                slope = config.slope if config.slope is not None else 0.0
                intercept = config.intercept if config.intercept is not None else 0.0
                delta = slope * t + intercept

            elif drift_type in ["cycle", "sinusoidal", "cyclic", "seasonal"]:
                period = config.period if config.period is not None else 100.0
                amplitude = config.amplitude if config.amplitude is not None else 1.0
                phase = config.phase if config.phase is not None else 0.0
                delta = amplitude * np.sin(2 * np.pi * t / period + phase)

            elif drift_type == "sigmoid":
                center = config.center if config.center is not None else len(t) / 2
                width = config.width if config.width is not None else len(t) / 10
                amplitude = config.amplitude if config.amplitude is not None else 1.0
                z = (t - center) / (width + 1e-9)
                sigmoid = 1.0 / (1.0 + np.exp(-z))
                delta = amplitude * sigmoid

            elif drift_type == "exponential_growth":
                rate = config.rate if config.rate is not None else 0.01
                # y = x * (1 + rate)^t  => delta = x * ((1+rate)^t - 1)
                # But here we apply delta + INITIAL values.
                # To be consistent with additive logic, we calculate delta from base
                base_values = df_evolved[col].values
                delta = base_values * ((1 + rate) ** t - 1)

            elif drift_type == "decay":
                rate = config.rate if config.rate is not None else 0.01
                base_values = df_evolved[col].values
                delta = base_values * ((1 - rate) ** t - 1)

            elif drift_type == "step":
                step_val = (
                    config.step if config.step is not None else len(t) / 2
                )
                amplitude = (
                    config.amplitude if config.amplitude is not None else 1.0
                )
                delta = np.where(t >= step_val, amplitude, 0.0)

            elif drift_type == "noise":
                scale = (
                    config.noise_std
                    if config.noise_std is not None
                    else (config.amplitude if config.amplitude is not None else 1.0)
                )
                delta = self.rng.normal(0, scale, size=len(t))

            elif drift_type == "random_walk":
                step_std = (
                    config.step_std if config.step_std is not None else 0.1
                )
                steps = self.rng.normal(0, step_std, size=len(t))
                delta = np.cumsum(steps)

            elif drift_type == "driven_by":
                driver_col = getattr(config, "driver_col", None)
                if not driver_col or driver_col not in df_evolved.columns:
                    raise ValueError(
                        f"evolve_type 'driven_by' for column '{col}' requires a valid "
                        f"'driver_col' pointing to an existing DataFrame column. "
                        f"Got driver_col={driver_col!r}."
                    )
                driver_values = df_evolved[driver_col].values.astype(float)
                func = getattr(config, "func", None) or "linear"
                func_params = getattr(config, "func_params", None) or {}
                delta = apply_func(func, func_params, driver_values)

            # Apply delta
            # Assuming additive drift for now. Could add 'mode': 'multiplicative' later.
            # Calculate current values before update for delta calculation if needed
            # But here delta is already the change vector for the whole column 't'

            # Pre-calculate std for propagation BEFORE applying delta to df
            driver_std = float(df_evolved[col].std())

            # Apply to main column
            df_evolved[col] = df_evolved[col] + delta

            # Propagate to correlated columns
            if correlations is not None and correlations is not False:
                self._propagate_numeric_drift(
                    df_evolved,
                    df_evolved.index,
                    col,
                    delta,
                    correlations,
                    driver_std=driver_std,
                )

        if auto_report and output_dir:
            from calm_data_generator.reports.QualityReporter import (
                QualityReporter,
            )

            reporter = QualityReporter(verbose=True, minimal=self.minimal_report)
            affected_cols = list(evolution_configs.keys())

            # Dump for reporting
            evolution_config_dump = {k: v.model_dump() for k, v in evolution_configs.items()}

            drift_config = {
                "generator_name": generator_name,
                "feature_cols": affected_cols,
                "drift_type": "Scenario Evolution",
                "drift_magnitude": "See evolution_config",
                "affected_columns": ", ".join(affected_cols),
                "evolution_config": evolution_config_dump,
            }
            # Create output dir if needed
            os.makedirs(output_dir, exist_ok=True)

            # Use the new updated report method which focuses on feature_cols
            reporter.update_report_after_drift(
                original_df=df,
                drifted_df=df_evolved,
                output_dir=output_dir,
                drift_config=drift_config,
                time_col=time_col,
                resample_rule=resample_rule,
            )

            # Generate evolution-specific visualization
            from calm_data_generator.reports.Visualizer import Visualizer

            Visualizer.generate_evolution_plot(
                original_df=df,
                evolved_df=df_evolved,
                evolution_config=evolution_config,
                output_dir=output_dir,
                time_col=time_col,
            )

        return df_evolved

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

    def construct_target(
        self,
        df: pd.DataFrame,
        target_col: str,
        formula: Union[str, Callable],
        noise_std: float = 0.0,
        task_type: str = "regression",
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Constructs or overwrites a target variable based on a formula.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str): Name of the target column to create/overwrite.
            formula (str or callable): Formula to calculate the raw target score.
                - If str: Used in df.eval(). Example: "0.5 * Age + 2.0 * Income"
                - If callable: Accepts df and returns a Series/array.
            noise_std (float): Standard deviation of Gaussian noise to add.
            task_type (str): 'regression' or 'classification'.
            threshold (float, optional): Threshold for binary classification.
                                         If None and task_type='classification', defaults to 0 (assuming raw score is logit-like)
                                         or requires sigmoid probability sampling.
                                         Here we implement simple thresholding: Y = 1 if Score > threshold else 0.

        Returns:
            pd.DataFrame: DataFrame with the new target column.
        """
        df_target = df.copy()

        # 1. Calculate Raw Score
        if isinstance(formula, str):
            # Validate the formula before calling eval().
            # Only allow identifiers that are actual column names plus arithmetic
            # operators and numeric literals — this prevents arbitrary code execution.
            import re as _re
            _allowed_pattern = _re.compile(
                r"^[\w\s\.\+\-\*\/\(\)\,\d\.\>\<\=\!]+$"
            )
            if not _allowed_pattern.match(formula):
                raise ValueError(
                    f"Formula '{formula}' contains disallowed characters. "
                    "Only column names, numeric literals, and arithmetic operators "
                    "(+, -, *, /, parentheses, inequalities) are permitted."
                )
            # Additionally verify that every identifier in the formula is an
            # existing column name or a numeric token, not an arbitrary name.
            _tokens = _re.findall(r"[A-Za-z_]\w*", formula)
            _unknown = [t for t in _tokens if t not in df_target.columns]
            if _unknown:
                raise ValueError(
                    f"Formula references unknown identifiers: {_unknown}. "
                    f"Available columns: {list(df_target.columns)}"
                )
            try:
                raw_score = df_target.eval(formula, engine="python")
            except Exception as e:
                raise ValueError(f"Error evaluating formula '{formula}': {e}")
        elif callable(formula):
            raw_score = formula(df_target)
        else:
            raise ValueError("Formula must be a string or a callable.")

        # Ensure raw_score is numeric
        raw_score = np.array(raw_score, dtype=float)

        # 2. Add Noise
        if noise_std > 0:
            noise = self.rng.normal(0, noise_std, size=len(df_target))
            raw_score += noise

        # 3. Finalize Target based on Task Type
        if task_type == "regression":
            df_target[target_col] = raw_score

        elif task_type == "classification":
            if threshold is None:
                # Default threshold 0 (e.g. if formula outputs log-odds or centered score)
                # Or we could use mean/median if we wanted balanced classes dynamically?
                # Let's stick to 0.0 as default for explicit formulas.
                threshold = 0.0

            df_target[target_col] = (raw_score > threshold).astype(int)

        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        return df_target

    def project_to_future_period(
        self,
        df: pd.DataFrame,
        periods: int = 1,
        trend_config: Optional[Dict[str, Dict]] = None,
        time_col: Optional[str] = None,
        block_col: Optional[str] = None,
        base_strategy: str = "last_period",
        window_size: int = 1,
        generator: Optional[object] = None,
        generator_method: str = "ctgan",
        n_samples_per_period: Optional[int] = None,
        output_dir: Optional[str] = None,
        auto_report: bool = True,
        generator_name: str = "FutureProjection",
    ) -> pd.DataFrame:
        """
        Projects data to future periods by generating synthetic data and applying trends.

        Args:
            df: Input DataFrame with historical data.
            periods: Number of future periods to generate.
            trend_config: Dict mapping feature -> trend params.
                Example: {"price": {"type": "linear", "slope": 0.03}}
            time_col: Column containing timestamps.
            block_col: Column containing block/period identifiers.
            base_strategy: How to select base data for generation:
                - 'last_period': Only last block/time period
                - 'all': Use entire dataset
                - 'window_n': Use last N periods (uses window_size)
            window_size: N for window_n strategy.
            generator: RealGenerator instance for synthetic data generation (optional).
            generator_method: Method to use with generator ('ctgan', 'tvae', etc).
            n_samples_per_period: Number of samples per projected period.
                If None, uses same count as base data.
            output_dir: Directory for report output.
            auto_report: Whether to generate drift analysis report.
            generator_name: Name for labeling the generator.

        Returns:
            DataFrame with original + projected data.
        """
        # 1. Select base data based on strategy
        if block_col and block_col in df.columns:
            unique_blocks = sorted(df[block_col].unique())
            max_block = unique_blocks[-1]

            if base_strategy == "last_period":
                base_df = df[df[block_col] == max_block].copy()
            elif base_strategy == "window_n":
                cutoff = unique_blocks[max(0, len(unique_blocks) - window_size)]
                base_df = df[df[block_col] >= cutoff].copy()
            else:  # 'all'
                base_df = df.copy()

            next_block = max_block + 1

        elif time_col and time_col in df.columns:
            # Time-based: use last portion
            if base_strategy == "last_period":
                # Use last 10% of data
                cutoff_idx = int(len(df) * 0.9)
                base_df = df.iloc[cutoff_idx:].copy()
            elif base_strategy == "window_n":
                cutoff_idx = int(len(df) * (1 - window_size * 0.1))
                base_df = df.iloc[max(0, cutoff_idx) :].copy()
            else:
                base_df = df.copy()

            next_block = 1  # Will be used for labeling
        else:
            base_df = df.copy()
            next_block = 1

        # Determine samples per period
        if n_samples_per_period is None:
            n_samples_per_period = len(base_df)

        # 2. Generate projected data for each period
        projected_dfs = []

        for period_idx in range(periods):
            period_label = next_block + period_idx

            # Generate synthetic data if generator provided
            if generator is not None:
                try:
                    synth_df = generator.generate(
                        data=base_df,
                        n_samples=n_samples_per_period,
                        method=generator_method,
                        auto_report=False,
                    )
                except Exception as e:
                    print(f"Generator failed, using resampling: {e}")
                    synth_df = base_df.sample(
                        n=n_samples_per_period, replace=True
                    ).reset_index(drop=True)
            else:
                # Simple resampling if no generator
                synth_df = base_df.sample(
                    n=n_samples_per_period, replace=True
                ).reset_index(drop=True)

            # 3. Apply trends for this period
            if trend_config:
                # Scale trends by period index (cumulative effect)
                scaled_config = {}
                for col, config in trend_config.items():
                    scaled_config[col] = config.copy()
                    if config.get("type") == "linear":
                        # Apply slope * period_idx to accumulate
                        scaled_config[col]["intercept"] = config.get("slope", 0) * (
                            period_idx + 1
                        )
                        scaled_config[col]["slope"] = 0  # Already accumulated

                synth_df = self.evolve_features(
                    df=synth_df,
                    evolution_config=scaled_config,
                    auto_report=False,
                )

            # Update block/time column
            if block_col and block_col in synth_df.columns:
                synth_df[block_col] = period_label

            if time_col and time_col in df.columns:
                # Offset time by period
                if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                    last_time = pd.to_datetime(df[time_col].max())
                    time_delta = last_time - pd.to_datetime(df[time_col].min())
                    synth_df[time_col] = pd.date_range(
                        start=last_time + pd.Timedelta(days=1),
                        periods=len(synth_df),
                        freq=time_delta / max(1, len(df) - 1),
                    )

            projected_dfs.append(synth_df)

        # 4. Combine original + projected
        result_df = pd.concat([df] + projected_dfs, ignore_index=True)

        # 5. Generate report if requested
        if auto_report and output_dir:
            from calm_data_generator.reports.QualityReporter import (
                QualityReporter,
            )

            os.makedirs(output_dir, exist_ok=True)

            projected_combined = pd.concat(projected_dfs, ignore_index=True)

            reporter = QualityReporter(verbose=True, minimal=self.minimal_report)
            reporter.generate_comprehensive_report(
                real_df=df,
                synthetic_df=projected_combined,
                generator_name=generator_name,
                output_dir=output_dir,
                block_column=block_col,
                time_col=time_col,
            )

        return result_df
