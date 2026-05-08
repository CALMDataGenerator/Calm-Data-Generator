import os
import random
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from calm_data_generator.generators.base import BaseGenerator
from calm_data_generator.generators.configs import DateConfig, DriftConfig, ReportConfig
from calm_data_generator.generators.drift.DriftInjector import DriftInjector
from calm_data_generator.generators.dynamics.ScenarioInjector import ScenarioInjector

from .StreamReporter import StreamReporter  # reporter to save JSON reports

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class StreamGenerator(BaseGenerator):
    """
    Synthetic data generator using River-backed generators with detailed configuration and reporting.

    This class orchestrates the generation of synthetic datasets, handling various types of drift,
    data balancing, timestamp injection, and comprehensive reporting.

    Key Features:
    - **Drift Simulation**: Supports 'none', 'virtual', 'gradual', 'incremental', and 'abrupt' drift types.
    - **Data Balancing**: Can balance the class distribution of the generated dataset.
    - **Timestamp Injection**: Adds a configurable timestamp column to simulate time-series data.
    - **Flexible Drift Control**: Allows fine-grained control over drift characteristics like position, width, and inconsistency.
    - **Comprehensive Reporting**: Automatically generates a detailed JSON report and visualizations using `StreamReporter`.
    """

    DEFAULT_OUTPUT_DIR = "real_time_output"

    @classmethod
    def set_default_output_dir(cls, path: str):
        """Sets the default directory for saving generated files."""
        cls.DEFAULT_OUTPUT_DIR = path

    @classmethod
    def get_default_output_dir(cls):
        """Gets the default directory for saving generated files."""
        return cls.DEFAULT_OUTPUT_DIR

    def __init__(
        self,
        random_state: Optional[int] = None,
        auto_report: bool = True,
        minimal_report: bool = False,
    ):
        """
        Initializes the StreamGenerator.

        Args:
            random_state (Optional[int]): Seed for the random number generator for reproducibility.
            auto_report (bool): Whether to automatically generate a report after generation.
            minimal_report (bool): If True, generates minimal reports (faster, no correlations/PCA).
        """
        super().__init__(
            random_state=random_state,
            auto_report=auto_report,
            minimal_report=minimal_report,
        )

    def generate(
        self,
        generator_instance,
        n_samples: int,
        filename: str = "synthetic_data.csv",
        output_dir: Optional[str] = None,
        target_col: str = "target",
        balance: bool = False,
        date_config: Optional[DateConfig] = None,  # New config object
        drift_type: str = "none",  # Kept for backward compat or ease of use if config not passed
        drift_options: Optional[Dict] = None,
        drift_config: Optional[List[Union[Dict, DriftConfig]]] = None,  # Explicit arg
        save_dataset: bool = False,  # Changed default to False
        generate_report: Optional[bool] = None,
        report_config: Optional[Union[ReportConfig, Dict]] = None,
        # ... other specialized args can remain optionally or be grouped later
        constraints: Optional[List[Dict]] = None,
        sequence_config: Optional[Dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Main public method to generate a synthetic dataset.

        Args:
           ...

        Returns:
            pd.DataFrame: The generated DataFrame.
        """
        out_dir = self._resolve_output_dir(output_dir) if output_dir else None

        # Resolve ReportConfig
        if report_config:
            if isinstance(report_config, dict):
                report_config = ReportConfig(**report_config)
            # Update output_dir matches
            if out_dir:
                report_config.output_dir = out_dir

        # Backward compatibility for date args if unpacked params are used (simple logic)
        # Ideally we prefer date_config.

        if date_config is None:
            # Try to construct from kwargs for backward compat
            if kwargs.get("date_start"):
                from calm_data_generator.generators.configs import DateConfig

                date_config = DateConfig(
                    start_date=kwargs.get("date_start"),
                    frequency=kwargs.get("date_every", 1),
                    step=kwargs.get("date_step"),
                    date_col=kwargs.get("date_col", "timestamp"),
                )

        df = self._generate_internal(
            generator_instance=generator_instance,
            n_samples=n_samples,
            target_col=target_col,
            balance=balance,
            drift_type=drift_type,
            drift_options=drift_options,
            drift_config=drift_config,  # Pass explicit arg
            date_config=date_config,
            output_dir=out_dir,
            report_config=report_config,  # Pass report config
            generate_report=generate_report,
            filename=filename,
            constraints=constraints,
            sequence_config=sequence_config,
            **kwargs,
        )

        if save_dataset and out_dir:
            full_csv_path = os.path.join(out_dir, filename)
            df.to_csv(full_csv_path, index=False)
            self.logger.info(f"Data generated and saved at: {full_csv_path}")

        return df

    def _generate_internal(self, **kwargs) -> pd.DataFrame:
        """Internal generation logic that constructs the DataFrame and triggers reporting."""
        n_samples = kwargs["n_samples"]
        generator_instance = kwargs["generator_instance"]
        balance = kwargs["balance"]
        kwargs["target_col"]
        drift_type = kwargs["drift_type"]

        # Extract date config
        date_config = kwargs.get("date_config")
        if date_config:
            date_start = date_config.start_date
            date_every = date_config.frequency
            date_step = date_config.step
            date_col = date_config.date_col
        else:
            date_start = kwargs.get("date_start")
            date_every = kwargs.get("date_every", 1)
            date_step = kwargs.get("date_step")
            date_col = kwargs.get("date_col", "timestamp")

        # Update kwargs for downstream usage (like _inject_dates if it used kwargs directly, or just local vars)
        # Actually _inject_dates is called below with specific args, so we just use local vars.

        data_gen_instance = generator_instance

        if drift_type == "virtual_drift":
            drift_options = kwargs.get("drift_options", {})
            pos = kwargs.get("position_of_drift") or n_samples // 2
            missing_fraction = drift_options.get("missing_fraction", 0.1)
            feature_cols = drift_options.get("feature_cols")
            data_gen_instance = self._virtual_drift_generator(
                generator_instance, pos, missing_fraction, feature_cols
            )

        if drift_type in ["gradual", "incremental", "abrupt"]:
            A = generator_instance
            B = kwargs["generator_instance_drift"]
            if not B:
                raise ValueError(
                    f"drift_generator must be provided for {drift_type} drift"
                )

            pos = kwargs.get("position_of_drift") or n_samples // 2
            width = kwargs.get("transition_width")
            inconsistency = kwargs.get("inconsistency", 0.0)

            if drift_type == "gradual":
                width = width if width is not None else n_samples // 10
            elif drift_type == "incremental":
                width = n_samples
                pos = n_samples // 2
            elif drift_type == "abrupt":
                data = list(A.take(pos)) + list(B.take(n_samples - pos))
                data = [list(x.values()) + [y] for x, y in data]
                data_gen_instance = None  # Data is already generated

            if data_gen_instance:
                data = self._build_drifted_rows(
                    A, B, n_samples, pos, width, inconsistency
                )
                data_gen_instance = None  # Data is already generated

        if data_gen_instance:
            data = (
                self._generate_balanced(data_gen_instance, n_samples)
                if balance
                else self._generate_data(data_gen_instance, n_samples)
            )

        # Infer column names
        columns = []
        try:
            # Use the specific metadata generator if provided, otherwise fallback to the main one.
            gen_for_meta = (
                kwargs.get("metadata_generator_instance")
                or kwargs["generator_instance"]
            )

            if hasattr(gen_for_meta, "take"):
                first_sample_features, _ = next(iter(gen_for_meta.take(1)))
                columns = list(first_sample_features.keys())
            else:
                raise AttributeError(
                    "Metadata generator is an iterator and does not have .take() method."
                )

        except Exception as e:
            self.logger.warning(
                f"Could not infer feature names from generator: {e}. Falling back to generic names."
            )
            if data:
                n_features = len(data[0]) - 1
                columns = [f"x{i}" for i in range(n_features)]

        final_columns = columns + [kwargs["target_col"]]
        df = pd.DataFrame(data, columns=final_columns)
        df = self._inject_dates(
            df,
            date_col,
            date_start,
            date_every,
            date_step,
            sequence_config=kwargs.get("sequence_config"),
        )

        # --- Dynamics Injection ---
        dynamics_config = kwargs.get("dynamics_config")
        if dynamics_config:
            self.logger.info("Applying dynamics injection...")
            # For StreamGenerator, random_state is initialized in __init__ -> self.rng
            # We can pick a seed from self.rng or just use self.rng if DynamicsInjector supported it.
            # DynamicsInjector uses a seed int.
            # self.rng is a Generator. We can allow DynamicsInjector to use its own random state.
            injector = ScenarioInjector()

            if "evolve_features" in dynamics_config:
                self.logger.info("Evolving features...")
                evolve_args = dynamics_config["evolve_features"]
                # Use the injected date column if applicable
                df = injector.evolve_features(
                    df, time_col=kwargs["date_col"], **evolve_args
                )

            if "construct_target" in dynamics_config:
                self.logger.info("Constructing dynamic target...")
                target_args = dynamics_config["construct_target"]
                df = injector.construct_target(df, **target_args)

        # --- Drift Injection ---
        drift_config_list = kwargs.get(
            "drift_config"
        )  # Can be passed via kwargs or explicit arg mapping in _generate_internal

        if drift_config_list:
            self.logger.info("Applying drift injection...")
            injector = DriftInjector(
                original_df=df,
                output_dir=kwargs.get("output_dir") or ".",
                generator_name="StreamGenerator_Drifted",
                target_column=kwargs.get("target_col"),
                block_column=kwargs.get("block_column"),
                time_col=kwargs.get("date_col"),
            )

            for drift_conf in drift_config_list:
                # Determine method and params
                method_name = "inject_feature_drift"  # Default
                params = {}
                drift_obj = None

                if isinstance(drift_conf, DriftConfig):
                    method_name = drift_conf.method
                    drift_obj = drift_conf
                    params = drift_conf.params or {}
                elif isinstance(drift_conf, dict):
                    if "method" in drift_conf and "params" in drift_conf:
                        method_name = drift_conf.get("method")
                        params = drift_conf.get("params", {})
                    else:
                        method_name = drift_conf.get(
                            "drift_method",
                            drift_conf.get("method", "inject_feature_drift"),
                        )
                        params = drift_conf

                if hasattr(injector, method_name):
                    self.logger.info(f"Injecting drift: {method_name}")
                    drift_method = getattr(injector, method_name)
                    try:
                        if "df" not in params:
                            params["df"] = df

                        if drift_obj:
                            res = drift_method(drift_config=drift_obj, **params)
                        else:
                            res = drift_method(**params)

                        if isinstance(res, pd.DataFrame):
                            df = res
                    except Exception as e:
                        self.logger.error(f"Failed to apply drift {method_name}: {e}")
                        raise e
                else:
                    self.logger.warning(
                        f"Drift method '{method_name}' not found in DriftInjector."
                    )

        # Resolve generate_report
        should_report = kwargs.get("generate_report")
        if should_report is None:
            should_report = self.auto_report

        if should_report:
            report_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "save_dataset",
                    "output_dir",
                    "generate_report",
                    "report_config",
                ]
            }
            # Ensure the report gets the actual generator instance, not the iterator
            report_kwargs["generator_instance"] = (
                kwargs.get("metadata_generator_instance")
                or kwargs["generator_instance"]
            )

            # Build drift_config for report if drift was applied
            # Build drift_config for report if drift was applied
            drift_config_list = kwargs.get("drift_config")
            if drift_config_list:
                drift_methods = []
                for d in drift_config_list:
                    if isinstance(d, DriftConfig):
                        drift_methods.append(d.method)
                    else:
                        drift_methods.append(
                            d.get("method", d.get("drift_method", "unknown"))
                        )

                report_kwargs["drift_config"] = {
                    "drift_type": ", ".join(drift_methods),
                    "drift_magnitude": "See config",
                    "affected_columns": "Multiple (via drift_config)",
                }

            # Pass report_config
            report_config = kwargs.get("report_config")

            self._save_report_json(
                df=df,
                output_dir=kwargs.get("output_dir"),
                report_config=report_config,
                **report_kwargs,
            )
        return df

    def generate_longitudinal_data(
        self,
        n_samples: int,
        longitudinal_config: Dict[str, Any],
        date_config: Optional[DateConfig] = None,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generates longitudinal clinical data (multi-visit).
        """
        self.logger.info("Generating longitudinal clinical data...")

        # 1. Generate base line data (Visit 0)
        base_data = self.generate(n_samples=n_samples, **kwargs)

        # Extract components
        demographics = base_data.get("demographics")
        omics = base_data.get("genes")  # Assuming genes for now

        if demographics is None:
            raise ValueError("Failed to generate base demographics.")

        # 2. Longitudinal Loop
        all_visits = []
        n_visits = longitudinal_config.get("n_visits", 3)
        time_step = longitudinal_config.get("time_step_days", 30)

        # Add patient ID if not present
        if "Patient_ID" not in demographics.columns:
            demographics["Patient_ID"] = [f"P_{i}" for i in range(len(demographics))]

        # Visit 0
        v0 = demographics.copy()
        v0["Visit_ID"] = 0
        v0["Days_Since_Start"] = 0
        if date_config and date_config.start_date:
            v0["Visit_Date"] = pd.to_datetime(date_config.start_date)

        all_visits.append(v0)

        # Subsequent Visits
        evolution_config = longitudinal_config.get("evolution_config", {})
        features_to_evolve = evolution_config.get("features", [])
        trend = evolution_config.get("trend", 0.0)
        noise = evolution_config.get("noise", 0.0)

        current_visit = v0.copy()

        for v in range(1, n_visits):
            next_visit = current_visit.copy()
            next_visit["Visit_ID"] = v
            next_visit["Days_Since_Start"] = v * time_step

            if date_config and date_config.start_date:
                next_visit["Visit_Date"] = pd.to_datetime(
                    date_config.start_date
                ) + pd.to_timedelta(v * time_step, unit="D")

            # Evolve features
            for col in features_to_evolve:
                if col in next_visit.columns and pd.api.types.is_numeric_dtype(
                    next_visit[col]
                ):
                    # Simple linear trend + noise
                    delta = trend * (
                        1 + np.random.normal(0, noise, size=len(next_visit))
                    )
                    next_visit[col] = next_visit[col] * (1 + delta)

            all_visits.append(next_visit)
            current_visit = next_visit

        longitudinal_df = pd.concat(all_visits, ignore_index=True)

        return {
            "longitudinal": longitudinal_df,
            "base_demographics": demographics,
            "base_omics": omics,
        }

    def inject_drift_group_transition(
        self,
        generator: Iterator,
        position_of_drift: int,
        missing_fraction: float,
        feature_cols: Optional[List[str]],
    ) -> Iterator[Tuple[Dict, int]]:
        """A generator that injects missing values (NaN) after a certain position."""
        feature_names = feature_cols
        for i, (x, y) in enumerate(generator):
            if i < position_of_drift:
                yield x, y
            else:
                if feature_names is None:
                    feature_names = list(x.keys())

                x_drifted = x.copy()
                for col in feature_names:
                    if self.rng.random() < missing_fraction:
                        x_drifted[col] = np.nan
                yield x_drifted, y

    def _virtual_drift_generator(
        self,
        generator: Iterator,
        position_of_drift: int,
        missing_fraction: float,
        feature_cols: Optional[List[str]],
    ) -> Iterator[Tuple[Dict, int]]:
        """A generator that injects missing values (NaN) after a certain position."""
        feature_names = feature_cols
        for i, (x, y) in enumerate(generator):
            if i < position_of_drift:
                yield x, y
            else:
                if feature_names is None:
                    feature_names = list(x.keys())

                x_drifted = x.copy()
                for col in feature_names:
                    if self.rng.random() < missing_fraction:
                        x_drifted[col] = np.nan
                yield x_drifted, y

    def _window_weights(
        self,
        n: int,
        center: float,
        width: int,
        profile: str = "sigmoid",
        k: float = 1.0,
    ) -> np.ndarray:
        """Generates a window of weights for smooth transitions between concepts."""
        if n <= 0:
            return np.zeros(0, dtype=float)
        i = np.arange(n, dtype=float)
        width = max(1, int(width))
        center = float(center)
        if profile == "sigmoid":
            base_scale = width / 4.0
            scale = max(1e-9, base_scale / max(1e-9, float(k)))
            z = (i - center) / scale
            w = 1.0 / (1.0 + np.exp(-z))
        else:
            left = center - width / 2.0
            right = center + width / 2.0
            w = np.clip((i - left) / max(1e-9, (right - left)), 0.0, 1.0)
        return w

    def _build_drifted_rows(
        self, base, drift, n_samples, position, width, inconsistency
    ) -> List[List]:
        """Builds a dataset with a gradual transition from a base generator to a drift generator."""
        w = self._window_weights(n_samples, center=position, width=width)
        if inconsistency > 0:
            noise = self.rng.normal(0, 0.1 * inconsistency, n_samples)
            walk = np.cumsum(noise)
            walk -= np.mean(walk)
            if np.max(np.abs(walk)) > 1e-9:
                walk /= np.max(np.abs(walk))
            sin_wave = np.sin(
                np.linspace(0, self.rng.uniform(1, 5) * 2 * np.pi, n_samples)
            )
            w = np.clip(w + (walk + sin_wave) * 0.5 * inconsistency, 0.0, 1.0)

        try:
            base_iter = base.take(n_samples)
            drift_iter = drift.take(n_samples)
        except AttributeError:
            base_iter = iter(base)
            drift_iter = iter(drift)

        rows = []
        for i in range(n_samples):
            it = drift_iter if self.rng.random() < w[i] else base_iter
            try:
                x, y = next(it)
            except StopIteration:
                break
            rows.append(list(x.values()) + [y])
        return rows

    def _generate_balanced(self, gen, n_samples, use_smote: bool = False) -> List[List]:
        """Generates samples and balances the classes to have roughly equal representation.

        Args:
            use_smote: If True, applies SMOTE to balance classes instead of random oversampling.
        """

        class_samples = defaultdict(list)
        max_samples = max(n_samples * 5, n_samples)

        # Handle both River objects and Python generators
        if hasattr(gen, "take"):
            data_iterator = gen.take(max_samples)
        else:
            data_iterator = (next(gen) for _ in range(max_samples))

        for x, y in data_iterator:
            class_samples[y].append(list(x.values()) + [y])

        data = []
        per_class = n_samples // len(class_samples) if class_samples else n_samples

        for samples in class_samples.values():
            data.extend(samples[:per_class])

        if len(data) < n_samples and data:
            if use_smote:
                try:
                    from imblearn.over_sampling import SMOTE

                    df_temp = pd.DataFrame(data)
                    X = df_temp.iloc[:, :-1].values
                    y = df_temp.iloc[:, -1].values

                    min_class_count = (
                        min(np.bincount(y.astype(int))) if len(np.unique(y)) > 1 else 1
                    )
                    k = min(5, min_class_count - 1) if min_class_count > 1 else 1

                    if k >= 1 and len(np.unique(y)) > 1:
                        smote = SMOTE(k_neighbors=k, random_state=42)
                        X_res, y_res = smote.fit_resample(X, y)
                        result = np.column_stack([X_res, y_res])
                        data = result[:n_samples].tolist()
                    else:
                        data.extend(random.choices(data, k=n_samples - len(data)))
                except ImportError:
                    data.extend(random.choices(data, k=n_samples - len(data)))
            else:
                data.extend(random.choices(data, k=n_samples - len(data)))

        return data[:n_samples] if data else []

    def _generate_data(self, gen, n_samples) -> List[List]:
        """Generates n_samples from a River generator or a standard Python iterator."""

        if hasattr(gen, "take"):
            data_iterator = gen.take(n_samples)
        else:
            # For standard Python generators (like _virtual_drift_generator)
            if not hasattr(gen, "__next__") and hasattr(gen, "__iter__"):
                gen = iter(gen)
            data_iterator = (next(gen) for _ in range(n_samples))

        return [list(x.values()) + [y] for x, y in data_iterator]

    def _stateful_generator(
        self,
        gen,
        n_samples: int,
        state_config: Optional[Dict] = None,
    ) -> List[List]:
        """
        Generates data where row t depends on row t-1 (stateful generation).

        Args:
            state_config: Dict with 'carry_cols' (column indices to carry forward),
                         'evolution_factor' (how much to modify carried values),
                         'noise_level' (random noise to add).
        """
        if not state_config:
            state_config = {}

        carry_cols = state_config.get("carry_cols", [])  # Column indices to carry
        evolution_factor = state_config.get("evolution_factor", 0.1)
        noise_level = state_config.get("noise_level", 0.05)

        data = []
        prev_row = None

        # Handle both River objects and Python generators
        if hasattr(gen, "take"):
            data_iterator = gen.take(n_samples)
        else:
            data_iterator = (next(gen) for _ in range(n_samples))

        for i, (x, y) in enumerate(data_iterator):
            row_values = list(x.values())

            if prev_row is not None and carry_cols:
                # Apply stateful evolution: new_value = prev_value * (1 + evolution) + noise
                for col_idx in carry_cols:
                    if col_idx < len(row_values) and isinstance(
                        row_values[col_idx], (int, float)
                    ):
                        prev_val = prev_row[col_idx]
                        evolved = prev_val * (
                            1 + self.rng.uniform(-evolution_factor, evolution_factor)
                        )
                        noise = (
                            self.rng.normal(0, abs(evolved) * noise_level)
                            if evolved != 0
                            else 0
                        )
                        row_values[col_idx] = evolved + noise

            row = row_values + [y]
            data.append(row)
            prev_row = row_values

        return data

    def _inject_dates(
        self,
        df,
        date_col,
        date_start,
        date_every,
        date_step,
        sequence_config: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Injects a date column. Supports simple continuous time or entity-based sequences.

        Args:
            sequence_config: Dict with 'entity_col', 'events_per_entity' (avg), 'session_gap_minutes', etc.
        """
        if not date_start:
            return df

        n_rows = len(df)

        if sequence_config:
            # Sequence Mode: Generate IDs and timestamps per ID
            entity_col = sequence_config.get("entity_col", "entity_id")
            avg_events = sequence_config.get("events_per_entity", 10)

            # Approximate number of entities (used to assign IDs below)

            # Assign IDs
            ids = []
            current_id = 0
            while len(ids) < n_rows:
                # Randomize length slightly
                length = max(
                    1, int(self.rng.normal(avg_events, max(1, avg_events / 5)))
                )
                ids.extend([f"User_{current_id}"] * length)
                current_id += 1

            ids = ids[:n_rows]
            df[entity_col] = ids

            # Generate timestamps per entity
            # We assume users start at random times within a window, or all at start_date
            base_start = pd.to_datetime(date_start)

            # Group by ID (simple iteration for now, optimized vectorization possible but complex)
            # To be fast, we process by group
            df_temp = pd.DataFrame({entity_col: ids})
            groups = df_temp.groupby(entity_col)

            # Global time or Per-User time?
            # Usually sequences are independent or interleaved.
            # Let's assume interleaved global time for simplicity OR per-session relative time.
            # Plan says: "generar secuencias de eventos por entidad".

            # Optimization: Assign start times for each entity
            entity_starts = base_start + pd.to_timedelta(
                self.rng.integers(0, 30, size=current_id + 1), unit="D"
            )

            # For each row, we need look up its entity's start + offset
            # Vectorized approach:
            # 1. Calculate cumulative count per group
            cumcounts = groups.cumcount()

            # 2. Map IDs to start times
            # Extract ID integer suffix
            id_series = pd.Series(ids)
            id_ints = id_series.str.split("_").str[-1].astype(int)

            # 3. Calculate offsets
            # step defaults
            step_unit = "D"
            step_val = 1
            if date_step:
                step_unit = list(date_step.keys())[0]
                step_val = list(date_step.values())[0]
                if step_unit == "days":
                    step_unit = "D"
                elif step_unit == "minutes":
                    step_unit = "m"
                elif step_unit == "seconds":
                    step_unit = "s"

            # Offsets = cumcount * step
            offsets = pd.to_timedelta(cumcounts * step_val, unit=step_unit)

            # 4. Final Dates = Entity Start + Offset
            # Need to map row -> entity_start
            # Using transform is slow? map is faster?
            # row_starts = id_ints.map(lambda i: entity_starts[i]) # Slower
            row_starts = entity_starts[id_ints.values]

            series = row_starts + offsets

        else:
            # Standard Continuous Mode
            time_deltas = np.arange(len(df)) // date_every
            if date_step:
                offset = pd.DateOffset(**date_step)
                series = pd.to_datetime(date_start) + time_deltas * offset
            else:
                series = pd.to_datetime(date_start) + pd.to_timedelta(
                    time_deltas, unit="D"
                )

        df[date_col] = series
        return df

    def _apply_constraints(
        self, df: pd.DataFrame, constraints: List[Dict]
    ) -> pd.DataFrame:
        """
        Applies post-hoc constraints to the Synthetic DataFrame.
        """
        if not constraints:
            return df

        initial_count = len(df)
        valid_mask = pd.Series(True, index=df.index)

        self.logger.info(f"Applying {len(constraints)} constraints...")

        for const in constraints:
            col = const.get("col")
            op = const.get("op")
            val = const.get("val")

            if col not in df.columns:
                continue

            if op == ">":
                valid_mask &= df[col] > val
            elif op == "<":
                valid_mask &= df[col] < val
            elif op == ">=":
                valid_mask &= df[col] >= val
            elif op == "<=":
                valid_mask &= df[col] <= val
            elif op == "==":
                valid_mask &= df[col] == val
            elif op == "!=":
                valid_mask &= df[col] != val

        filtered_df = df[valid_mask].copy()
        dropped = initial_count - len(filtered_df)
        if dropped > 0:
            self.logger.warning(
                f"Constraints dropped {dropped} samples ({dropped / initial_count:.1%})."
            )

        return filtered_df

    def _save_report_json(
        self,
        df: pd.DataFrame,
        output_dir: str,
        report_config: Optional[ReportConfig] = None,
        **kwargs,
    ):
        """Saves a comprehensive JSON report of the generated data and its properties."""
        # Map kwargs to StreamReporter signature
        report_kwargs = {
            "target_column": kwargs.get("target_col"),
            "time_col": kwargs.get("date_col"),
            "block_column": kwargs.get("block_column"),
            "drift_config": {
                "drift_type": kwargs.get("drift_type"),
                "position_of_drift": kwargs.get("position_of_drift"),
                "transition_width": kwargs.get("transition_width"),
                "drift_options": kwargs.get("drift_options"),
            },
        }

        # Filter out None values to avoid passing them
        report_kwargs_filtered = {
            k: v for k, v in report_kwargs.items() if v is not None
        }

        # Prepare ReportConfig for StreamReporter

        # Prepare ReportConfig for StreamReporter
        # If report_config is passed, use it. But we also have kwargs that might override or supplement?
        # StreamReporter.generate_report takes report_config OR individual args.
        # We can pass report_config and filter kwargs.

        try:
            reporter = StreamReporter(verbose=True, minimal_report=self.minimal_report)
            reporter.generate_report(
                synthetic_df=df,
                generator_name=kwargs.get(
                    "generator_instance", self
                ).__class__.__name__,  # kwargs["generator_instance"] was set in generate_report call
                output_dir=output_dir,
                report_config=report_config,
                **report_kwargs_filtered,
            )
        except Exception as e:
            self.logger.error(f"Could not generate report: {e}", exc_info=True)

    def validate_params(self, **kwargs):
        """Validates input parameters for the generate method."""
        if not (
            isinstance(kwargs.get("n_samples"), int) and kwargs.get("n_samples") > 0
        ):
            raise ValueError("n_samples must be a positive integer")

    def _resolve_output_dir(self, path: Optional[str]) -> str:
        """Resolves the output directory path, creating it if it doesn't exist."""
        out = os.path.abspath(path or self.DEFAULT_OUTPUT_DIR)
        os.makedirs(out, exist_ok=True)
        return out
