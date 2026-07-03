"""Demographic data generation mixin for ClinicalDataGenerator."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from calm_data_generator.generators.drift.DriftInjector import DriftInjector
from calm_data_generator.generators.dynamics.ScenarioInjector import ScenarioInjector


class _DemographicMixin:
    """Mixin providing demographic data generation for ClinicalDataGenerator."""

    def generate_demographic_data(
        self,
        n_samples: int,
        control_disease_ratio: float = 0.5,
        demographic_correlations: np.ndarray = None,
        custom_demographic_columns: dict = None,
        date_column_name: str = None,
        date_value: str = None,
        class_assignment_function: callable = None,
        drift_injection_config: Optional[List[Dict]] = None,
        dynamics_config: Optional[Dict] = None,
        constraints: Optional[List[Dict]] = None,
    ):
        """
        Generates synthetic demographic data for a given number of samples.

        Args:
            n_samples (int): The number of patient samples to generate.
            control_disease_ratio (float): The ratio of control patients (e.g., 0.5 for 50% control, 50% disease).
            demographic_correlations (np.ndarray): A correlation matrix for demographic features.
            custom_demographic_columns (dict): A dictionary where keys are column names and values are scipy.stats distributions.
            date_column_name (str, optional): Name for a new date column. If provided, date_value must also be provided.
            date_value (str, optional): The date value to fill the new date column with (e.g., '2023-01-15').
            class_assignment_function (callable, optional): A function that takes the generated demographic DataFrame
                and returns a Series of strings (for subgroups) or a binary array (0 or 1) for class assignment.

        Returns:
            (pd.DataFrame, pd.DataFrame): A tuple containing:
                - df_temp: The main demographic DataFrame with categorical values.
                - raw_demographic_data: A DataFrame with raw numerical/binary values for correlation.
        """
        # Default marginals for age, sex, and propensity score
        import scipy.stats as stats

        default_marginals = {
            "Age": stats.norm(loc=60, scale=12),
            "Sex": stats.binom(n=1, p=0.52),
            "Propensity": stats.norm(),  # Propensity score for group assignment
        }

        # Combine default and custom marginals
        marginals_to_use = default_marginals.copy()
        if custom_demographic_columns:
            for col_name, distribution_spec in custom_demographic_columns.items():
                if isinstance(distribution_spec, dict):
                    # If it's a dictionary, interpret as distribution parameters
                    dist_type = distribution_spec.get("distribution")
                    params = {
                        k: v
                        for k, v in distribution_spec.items()
                        if k != "distribution"
                    }

                    if dist_type == "norm":
                        marginals_to_use[col_name] = stats.norm(**params)
                    elif dist_type == "binom":
                        marginals_to_use[col_name] = stats.binom(**params)
                    elif dist_type == "uniform":
                        marginals_to_use[col_name] = stats.uniform(**params)
                    elif dist_type == "poisson":
                        marginals_to_use[col_name] = stats.poisson(**params)
                    elif dist_type == "randint":
                        marginals_to_use[col_name] = stats.randint(**params)
                    elif dist_type == "truncnorm":
                        marginals_to_use[col_name] = stats.truncnorm(**params)
                    else:
                        raise ValueError(
                            f"Unsupported distribution type '{dist_type}' for demographic column '{col_name}'."
                        )
                else:
                    # Otherwise, assume it's a scipy.stats distribution object
                    marginals_to_use[col_name] = distribution_spec

        if "Propensity" not in marginals_to_use:
            marginals_to_use["Propensity"] = stats.norm()

        ordered_col_names = list(marginals_to_use.keys())
        marginals_list = [marginals_to_use[col] for col in ordered_col_names]
        n_mod_vars = len(marginals_list)

        if demographic_correlations is None:
            demographic_correlations = np.identity(n_mod_vars)
        else:
            if demographic_correlations.shape != (n_mod_vars, n_mod_vars):
                raise ValueError(
                    f"demographic_correlations matrix shape {demographic_correlations.shape} does not match the number of demographic variables ({n_mod_vars})."
                )

        X_demo_raw = self._generate_correlated_module(
            n_samples, marginals_list, demographic_correlations
        )
        df_temp = pd.DataFrame(X_demo_raw, columns=ordered_col_names)

        if "Age" in df_temp.columns:
            df_temp["Age"] = np.round(df_temp["Age"]).astype(int)
        if "Sex" in df_temp.columns:
            df_temp["Sex_Binario"] = np.round(df_temp["Sex"]).astype(int)
            df_temp["Sex"] = df_temp["Sex_Binario"].map({0: "Female", 1: "Male"})

        patient_ids = [
            f"PAT_{np.random.randint(10000, 99999)}_{i}" for i in range(n_samples)
        ]
        df_temp["Patient_ID"] = patient_ids
        df_temp = df_temp.set_index("Patient_ID")

        if date_column_name and date_value:
            try:
                df_temp[date_column_name] = pd.to_datetime(date_value)
            except ValueError:
                df_temp[date_column_name] = date_value

        # --- Group and Subgroup Assignment ---
        if class_assignment_function:
            subgroups = class_assignment_function(df_temp)
            if (
                not isinstance(subgroups, (np.ndarray, pd.Series))
                or subgroups.shape[0] != n_samples
            ):
                raise ValueError(
                    "The class_assignment_function must return a numpy array or pandas Series of length n_samples."
                )

            # Store detailed subgroups
            df_temp["Disease_Subgroup"] = subgroups

            # For backward compatibility, create the binary 'Group' column
            group_final = (subgroups != "Control").astype(int)

        else:
            propensity_scores = df_temp["Propensity"].values
            n_control = int(n_samples * control_disease_ratio)
            n_disease = n_samples - n_control
            sorted_indices = np.argsort(propensity_scores)
            group_final = np.zeros(n_samples, dtype=int)
            group_final[sorted_indices[-n_disease:]] = 1
            df_temp["Disease_Subgroup"] = np.where(
                group_final == 1, "Disease", "Control"
            )

        df_temp["Binary_Group"] = group_final
        df_temp["Group"] = df_temp["Binary_Group"].map({0: "Control", 1: "Disease"})

        # Store raw numerical data
        raw_demographic_data = df_temp.copy()
        if "Group" in raw_demographic_data.columns:
            raw_demographic_data = raw_demographic_data.drop(columns=["Group"])
        if "Disease_Subgroup" in raw_demographic_data.columns:
            raw_demographic_data = raw_demographic_data.drop(
                columns=["Disease_Subgroup"]
            )
        if "Propensity" in raw_demographic_data.columns:
            raw_demographic_data = raw_demographic_data.drop(columns=["Propensity"])
        if (
            "Sex" in raw_demographic_data.columns
            and "Sex_Binario" in raw_demographic_data.columns
        ):
            # Preserve Sex's column position: Sex_Binario must land at the same index
            # so that demographic_gene_correlations matrix rows align correctly.
            sex_pos = raw_demographic_data.columns.get_loc("Sex")
            raw_demographic_data = raw_demographic_data.drop(columns=["Sex"])
            cols = list(raw_demographic_data.columns)
            cols.remove("Sex_Binario")
            cols.insert(sex_pos, "Sex_Binario")
            raw_demographic_data = raw_demographic_data[cols]

        # Clean up final demographic df
        df_temp = df_temp.drop(columns=["Binary_Group", "Propensity"])
        if "Sex_Binario" in df_temp.columns:
            df_temp = df_temp.drop(columns=["Sex_Binario"])

        # --- Dynamics Injection ---
        if dynamics_config:
            injector = ScenarioInjector()
            if "evolve_features" in dynamics_config:
                evolve_args = dynamics_config["evolve_features"]
                # If date_column_name was used, use it as time_col
                df_temp = injector.evolve_features(
                    df_temp, time_col=date_column_name, evolution_config=evolve_args
                )
            if "construct_target" in dynamics_config:
                target_args = dynamics_config["construct_target"]
                df_temp = injector.construct_target(df_temp, **target_args)

        # --- Drift Injection ---
        if drift_injection_config:
            injector = DriftInjector(
                original_df=df_temp,
                generator_name="ClinicalDataGenerator_Demographic",
                time_col=date_column_name,
            )
            df_temp = injector.inject_multiple_types_of_drift(
                df=df_temp, schedule=drift_injection_config, time_col=date_column_name
            )

        # --- Constraints Application ---
        if constraints:
            df_temp = self._apply_constraints(df_temp, constraints)

            # Sync raw_demographic_data with filtered df_temp
            # We need to drop rows from raw that were dropped from df_temp
            if len(df_temp) < len(raw_demographic_data):
                raw_demographic_data = raw_demographic_data.loc[df_temp.index]

        return df_temp, raw_demographic_data

    def _apply_constraints(self, df: pd.DataFrame, constraints: List[Dict]) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        for c in constraints:
            col, op, val = c.get("col"), c.get("op"), c.get("val")
            if col not in df.columns:
                continue
            if op == ">=":
                mask &= df[col] >= val
            elif op == "<=":
                mask &= df[col] <= val
            elif op == "==":
                mask &= df[col] == val
            elif op == ">":
                mask &= df[col] > val
            elif op == "<":
                mask &= df[col] < val
        return df[mask].copy()

    def _prepare_demographic_context(
        self,
        demographic_df,
        demographic_id_col,
        raw_demographic_data,
        n_samples_default=100,
        control_disease_ratio=0.5,
        use_correlation=False,
    ):
        """
        Helper to extract or generate demographic context for omics data generation.
        """
        if demographic_df is not None:
            if demographic_id_col is None or (
                demographic_id_col not in demographic_df.columns
                and demographic_id_col != demographic_df.index.name
            ):
                raise ValueError(
                    "demographic_id_col must be provided and exist in demographic_df when demographic_df is provided."
                )

            n_samples = len(demographic_df)
            if demographic_id_col == demographic_df.index.name:
                patient_ids = np.array(demographic_df.index.values)
            else:
                patient_ids = np.array(demographic_df[demographic_id_col].values)

            groups = (
                demographic_df["Group"].values
                if "Group" in demographic_df.columns
                else np.array(["Control"] * n_samples)
            )

            if use_correlation and raw_demographic_data is None:
                raise ValueError(
                    "raw_demographic_data must be provided when correlation is used."
                )

            demographic_marginals_for_corr = []
            if raw_demographic_data is not None:
                for col in raw_demographic_data.columns:
                    if (
                        col != demographic_id_col
                        and col not in self._EXCLUDED_DEMO_COLS
                    ):
                        if pd.api.types.is_numeric_dtype(raw_demographic_data[col]):
                            if raw_demographic_data[col].nunique() <= 2:
                                p_val = raw_demographic_data[col].mean()
                                demographic_marginals_for_corr.append(
                                    stats.binom(n=1, p=p_val)
                                )
                            else:
                                loc_val = raw_demographic_data[col].mean()
                                scale_val = raw_demographic_data[col].std()
                                demographic_marginals_for_corr.append(
                                    stats.norm(loc=loc_val, scale=scale_val)
                                )
        else:
            n_samples = n_samples_default
            patient_ids = np.array(
                [f"PAT_{np.random.randint(10000, 99999)}_{i}" for i in range(n_samples)]
            )
            n_control = int(n_samples * control_disease_ratio)
            n_disease = n_samples - n_control
            groups = np.array(["Control"] * n_control + ["Disease"] * n_disease)
            np.random.shuffle(groups)

            demographic_marginals_for_corr = []
            p_val_disease = n_disease / n_samples
            demographic_marginals_for_corr.append(stats.binom(n=1, p=p_val_disease))

        idx_control = np.where(groups == "Control")[0]
        idx_disease = np.where(groups == "Disease")[0]

        return (
            patient_ids,
            groups,
            idx_control,
            idx_disease,
            demographic_marginals_for_corr,
        )
