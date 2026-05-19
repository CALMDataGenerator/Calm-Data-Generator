import io
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from calm_data_generator.generators.clinical.ClinicReporter import ClinicReporter
from calm_data_generator.generators.complex.ComplexGenerator import ComplexGenerator
from calm_data_generator.generators.configs import DateConfig, DriftConfig
from calm_data_generator.generators.drift.DriftInjector import DriftInjector
from calm_data_generator.generators.dynamics.ScenarioInjector import ScenarioInjector

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logger = logging.getLogger(__name__)


class ClinicalDataGenerator(ComplexGenerator):
    """
    A class to generate synthetic clinical data including demographic, gene expression, and protein data.
    """

    def __init__(self, seed=42, auto_report=True, minimal_report=False):
        """
        Initializes the ClinicalDataGenerator with a given random seed for reproducibility.

        Args:
            seed: Random seed for reproducibility.
            auto_report: If True, automatically generates reports after generation.
            minimal_report: If True, generates minimal reports (faster, no correlations/PCA).
        """
        super().__init__(
            random_state=seed,
            auto_report=auto_report,
            minimal_report=minimal_report,
        )
        # ClinicalDataGenerator also uses np.random.seed globally for scipy dependencies
        np.random.seed(seed)

    def generate(
        self,
        n_samples: int = 100,
        n_genes: int = 200,
        n_proteins: int = 50,
        date_config: Optional["DateConfig"] = None,
        output_dir: Optional[str] = None,
        save_dataset: bool = False,
        # ... forward other args as needed or keep simple for now
        constraints: Optional[List[Dict]] = None,
        longitudinal_config: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """
        Main entry point for generating clinical datasets (Demographics + Omics).
        Returns a dictionary of DataFrames.
        """

        # 1. Resolve date params
        date_col = "timestamp"
        date_val = None
        if date_config:
            date_col = date_config.date_col
            if date_config.start_date:
                date_val = date_config.start_date

        # 2. Generate Demographics
        demo_df, raw_demo = self.generate_demographic_data(
            n_samples=n_samples,
            date_column_name=date_col,
            date_value=date_val,
            # Pass through other potential kwargs if they match
            control_disease_ratio=kwargs.get("control_disease_ratio", 0.5),
            demographic_correlations=kwargs.get("demographic_correlations"),
            custom_demographic_columns=kwargs.get("custom_demographic_columns"),
            drift_injection_config=kwargs.get("demographics_drift_config"),
            dynamics_config=kwargs.get("demographics_dynamics_config"),
            constraints=constraints,
        )

        # 3. Generate Genes (RNA-Seq by default if not specified)
        genes_df = self.generate_gene_data(
            n_genes=n_genes,
            gene_type=kwargs.get("gene_type", "RNA-Seq"),
            demographic_df=demo_df,
            demographic_id_col="Patient_ID",  # Default from generate_demographic_data
            raw_demographic_data=raw_demo,
            n_samples=n_samples,  # Redundant but required by signature
            demographic_gene_correlations=kwargs.get("demographic_gene_correlations"),
            gene_correlations=kwargs.get("gene_correlations"),
            drift_injection_config=kwargs.get("genes_drift_config"),
            dynamics_config=kwargs.get("genes_dynamics_config"),
            disease_effects_config=kwargs.get("disease_effects_config"),
        )

        # 4. Generate Proteins
        proteins_df = self.generate_protein_data(
            n_proteins=n_proteins,
            demographic_df=demo_df,
            demographic_id_col="Patient_ID",
            raw_demographic_data=raw_demo,
            n_samples=n_samples,
            drift_injection_config=kwargs.get("proteins_drift_config"),
            dynamics_config=kwargs.get("proteins_dynamics_config"),
            disease_effects_config=kwargs.get("disease_effects_config"),
        )

        res = {"demographics": demo_df, "genes": genes_df, "proteins": proteins_df}

        # Generate target variable Y if config provided
        target_config = kwargs.get("target_variable_config")
        if target_config:
            omics_dfs = [df for df in [genes_df, proteins_df] if df is not None and not df.empty]
            try:
                Y = self.generate_target_variable(
                    demographic_df=demo_df,
                    omics_dfs=omics_dfs if omics_dfs else demo_df,
                    weights=target_config.get("weights", {}),
                    noise_std=target_config.get("noise_std", 0.1),
                    binary_threshold=target_config.get("binary_threshold"),
                )
                y_name = target_config.get("name", "Y")
                Y.name = y_name
                demo_df[y_name] = Y
                res["demographics"] = demo_df
                logger.info("Target variable '%s' generated and added to demographics.", y_name)
            except Exception as e:
                logger.error("Failed to generate target variable: %s", e)

        if save_dataset and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if demo_df is not None:
                demo_df.to_csv(os.path.join(output_dir, "demographics.csv"))
            if genes_df is not None:
                genes_df.to_csv(os.path.join(output_dir, "genes.csv"))
            if proteins_df is not None:
                proteins_df.to_csv(os.path.join(output_dir, "proteins.csv"))

        # Prepare unified dataset for reporting (and saving if requested)
        dfs_to_merge = []
        if demo_df is not None:
            dfs_to_merge.append(demo_df)
        if genes_df is not None:
            dfs_to_merge.append(genes_df)
        if proteins_df is not None:
            dfs_to_merge.append(proteins_df)

        unified_df = None
        if dfs_to_merge:
            unified_df = dfs_to_merge[0]
            for df_merge in dfs_to_merge[1:]:
                unified_df = unified_df.join(df_merge, rsuffix="_dup")

        if save_dataset and output_dir and unified_df is not None:
            unified_df.to_csv(os.path.join(output_dir, "unified_clinical_data.csv"))

        should_report = kwargs.get("generate_report")
        if should_report is None:
            should_report = self.auto_report

        if should_report and output_dir:
            # Build aggregated drift_config if any drift was applied
            drift_sources = []
            if kwargs.get("demographics_drift_config"):
                drift_sources.append("Demographics")
            if kwargs.get("genes_drift_config"):
                drift_sources.append("Genes")
            if kwargs.get("proteins_drift_config"):
                drift_sources.append("Proteins")

            if drift_sources:
                kwargs["drift_config"] = {
                    "drift_type": "Clinical Drift",
                    "drift_magnitude": "See config",
                    "affected_columns": ", ".join(drift_sources),
                }

            if demo_df is not None:
                self._generate_report(
                    demo_df,
                    "Demographics",
                    output_dir,
                    date_col,
                    "demographics_report",
                    **kwargs,
                )
            if genes_df is not None:
                self._generate_report(
                    genes_df, "Genes", output_dir, date_col, "genes_report", **kwargs
                )
            if proteins_df is not None:
                self._generate_report(
                    proteins_df,
                    "Proteins",
                    output_dir,
                    date_col,
                    "proteins_report",
                    **kwargs,
                )
            if unified_df is not None:
                self._generate_report(
                    unified_df,
                    "Unified_Clinical_Data",
                    output_dir,
                    date_col,
                    "unified_report",
                    **kwargs,
                )

        return res

    def _generate_report(self, df, name, base_output_dir, time_col, sub_dir, **kwargs):
        """Helper to generate a report for a specific dataframe using StreamReporter."""
        try:
            reporter = ClinicReporter(verbose=True)
            report_dir = os.path.join(base_output_dir, sub_dir)

            # Extract target column if available
            target_config = kwargs.get("target_variable_config", {})
            target_col = target_config.get("name", "diagnosis")
            if target_col not in df.columns:
                target_col = None

            # Extract drift_config if available
            drift_config = kwargs.get("drift_config")

            reporter.generate_report(
                synthetic_df=df,
                generator_name=f"Clinical_{name}",
                output_dir=report_dir,
                target_column=target_col,
                time_col=time_col if time_col in df.columns else None,
                drift_config=drift_config,
            )
            print(f"✅ Generated report for {name} in {report_dir}")
        except Exception as e:
            print(f"⚠️ Failed to generate report for {name}: {e}")

    def _design_gene_params_rnaseq(self, n_genes, gene_mean_log_center):
        """
        Designs r (size) and p (prob) parameters for Negative Binomial distribution (RNA-Seq).
        """
        log_means = np.random.normal(loc=gene_mean_log_center, scale=1.5, size=n_genes)
        means = np.round(np.exp(log_means))

        dispersions = np.random.uniform(low=0.1, high=1.0, size=n_genes)
        sizes = 1 / dispersions

        probs = sizes / (sizes + means)

        valid_mask = (probs > 0) & (probs < 1)

        return sizes, probs, valid_mask

    def _design_protein_params(self, n_proteins):
        """
        Designs log_means and log_stds parameters for Log-Normal distribution (Proteins).
        """
        log_means = np.random.normal(loc=3.0, scale=0.8, size=n_proteins)
        log_stds = np.random.uniform(low=0.1, high=0.4, size=n_proteins)
        return log_stds, log_means  # Return log_stds and log_means, not log_scales

    def _design_gene_params_normal(self, n_genes, gene_mean_loc_center):
        """
        Designs loc (mean) and scale (std dev) parameters for Normal distribution (Microarray).
        """
        locs = np.random.normal(loc=gene_mean_loc_center, scale=1.0, size=n_genes)
        scales = np.random.uniform(low=0.5, high=2.0, size=n_genes)
        return locs, scales

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
            raw_demographic_data = raw_demographic_data.drop(columns=["Sex"])

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
                        and col != "Group"
                        and col != "Binary_Group"
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

    def generate_gene_data(
        self,
        n_genes: int,
        gene_type: str,  # "RNA-Seq" or "Microarray"
        demographic_df: pd.DataFrame = None,
        demographic_id_col: str = None,
        raw_demographic_data: pd.DataFrame = None,
        gene_correlations: np.ndarray = None,
        demographic_gene_correlations: np.ndarray = None,
        disease_effects_config: dict = None,  # New parameter
        subgroup_col: str = None,  # New parameter for subgroup-based effects
        gene_mean_log_center: float = np.log(80),  # For RNA-Seq
        gene_mean_loc_center: float = 7.0,  # For Microarray
        control_disease_ratio: float = 0.5,
        custom_gene_parameters: dict = None,
        n_samples: int = 100,
        random_state: int = 42,  # Added for completeness/drift
        drift_injection_config: Optional[List[Dict]] = None,
        dynamics_config: Optional[Dict] = None,
    ):
        """
        Generates synthetic gene expression data.
        Supports heterogeneous disease effects via a structured `disease_effects_config`.
        """
        if gene_type.lower() not in ["rna-seq", "microarray"]:
            raise ValueError("gene_type must be 'RNA-Seq' or 'Microarray'.")

        # --- 1. Handle Demographic Data ---
        patient_ids, groups, idx_control, idx_disease, demographic_marginals = (
            self._prepare_demographic_context(
                demographic_df,
                demographic_id_col,
                raw_demographic_data,
                n_samples_default=n_samples,
                control_disease_ratio=control_disease_ratio,
                use_correlation=(demographic_gene_correlations is not None),
            )
        )
        n_total_samples = len(patient_ids)

        # --- 2. Design Base Gene Parameters ---
        base_gene_marginals = [None] * n_genes
        for i in range(n_genes):
            if gene_type.lower() == "microarray":
                loc = np.random.normal(loc=gene_mean_loc_center, scale=1.0)
                scale = np.random.uniform(low=0.5, high=2.0)
                scale = np.random.uniform(low=0.5, high=2.0)
                base_gene_marginals[i] = stats.norm(loc=loc, scale=scale)
            else:  # RNA-Seq
                log_mean = np.random.normal(loc=gene_mean_log_center, scale=1.5)
                mean = np.round(np.exp(log_mean))
                dispersion = np.random.uniform(low=0.1, high=1.0)
                r_val = 1 / dispersion
                p_val = r_val / (r_val + mean)
                if not (0 < p_val < 1):
                    p_val = 0.5
                base_gene_marginals[i] = stats.nbinom(n=r_val, p=p_val)

        # --- 3. Generate Base Correlated Data for ALL Samples ---
        if demographic_gene_correlations is not None:
            # Prepare conditioning data from raw_demographic_data
            # Filter columns exactly as _prepare_demographic_context does
            cond_cols = [
                c
                for c in raw_demographic_data.columns
                if c != demographic_id_col
                and c != "Group"
                and c != "Binary_Group"
                and pd.api.types.is_numeric_dtype(raw_demographic_data[c])
            ]
            conditioning_data = raw_demographic_data[cond_cols].values

            X_genes_base = self._generate_conditional_data(
                n_samples=n_total_samples,
                conditioning_data=conditioning_data,
                conditioning_marginals=demographic_marginals,
                target_marginals=base_gene_marginals,
                full_covariance=demographic_gene_correlations,
            )
        else:
            X_genes_base = self._generate_correlated_module(
                n_total_samples,
                base_gene_marginals,
                gene_correlations
                if gene_correlations is not None
                else np.identity(n_genes),
            )

        df_genes = pd.DataFrame(
            X_genes_base, columns=[f"G_{i}" for i in range(n_genes)], index=patient_ids
        )

        # --- 4. Apply Heterogeneous Disease Effects via Subgroups ---
        if disease_effects_config and len(idx_disease) > 0:
            # --- New logic for subgroup-based effect application ---
            if subgroup_col and demographic_df is not None:
                if subgroup_col not in demographic_df.columns:
                    raise ValueError(
                        f"subgroup_col '{subgroup_col}' not found in demographic_df."
                    )

                effect_definitions = disease_effects_config.get("effects", {})
                patient_subgroups_config = disease_effects_config.get(
                    "patient_subgroups", []
                )

                for sub_config in patient_subgroups_config:
                    subgroup_name = sub_config["name"]
                    effects_to_apply = sub_config.get("apply_effects", [])

                    # Find patients belonging to this subgroup
                    subgroup_patient_ids = demographic_df[
                        demographic_df[subgroup_col] == subgroup_name
                    ].index
                    subgroup_patient_ids = subgroup_patient_ids.intersection(
                        df_genes.index
                    )  # Ensure they are in the current gene df

                    if subgroup_patient_ids.empty:
                        continue

                    # Apply all effects listed for this subgroup
                    for effect_name in effects_to_apply:
                        effect = effect_definitions.get(effect_name)
                        if not effect:
                            raise ValueError(
                                f"Effect '{effect_name}' not found in definitions."
                            )
                        self.apply_stochastic_effects(
                            df_genes, subgroup_patient_ids, effect
                        )

            # --- Old logic for random assignment (backward compatibility) ---
            else:
                if isinstance(disease_effects_config, list):
                    # Auto-generate names if not provided for backward compatibility
                    effect_definitions = {}
                    for idx, effect in enumerate(disease_effects_config):
                        # Use provided name or generate one
                        effect_name = effect.get("name", f"effect_{idx}")
                        effect_definitions[effect_name] = effect

                    subgroups = [
                        {
                            "name": "all_disease",
                            "remainder": True,
                            "apply_effects": list(effect_definitions.keys()),
                        }
                    ]
                elif isinstance(disease_effects_config, dict):
                    effect_definitions = disease_effects_config["effects"]
                    subgroups = disease_effects_config["patient_subgroups"]
                else:
                    raise TypeError(
                        "disease_effects_config must be a list (old format) or a dict (new format)."
                    )

                all_disease_patient_ids = patient_ids[idx_disease].copy()
                np.random.shuffle(
                    all_disease_patient_ids
                )  # Shuffle for random assignment
                patient_idx_start = 0

                for subgroup in subgroups:
                    n_total_disease = len(all_disease_patient_ids)

                    if "count" in subgroup:
                        num_patients = subgroup["count"]
                    elif "percentage" in subgroup:
                        num_patients = int(subgroup["percentage"] * n_total_disease)
                    elif "remainder" in subgroup and subgroup["remainder"]:
                        num_patients = n_total_disease - patient_idx_start
                    else:
                        continue

                    patient_idx_end = patient_idx_start + num_patients
                    subgroup_patient_ids = all_disease_patient_ids[
                        patient_idx_start:patient_idx_end
                    ]
                    patient_idx_start = patient_idx_end

                    if len(subgroup_patient_ids) == 0:
                        continue

                    for effect_name in subgroup.get("apply_effects", []):
                        effect = effect_definitions.get(effect_name)
                        if not effect:
                            raise ValueError(
                                f"Effect '{effect_name}' not found in definitions."
                            )
                        self.apply_stochastic_effects(
                            df_genes, subgroup_patient_ids, effect
                        )

        if gene_type.lower() == "rna-seq":
            df_genes = df_genes.round(0).astype(int)

        # --- Dynamics Injection ---
        if dynamics_config:
            injector = ScenarioInjector()
            if "evolve_features" in dynamics_config:
                evolve_args = dynamics_config["evolve_features"]
                df_genes = injector.evolve_features(
                    df_genes, evolution_config=evolve_args
                )
            if "construct_target" in dynamics_config:
                target_args = dynamics_config["construct_target"]
                df_genes = injector.construct_target(df_genes, **target_args)

        # --- Drift Injection ---
        if drift_injection_config:
            injector = DriftInjector(
                original_df=df_genes,
                generator_name="ClinicalDataGenerator_Gene",
                # Genes usually don't have time column unless passed or index logic
            )
            df_genes = injector.inject_multiple_types_of_drift(
                df=df_genes, schedule=drift_injection_config
            )

        return df_genes

    def generate_protein_data(
        self,
        n_proteins: int,
        demographic_df: pd.DataFrame = None,
        demographic_id_col: str = None,
        raw_demographic_data: pd.DataFrame = None,
        protein_correlations: np.ndarray = None,
        demographic_protein_correlations: np.ndarray = None,
        disease_effects_config: list = None,
        control_disease_ratio: float = 0.5,
        custom_protein_parameters: dict = None,
        protein_mean_log_center: float = 3.0,
        n_samples: int = 100,
        drift_injection_config: Optional[List[Dict]] = None,
        dynamics_config: Optional[Dict] = None,
    ) -> pd.DataFrame:
        if n_proteins <= 0:
            return pd.DataFrame()  # Return empty DF
        """
        Generates synthetic protein expression data using post-generation stochastic effects.
        """
        # --- 1. Handle Demographic Data ---
        patient_ids, groups, idx_control, idx_disease, demographic_marginals = (
            self._prepare_demographic_context(
                demographic_df,
                demographic_id_col,
                raw_demographic_data,
                n_samples_default=n_samples,
                control_disease_ratio=control_disease_ratio,
                use_correlation=(demographic_protein_correlations is not None),
            )
        )
        n_total_samples = len(patient_ids)

        # --- 2. Design Base Protein Parameters ---
        base_protein_marginals = [None] * n_proteins
        for i in range(n_proteins):
            # Simplified parameter design
            log_mean = np.random.normal(loc=protein_mean_log_center, scale=0.8)
            log_std = np.random.uniform(low=0.1, high=0.4)
            base_protein_marginals[i] = stats.lognorm(s=log_std, scale=np.exp(log_mean))

        # --- 3. Generate Base Correlated Data for ALL Samples ---
        if demographic_protein_correlations is not None:
            # Prepare conditioning data from raw_demographic_data
            cond_cols = [
                c
                for c in raw_demographic_data.columns
                if c != demographic_id_col and c != "Group"
            ]
            conditioning_data = raw_demographic_data[cond_cols].values

            X_proteins_base = self._generate_conditional_data(
                n_samples=n_total_samples,
                conditioning_data=conditioning_data,
                conditioning_marginals=demographic_marginals,
                target_marginals=base_protein_marginals,
                full_covariance=demographic_protein_correlations,
            )
        else:
            X_proteins_base = self._generate_correlated_module(
                n_total_samples,
                base_protein_marginals,
                protein_correlations
                if protein_correlations is not None
                else np.identity(n_proteins),
            )

        df_proteins = pd.DataFrame(
            X_proteins_base,
            columns=[f"P_{i}" for i in range(n_proteins)],
            index=patient_ids,
        )

        # --- 4. Apply Stochastic, Per-Patient Disease Effects ---
        if disease_effects_config and len(idx_disease) > 0:
            disease_patient_ids = patient_ids[idx_disease]

            for effect in disease_effects_config:
                # Validate required keys
                if "index" not in effect:
                    raise ValueError(
                        f"Invalid disease effect config. Must include 'index' key in {effect}."
                    )
                if "effect_type" not in effect or "effect_value" not in effect:
                    raise ValueError(
                        f"Invalid disease effect config. Missing 'effect_type' or 'effect_value' in {effect}."
                    )

                if effect.get("effect_type") == "additive_shift":
                    self.logger.warning(
                        "effect_type 'additive_shift' on protein data previously applied shifts "
                        "in log-space (equivalent to fold_change). Use 'fold_change' for "
                        "lognormal data, or 'simple_additive_shift' for direct additive shifts."
                    )

                self.apply_stochastic_effects(df_proteins, disease_patient_ids, effect)

        # --- Dynamics Injection ---
        if dynamics_config:
            injector = ScenarioInjector()
            if "evolve_features" in dynamics_config:
                evolve_args = dynamics_config["evolve_features"]
                df_proteins = injector.evolve_features(
                    df_proteins, evolution_config=evolve_args
                )
            if "construct_target" in dynamics_config:
                target_args = dynamics_config["construct_target"]
                df_proteins = injector.construct_target(df_proteins, **target_args)

        # --- Drift Injection ---
        if drift_injection_config:
            injector = DriftInjector(
                original_df=df_proteins, generator_name="ClinicalDataGenerator_Protein"
            )
            df_proteins = injector.inject_multiple_types_of_drift(
                df=df_proteins, schedule=drift_injection_config
            )

        return df_proteins

    def generate_target_variable(
        self,
        demographic_df: pd.DataFrame,
        omics_dfs: Union[List[pd.DataFrame], pd.DataFrame],
        weights: dict,
        noise_std: float = 0.1,
        binary_threshold: Optional[Union[float, str]] = None,
    ) -> pd.Series:
        """
        Generates a target variable Y as a linear combination of demographic and omics features.

        Args:
            demographic_df (pd.DataFrame): Demographic data.
            omics_dfs (list[pd.DataFrame] | pd.DataFrame): One or more omics dataframes.
            weights (dict): Dictionary mapping column names (or regex patterns) to coefficients.
                            Example: {'Age': 0.3, 'Sex': 0.1, 'G_.*': 0.01}
                            When a pattern matches multiple columns, weight is applied to their mean.
            noise_std (float): Standard deviation of the Gaussian noise added to Y.
            binary_threshold (float | 'median' | None): If float, binarizes Y at that value.
                            If 'median', binarizes at the median (guarantees 50/50 balance).
                            If None, returns continuous Y.

        Returns:
            pd.Series: The generated target variable Y.
        """
        if isinstance(omics_dfs, pd.DataFrame):
            omics_dfs = [omics_dfs]

        full_df = demographic_df.copy()
        for df in omics_dfs:
            if not df.index.equals(full_df.index):
                # Index names may differ (e.g. "Patient_ID" vs None) while values match
                if not (df.index == full_df.index).all():
                    raise ValueError(
                        f"Indices of demographic_df and omics_dfs must match. "
                        f"demographic_df index sample: {full_df.index[:3].tolist()}, "
                        f"omics index sample: {df.index[:3].tolist()}"
                    )
                df = df.copy()
                df.index = full_df.index
            full_df = pd.concat([full_df, df], axis=1)

        if "Sex" in full_df.columns and full_df["Sex"].dtype == "object":
            full_df["Sex"] = full_df["Sex"].map({"Male": 1, "Female": 0})

        n_samples = len(full_df)
        Y = np.zeros(n_samples)

        for pattern, weight in weights.items():
            regex = re.compile(pattern)
            matched_cols = [col for col in full_df.columns if regex.match(col)]

            if not matched_cols:
                logger.warning("No columns matched pattern '%s'. Available columns sample: %s",
                               pattern, list(full_df.columns[:10]))
                continue

            numeric_cols = [c for c in matched_cols if pd.api.types.is_numeric_dtype(full_df[c])]
            non_numeric = set(matched_cols) - set(numeric_cols)
            if non_numeric:
                logger.warning("Skipping non-numeric columns matching '%s': %s", pattern, non_numeric)

            if not numeric_cols:
                continue

            # Apply weight to the mean of matched columns (avoids inflating contribution
            # when pattern matches many columns, e.g. 'G_.*' matching 1000 genes)
            group_data = full_df[numeric_cols].mean(axis=1)
            std = group_data.std()
            group_std = (group_data - group_data.mean()) / std if std > 0 else group_data - group_data.mean()
            Y += weight * group_std

        Y += np.random.normal(0, noise_std, n_samples)

        Y_series = pd.Series(Y, index=full_df.index, name="Target_Y")

        if binary_threshold == "median":
            Y_series = (Y_series > Y_series.median()).astype(int)
        elif binary_threshold is not None:
            Y_series = (Y_series > binary_threshold).astype(int)

        return Y_series

    def generate_custom_correlated_omics_data(
        self,
        demographic_df: pd.DataFrame,
        omics_data_df: pd.DataFrame,
        patient_filter: dict = None,
        omics_subset_indices: list = None,
        correlation_matrix: np.ndarray = None,
        omics_type: str = "genes",
        gene_type: str = None,
        disease_effect_type: str = None,
        disease_effect_value: float = None,
        n_genes_total: int = 0,
        n_proteins_total: int = 0,
    ) -> pd.DataFrame:
        """
        Generates correlated omics data for a custom subset of patients and omics features.
        This method will RE-GENERATE and OVERWRITE data for the specified subset.
        """
        # --- 1. Validate Inputs ---
        if omics_type not in ["genes", "proteins", "both"]:
            raise ValueError("omics_type must be 'genes', 'proteins', or 'both'.")
        if omics_type in ["genes", "both"] and gene_type not in [
            "RNA-Seq",
            "Microarray",
        ]:
            raise ValueError(
                "gene_type must be 'RNA-Seq' or 'Microarray' if omics_type is 'genes' or 'both'."
            )
        if omics_subset_indices is None or not omics_subset_indices:
            raise ValueError("omics_subset_indices cannot be None or empty.")
        if correlation_matrix is None:
            raise ValueError("correlation_matrix cannot be None.")
        if omics_type == "both":
            raise NotImplementedError(
                "Handling 'both' omics_type is not yet implemented for custom correlations."
            )

        # --- 2. Filter Patients ---
        # If no demographic_df is provided, use the omics_data_df index
        if demographic_df is None:
            demographic_df = pd.DataFrame(index=omics_data_df.index)

        filtered_patients_df = demographic_df.copy()
        if patient_filter:
            for col, value in patient_filter.items():
                if col in filtered_patients_df.columns:
                    if isinstance(value, (list, np.ndarray)):
                        filtered_patients_df = filtered_patients_df[
                            filtered_patients_df[col].isin(value)
                        ]
                    else:
                        filtered_patients_df = filtered_patients_df[
                            filtered_patients_df[col] == value
                        ]
                else:
                    raise ValueError(
                        f"Filter column '{col}' not found in demographic_df."
                    )

        patient_ids_to_modify = filtered_patients_df.index.intersection(
            omics_data_df.index
        )
        if patient_ids_to_modify.empty:
            print("No matching patient IDs. Returning original omics_data_df.")
            return omics_data_df

        # --- 3. Prepare Omics Subset ---
        n_samples_filtered = len(patient_ids_to_modify)
        n_omics_subset = len(omics_subset_indices)

        if correlation_matrix.shape != (n_omics_subset, n_omics_subset):
            raise ValueError(
                f"correlation_matrix shape {correlation_matrix.shape} does not match subset size ({n_omics_subset})."
            )

        # --- 4. Design Omics Parameters for the Subset ---
        control_marginals_subset = []
        disease_marginals_subset = []  # In case we want to apply DE to a subset

        for i, omics_idx in enumerate(omics_subset_indices):
            data_col = omics_data_df.iloc[:, omics_idx]

            if omics_type == "genes":
                if gene_type == "Microarray":
                    loc = data_col.mean()
                    scale = data_col.std() if data_col.std() > 0 else 1.0
                    base_dist = stats.norm(loc=loc, scale=scale)
                    control_marginals_subset.append(base_dist)
                    if (
                        disease_effect_type == "additive_shift"
                        and disease_effect_value is not None
                    ):
                        disease_dist = stats.norm(
                            loc=loc + disease_effect_value, scale=scale
                        )
                        disease_marginals_subset.append(disease_dist)
                    else:
                        disease_marginals_subset.append(base_dist)
                else:  # RNA-Seq
                    mean = data_col.mean()
                    variance = data_col.var()
                    if variance > mean and mean > 0:
                        r = mean**2 / (variance - mean)
                        p = mean / variance
                        if r > 0 and 0 < p < 1:
                            base_dist = stats.nbinom(n=r, p=p)
                            control_marginals_subset.append(base_dist)
                            if (
                                disease_effect_type == "fold_change"
                                and disease_effect_value is not None
                            ):
                                new_mean = mean * disease_effect_value
                                new_p = r / (r + new_mean)
                                if r > 0 and 0 < new_p < 1:
                                    disease_marginals_subset.append(
                                        stats.nbinom(n=r, p=new_p)
                                    )
                                else:
                                    disease_marginals_subset.append(
                                        base_dist
                                    )  # Fallback
                            else:
                                disease_marginals_subset.append(base_dist)
                        else:
                            base_dist = stats.poisson(mu=max(1, mean))  # Fallback
                            control_marginals_subset.append(base_dist)
                            disease_marginals_subset.append(base_dist)
                    else:
                        base_dist = stats.poisson(mu=max(1, mean))  # Fallback
                        control_marginals_subset.append(base_dist)
                        disease_marginals_subset.append(base_dist)

            elif omics_type == "proteins":
                log_data = np.log(data_col[data_col > 0])
                if not log_data.empty:
                    s = log_data.std() if log_data.std() > 0 else 1.0
                    loc = log_data.mean()
                    base_dist = stats.lognorm(s=s, scale=np.exp(loc))
                    control_marginals_subset.append(base_dist)
                    if (
                        disease_effect_type == "additive_shift"
                        and disease_effect_value is not None
                    ):
                        disease_dist = stats.lognorm(
                            s=s, scale=np.exp(loc + disease_effect_value)
                        )
                        disease_marginals_subset.append(disease_dist)
                    else:
                        disease_marginals_subset.append(base_dist)
                else:
                    base_dist = stats.norm(loc=0, scale=1)  # Fallback
                    control_marginals_subset.append(base_dist)
                    disease_marginals_subset.append(base_dist)

        # --- 5. Generate Correlated Data ---
        # Use disease marginals if DE is specified, otherwise use control marginals
        marginals_to_use = (
            disease_marginals_subset
            if (disease_effect_type and disease_effect_value is not None)
            else control_marginals_subset
        )

        # *** THIS IS THE CORRECTED CALL ***
        generated_data_subset = self._generate_correlated_module(
            n_samples_filtered,
            marginals_to_use,
            correlation_matrix,
            is_gene_module=(omics_type == "genes" and gene_type == "RNA-Seq"),
            n_gene_vars=n_omics_subset,
        )

        # --- 6. Update Omics Data ---
        updated_omics_data_df = omics_data_df.copy()
        updated_omics_data_df.loc[
            patient_ids_to_modify, updated_omics_data_df.columns[omics_subset_indices]
        ] = generated_data_subset

        return updated_omics_data_df

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
        print("Generating longitudinal clinical data...")

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
        demographic_df: pd.DataFrame,
        omics_data_df: pd.DataFrame,
        transition_type: str,  # 'control_to_disease', 'disease_to_control', 'bidirectional'
        selection_criteria: dict,
        omics_type: str,
        gene_type: str = None,
        disease_gene_indices: list = None,
        disease_protein_indices: list = None,
        disease_effect_type: str = None,
        disease_effect_value: float = None,
        n_genes_total: int = None,
        n_proteins_total: int = None,
    ):
        """
        Injects drift by transitioning patients between control and disease groups
        and regenerating their omics data.
        """
        updated_demographic_df = demographic_df.copy()
        updated_omics_data_df = omics_data_df.copy()

        if transition_type not in [
            "control_to_disease",
            "disease_to_control",
            "bidirectional",
        ]:
            raise ValueError(
                "transition_type must be one of 'control_to_disease', 'disease_to_control', 'bidirectional'."
            )

        def get_patient_ids_for_transition(source_group: str):
            source_patients = updated_demographic_df[
                updated_demographic_df["Group"] == source_group
            ].index.tolist()
            num_to_transition = 0
            if "percentage" in selection_criteria:
                num_to_transition = int(
                    len(source_patients) * selection_criteria["percentage"]
                )
            elif "random" in selection_criteria:
                num_to_transition = min(
                    selection_criteria["random"], len(source_patients)
                )
            elif "patient_ids" in selection_criteria:
                return [
                    pid
                    for pid in selection_criteria["patient_ids"]
                    if pid in source_patients
                ]

            return np.random.choice(
                source_patients, num_to_transition, replace=False
            ).tolist()

        patient_ids_to_modify = []
        if (
            transition_type == "control_to_disease"
            or transition_type == "bidirectional"
        ):
            patient_ids_to_modify.extend(get_patient_ids_for_transition("Control"))
        if (
            transition_type == "disease_to_control"
            or transition_type == "bidirectional"
        ):
            patient_ids_to_modify.extend(get_patient_ids_for_transition("Disease"))

        if not patient_ids_to_modify:
            return updated_demographic_df, updated_omics_data_df

        # Update demographic data (vectorized)
        modify_idx = pd.Index(patient_ids_to_modify)
        mask_control = updated_demographic_df.loc[modify_idx, "Group"] == "Control"
        mask_disease = ~mask_control
        updated_demographic_df.loc[modify_idx[mask_control], "Group"] = "Disease"
        updated_demographic_df.loc[modify_idx[mask_disease], "Group"] = "Control"
        if "Binary_Group" in updated_demographic_df.columns:
            updated_demographic_df.loc[modify_idx[mask_control], "Binary_Group"] = 1
            updated_demographic_df.loc[modify_idx[mask_disease], "Binary_Group"] = 0

        # Regenerate omics data for all transitioned patients
        if patient_ids_to_modify:
            control_to_disease_ids = [
                pid
                for pid in patient_ids_to_modify
                if updated_demographic_df.loc[pid, "Group"] == "Disease"
            ]
            disease_to_control_ids = [
                pid
                for pid in patient_ids_to_modify
                if updated_demographic_df.loc[pid, "Group"] == "Control"
            ]

            # Process transitions to Disease
            if control_to_disease_ids:
                temp_demographic_df_d = updated_demographic_df.loc[
                    control_to_disease_ids
                ].reset_index()
                id_col_d = temp_demographic_df_d.columns[0]

                if omics_type == "genes" or omics_type == "both":
                    gene_effects = []
                    if (
                        disease_gene_indices
                        and disease_effect_type
                        and disease_effect_value is not None
                    ):
                        gene_effects.append(
                            {
                                "name": "transition_effect",
                                "index": disease_gene_indices,
                                "effect_type": disease_effect_type,
                                "effect_value": disease_effect_value,
                            }
                        )
                    gene_df_d = self.generate_gene_data(
                        n_genes=n_genes_total,
                        gene_type=gene_type,
                        demographic_df=temp_demographic_df_d,
                        demographic_id_col=id_col_d,
                        disease_effects_config=gene_effects,
                    )
                    gene_cols = gene_df_d.columns
                    updated_omics_data_df.loc[control_to_disease_ids, gene_cols] = (
                        gene_df_d.loc[control_to_disease_ids, gene_cols].values
                    )

                if omics_type == "proteins" or omics_type == "both":
                    protein_effects = []
                    if (
                        disease_protein_indices
                        and disease_effect_type
                        and disease_effect_value is not None
                    ):
                        protein_effects.append(
                            {
                                "name": "transition_effect",
                                "index": disease_protein_indices,
                                "effect_type": disease_effect_type,
                                "effect_value": disease_effect_value,
                            }
                        )
                    protein_df_d = self.generate_protein_data(
                        n_proteins=n_proteins_total,
                        demographic_df=temp_demographic_df_d,
                        demographic_id_col=id_col_d,
                        disease_effects_config=protein_effects,
                    )
                    protein_cols_d = protein_df_d.columns
                    updated_omics_data_df.loc[
                        control_to_disease_ids, protein_cols_d
                    ] = protein_df_d.loc[control_to_disease_ids, protein_cols_d].values

            # Process transitions to Control
            if disease_to_control_ids:
                temp_demographic_df_c = updated_demographic_df.loc[
                    disease_to_control_ids
                ].reset_index()
                id_col_c = temp_demographic_df_c.columns[0]

                if omics_type == "genes" or omics_type == "both":
                    gene_df_c = self.generate_gene_data(
                        n_genes=n_genes_total,
                        gene_type=gene_type,
                        demographic_df=temp_demographic_df_c,
                        demographic_id_col=id_col_c,
                        disease_effects_config=[],
                    )
                    gene_cols_c = gene_df_c.columns
                    updated_omics_data_df.loc[disease_to_control_ids, gene_cols_c] = (
                        gene_df_c.loc[disease_to_control_ids, gene_cols_c].values
                    )

                if omics_type == "proteins" or omics_type == "both":
                    protein_df_c = self.generate_protein_data(
                        n_proteins=n_proteins_total,
                        demographic_df=temp_demographic_df_c,
                        demographic_id_col=id_col_c,
                        disease_effects_config=[],
                    )
                    protein_cols_c = protein_df_c.columns
                    updated_omics_data_df.loc[
                        disease_to_control_ids, protein_cols_c
                    ] = protein_df_c.loc[disease_to_control_ids, protein_cols_c].values

        return updated_demographic_df, updated_omics_data_df

    def inject_drift_correlated_modules(
        self,
        omics_data_df: pd.DataFrame,
        module_indices: list,
        new_correlation_matrix: np.ndarray = None,
        add_indices: list = None,
        remove_indices: list = None,
        omics_type: str = "genes",
        gene_type: str = None,
    ) -> pd.DataFrame:
        """
        Injects drift by modifying correlated modules of omics features.
        This will RE-GENERATE data for ALL patients for the specified module indices.
        """
        updated_omics_df = omics_data_df.copy()

        if remove_indices:
            module_indices = [
                idx for idx in module_indices if idx not in remove_indices
            ]
        if add_indices:
            module_indices.extend(add_indices)
            module_indices = sorted(list(set(module_indices)))

        if not module_indices:
            return updated_omics_df

        if new_correlation_matrix is None:
            # If no new matrix, just re-generate with existing correlations
            new_correlation_matrix = (
                updated_omics_df.iloc[:, module_indices].corr().values
            )

        # Use generate_custom_correlated_omics_data to re-generate the module for ALL patients
        updated_omics_df = self.generate_custom_correlated_omics_data(
            demographic_df=pd.DataFrame(
                index=updated_omics_df.index
            ),  # Pass demographic_df to get patient IDs
            omics_data_df=updated_omics_df,
            patient_filter=None,  # Apply to all patients
            omics_subset_indices=module_indices,
            correlation_matrix=new_correlation_matrix,
            omics_type=omics_type,
            gene_type=gene_type,
        )

        return updated_omics_df

    def generate_additional_time_step_data(
        self,
        n_samples: int,
        date_value: str,
        omics_to_generate: list,  # e.g., ['genes', 'proteins']
        n_genes: int = 0,
        n_proteins: int = 0,
        gene_type: str = None,
        parameter_drift_config: dict = None,
        transition_drift_config: dict = None,
        module_drift_config: dict = None,
        **kwargs,
    ):
        """
        Generates data for a new time step, with options to inject various types of drift.
        """
        # 1. Generate baseline demographic and omics data
        demographic_df, raw_demographic_data = self.generate_demographic_data(
            n_samples=n_samples,
            date_column_name="Date",
            date_value=date_value,
            **kwargs.get("demographic_params", {}),
        )

        omics_df = pd.DataFrame(index=demographic_df.index)

        # Apply parameter drift (by passing custom parameters)
        custom_gene_params = (
            parameter_drift_config.get("custom_gene_parameters", {})
            if parameter_drift_config
            else {}
        )
        custom_protein_params = (
            parameter_drift_config.get("custom_protein_parameters", {})
            if parameter_drift_config
            else {}
        )

        if "genes" in omics_to_generate:
            gene_df = self.generate_gene_data(
                n_genes=n_genes,
                gene_type=gene_type,
                demographic_df=demographic_df,
                demographic_id_col=demographic_df.index.name,
                raw_demographic_data=raw_demographic_data,
                custom_gene_parameters=custom_gene_params,
                **kwargs.get("gene_params", {}),
            )
            omics_df = pd.concat([omics_df, gene_df], axis=1)

        if "proteins" in omics_to_generate:
            protein_df = self.generate_protein_data(
                n_proteins=n_proteins,
                demographic_df=demographic_df,
                demographic_id_col=demographic_df.index.name,
                raw_demographic_data=raw_demographic_data,
                custom_protein_parameters=custom_protein_params,
                **kwargs.get("protein_params", {}),
            )
            omics_df = pd.concat([omics_df, protein_df], axis=1)

        # 2. Apply group transition drift
        if transition_drift_config:
            demographic_df, omics_df = self.inject_drift_group_transition(
                demographic_df=demographic_df,
                omics_data_df=omics_df,
                n_genes_total=n_genes,
                n_proteins_total=n_proteins,
                gene_type=gene_type,
                **transition_drift_config,
            )

        # 3. Apply correlated module drift
        if module_drift_config:
            omics_df = self.inject_drift_correlated_modules(
                omics_data_df=omics_df, gene_type=gene_type, **module_drift_config
            )

        return demographic_df, omics_df

    def _generate_text_report(self, dfs_with_titles: list, report_title: str) -> str:
        """Helper to create a text report from a list of (title, dataframe) tuples."""
        report_stream = io.StringIO()
        report_stream.write(f"--- {report_title} ---\n\n")
        for title, df in dfs_with_titles:
            report_stream.write(f"--- {title} ---\n")
            # Use to_string to capture the full dataframe in the report
            report_stream.write(df.to_string() + "\n\n")
        return report_stream.getvalue()

    def _summarize_longitudinal_transition(
        self,
        df_demo_t2,
        idx_transicion,
        df_genes_t1,
        df_genes_t2,
        df_proteins_t1,
        df_proteins_t2,
        gene_indices,
        protein_indices,
    ):
        """Helper to summarize changes in the transition cohort."""
        summary_stream = io.StringIO()
        summary_stream.write(
            "--- LONGITUDINAL TRANSITION (DRIFT) COHORT ANALYSIS ---\n"
        )
        summary_stream.write(
            f"Number of patients transitioned: {len(idx_transicion)}\n"
        )

        if len(idx_transicion) > 0:
            summary_stream.write(
                "Gene expression changes for transitioned patients (Module A):\n"
            )
            gene_cols = df_genes_t1.columns[gene_indices]
            gene_changes = (
                df_genes_t2.loc[idx_transicion, gene_cols].mean()
                - df_genes_t1.loc[idx_transicion, gene_cols].mean()
            )
            summary_stream.write(f"Mean gene changes:\n{gene_changes.to_string()}\n\n")

            summary_stream.write(
                "Protein expression changes for transitioned patients (Module A):\n"
            )
            protein_cols = df_proteins_t1.columns[protein_indices]
            protein_changes = (
                df_proteins_t2.loc[idx_transicion, protein_cols].mean()
                - df_proteins_t1.loc[idx_transicion, protein_cols].mean()
            )
            summary_stream.write(
                f"Mean protein changes:\n{protein_changes.to_string()}\n"
            )

        return summary_stream.getvalue()


# ---------------------------------------------------------------------------
# SCRIPT/SIMULATION FUNCTION (Moved out of the class)
# ---------------------------------------------------------------------------


def replicate_genes_proteins(
    generator: ClinicalDataGenerator,
    mode,
    output_dir,
    n_samples=100,
    factor_escala=1.0,
    gene_mean_loc_center=7.0,
    gene_mean_log_center=np.log(80),
):
    """
    Runs a specific T1 vs T2 longitudinal simulation using the ClinicalDataGenerator.
    This script is now a flexible template for designing complex simulations.
    """
    if mode not in ["microarray", "rna-seq"]:
        raise ValueError("Mode must be 'microarray' or 'rna-seq'")

    n_genes = int(100 * factor_escala)  # Reduced for clarity in example
    n_proteins = int(60 * factor_escala)

    print(f"--- STARTING {mode.upper()} LONGITUDINAL SIMULATION (T1 & T2) ---")
    print(f"Simulating T1 (50/50) and T2 (Drift) for {n_samples} patients.")
    print(f"Genes: {n_genes}, Proteins: {n_proteins}")

    # --- 1. Demographic Generation (T1) ---
    df_demo_t1, raw_demo_t1 = generator.generate_demographic_data(
        n_samples, control_disease_ratio=0.5
    )

    # --- 2. Define Simulation Design: Modules, Effects, and Correlations ---

    # A. Define Modules (gene and protein indices)
    gene_indices_modA = list(range(0, 20))
    gene_indices_modB = list(range(20, 40))
    gene_indices_modC = list(range(40, 60))

    protein_indices_modA = list(range(0, 10))
    protein_indices_modB = list(range(10, 20))
    protein_indices_modC = list(range(20, 30))

    # B. Define Disease Effects Configuration for each module
    if mode == "microarray":
        gene_effects_config = [
            {
                "name": "Module_A",
                "index": gene_indices_modA,
                "effect_type": "additive_shift",
                "effect_value": [0.8, 1.2],
            },
            {
                "name": "Module_B",
                "index": gene_indices_modB,
                "effect_type": "variance_scale",
                "effect_value": [1.5, 2.0],
            },
            {
                "name": "Module_C",
                "index": gene_indices_modC,
                "effect_type": "additive_shift",
                "effect_value": [-0.6, -0.4],
            },
        ]
    else:  # rnaseq
        gene_effects_config = [
            {
                "name": "Module_A",
                "index": gene_indices_modA,
                "effect_type": "fold_change",
                "effect_value": [2.0, 3.0],
            },
            {
                "name": "Module_B",
                "index": gene_indices_modB,
                "effect_type": "fold_change",
                "effect_value": [0.5, 0.7],
            },
            # Module C has no effect in this scenario
        ]

    protein_effects_config = [
        {
            "name": "Module_A",
            "index": protein_indices_modA,
            "effect_type": "additive_shift",
            "effect_value": [np.log(1.8), np.log(2.2)],
        },
        {
            "name": "Module_B",
            "index": protein_indices_modB,
            "effect_type": "variance_scale",
            "effect_value": [1.2, 1.5],
        },
    ]

    # C. Define Block-Correlation Matrix
    def fill_block(matrix, indices, corr_value):
        for i in indices:
            for j in indices:
                if i < j:
                    matrix[i, j] = matrix[j, i] = np.random.uniform(
                        corr_value[0], corr_value[1]
                    )

    def fill_inter_block(matrix, indices1, indices2, corr_value):
        for i in indices1:
            for j in indices2:
                matrix[i, j] = matrix[j, i] = np.random.uniform(
                    corr_value[0], corr_value[1]
                )

    # Gene-Gene Correlations
    gene_correlations = np.identity(n_genes)
    fill_block(
        gene_correlations, gene_indices_modA, [0.6, 0.8]
    )  # High correlation within Module A
    fill_block(
        gene_correlations, gene_indices_modB, [0.5, 0.7]
    )  # High correlation within Module B
    fill_block(
        gene_correlations, gene_indices_modC, [0.3, 0.5]
    )  # Low correlation within Module C
    fill_inter_block(
        gene_correlations, gene_indices_modA, gene_indices_modB, [0.2, 0.4]
    )  # Medium correlation between A and B

    # Protein-Protein Correlations
    protein_correlations = np.identity(n_proteins)
    fill_block(protein_correlations, protein_indices_modA, [0.5, 0.7])
    fill_block(protein_correlations, protein_indices_modB, [0.4, 0.6])

    # --- 3. Generate T1 Data ---
    print("\n--- Generating T1 Data ---")
    df_genes_t1 = generator.generate_gene_data(
        n_genes=n_genes,
        gene_type=mode,
        demographic_df=df_demo_t1,
        demographic_id_col=df_demo_t1.index.name,
        gene_correlations=gene_correlations,
        disease_effects_config=gene_effects_config,
        gene_mean_loc_center=gene_mean_loc_center,
        gene_mean_log_center=gene_mean_log_center,
    )

    df_proteins_t1 = generator.generate_protein_data(
        n_proteins=n_proteins,
        demographic_df=df_demo_t1,
        demographic_id_col=df_demo_t1.index.name,
        protein_correlations=protein_correlations,
        disease_effects_config=protein_effects_config,
    )

    # --- 4. T1 Report and Save ---
    os.makedirs(output_dir, exist_ok=True)
    report_title_t1 = f"T1 DATA ({mode.upper()}) - Modules A, B, C"
    df_list_t1 = [
        ("DEMOGRAPHIC T1", df_demo_t1.copy()),
        ("GENES T1", df_genes_t1.copy()),
        ("PROTEINS T1", df_proteins_t1.copy()),
    ]
    report_t1 = generator._generate_text_report(df_list_t1, report_title_t1)

    with open(
        os.path.join(output_dir, f"report_t1_{mode}.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(report_t1)

    df_demo_t1.to_csv(
        os.path.join(output_dir, f"demographic_dataset_t1_{mode}.csv"), index=True
    )
    df_genes_t1.to_csv(
        os.path.join(output_dir, f"genes_dataset_t1_{mode}.csv"), index=True
    )
    df_proteins_t1.to_csv(
        os.path.join(output_dir, f"dataset_proteins_t1_{mode}.csv"), index=True
    )
    print(f"T1 files saved to: {output_dir}")

    # --- 5. T2 DRIFT GENERATION ---
    print("\n" + "=" * 50)
    print("  STEP 5: GENERATING DRIFT TO T2")
    print("=" * 50)

    idx_control_t1 = df_demo_t1[df_demo_t1["Group"] == "Control"].index
    n_control_to_disease = len(idx_control_t1) // 2
    idx_transicion = np.random.choice(
        idx_control_t1, n_control_to_disease, replace=False
    )

    df_demo_t2 = df_demo_t1.copy()
    df_demo_t2.loc[idx_transicion, "Group"] = "Disease"

    print(
        f"T2 Cohort: {len(df_demo_t2[df_demo_t2['Group'] == 'Control'])} control, {len(df_demo_t2[df_demo_t2['Group'] == 'Disease'])} disease."
    )
    print(f"{len(idx_transicion)} patients transitioned from Control to Disease.")

    # --- 6. Generate T2 Data (with new demographic context) ---
    print("\n--- Generating T2 Data (with drift) ---")
    df_genes_t2 = generator.generate_gene_data(
        n_genes=n_genes,
        gene_type=mode,
        demographic_df=df_demo_t2,
        demographic_id_col=df_demo_t2.index.name,
        gene_correlations=gene_correlations,
        disease_effects_config=gene_effects_config,
        gene_mean_loc_center=gene_mean_loc_center,
        gene_mean_log_center=gene_mean_log_center,
    )

    df_proteins_t2 = generator.generate_protein_data(
        n_proteins=n_proteins,
        demographic_df=df_demo_t2,
        demographic_id_col=df_demo_t2.index.name,
        protein_correlations=protein_correlations,
        disease_effects_config=protein_effects_config,
    )

    # --- 7. T2 Report and Save ---
    report_title_t2 = f"T2 DATA ({mode.upper()}) - WITH LONGITUDINAL DRIFT"
    df_list_t2 = [
        ("DEMOGRAPHIC T2", df_demo_t2.copy()),
        ("GENES T2", df_genes_t2.copy()),
        ("PROTEINS T2", df_proteins_t2.copy()),
    ]
    report_t2 = generator._generate_text_report(df_list_t2, report_title_t2)

    # For the summary, we can provide all affected indices
    all_gene_indices = sorted(
        list(set(gene_indices_modA + gene_indices_modB + gene_indices_modC))
    )
    all_protein_indices = sorted(
        list(set(protein_indices_modA + protein_indices_modB + protein_indices_modC))
    )

    transition_summary = generator._summarize_longitudinal_transition(
        df_demo_t2,
        idx_transicion,
        df_genes_t1,
        df_genes_t2,
        df_proteins_t1,
        df_proteins_t2,
        all_gene_indices,
        all_protein_indices,
    )
    report_t2 += "\n\n" + transition_summary

    with open(
        os.path.join(output_dir, f"report_t2_{mode}.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(report_t2)

    df_demo_t2.to_csv(
        os.path.join(output_dir, f"demographic_dataset_t2_{mode}.csv"), index=True
    )
    df_genes_t2.to_csv(
        os.path.join(output_dir, f"genes_dataset_t2_{mode}.csv"), index=True
    )
    df_proteins_t2.to_csv(
        os.path.join(output_dir, f"dataset_proteins_t2_{mode}.csv"), index=True
    )

    print(f"\nT2 files saved to: {output_dir}")
    print("\n--- LONGITUDINAL SIMULATION COMPLETED! ---")

    return df_demo_t2, df_genes_t2, df_proteins_t2
