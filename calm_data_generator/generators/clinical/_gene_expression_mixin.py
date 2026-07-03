"""Gene expression data generation mixin for ClinicalDataGenerator."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from calm_data_generator.generators.drift.DriftInjector import DriftInjector
from calm_data_generator.generators.dynamics.ScenarioInjector import ScenarioInjector


class _GeneExpressionMixin:
    """Mixin providing gene expression data generation for ClinicalDataGenerator."""

    def generate_gene_data(
        self,
        n_genes: int,
        gene_type: str,  # "RNA-Seq" or "Microarray"
        demographic_df: pd.DataFrame = None,
        demographic_id_col: str = None,
        raw_demographic_data: pd.DataFrame = None,
        gene_correlations: np.ndarray = None,
        demographic_gene_correlations: np.ndarray = None,
        disease_effects_config: dict = None,
        subgroup_col: str = None,
        gene_mean_log_center: float = np.log(80),
        gene_mean_loc_center: float = 7.0,
        control_disease_ratio: float = 0.5,
        custom_gene_parameters: dict = None,
        n_samples: int = 100,
        random_state: int = 42,
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
                and c not in self._EXCLUDED_DEMO_COLS
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
