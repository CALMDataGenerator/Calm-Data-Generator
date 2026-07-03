"""Protein expression data generation mixin for ClinicalDataGenerator."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from calm_data_generator.generators.drift.DriftInjector import DriftInjector
from calm_data_generator.generators.dynamics.ScenarioInjector import ScenarioInjector


class _ProteinExpressionMixin:
    """Mixin providing protein expression data generation for ClinicalDataGenerator."""

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
        """
        Generates synthetic protein expression data using post-generation stochastic effects.
        """
        if n_proteins <= 0:
            return pd.DataFrame()  # Return empty DF
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
                if c != demographic_id_col
                and c not in self._EXCLUDED_DEMO_COLS
                and pd.api.types.is_numeric_dtype(raw_demographic_data[c])
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
