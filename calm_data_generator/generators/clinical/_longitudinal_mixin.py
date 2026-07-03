"""Longitudinal (multi-visit) and time-step drift mixin for ClinicalDataGenerator."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from calm_data_generator.generators.configs import DateConfig

logger = logging.getLogger(__name__)


class _LongitudinalMixin:
    """Mixin providing longitudinal/multi-visit data generation and drift for ClinicalDataGenerator."""

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
        logger.info("Generating longitudinal clinical data...")

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
