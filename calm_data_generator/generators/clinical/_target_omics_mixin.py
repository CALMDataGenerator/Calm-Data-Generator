"""Target-variable synthesis and custom correlated omics regeneration mixin
for ClinicalDataGenerator.
"""

import logging
import re
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class _TargetOmicsMixin:
    """Mixin providing target-variable and custom correlated omics generation for ClinicalDataGenerator."""

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
        # Backward compat: expose numeric Sex as Sex_Binario if callers use that key in weights
        if "Sex_Binario" not in full_df.columns and "Sex" in full_df.columns:
            full_df["Sex_Binario"] = full_df["Sex"]

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
            logger.warning("No matching patient IDs. Returning original omics_data_df.")
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
