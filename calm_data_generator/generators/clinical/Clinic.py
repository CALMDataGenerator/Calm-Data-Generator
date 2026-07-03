import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from calm_data_generator.generators.complex.ComplexGenerator import ComplexGenerator
from calm_data_generator.generators.configs import DateConfig

from ._demographic_mixin import _DemographicMixin
from ._gene_expression_mixin import _GeneExpressionMixin
from ._longitudinal_mixin import _LongitudinalMixin
from ._omics_params_mixin import _OmicsParamsMixin
from ._protein_expression_mixin import _ProteinExpressionMixin
from ._reporting_mixin import _ReportingMixin
from ._target_omics_mixin import _TargetOmicsMixin

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logger = logging.getLogger(__name__)


class ClinicalDataGenerator(
    ComplexGenerator,
    _DemographicMixin,
    _OmicsParamsMixin,
    _GeneExpressionMixin,
    _ProteinExpressionMixin,
    _TargetOmicsMixin,
    _LongitudinalMixin,
    _ReportingMixin,
):
    """
    A class to generate synthetic clinical data including demographic, gene expression, and protein data.

    Implementation is split into mixins (_demographic_mixin.py, _omics_params_mixin.py,
    _gene_expression_mixin.py, _protein_expression_mixin.py, _target_omics_mixin.py,
    _longitudinal_mixin.py, _reporting_mixin.py) — this file keeps the constructor and
    the main `generate()` orchestrator. See ARCHITECTURE.md for the module map.
    """

    _EXCLUDED_DEMO_COLS = {"Group", "Binary_Group", "Disease_Subgroup"}

    @staticmethod
    def get_conditioning_columns(raw_demographic_data: "pd.DataFrame",
                                  demographic_id_col: str = "Patient_ID") -> list:
        """Returns the ordered list of numeric demographic columns used as conditioning input.
        Use len() of this list as n_demo when calling build_correlation_matrix."""
        return [
            c for c in raw_demographic_data.columns
            if c != demographic_id_col
            and c not in ClinicalDataGenerator._EXCLUDED_DEMO_COLS
            and pd.api.types.is_numeric_dtype(raw_demographic_data[c])
        ]

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

    logger.info(
        "STARTING %s LONGITUDINAL SIMULATION (T1 & T2) | %d patients | Genes: %d, Proteins: %d",
        mode.upper(), n_samples, n_genes, n_proteins,
    )

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
    logger.info("Generating T1 Data...")
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
    logger.info("T1 files saved to: %s", output_dir)

    # --- 5. T2 DRIFT GENERATION ---
    logger.info("STEP 5: GENERATING DRIFT TO T2")

    idx_control_t1 = df_demo_t1[df_demo_t1["Group"] == "Control"].index
    n_control_to_disease = len(idx_control_t1) // 2
    idx_transicion = np.random.choice(
        idx_control_t1, n_control_to_disease, replace=False
    )

    df_demo_t2 = df_demo_t1.copy()
    df_demo_t2.loc[idx_transicion, "Group"] = "Disease"

    logger.info(
        "T2 Cohort: %d control, %d disease. %d patients transitioned.",
        len(df_demo_t2[df_demo_t2["Group"] == "Control"]),
        len(df_demo_t2[df_demo_t2["Group"] == "Disease"]),
        len(idx_transicion),
    )

    # --- 6. Generate T2 Data (with new demographic context) ---
    logger.info("Generating T2 Data (with drift)...")
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

    logger.info("T2 files saved to: %s | LONGITUDINAL SIMULATION COMPLETED", output_dir)

    return df_demo_t2, df_genes_t2, df_proteins_t2
