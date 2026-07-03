"""Reporting and summary-text helpers mixin for ClinicalDataGenerator."""

import io
import logging
import os

from calm_data_generator.generators.clinical.ClinicReporter import ClinicReporter

logger = logging.getLogger(__name__)


class _ReportingMixin:
    """Mixin providing report generation and text-summary helpers for ClinicalDataGenerator."""

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
            logger.info("Generated report for %s in %s", name, report_dir)
        except Exception as e:
            logger.warning("Failed to generate report for %s: %s", name, e)

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
