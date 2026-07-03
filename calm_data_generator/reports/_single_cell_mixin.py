"""Single-cell (scGFT) evaluation mixin for QualityReporter."""

import contextlib
import io
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("QualityReporter")


class _SingleCellMixin:
    """Mixin providing scGFT single-cell evaluation for QualityReporter."""

    def generate_scgft_report(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        output_dir: str,
        target_column: Optional[str] = None,
    ) -> None:
        """
        Runs only the scGFT single-cell evaluation without the full report pipeline.
        """
        os.makedirs(output_dir, exist_ok=True)
        self._run_scgft_evaluation(real_df, synthetic_df, output_dir, target_column)

    def _run_scgft_evaluation(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        output_dir: str,
        target_col: Optional[str] = None
    ) -> None:
        """
        Runs the scGFT_evaluador single-cell validation and saves output to HTML.
        """
        try:
            import anndata as ad
            import scanpy as sc
            from scgft_evaluator import ScGFT_Evaluator
        except ImportError:
            if self.verbose:
                logger.warning(
                    "scgft-evaluator not found. "
                    "Install with: pip install git+https://github.com/nasim23ea/scgft-evaluator.git"
                )
            return

        if self.verbose:
            logger.info("RUNNING scGFT SINGLE-CELL EVALUATION")

        try:
            # 1. Convert to AnnData
            # Assume all numeric columns are gene expression
            numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col and target_col in numeric_cols:
                numeric_cols.remove(target_col)

            adata_real = ad.AnnData(real_df[numeric_cols])
            adata_synth = ad.AnnData(synthetic_df[numeric_cols])

            if target_col and target_col in real_df.columns:
                adata_real.obs["cell_type"] = real_df[target_col].astype(str).values
                adata_synth.obs["cell_type"] = synthetic_df[target_col].astype(str).values
            else:
                # Mock cell types if not provided
                adata_real.obs["cell_type"] = "unknown"
                adata_synth.obs["cell_type"] = "unknown"

            # 2. Basic Preprocessing for scvi metrics (PCA is required)
            if self.verbose:
                logger.info("Preprocessing AnnData (PCA)...")

            sc.pp.pca(adata_real)
            sc.pp.pca(adata_synth)

            # 3. Determine groups and gene list
            genes_top = numeric_cols
            # Sort groups for deterministic orden regardless of appearance order
            grupos = sorted([str(g) for g in adata_real.obs["cell_type"].unique().tolist()])
            if len(grupos) < 2:
                raise ValueError("scGFT evaluation requires at least 2 groups in target_col.")
            grupo_a, grupo_b = grupos[0], grupos[1]

            # 4. Run evaluator
            # Seed global RNG before MMD permutation test for reproducibility
            np.random.seed(42)
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                results = ScGFT_Evaluator.run_all(
                    adata_real, adata_synth,
                    genes_top=genes_top,
                    col_grupo="cell_type",
                    grupo_a=grupo_a,
                    grupo_b=grupo_b,
                )

            output_text = f.getvalue()

            if self.verbose:
                logger.info("scGFT output:\n%s\n%s", output_text, results.to_string(index=False))

            # 4. Save to HTML Report
            scgft_report_path = os.path.join(output_dir, "scgft_report.html")

            html_content = f"""
            <html>
            <head>
                <title>scGFT Single-Cell Evaluation</title>
                <style>
                    body {{ font-family: 'Inter', sans-serif; background: #f8fafc; padding: 40px; color: #1e293b; }}
                    .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }}
                    h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 15px; }}
                    pre {{ background: #1e293b; color: #f8fafc; padding: 20px; border-radius: 8px; overflow-x: auto; font-size: 14px; line-height: 1.5; }}
                    .metric-box {{ background: #f1f5f9; border-left: 4px solid #3b82f6; padding: 15px; margin: 20px 0; }}
                    .footer {{ margin-top: 30px; font-size: 12px; color: #64748b; text-align: center; }}
                    .results-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
                    .results-table th {{ background: #3b82f6; color: white; padding: 8px 12px; text-align: left; }}
                    .results-table td {{ padding: 8px 12px; border-bottom: 1px solid #e2e8f0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>scGFT Single-Cell Evaluation Report</h1>
                    <div class="metric-box">
                        <strong>Evaluation Methodology:</strong> Graph Fourier Transform based manifold preservation.
                    </div>
                    <pre>{output_text}</pre>
                    <h2>Results</h2>
                    {results.to_html(index=False, border=0, classes="results-table")}
                    <div class="footer">
                        Generated by calm_data_generator with scgft-evaluator support.
                    </div>
                </div>
            </body>
            </html>
            """

            with open(scgft_report_path, "w") as html_file:
                html_file.write(html_content)

            if self.verbose:
                logger.info("scGFT report saved to: %s", scgft_report_path)

        except Exception as e:
            logger.error("scGFT evaluation failed: %s", e)
