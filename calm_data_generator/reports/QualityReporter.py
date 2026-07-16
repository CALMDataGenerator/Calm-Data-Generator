"""
Static, File-Based Real Data Reporter

This module provides the QualityReporter class, which generates a detailed,
static report comparing a real dataset with a synthetic one.
Uses YData Profiling for analysis and Plotly for interactive visualizations.

Implementation is split into mixins (_statistical_similarity_mixin.py,
_quality_scoring_mixin.py, _privacy_metrics_mixin.py, _single_cell_mixin.py,
_ml_utility_mixin.py) — this file keeps the orchestration entry points. See
ARCHITECTURE.md for the module map.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from calm_data_generator.generators.configs import ReportConfig
from calm_data_generator.reports.base import BaseReporter
from calm_data_generator.reports.DiscriminatorReporter import DiscriminatorReporter
from calm_data_generator.reports.ExternalReporter import ExternalReporter
from calm_data_generator.reports.LocalIndexGenerator import LocalIndexGenerator
from calm_data_generator.reports.Visualizer import Visualizer

from ._ml_utility_mixin import _MLUtilityMixin
from ._privacy_metrics_mixin import _PrivacyMetricsMixin
from ._quality_scoring_mixin import _QualityScoringMixin
from ._single_cell_mixin import _SingleCellMixin
from ._statistical_similarity_mixin import _StatisticalSimilarityMixin

logger = logging.getLogger("QualityReporter")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)


class QualityReporter(
    BaseReporter,
    _StatisticalSimilarityMixin,
    _QualityScoringMixin,
    _PrivacyMetricsMixin,
    _SingleCellMixin,
    _MLUtilityMixin,
):
    """
    Generates a static, file-based report comparing a real dataset and its synthetic counterpart.
    Uses YData Profiling and Plotly for visualizations.
    """

    def __init__(self, verbose: bool = True, minimal: bool = False):
        """
        Initializes the QualityReporter.

        Args:
            verbose (bool): If True, prints progress messages to the console.
            minimal (bool): If True, skips expensive computations (PCA/UMAP, full correlations).
        """
        super().__init__(verbose=verbose, minimal=minimal)
        self.discriminator_reporter = DiscriminatorReporter(verbose=verbose)

    def calculate_quality_metrics(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculates quality metrics (SDMetrics) for two datasets without generating a full report.

        Args:
           real_df (pd.DataFrame): The original/real dataset.
           synthetic_df (pd.DataFrame): The generated/synthetic dataset.

        Returns:
           Dict[str, float]: A dictionary containing 'overall_quality_score' and 'weighted_quality_score'.
                             Returns {'error': ...} if SDMetrics is not available or fails.
        """
        return self._assess_quality_scores(real_df, synthetic_df)

    def calculate_ari(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        target_col: str
    ) -> Dict[str, float]:
        """
        Public method to calculate ARI metrics without generating a full report.
        """
        return self._calculate_ari_metrics(real_df, synthetic_df, target_col) or {}

    def generate_report(self, *args, **kwargs):
        """Wrapper for generate_comprehensive_report to satisfy BaseReporter contract."""
        return self.generate_comprehensive_report(*args, **kwargs)

    def evaluate(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Lightweight, in-memory fidelity evaluation — no files written.

        Computes the same metrics as `generate_comprehensive_report` (SDMetrics quality
        scores, statistical similarity tests, and TSTR if a target is given) without
        writing any HTML or JSON to disk. Use this for programmatic checks (e.g. inside
        a training loop or a test), and `generate_comprehensive_report` when you want the
        full HTML dashboard.

        Args:
            real_df (pd.DataFrame): The original/real dataset.
            synthetic_df (pd.DataFrame): The generated/synthetic dataset.
            target_column (Optional[str]): If provided, also runs TSTR (Train Synthetic,
                Test Real) for this column.

        Returns:
            Dict[str, Any]: {
                "quality_scores": SDMetrics overall/weighted scores,
                "statistical_metrics": MMD, KS, Wasserstein, Levene per numeric column,
                "tstr_metrics": TSTR metrics (only if target_column is given), or None,
            }
        """
        quality_scores = self._assess_quality_scores(real_df, synthetic_df)
        statistical_metrics = self._compute_statistical_tests(real_df, synthetic_df)

        tstr_metrics = None
        if target_column:
            tstr_result = self._compute_tstr(real_df, synthetic_df, target_column)
            if tstr_result is not None:
                tstr_metrics, _ = tstr_result

        return {
            "quality_scores": quality_scores,
            "statistical_metrics": statistical_metrics,
            "tstr_metrics": tstr_metrics,
        }

    def generate_comprehensive_report(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        generator_name: str,
        output_dir: str,
        target_column: Optional[str] = None,
        block_column: Optional[str] = None,
        focus_cols: Optional[List[str]] = None,
        drift_config: Optional[Dict[str, Any]] = None,
        time_col: Optional[str] = None,
        drift_history: Optional[List[Dict[str, Any]]] = None,
        resample_rule: Optional[Union[str, int]] = None,
        constraints_stats: Optional[Dict[str, int]] = None,
        privacy_check: bool = False,
        minimal: Optional[bool] = None,
        discriminator: bool = False,
        tstr: bool = False,
        spearman: bool = False,
        report_config: Optional[Union[ReportConfig, Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive file-based report comparing real and synthetic data.
        Can use ReportConfig or individual arguments.

        Merge semantics when both `report_config` and individual arguments are given:
        - Non-boolean settings (target_column, block_column, focus_cols, time_col,
          resample_rule, constraints_stats, minimal, use_scgft) come from `report_config`;
          the individual arguments are only used to build `report_config` when it is not
          passed at all.
        - Boolean feature flags (privacy_check, discriminator, tstr, spearman) and
          `output_dir` use OR semantics: an explicitly-True/non-default argument always
          wins, even if `report_config` says otherwise. This avoids silently dropping a
          flag the caller explicitly asked for.

        Returns:
            Dict[str, Any]: The same results dict written to report_results.json
            (quality scores, statistical tests, TSTR, privacy, ARI, etc.).
        """
        # Resolve Configuration
        if report_config:
            if isinstance(report_config, dict):
                report_config = ReportConfig(**report_config)
        else:
            # Build config from individual arguments
            report_config = ReportConfig(
                output_dir=output_dir,
                target_column=target_column,
                block_column=block_column,
                focus_columns=focus_cols,
                time_col=time_col,
                resample_rule=resample_rule,
                constraints_stats=constraints_stats,
                privacy_check=privacy_check,
                minimal=minimal if minimal is not None else self.minimal,
                discriminator=discriminator,
                tstr=tstr,
                spearman=spearman,
                auto_report=True,  # implied
            )

        # Explicit non-default arguments always win for output_dir and boolean flags,
        # even when a report_config was also passed (see merge semantics above).
        if output_dir != report_config.output_dir and output_dir != "output":
            report_config.output_dir = output_dir

        # Use config values
        output_dir = report_config.output_dir
        target_column = report_config.target_column
        block_column = report_config.block_column
        focus_cols = report_config.focus_columns
        time_col = report_config.time_col
        resample_rule = report_config.resample_rule
        constraints_stats = report_config.constraints_stats
        privacy_check = report_config.privacy_check or privacy_check
        use_minimal = report_config.minimal
        discriminator = report_config.discriminator or discriminator
        tstr = report_config.tstr or tstr
        spearman = report_config.spearman or spearman
        use_scgft = report_config.use_scgft

        if self.verbose:
            logger.info(
                "COMPREHENSIVE REAL DATA GENERATION REPORT | Generator: %s | Timestamp: %s",
                generator_name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

        os.makedirs(output_dir, exist_ok=True)

        # Determine time column
        final_time_col = (
            time_col
            if time_col and time_col in real_df.columns
            else "timestamp"
            if "timestamp" in real_df.columns
            else None
        )

        # === Resampling Logic ===
        if resample_rule is not None:
            real_df_for_report = self._apply_resampling(
                real_df, final_time_col, block_column, resample_rule
            )
            synthetic_df_for_report = self._apply_resampling(
                synthetic_df, final_time_col, block_column, resample_rule
            )
        else:
            real_df_for_report = real_df
            synthetic_df_for_report = synthetic_df

        # === TSTR ===
        tstr_metrics = None
        if tstr and target_column:
            tstr_metrics = self._run_tstr(real_df, synthetic_df, target_column, output_dir)
        elif tstr and not target_column:
            logger.warning("tstr=True but no target_column was given; TSTR skipped.")

        # === Statistical Tests (MMD, KS, Levene) ===
        statistical_metrics = self._run_statistical_tests(real_df_for_report, synthetic_df_for_report, output_dir)

        # === Quality Assessment ===
        sdmetrics_quality, sequential_quality, privacy_metrics = self._run_quality_assessment(
            real_df, real_df_for_report, synthetic_df, synthetic_df_for_report,
            generator_name, block_column, final_time_col, privacy_check,
            output_dir, target_column,
        )

        # === Quality Scores by Block (for evolution plot) ===
        quality_scores_by_block = []
        if block_column and block_column in real_df.columns:
            quality_scores_by_block = self._calculate_quality_by_block(
                real_df, synthetic_df, block_column
            )

            if quality_scores_by_block:
                block_labels = [s["block"] for s in quality_scores_by_block]
                Visualizer.generate_quality_evolution_plot(
                    scores=quality_scores_by_block,
                    output_dir=output_dir,
                    x_labels=block_labels,
                )

        # === Save Results JSON ===
        results = {
            "generator_name": generator_name,
            "generation_timestamp": datetime.now().isoformat(),
            "real_rows": len(real_df),
            "synthetic_rows": len(synthetic_df),
            "quality_scores": sdmetrics_quality,
            "quality_by_block": quality_scores_by_block if quality_scores_by_block else None,
            "compared_data_files": {"original": "real_data", "generated": "synthetic_data"},
            "sequential_quality": sequential_quality,
            "privacy_metrics": privacy_metrics,
            "constraints_stats": constraints_stats,
            "ari_metrics": self._calculate_ari_metrics(real_df_for_report, synthetic_df_for_report, target_column),
            "tstr_metrics": tstr_metrics,
            "statistical_metrics": statistical_metrics,
        }

        results_path = os.path.join(output_dir, "report_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        # === Generate Plotly Visualizations ===
        self._run_visualizations(
            real_df_for_report, synthetic_df_for_report,
            output_dir, focus_cols, target_column, drift_config,
            use_minimal, discriminator, spearman,
        )

        # === Generate YData Reports ===
        self._run_ydata_reports(
            real_df_for_report, synthetic_df_for_report,
            output_dir, final_time_col, block_column, use_minimal,
        )

        # === scGFT Single-Cell Evaluation ===
        if use_scgft:
            self._run_scgft_evaluation(real_df_for_report, synthetic_df_for_report, output_dir, target_column)

        # === Generate Dashboard ===
        LocalIndexGenerator.create_index(output_dir)

        if self.verbose:
            logger.info("Report generated at: %s", output_dir)

        return results

    def _run_quality_assessment(
        self,
        real_df: pd.DataFrame,
        real_df_for_report: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        synthetic_df_for_report: pd.DataFrame,
        generator_name: str,
        block_column: Optional[str],
        final_time_col: Optional[str],
        privacy_check: bool,
        output_dir: str,
        target_column: Optional[str],
    ):
        """Runs SDMetrics, sequential quality, and privacy checks. Returns (sdmetrics_quality, sequential_quality, privacy_metrics)."""
        sdmetrics_quality = self._assess_quality_scores(real_df_for_report, synthetic_df_for_report)

        sequential_quality = None
        _seq_available = True
        try:
            from sdmetrics.reports.sequential import QualityReport as _  # noqa: F401
        except ImportError:
            _seq_available = False
        if final_time_col and block_column and _seq_available:
            sequential_quality = self._assess_sequential_quality(
                real_df, synthetic_df, block_column, final_time_col
            )

        privacy_metrics = None
        if privacy_check or "dp" in generator_name.lower():
            privacy_metrics = self._calculate_dcr_privacy(real_df, synthetic_df)
            # Singling-Out risk (anonymeter) is optional — None if the dependency
            # isn't installed. Nested under its own key so it never disturbs the
            # existing DCR/NNDR fields at the top level.
            singling_out = self._calculate_singling_out_risk(real_df, synthetic_df)
            if singling_out is not None:
                privacy_metrics = privacy_metrics or {}
                privacy_metrics["singling_out"] = singling_out

        if "overall_quality_score" in sdmetrics_quality:
            if self.verbose:
                logger.info("Generating Quality Scores Card...")
            Visualizer.generate_quality_scores_card(
                overall_score=sdmetrics_quality["overall_quality_score"],
                weighted_score=sdmetrics_quality["weighted_quality_score"],
                output_dir=output_dir,
            )

        return sdmetrics_quality, sequential_quality, privacy_metrics

    def _run_visualizations(
        self,
        real_df_for_report: pd.DataFrame,
        synthetic_df_for_report: pd.DataFrame,
        output_dir: str,
        focus_cols: Optional[List[str]],
        target_column: Optional[str],
        drift_config: Optional[Dict[str, Any]],
        use_minimal: bool,
        discriminator: bool,
        spearman: bool,
    ) -> None:
        """Generates Plotly visualizations: density, PCA, comparison, discriminator."""
        if self.verbose:
            logger.info("Generating Plotly Visualizations...")

        Visualizer.generate_density_plots(
            df=synthetic_df_for_report,
            output_dir=output_dir,
            columns=focus_cols,
            color_col=target_column,
        )

        if not use_minimal:
            combined_df = pd.concat(
                [
                    real_df_for_report.assign(_source="Real"),
                    synthetic_df_for_report.assign(_source="Synthetic"),
                ],
                ignore_index=True,
            )
            Visualizer.generate_dimensionality_plot(
                df=combined_df,
                output_dir=output_dir,
                color_col="_source",
            )
        elif self.verbose:
            logger.info("Skipping PCA/UMAP (minimal mode)")

        if use_minimal and self.verbose:
            logger.info("Skipping full quality assessment (minimal mode)")

        Visualizer.generate_comparison_plots(
            original_df=real_df_for_report,
            drifted_df=synthetic_df_for_report,
            output_dir=output_dir,
            columns=focus_cols,
            drift_config=drift_config,
        )

        if spearman:
            Visualizer.generate_spearman_heatmaps(
                real_df=real_df_for_report,
                synthetic_df=synthetic_df_for_report,
                output_dir=output_dir,
            )

        Visualizer.generate_qq_plots(
            real_df=real_df_for_report,
            synthetic_df=synthetic_df_for_report,
            output_dir=output_dir,
        )

        if discriminator and not use_minimal:
            try:
                self.discriminator_reporter.generate_report(
                    real_df=real_df_for_report,
                    synthetic_df=synthetic_df_for_report,
                    output_dir=output_dir,
                    label_real="Real/Original",
                    label_synthetic="Synthetic/Drifted",
                )
            except Exception as e:
                self.logger.error(f"Discriminator report generation failed: {e}")

    def _run_ydata_reports(
        self,
        real_df_for_report: pd.DataFrame,
        synthetic_df_for_report: pd.DataFrame,
        output_dir: str,
        final_time_col: Optional[str],
        block_column: Optional[str],
        use_minimal: bool,
    ) -> None:
        """Generates YData Profiling comparison and per-block reports."""
        if self.verbose:
            logger.info("Generating YData Reports...")

        ExternalReporter.generate_comparison(
            ref_df=real_df_for_report,
            curr_df=synthetic_df_for_report,
            output_dir=output_dir,
            ref_name="Original / Real",
            curr_name="Generated / Synthetic",
            time_col=final_time_col,
            block_col=block_column,
            minimal=use_minimal,
        )

        ExternalReporter.generate_profile(
            df=synthetic_df_for_report,
            output_dir=output_dir,
            filename="generated_profile.html",
            time_col=final_time_col,
            block_col=block_column,
            title="Generated Data Profile",
            minimal=use_minimal,
        )

        if block_column and block_column in real_df_for_report.columns:
            if self.verbose:
                logger.info("Generating Per-Block Reports (Block Col: %s)...", block_column)

            blocks = sorted(real_df_for_report[block_column].unique(), key=str)
            blocks_dir = os.path.join(output_dir, "blocks_reports")
            os.makedirs(blocks_dir, exist_ok=True)

            for block_id in blocks:
                real_blk = real_df_for_report[real_df_for_report[block_column] == block_id]
                synth_blk = synthetic_df_for_report[synthetic_df_for_report[block_column] == block_id]

                if real_blk.empty or synth_blk.empty:
                    continue

                block_out_dir = os.path.join(blocks_dir, f"block_{str(block_id)}")
                os.makedirs(block_out_dir, exist_ok=True)

                ExternalReporter.generate_comparison(
                    ref_df=real_blk,
                    curr_df=synth_blk,
                    output_dir=block_out_dir,
                    ref_name=f"Block {block_id} (Real)",
                    curr_name=f"Block {block_id} (Synth)",
                    time_col=final_time_col,
                    minimal=use_minimal,
                )

    def update_report_after_drift(
        self,
        original_df: pd.DataFrame,
        drifted_df: pd.DataFrame,
        output_dir: str,
        drift_config: Optional[Dict[str, Any]] = None,
        time_col: Optional[str] = None,
        block_column: Optional[str] = None,
        resample_rule: Optional[Union[str, int]] = None,
    ) -> None:
        """
        Updates the report after drift injection.
        """
        generator_name = (
            drift_config.get("generator_name", "DriftInjector")
            if drift_config
            else "DriftInjector"
        )

        self.generate_comprehensive_report(
            real_df=original_df,
            synthetic_df=drifted_df,
            generator_name=generator_name,
            output_dir=output_dir,
            focus_cols=drift_config.get("feature_cols") if drift_config else None,
            drift_config=drift_config,
            time_col=time_col,
            block_column=block_column,
            resample_rule=resample_rule,
        )
