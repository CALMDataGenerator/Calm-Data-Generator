"""
Static, File-Based Real Data Reporter

This module provides the QualityReporter class, which generates a detailed,
static report comparing a real dataset with a synthetic one.
Uses YData Profiling for analysis and Plotly for interactive visualizations.
"""

import contextlib
import io
import json
import logging
import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from calm_data_generator.generators.configs import ReportConfig
from calm_data_generator.reports.base import BaseReporter
from calm_data_generator.reports.DiscriminatorReporter import DiscriminatorReporter
from calm_data_generator.reports.ExternalReporter import ExternalReporter
from calm_data_generator.reports.LocalIndexGenerator import LocalIndexGenerator
from calm_data_generator.reports.Visualizer import Visualizer

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


class QualityReporter(BaseReporter):
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
    ) -> None:
        """
        Generates a comprehensive file-based report comparing real and synthetic data.
        Can use ReportConfig or individual arguments.
        """
        # Resolve Configuration
        # Defaults if not provided in args
        if report_config:
            if isinstance(report_config, dict):
                report_config = ReportConfig(**report_config)
        else:
            # Create from args
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
                auto_report=True,  # implied
            )

        # Override config with explicit non-None args (if mixed usage)
        # But for simplicity, let's assume if report_config is passed, it is the source of truth,
        # unless args are explicitly provided to override?
        # A simple merge approach:
        if (
            output_dir != report_config.output_dir and output_dir != "output"
        ):  # "output" is default
            report_config.output_dir = output_dir

        # Use config values
        output_dir = report_config.output_dir
        target_column = report_config.target_column
        block_column = report_config.block_column
        focus_cols = report_config.focus_columns
        time_col = report_config.time_col
        resample_rule = report_config.resample_rule
        constraints_stats = report_config.constraints_stats
        privacy_check = report_config.privacy_check
        use_minimal = report_config.minimal
        discriminator = report_config.discriminator
        use_scgft = report_config.use_scgft

        # Force minimal override if self.minimal is explicitly True?
        # If config says False but self.minimal is True...
        # Let's say config wins for this execution.

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

    @staticmethod
    def _build_sdmetrics_metadata(df: pd.DataFrame) -> Dict[str, Any]:
        """Infers SDMetrics column metadata from DataFrame dtypes."""
        cols = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                cols[col] = {"sdtype": "numerical"}
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                cols[col] = {"sdtype": "datetime"}
            else:
                cols[col] = {"sdtype": "categorical"}
        return {"columns": cols}

    def _assess_quality_scores(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Assesses the quality of synthetic data using SDMetrics."""
        try:
            from sdmetrics.reports.single_table import QualityReport
        except ImportError:
            return {"error": "SDMetrics not available"}

        try:
            if self.verbose:
                logger.info("Running SDMetrics Quality Assessment...")

            common_cols = list(set(real_df.columns) & set(synthetic_df.columns))
            if len(common_cols) < len(real_df.columns) and self.verbose:
                dropped = set(real_df.columns) - set(common_cols)
                logger.info("Aligning columns (dropped: %s)", dropped)

            real_aligned = real_df[common_cols]
            synth_aligned = synthetic_df[common_cols]

            md_dict = self._build_sdmetrics_metadata(real_aligned)

            report = QualityReport()
            report.generate(real_aligned, synth_aligned, md_dict)

            overall_score = report.get_score()
            weighted_score = self._get_weighted_score(
                real_df, synthetic_df, overall_score
            )

            if self.verbose:
                logger.info(
                    "SDMetrics Assessment complete. Overall: %.2f, Weighted: %.2f",
                    overall_score,
                    weighted_score,
                )

            return {
                "overall_quality_score": round(overall_score, 4),
                "weighted_quality_score": round(weighted_score, 4),
            }

        except Exception as e:
            self.logger.error(f"quality assessment failed: {e}")
            return {"error": str(e)}

    def _calculate_quality_by_block(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        block_column: str,
    ) -> List[Dict[str, Any]]:
        """
        Calculates  quality scores for each block.
        """
        try:
            from sdmetrics.reports.single_table import QualityReport
        except ImportError:
            return []

        scores = []
        try:
            unique_blocks = sorted(real_df[block_column].unique(), key=str)
            # Build metadata once — all blocks share the same schema
            md_dict = self._build_sdmetrics_metadata(real_df)

            for block_id in unique_blocks:
                real_block = real_df[real_df[block_column] == block_id]
                synth_block = synthetic_df[synthetic_df[block_column] == block_id]

                if real_block.empty or synth_block.empty:
                    continue

                try:

                    report = QualityReport()
                    report.generate(
                        real_block,
                        synth_block,
                        md_dict,
                    )

                    overall = report.get_score()
                    weighted = self._get_weighted_score(
                        real_block, synth_block, overall
                    )

                    scores.append(
                        {
                            "block": str(block_id),
                            "overall": round(overall, 4),
                            "weighted": round(weighted, 4),
                        }
                    )

                except Exception as e:
                    self.logger.warning(f" failed for block {block_id}: {e}")

        except Exception as e:
            self.logger.error(f"Block-wise calculation failed: {e}")

        return scores

    def _get_weighted_score(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, base_score: float
    ) -> float:
        """
        Calculates a weighted score, penalized by data duplication and null values.
        """
        if synthetic_df.empty:
            return 0.0

        # Internal duplicates
        internal_dup_count = synthetic_df.duplicated().sum()

        # Cross duplicates
        try:
            shared_cols = [c for c in synthetic_df.columns if c in real_df.columns]
            real_unique = real_df[shared_cols].drop_duplicates()
            synth_cast = synthetic_df[shared_cols].copy()
            for col in shared_cols:
                try:
                    synth_cast[col] = synth_cast[col].astype(real_unique[col].dtype, errors="ignore")
                except (ValueError, TypeError):
                    pass
            merged = synth_cast.merge(real_unique, on=shared_cols, how="left", indicator=True)
            cross_dup_count = (merged["_merge"] == "both").sum()
        except Exception as e:
            self.logger.warning(f"Could not compute cross-duplication count; defaulting to 0. Reason: {e}")
            cross_dup_count = 0

        total_bad = internal_dup_count + cross_dup_count
        duplication_penalty = (
            total_bad / len(synthetic_df) if len(synthetic_df) > 0 else 0.0
        )

        null_penalty = (
            synthetic_df.isnull().sum().sum() / synthetic_df.size
            if synthetic_df.size > 0
            else 0.0
        )

        base_score = 0.0 if pd.isna(base_score) else base_score
        weighted_score = base_score * (1 - duplication_penalty) * (1 - null_penalty)

        return weighted_score

    def _apply_resampling(
        self,
        df: pd.DataFrame,
        time_col: Optional[str],
        block_column: Optional[str],
        resample_rule: Union[str, int],
    ) -> pd.DataFrame:
        """
        Applies resampling/aggregation based on time or block column.
        """
        if self.verbose:
            logger.info("Applying resampling rule: %s", resample_rule)

        def agg_mode(x):
            m = x.mode()
            return m.iloc[0] if not m.empty else np.nan

        exclude_cols = [c for c in [time_col, block_column] if c]
        cols_to_agg = [c for c in df.columns if c not in exclude_cols]

        agg_dict = {}
        for col in cols_to_agg:
            if pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = "mean"
            else:
                agg_dict[col] = agg_mode

        if time_col and time_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

            df = df.set_index(time_col)
            df = df.resample(resample_rule).agg(agg_dict).reset_index()
            df = df.dropna(how="all")

        elif (
            block_column
            and block_column in df.columns
            and isinstance(resample_rule, int)
        ):
            df["_block_group"] = df[block_column] // resample_rule
            df = df.groupby("_block_group").agg(agg_dict).reset_index(drop=True)

        return df

    def _assess_sequential_quality(self, real_df, synthetic_df, entity_col, time_col):
        """Assess quality of sequential data using SDMetrics."""
        try:
            from sdmetrics.reports.sequential import QualityReport as SequentialQualityReport
        except ImportError:
            return None

        try:
            if self.verbose:
                logger.info("Running Sequential Quality Assessment...")

            # Prepare metadata dict for SDMetrics
            # We need to construct it manually if SingleTableMetadata doesn't support sequential well directly
            # or usage is different.

            # Simple metadata construction
            cols = {}
            for c in real_df.columns:
                if pd.api.types.is_numeric_dtype(real_df[c]):
                    cols[c] = {"sdtype": "numerical"}
                elif pd.api.types.is_datetime64_any_dtype(real_df[c]):
                    cols[c] = {"sdtype": "datetime"}
                else:
                    cols[c] = {"sdtype": "categorical"}

            metadata = {
                "columns": cols,
                "sequence_key": entity_col,
                "sequence_index": time_col,
            }

            report = SequentialQualityReport()
            report.generate(real_df, synthetic_df, metadata)

            return {
                "score": report.get_score(),
                "details": report.get_properties().to_dict(),
            }

        except Exception as e:
            self.logger.warning(f"Sequential assessment failed: {e}")
            return None

    def _calculate_dcr_privacy(self, real_df, synthetic_df, sample_size=1000):
        """
        Calculates Distance to Closest Record (DCR).
        Simple implementation: Euclidean distance on numeric columns.
        """
        try:
            if self.verbose:
                logger.info("Calculating Privacy Metrics (DCR)...")

            # preprocessing: dummy encoding for categorical, fillna for numeric
            # Use only numeric for simplicity in DCR or simple encoding
            numerics = real_df.select_dtypes(include=[np.number]).columns

            if len(numerics) == 0:
                return {"error": "No numeric columns for DCR"}

            real_num = real_df[numerics].fillna(0).values
            synth_num = synthetic_df[numerics].fillna(0).values

            # Downsample if too large for N^2 complexity
            if len(real_num) > sample_size:
                indices = np.random.choice(len(real_num), sample_size, replace=False)
                real_num = real_num[indices]
            if len(synth_num) > sample_size:
                indices = np.random.choice(len(synth_num), sample_size, replace=False)
                synth_num = synth_num[indices]

            # Normalize
            min_val = np.min(real_num, axis=0)
            max_val = np.max(real_num, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1

            real_norm = (real_num - min_val) / range_val
            synth_norm = (synth_num - min_val) / range_val

            # Compute min distance from each synthetic record to any real record
            from sklearn.metrics import pairwise_distances

            dists = pairwise_distances(synth_norm, real_norm, metric="euclidean")
            min_dists = np.min(dists, axis=1)  # Min dist for each synthetic row

            dcr_5th = np.percentile(min_dists, 5)
            dcr_mean = np.mean(min_dists)

            return {
                "dcr_5th_percentile": dcr_5th,
                "dcr_mean": dcr_mean,
                "interpretation": "Lower 5th percentile means higher risk of re-identification (records too close to real data).",
            }
        except Exception as e:
            self.logger.error(f"Privacy check failed: {e}")
            return None

    def _calculate_ari_metrics(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        target_col: Optional[str]
    ) -> Optional[Dict[str, float]]:
        """
        Calculates Adjusted Rand Index (ARI) using KMeans (k=2) to assess class separability.
        """
        if not SKLEARN_AVAILABLE or not target_col or target_col not in real_df.columns:
            return None

        try:
            if self.verbose:
                logger.info("Calculating ARI metrics (class separability)...")

            def get_ari(df, t_col):
                features = df.select_dtypes(include=[np.number]).drop(columns=[t_col], errors='ignore')
                if features.empty:
                    return None
                X = features.fillna(0).values
                if len(X) < 2:
                    return 0.0
                kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
                cluster_labels = kmeans.fit_predict(X)
                true_labels = pd.Categorical(df[t_col]).codes
                return float(adjusted_rand_score(true_labels, cluster_labels))

            ari_real = get_ari(real_df, target_col)
            ari_synth = get_ari(synthetic_df, target_col)

            return {
                "ari_original": ari_real,
                "ari_synthetic": ari_synth,
                "ari_improvement": (ari_synth - ari_real) if (ari_real is not None and ari_synth is not None) else 0.0
            }
        except Exception as e:
            logger.error(f"ARI calculation failed: {e}")
            return None

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

    def _run_statistical_tests(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        output_dir: str,
    ) -> Optional[Dict[str, Any]]:
        """MMD, KS per column, Levene variance test per column. Saves statistical_tests.html."""
        try:
            from scipy.stats import ks_2samp, levene
        except ImportError:
            logger.warning("scipy required for statistical tests.")
            return None

        try:
            num_cols = real_df.select_dtypes(include="number").columns.tolist()
            if not num_cols:
                return None

            results_per_col: Dict[str, Dict] = {}

            for col in num_cols:
                r = real_df[col].dropna().values
                s = synthetic_df[col].dropna().values
                if len(r) < 2 or len(s) < 2:
                    continue

                ks_stat, ks_p = ks_2samp(r, s)
                lev_stat, lev_p = levene(r, s)

                results_per_col[col] = {
                    "ks_statistic": round(float(ks_stat), 4),
                    "ks_pvalue": round(float(ks_p), 4),
                    "levene_statistic": round(float(lev_stat), 4),
                    "levene_pvalue": round(float(lev_p), 4),
                    "var_real": round(float(np.var(r)), 4),
                    "var_synthetic": round(float(np.var(s)), 4),
                    "var_ratio": round(float(np.var(s) / np.var(r)) if np.var(r) > 0 else float("nan"), 4),
                }

            # MMD (global, all numeric cols together)
            mmd_score = self._compute_mmd(
                real_df[num_cols].fillna(0).values,
                synthetic_df[num_cols].fillna(0).values,
            )

            output = {"mmd": round(mmd_score, 6), "per_column": results_per_col}

            if self.verbose:
                logger.info("Statistical tests — MMD: %.4f, columns tested: %d", mmd_score, len(results_per_col))

            self._save_statistical_tests_html(output, output_dir)
            return output

        except Exception as e:
            logger.error("Statistical tests failed: %s", e)
            return None

    @staticmethod
    def _compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
        """Maximum Mean Discrepancy with RBF kernel. Lower = more similar."""
        # Downsample for speed
        n = min(500, len(X), len(Y))
        rng = np.random.default_rng(42)
        X = X[rng.choice(len(X), n, replace=False)]
        Y = Y[rng.choice(len(Y), n, replace=False)]

        # Normalize
        std = X.std(axis=0)
        std[std == 0] = 1
        X = (X - X.mean(axis=0)) / std
        Y = (Y - X.mean(axis=0)) / std

        def rbf(A, B):
            diff = A[:, None, :] - B[None, :, :]
            return np.exp(-gamma * np.sum(diff ** 2, axis=-1)).mean()

        return float(rbf(X, X) - 2 * rbf(X, Y) + rbf(Y, Y))

    def _save_statistical_tests_html(self, data: Dict, output_dir: str) -> None:
        """Saves statistical_tests.html with MMD + per-column KS/Levene table."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            return

        try:
            cols = list(data["per_column"].keys())
            if not cols:
                return

            ks_stats = [data["per_column"][c]["ks_statistic"] for c in cols]
            ks_ps = [data["per_column"][c]["ks_pvalue"] for c in cols]
            lev_ps = [data["per_column"][c]["levene_pvalue"] for c in cols]
            var_real = [data["per_column"][c]["var_real"] for c in cols]
            var_synth = [data["per_column"][c]["var_synthetic"] for c in cols]

            ks_colors = ["#ef4444" if p < 0.05 else "#22c55e" for p in ks_ps]
            lev_colors = ["#ef4444" if p < 0.05 else "#22c55e" for p in lev_ps]

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "KS Statistic per Column (lower = more similar)",
                    "KS p-value (red = significant difference)",
                    "Levene p-value (red = variance differs significantly)",
                    "Variance: Real vs Synthetic",
                ],
                vertical_spacing=0.18,
                horizontal_spacing=0.10,
            )

            fig.add_trace(go.Bar(x=cols, y=ks_stats, marker_color="#3b82f6", name="KS stat"), row=1, col=1)
            fig.add_trace(go.Bar(x=cols, y=ks_ps, marker_color=ks_colors, name="KS p-value"), row=1, col=2)
            fig.add_trace(go.Bar(x=cols, y=lev_ps, marker_color=lev_colors, name="Levene p-value"), row=2, col=1)
            fig.add_trace(go.Bar(x=cols, y=var_real, name="Real variance", marker_color="#6366f1"), row=2, col=2)
            fig.add_trace(go.Bar(x=cols, y=var_synth, name="Synthetic variance", marker_color="#f59e0b"), row=2, col=2)

            # p=0.05 reference lines
            for (r, c) in [(1, 2), (2, 1)]:
                fig.add_hline(y=0.05, line=dict(color="#94a3b8", dash="dash", width=1), row=r, col=c)

            fig.update_layout(
                title=dict(
                    text=f"Statistical Tests — MMD: <b>{data['mmd']:.4f}</b>  "
                         f"<span style='font-size:13px;color:#64748b'>"
                         f"(MMD ≈ 0 = distributions match)</span>",
                    font=dict(size=18),
                ),
                height=700,
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="Inter, sans-serif", size=12),
                barmode="group",
                showlegend=True,
                margin=dict(t=100, b=60, l=60, r=40),
            )

            path = os.path.join(output_dir, "statistical_tests.html")
            fig.write_html(path, include_plotlyjs="cdn")
            logger.info("Statistical tests report saved to: %s", path)

        except Exception as e:
            logger.error("Failed to save statistical_tests.html: %s", e)

    def _run_tstr(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        target_col: str,
        output_dir: str,
    ) -> Optional[Dict[str, Any]]:
        """Train on Synthetic, Test on Real. RF classifier or regressor depending on target dtype."""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.metrics import (
                balanced_accuracy_score,
                f1_score,
                mean_absolute_percentage_error,
                mean_squared_error,
                r2_score,
                roc_auc_score,
            )
        except ImportError:
            logger.warning("scikit-learn required for TSTR. Install with: pip install scikit-learn")
            return None

        try:
            if target_col not in real_df.columns or target_col not in synthetic_df.columns:
                logger.warning("TSTR skipped: target_col '%s' not in both dataframes.", target_col)
                return None

            # Detect task
            is_classification = (
                real_df[target_col].dtype == object
                or (
                    real_df[target_col].nunique() <= 20
                    and real_df[target_col].dtype in (np.int64, np.int32, np.int8, int)
                )
            )

            # Prepare features — drop target, encode categoricals
            feature_cols = [c for c in real_df.columns if c != target_col]
            synth_X = pd.get_dummies(synthetic_df[feature_cols])
            real_X = pd.get_dummies(real_df[feature_cols])
            real_X = real_X.reindex(columns=synth_X.columns, fill_value=0)

            synth_y = synthetic_df[target_col]
            real_y = real_df[target_col]

            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(synth_X, synth_y)
                preds = model.predict(real_X)
                proba = model.predict_proba(real_X)
                classes = model.classes_
                if len(classes) == 2:
                    auc = roc_auc_score(real_y, proba[:, 1])
                else:
                    auc = roc_auc_score(real_y, proba, multi_class="ovr", average="macro")

                metrics = {
                    "task": "classification",
                    "roc_auc": round(float(auc), 4),
                    "balanced_accuracy": round(float(balanced_accuracy_score(real_y, preds)), 4),
                    "f1_macro": round(float(f1_score(real_y, preds, average="macro")), 4),
                }
                metric_labels = ["ROC AUC", "Balanced Accuracy", "F1 (macro)"]
                metric_values = [metrics["roc_auc"], metrics["balanced_accuracy"], metrics["f1_macro"]]
                metric_colors = [
                    "#3b82f6" if v >= 0.7 else "#f59e0b" if v >= 0.5 else "#ef4444"
                    for v in metric_values
                ]
                task_label = "Classification"
                interpretation = (
                    "ROC AUC near 0.5 means synthetic data preserves little predictive signal. "
                    "Higher values indicate the synthetic data is useful for training."
                )
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(synth_X, synth_y)
                preds = model.predict(real_X)

                metrics = {
                    "task": "regression",
                    "r2": round(float(r2_score(real_y, preds)), 4),
                    "mape": round(float(mean_absolute_percentage_error(real_y, preds)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(real_y, preds))), 4),
                }
                metric_labels = ["R²", "MAPE", "RMSE"]
                metric_values = [metrics["r2"], metrics["mape"], metrics["rmse"]]
                metric_colors = ["#3b82f6", "#f59e0b", "#6366f1"]
                task_label = "Regression"
                interpretation = (
                    "R² close to 1 means synthetic data preserves the target distribution well. "
                    "Lower MAPE and RMSE indicate better predictive utility."
                )

            if self.verbose:
                logger.info("TSTR metrics (%s): %s", task_label, metrics)

            # Generate HTML report
            self._save_tstr_html(
                metrics=metrics,
                metric_labels=metric_labels,
                metric_values=metric_values,
                metric_colors=metric_colors,
                task_label=task_label,
                interpretation=interpretation,
                target_col=target_col,
                n_train=len(synthetic_df),
                n_test=len(real_df),
                output_dir=output_dir,
            )

            return metrics

        except Exception as e:
            logger.error("TSTR failed: %s", e)
            return None

    def _save_tstr_html(
        self,
        metrics: Dict[str, Any],
        metric_labels: List[str],
        metric_values: List[float],
        metric_colors: List[str],
        task_label: str,
        interpretation: str,
        target_col: str,
        n_train: int,
        n_test: int,
        output_dir: str,
    ) -> None:
        """Saves tstr_report.html with a Plotly bar chart + metrics table."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("plotly required to generate TSTR HTML report.")
            return

        fig = go.Figure(go.Bar(
            x=metric_labels,
            y=metric_values,
            marker_color=metric_colors,
            text=[f"{v:.4f}" for v in metric_values],
            textposition="outside",
        ))
        fig.update_layout(
            title=dict(text=f"TSTR — {task_label} ({target_col})", font=dict(size=20)),
            yaxis=dict(range=[0, max(metric_values) * 1.25], title="Score"),
            xaxis=dict(title="Metric"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter, sans-serif", size=13),
            margin=dict(t=80, b=60, l=60, r=40),
        )

        rows_html = "".join(
            f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
            for k, v in metrics.items() if k != "task"
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TSTR Report</title>
<style>
  body {{ font-family: Inter, sans-serif; background: #f8fafc; padding: 32px; color: #1e293b; }}
  .card {{ background: white; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.08); padding: 28px; max-width: 860px; margin: 0 auto 24px; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 4px; }}
  .meta {{ color: #64748b; font-size: .9rem; margin-bottom: 20px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .95rem; }}
  th {{ background: #f1f5f9; text-align: left; padding: 10px 14px; font-weight: 600; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #e2e8f0; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 99px; font-size: .8rem; font-weight: 600; background: #dbeafe; color: #1d4ed8; }}
  .note {{ background: #f8fafc; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 4px; font-size: .9rem; color: #334155; margin-top: 0; }}
</style>
</head>
<body>
<div class="card">
  <h1>TSTR — Train on Synthetic, Test on Real</h1>
  <p class="meta">
    <span class="badge">{task_label}</span>&nbsp;
    Target: <strong>{target_col}</strong> &nbsp;|&nbsp;
    Train (synthetic): <strong>{n_train}</strong> rows &nbsp;|&nbsp;
    Test (real): <strong>{n_test}</strong> rows
  </p>
  {fig.to_html(full_html=False, include_plotlyjs="cdn")}
  <br>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {rows_html}
  </table>
</div>
<div class="card">
  <p class="note">{interpretation}</p>
</div>
</body>
</html>"""

        path = os.path.join(output_dir, "tstr_report.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("TSTR report saved to: %s", path)
