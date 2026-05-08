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

sc = None
ad = None
try:
    import anndata as ad
    import scanpy as sc
    from scgft_evaluator import ScGFT_Evaluator
    SCGFT_AVAILABLE = True
except ImportError:
    SCGFT_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from calm_data_generator.generators.configs import ReportConfig  # noqa: E402
from calm_data_generator.reports.base import BaseReporter  # noqa: E402
from calm_data_generator.reports.DiscriminatorReporter import DiscriminatorReporter  # noqa: E402
from calm_data_generator.reports.ExternalReporter import ExternalReporter  # noqa: E402
from calm_data_generator.reports.LocalIndexGenerator import LocalIndexGenerator  # noqa: E402
from calm_data_generator.reports.Visualizer import Visualizer  # noqa: E402

# Direct usage of sdmetrics
try:
    from sdmetrics.reports.single_table import QualityReport

    SDMETRICS_AVAILABLE = True
except ImportError:
    SDMETRICS_AVAILABLE = False

# Sequential quality report (optional, may not be available in all versions)
try:
    from sdmetrics.reports.sequential import (
        QualityReport as SequentialQualityReport,
    )

    SEQUENTIAL_SDMETRICS_AVAILABLE = True
except ImportError:
    SEQUENTIAL_SDMETRICS_AVAILABLE = False


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
        adversarial_validation: bool = False,
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
                adversarial_validation=adversarial_validation,
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
        adversarial_validation = report_config.adversarial_validation
        use_scgft = report_config.use_scgft

        # Force minimal override if self.minimal is explicitly True?
        # If config says False but self.minimal is True...
        # Let's say config wins for this execution.

        if self.verbose:
            print("=" * 80)
            print("COMPREHENSIVE REAL DATA GENERATION REPORT")
            print(f"Generator: {generator_name}")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)

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
        }

        results_path = os.path.join(output_dir, "report_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        # === Generate Plotly Visualizations ===
        self._run_visualizations(
            real_df_for_report, synthetic_df_for_report,
            output_dir, focus_cols, target_column, drift_config,
            use_minimal, adversarial_validation,
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
            print(f"\nReport generated at: {output_dir}")

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
        if final_time_col and block_column and SEQUENTIAL_SDMETRICS_AVAILABLE:
            sequential_quality = self._assess_sequential_quality(
                real_df, synthetic_df, block_column, final_time_col
            )

        privacy_metrics = None
        if privacy_check or "dp" in generator_name.lower():
            privacy_metrics = self._calculate_dcr_privacy(real_df, synthetic_df)

        if "overall_quality_score" in sdmetrics_quality:
            if self.verbose:
                print("\nGenerating Quality Scores Card...")
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
        adversarial_validation: bool,
    ) -> None:
        """Generates Plotly visualizations: density, PCA, comparison, discriminator."""
        if self.verbose:
            print("\nGenerating Plotly Visualizations...")

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
            print("   -> Skipping PCA/UMAP (minimal mode)")

        if use_minimal and self.verbose:
            print("   -> Skipping full quality assessment (minimal mode)")

        Visualizer.generate_comparison_plots(
            original_df=real_df_for_report,
            drifted_df=synthetic_df_for_report,
            output_dir=output_dir,
            columns=focus_cols,
            drift_config=drift_config,
        )

        if adversarial_validation and not use_minimal:
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
            print("\nGenerating YData Reports...")

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
                print(f"\nGenerating Per-Block Reports (Block Col: {block_column})...")

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
        if not SDMETRICS_AVAILABLE:
            return {"error": "SDMetrics not available"}

        try:
            if self.verbose:
                print("\nRunning SDMetrics Quality Assessment...")

            common_cols = list(set(real_df.columns) & set(synthetic_df.columns))
            if len(common_cols) < len(real_df.columns) and self.verbose:
                dropped = set(real_df.columns) - set(common_cols)
                print(f"   -> Aligning columns for (dropped: {dropped})")

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
                print(
                    f"SDMetrics Assessment complete. Overall: {overall_score:.2f}, Weighted: {weighted_score:.2f}"
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
        if not SDMETRICS_AVAILABLE:
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
            print(f"Applying resampling rule: {resample_rule}")

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
        if not SEQUENTIAL_SDMETRICS_AVAILABLE:
            return None

        try:
            if self.verbose:
                print("\nRunning Sequential Quality Assessment...")

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
                print("\nCalculating Privacy Metrics (DCR)...")

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
                print("Calculating ARI metrics (class separability)...")

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
        if not SCGFT_AVAILABLE:
            if self.verbose:
                print("\n[WARNING] scgft-evaluator not found. Install with: pip install git+https://github.com/nasim23ea/scgft-evaluator.git")
            return

        if self.verbose:
            print("\n" + "="*40)
            print("RUNNING scGFT SINGLE-CELL EVALUATION")
            print("="*40)

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
                print("   -> Preprocessing AnnData (PCA)...")

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
                print(output_text)
                print(results.to_string(index=False))

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
                print(f"   -> scGFT report saved to: {scgft_report_path}")

        except Exception as e:
            logger.error(f"scGFT evaluation failed: {e}")
            if self.verbose:
                print(f"   -> [ERROR] scGFT evaluation failed: {e}")
