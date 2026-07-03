"""Quality scoring mixin for QualityReporter: SDMetrics, sequential quality, resampling."""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("QualityReporter")


class _QualityScoringMixin:
    """Mixin providing SDMetrics-based quality scoring for QualityReporter."""

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
