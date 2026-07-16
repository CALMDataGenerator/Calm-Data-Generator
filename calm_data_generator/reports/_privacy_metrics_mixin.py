"""Privacy and clustering-separability metrics mixin for QualityReporter."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger("QualityReporter")

# anonymeter's own default (10_000_000) can make a single evaluate() call hang for
# minutes on low-cardinality/duplicate-heavy data, since it keeps retrying to find
# unique attack queries. Bound it hard so a report run never blocks on this — tested
# against a worst-case (low-cardinality, exact-duplicate) dataset: ~15s at this value,
# vs. multi-minute hangs at anonymeter's own default.
_SINGLING_OUT_MAX_ATTEMPTS_DEFAULT = 15_000


class _PrivacyMetricsMixin:
    """Mixin providing privacy (DCR) and class-separability (ARI) metrics for QualityReporter."""

    def _calculate_dcr_privacy(self, real_df, synthetic_df, sample_size=1000):
        """
        Calculates Distance to Closest Record (DCR) and Nearest Neighbor Distance
        Ratio (NNDR). Simple implementation: Euclidean distance on numeric columns.

        DCR alone is scale-dependent (a "small" distance means different things on
        different features). NNDR complements it: it's the ratio of the distance to
        the closest real record over the distance to the second-closest one. A ratio
        near 0 means a synthetic record sits right on top of one specific real record
        while being much farther from any other — a stronger, scale-invariant signal
        of re-identification risk than DCR alone.
        """
        try:
            if self.verbose:
                logger.info("Calculating Privacy Metrics (DCR, NNDR)...")

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

            # Compute distances from each synthetic record to every real record
            from sklearn.metrics import pairwise_distances

            dists = pairwise_distances(synth_norm, real_norm, metric="euclidean")
            min_dists = np.min(dists, axis=1)  # Min dist for each synthetic row

            dcr_5th = np.percentile(min_dists, 5)
            dcr_mean = np.mean(min_dists)

            result = {
                "dcr_5th_percentile": dcr_5th,
                "dcr_mean": dcr_mean,
                "interpretation": "Lower 5th percentile means higher risk of re-identification (records too close to real data).",
            }

            # NNDR needs at least 2 real records to compute a "2nd closest" distance
            if real_norm.shape[0] >= 2:
                k = min(2, dists.shape[1])
                nearest_two = np.partition(dists, k - 1, axis=1)[:, :k]
                nearest_two.sort(axis=1)
                d1 = nearest_two[:, 0]
                d2 = nearest_two[:, -1] if k == 2 else nearest_two[:, 0]
                nndr = d1 / (d2 + 1e-12)

                result["nndr_5th_percentile"] = float(np.percentile(nndr, 5))
                result["nndr_mean"] = float(np.mean(nndr))
                result["nndr_interpretation"] = (
                    "Ratio near 0 means a synthetic record is much closer to one real "
                    "record than to any other (higher re-identification risk); near 1 "
                    "means it's roughly equidistant to several real records (lower risk)."
                )

            return result
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

    def _calculate_singling_out_risk(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        control_df: Optional[pd.DataFrame] = None,
        n_attacks: int = 100,
        n_cols: int = 3,
        max_attempts: int = _SINGLING_OUT_MAX_ATTEMPTS_DEFAULT,
    ) -> Optional[Dict[str, Any]]:
        """
        Singling-Out risk via anonymeter (MIT-licensed: github.com/statice/anonymeter).

        Estimates the risk that an attacker can isolate ("single out") one specific
        real record using a combination of `n_cols` attributes learned from the
        synthetic data. Complements DCR/NNDR: those measure closeness in feature
        space, this measures whether the synthetic data leaks *combinations* of
        attribute values that are rare enough to identify one real individual.

        Optional dependency: `pip install calm-data-generator[privacy]`. Returns
        None (with an info-level log, not an error) if anonymeter isn't installed.

        Args:
            real_df, synthetic_df: The datasets to compare.
            control_df: Optional held-out real records (not seen during synthesis).
                Recommended by anonymeter to separate genuine leakage from risk that
                exists in the real population anyway. If omitted, risk is computed
                without this baseline correction.
            n_attacks: Number of singling-out attempts to simulate.
            n_cols: Number of attributes combined per attack (capped to the number
                of available columns).
            max_attempts: Hard cap on internal retries. anonymeter's own default
                (10,000,000) can hang for minutes on low-cardinality/duplicate-heavy
                data while it searches for unique attack queries — this keeps the
                call bounded regardless of the input data's cardinality.
        """
        try:
            from anonymeter.evaluators import SinglingOutEvaluator
        except ImportError:
            logger.info(
                "anonymeter not installed; skipping Singling-Out risk. "
                "Install with: pip install calm-data-generator[privacy]"
            )
            return None

        try:
            if self.verbose:
                logger.info("Calculating Singling-Out risk (anonymeter)...")

            evaluator = SinglingOutEvaluator(
                ori=real_df,
                syn=synthetic_df,
                control=control_df,
                n_attacks=min(n_attacks, len(synthetic_df)),
                n_cols=min(n_cols, real_df.shape[1]),
                max_attempts=max_attempts,
            )
            evaluator.evaluate(mode="multivariate")
            risk = evaluator.risk()

            return {
                "risk": round(float(risk.value), 4),
                "ci_low": round(float(risk.ci[0]), 4),
                "ci_high": round(float(risk.ci[1]), 4),
                "used_control": control_df is not None,
                "interpretation": (
                    "Risk near 0 means an attacker gains little advantage from the "
                    "synthetic data when trying to single out a specific real record. "
                    "Risk near 1 means the synthetic data makes it easy."
                ),
            }
        except Exception as e:
            logger.warning(f"Singling-Out risk calculation failed or was skipped: {e}")
            return None
