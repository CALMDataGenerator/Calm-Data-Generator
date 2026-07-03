"""Privacy and clustering-separability metrics mixin for QualityReporter."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger("QualityReporter")


class _PrivacyMetricsMixin:
    """Mixin providing privacy (DCR) and class-separability (ARI) metrics for QualityReporter."""

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
