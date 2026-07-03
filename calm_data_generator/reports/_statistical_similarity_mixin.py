"""Statistical similarity mixin for QualityReporter: KS, Levene, MMD tests."""

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("QualityReporter")


class _StatisticalSimilarityMixin:
    """Mixin providing statistical similarity tests for QualityReporter."""

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

        # Normalize both using X's statistics
        X_mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1
        X = (X - X_mean) / std
        Y = (Y - X_mean) / std

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
