"""Correlation heatmap plots for Visualizer."""

import logging
import os

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger("Visualizer")


def generate_spearman_heatmaps(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    output_dir: str,
    filename: str = "spearman_heatmaps.html",
):
    """Side-by-side Spearman correlation heatmaps: real vs synthetic."""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available, skipping Spearman heatmaps.")
        return None

    try:
        from plotly.subplots import make_subplots

        num_cols = real_df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            logger.warning("Spearman heatmaps skipped: need ≥2 numeric columns.")
            return None

        def _spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
            # Rank once per column, then Pearson on ranks — O(n²) BLAS vs O(n²) scipy calls
            ranked = df.rank(method="average")
            corr = np.corrcoef(ranked.values.T)
            return pd.DataFrame(corr, index=df.columns, columns=df.columns)

        real_num = real_df[num_cols].dropna()
        synth_num = synthetic_df[num_cols].dropna()

        real_corr = _spearman_matrix(real_num)
        synth_corr = _spearman_matrix(synth_num)
        diff_corr = synth_corr - real_corr

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Real (Spearman)", "Synthetic (Spearman)", "Difference (Synth − Real)"],
            horizontal_spacing=0.08,
        )

        colorscale = "RdBu"
        for col_idx, (matrix, title) in enumerate([
            (real_corr, "Real"),
            (synth_corr, "Synthetic"),
            (diff_corr, "Difference"),
        ], start=1):
            zmin, zmax = (-1, 1) if col_idx < 3 else (-0.5, 0.5)
            fig.add_trace(
                go.Heatmap(
                    z=matrix.values,
                    x=num_cols,
                    y=num_cols,
                    colorscale=colorscale,
                    zmin=zmin, zmax=zmax,
                    showscale=(col_idx == 3),
                    text=[[f"{v:.2f}" for v in row] for row in matrix.values],
                    texttemplate="%{text}",
                    hovertemplate="Row: %{y}<br>Col: %{x}<br>ρ: %{z:.3f}<extra></extra>",
                ),
                row=1, col=col_idx,
            )

        n = len(num_cols)
        fig.update_layout(
            title=dict(text="Spearman Correlation — Real vs Synthetic", font=dict(size=18)),
            height=max(400, 120 + n * 40),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="Inter, sans-serif", size=12),
            margin=dict(t=80, b=40, l=100, r=40),
        )

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.write_html(path, include_plotlyjs="cdn")
        return path

    except Exception as e:
        logger.error(f"Failed to generate Spearman heatmaps: {e}")
        return None
