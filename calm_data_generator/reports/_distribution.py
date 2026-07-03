"""Distribution plots for Visualizer: density histograms and QQ plots."""

import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger("Visualizer")


def generate_density_plots(
    df: pd.DataFrame,
    output_dir: str,
    columns: Optional[List[str]] = None,
    filename: str = "density_plots.html",
    color_col: Optional[str] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    labels: tuple = ("Generated", "Original"),
) -> Optional[str]:
    """
    Generates interactive density/distribution plots for numeric columns.

    Args:
        df: Primary DataFrame to plot.
        comparison_df: Optional second DataFrame for overlay comparison.
        labels: Labels for (primary, comparison) datasets.
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available. Skipping density plots.")
        return None

    try:
        # Select numeric columns
        if columns:
            numeric_cols = [
                c
                for c in columns
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            logger.warning("No numeric columns found for density plots.")
            return None

        # Limit to 12 columns max
        numeric_cols = numeric_cols[:12]

        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=numeric_cols,
            vertical_spacing=0.08,
            horizontal_spacing=0.06,
        )

        for idx, col in enumerate(numeric_cols):
            row = idx // n_cols + 1
            col_pos = idx % n_cols + 1

            # Primary dataset
            fig.add_trace(
                go.Histogram(
                    x=df[col].dropna(),
                    name=labels[0],
                    opacity=0.7 if comparison_df is not None else 1.0,
                    marker_color="#636EFA",
                    showlegend=(idx == 0),
                    legendgroup="primary",
                    histnorm="probability density"
                    if comparison_df is not None
                    else None,
                ),
                row=row,
                col=col_pos,
            )

            # Comparison dataset (if provided)
            if comparison_df is not None and col in comparison_df.columns:
                fig.add_trace(
                    go.Histogram(
                        x=comparison_df[col].dropna(),
                        name=labels[1],
                        opacity=0.7,
                        marker_color="#EF553B",
                        showlegend=(idx == 0),
                        legendgroup="comparison",
                        histnorm="probability density",
                    ),
                    row=row,
                    col=col_pos,
                )

        title = "Feature Distributions"
        if comparison_df is not None:
            title = f"Distribution Comparison: {labels[0]} vs {labels[1]}"

        fig.update_layout(
            title_text=title,
            height=300 * n_rows,
            showlegend=(comparison_df is not None),
            barmode="overlay",
        )

        output_path = os.path.join(output_dir, filename)
        pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)

        return output_path

    except Exception as e:
        logger.error(f"Failed to generate density plots: {e}")
        return None


def generate_qq_plots(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    output_dir: str,
    filename: str = "qq_plots.html",
    max_cols: int = 12,
) -> Optional[str]:
    """QQ plots per numeric column: real quantiles vs synthetic quantiles."""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available, skipping QQ plots.")
        return None

    try:
        from plotly.subplots import make_subplots

        num_cols = real_df.select_dtypes(include="number").columns.tolist()[:max_cols]
        if not num_cols:
            logger.warning("QQ plots skipped: no numeric columns.")
            return None

        ncols = min(3, len(num_cols))
        nrows = (len(num_cols) + ncols - 1) // ncols

        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=num_cols,
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        for i, col in enumerate(num_cols):
            row, col_pos = divmod(i, ncols)
            real_q = np.quantile(real_df[col].dropna(), np.linspace(0, 1, 100))
            synth_q = np.quantile(synthetic_df[col].dropna(), np.linspace(0, 1, 100))
            mn, mx = min(real_q.min(), synth_q.min()), max(real_q.max(), synth_q.max())

            # Diagonal reference line
            fig.add_trace(
                go.Scatter(
                    x=[mn, mx], y=[mn, mx],
                    mode="lines",
                    line=dict(color="#94a3b8", dash="dash", width=1),
                    showlegend=(i == 0),
                    name="Perfect match",
                    hoverinfo="skip",
                ),
                row=row + 1, col=col_pos + 1,
            )
            # QQ scatter
            fig.add_trace(
                go.Scatter(
                    x=real_q, y=synth_q,
                    mode="markers",
                    marker=dict(color="#3b82f6", size=4, opacity=0.7),
                    showlegend=(i == 0),
                    name="Quantiles",
                    hovertemplate=f"Real: %{{x:.3f}}<br>Synth: %{{y:.3f}}<extra>{col}</extra>",
                ),
                row=row + 1, col=col_pos + 1,
            )

        fig.update_layout(
            title=dict(text="QQ Plots — Real vs Synthetic Quantiles", font=dict(size=18)),
            height=max(400, nrows * 280),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="Inter, sans-serif", size=11),
            margin=dict(t=80, b=40, l=60, r=40),
        )
        fig.update_xaxes(title_text="Real quantiles", showgrid=True, gridcolor="#f1f5f9")
        fig.update_yaxes(title_text="Synthetic quantiles", showgrid=True, gridcolor="#f1f5f9")

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.write_html(path, include_plotlyjs="cdn")
        return path

    except Exception as e:
        logger.error(f"Failed to generate QQ plots: {e}")
        return None
