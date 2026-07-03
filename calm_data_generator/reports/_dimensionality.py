"""Dimensionality-reduction plots for Visualizer (PCA)."""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger("Visualizer")


def generate_dimensionality_plot(
    df: pd.DataFrame,
    output_dir: str,
    filename: str = "dimensionality_plot.html",
    color_col: Optional[str] = None,
) -> Optional[str]:
    """
    Generates combined PCA + UMAP visualization in a single HTML file.
    """
    if not PLOTLY_AVAILABLE or not SKLEARN_AVAILABLE:
        logger.warning("Required libraries not available.")
        return None

    try:
        # Select numeric columns and drop NaN
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if numeric_df.shape[1] < 2:
            logger.warning(
                "Not enough numeric columns for dimensionality reduction."
            )
            return None

        if numeric_df.shape[0] < 10:
            logger.warning("Not enough samples for dimensionality reduction.")
            return None

        if numeric_df.empty:
            logger.warning("No data remaining after dropping NaN rows.")
            return None

        # Standardize
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # === PCA ===
        # For very wide data, truncate to 500 components first to avoid memory/speed issues
        if scaled_data.shape[1] > 500:
            pre_pca = PCA(n_components=500, random_state=42)
            scaled_data = pre_pca.fit_transform(scaled_data)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        pca_df = pd.DataFrame(
            pca_result,
            columns=["PC1", "PC2"],
            index=numeric_df.index,
        )

        # Add color column if provided
        color_values = None
        if color_col and color_col in df.columns:
            color_values = df.loc[numeric_df.index, color_col].values

        # Only PCA
        fig = go.Figure()
        if color_values is not None:
            for val in np.unique(color_values):
                mask = color_values == val
                fig.add_trace(
                    go.Scatter(
                        x=pca_df.loc[mask, "PC1"],
                        y=pca_df.loc[mask, "PC2"],
                        mode="markers",
                        name=str(val),
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=pca_df["PC1"],
                    y=pca_df["PC2"],
                    mode="markers",
                    showlegend=False,
                )
            )

        fig.update_layout(
            title=f"PCA Visualization (Explained Variance: {sum(pca.explained_variance_ratio_):.1%})",
            xaxis_title="PC1",
            yaxis_title="PC2",
            height=500,
        )

        output_path = os.path.join(output_dir, filename)
        pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)

        return output_path

    except Exception as e:
        logger.error(f"Failed to generate dimensionality plot: {e}")
        return None
