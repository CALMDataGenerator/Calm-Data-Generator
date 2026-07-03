"""Sequence and evolution (before/after) plots for Visualizer."""

import logging
import os
from typing import Any, Dict, List, Optional

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


def generate_sequence_plot(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    output_dir: str,
    filename: str = "sequence_plot.html",
    feature_cols: Optional[List[str]] = None,
    n_entities: int = 5,
) -> Optional[str]:
    """
    Generates line plots for sequential data, showing trajectories of a few entities.
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available. Skipping sequence plot.")
        return None

    try:
        # Validate columns
        if entity_col not in df.columns or time_col not in df.columns:
            logger.warning(
                f"Entity col '{entity_col}' or Time col '{time_col}' not found."
            )
            return None

        # Select features
        if feature_cols:
            plot_cols = [
                c
                for c in feature_cols
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]
        else:
            plot_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            plot_cols = [
                c for c in plot_cols if c not in [entity_col]
            ]  # Exclude ID if numeric

        if not plot_cols:
            logger.warning("No numeric features to plot for sequences.")
            return None

        # Select top N entities
        entities = df[entity_col].unique()[:n_entities]
        subset = df[df[entity_col].isin(entities)].copy()

        if subset.empty:
            logger.warning("No data found for the selected entities.")
            return None

        # Sort by time
        subset = subset.sort_values(by=[entity_col, time_col])

        # Create subplots
        n_rows = min(len(plot_cols), 5)  # Max 5 features stacked
        plot_cols = plot_cols[:n_rows]

        fig = make_subplots(
            rows=n_rows,
            cols=1,
            subplot_titles=plot_cols,
            shared_xaxes=True,
            vertical_spacing=0.05,
        )

        for i, col in enumerate(plot_cols):
            for entity in entities:
                mask = subset[entity_col] == entity
                row_data = subset[mask]

                fig.add_trace(
                    go.Scatter(
                        x=row_data[time_col],
                        y=row_data[col],
                        mode="lines+markers",
                        name=str(entity),
                        legendgroup=str(entity),
                        showlegend=(i == 0),  # Only show legend once
                        line=dict(width=1.5),
                        marker=dict(size=4),
                    ),
                    row=i + 1,
                    col=1,
                )

        fig.update_layout(
            title=f"Sequence Trajectories (Top {len(entities)} Entities)",
            height=300 * n_rows,
            hovermode="x unified",
        )

        output_path = os.path.join(output_dir, filename)
        pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)
        return output_path

    except Exception as e:
        logger.error(f"Failed to generate sequence plot: {e}")
        return None


def generate_evolution_plot(
    original_df: pd.DataFrame,
    evolved_df: pd.DataFrame,
    evolution_config: Dict[str, Any],
    output_dir: str,
    time_col: Optional[str] = None,
    filename: str = "evolution_plot.html",
) -> Optional[str]:
    """
    Generates a before/after evolution plot for ScenarioInjector.
    Shows how features changed after applying evolution transformations.

    Args:
        original_df: Original DataFrame before evolution.
        evolved_df: DataFrame after evolution was applied.
        evolution_config: Dictionary mapping column names to evolution specs.
        output_dir: Directory to save the plot.
        time_col: Optional time column for x-axis.
        filename: Output filename.

    Returns:
        Path to the generated HTML file, or None on failure.
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available, skipping evolution plot")
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Get evolved columns
        evolved_cols = [
            c for c in evolution_config.keys() if c in original_df.columns
        ]
        if not evolved_cols:
            logger.warning("No evolved columns found in DataFrame")
            return None

        n_cols = len(evolved_cols)
        n_rows = min(n_cols, 4)  # Max 4 features

        fig = make_subplots(
            rows=n_rows,
            cols=2,
            subplot_titles=[
                f"{col} - Before" if i % 2 == 0 else f"{col} - After"
                for col in evolved_cols[:n_rows]
                for i in range(2)
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.12,
        )

        # Use time_col or index for x-axis
        if time_col and time_col in original_df.columns:
            x_orig = original_df[time_col]
            x_evolved = evolved_df[time_col]
        else:
            x_orig = np.arange(len(original_df))
            x_evolved = np.arange(len(evolved_df))

        for i, col in enumerate(evolved_cols[:n_rows]):
            config = evolution_config.get(col, {})
            evo_type = config.get("type", "unknown")

            # Before (original)
            fig.add_trace(
                go.Scatter(
                    x=x_orig,
                    y=original_df[col],
                    mode="lines",
                    name=f"{col} (Original)",
                    line=dict(color="#6c757d", width=1),
                    showlegend=(i == 0),
                    legendgroup="original",
                ),
                row=i + 1,
                col=1,
            )

            # After (evolved)
            fig.add_trace(
                go.Scatter(
                    x=x_evolved,
                    y=evolved_df[col],
                    mode="lines",
                    name=f"{col} (Evolved)",
                    line=dict(color="#007bff", width=1.5),
                    showlegend=(i == 0),
                    legendgroup="evolved",
                ),
                row=i + 1,
                col=2,
            )

            # Add annotation for evolution type
            fig.add_annotation(
                text=f"Evolution: {evo_type}",
                xref=f"x{i * 2 + 2} domain",
                yref=f"y{i * 2 + 2} domain",
                x=0.02,
                y=0.98,
                showarrow=False,
                font=dict(size=10, color="#666"),
                bgcolor="rgba(255,255,255,0.8)",
            )

        fig.update_layout(
            title="📈 Feature Evolution Analysis (ScenarioInjector)",
            height=250 * n_rows,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
        )

        output_path = os.path.join(output_dir, filename)
        pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)
        logger.info(f"Evolution plot saved to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to generate evolution plot: {e}")
        return None
