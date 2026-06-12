"""
Visualizer Module
Generates interactive Plotly HTML plots for data visualization.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

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


class Visualizer:
    """
    Generates interactive Plotly HTML visualizations for data reports.
    """

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def generate_quality_scores_card(
        overall_score: float,
        weighted_score: float,
        output_dir: str,
        filename: str = "quality_scores.html",
    ) -> Optional[str]:
        """
        Generates a clean, simple HTML card showing Quality scores.
        """
        try:
            # Determine color based on score
            def get_color(score):
                if score >= 0.75:
                    return "#28a745"  # Green
                elif score >= 0.50:
                    return "#ffc107"  # Yellow
                else:
                    return "#dc3545"  # Red

            html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        .score-card {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            display: flex;
            gap: 40px;
            padding: 20px;
            justify-content: center;
        }}
        .score-box {{
            text-align: center;
            padding: 20px 40px;
            border-radius: 12px;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .score-value {{
            font-size: 48px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .score-label {{
            font-size: 14px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
    </style>
</head>
<body>
    <div class="score-card">
        <div class="score-box">
            <div class="score-label">Overall Quality</div>
            <div class="score-value" style="color: {get_color(overall_score)}">
                {overall_score:.1%}
            </div>
        </div>
        <div class="score-box">
            <div class="score-label">Weighted Quality</div>
            <div class="score-value" style="color: {get_color(weighted_score)}">
                {weighted_score:.1%}
            </div>
        </div>
    </div>
</body>
</html>
"""
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w") as f:
                f.write(html)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate Quality scores card: {e}")
            return None

    @staticmethod
    def generate_quality_evolution_plot(
        scores: List[Dict[str, Any]],
        output_dir: str,
        filename: str = "quality_evolution.html",
        x_labels: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Generates Quality evolution plot showing overall and weighted scores.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping Quality evolution plot.")
            return None

        try:
            if not scores:
                logger.warning("No scores provided for Quality evolution plot.")
                return None

            overall_scores = [s.get("overall", 0) for s in scores]
            weighted_scores = [s.get("weighted", 0) for s in scores]

            if x_labels is None:
                x_labels = [f"Block {i + 1}" for i in range(len(scores))]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=overall_scores,
                    mode="lines+markers",
                    name="Overall Quality Score",
                    line=dict(color="blue", width=2),
                    marker=dict(size=10),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=weighted_scores,
                    mode="lines+markers",
                    name="Weighted Quality Score",
                    line=dict(color="green", width=2),
                    marker=dict(size=10),
                )
            )

            fig.update_layout(
                title="Quality Evolution",
                xaxis_title="Block / Time Period",
                yaxis_title="Quality Score",
                yaxis=dict(range=[0, 1]),
                height=500,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode="x unified",
            )

            output_path = os.path.join(output_dir, filename)
            pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate Quality evolution plot: {e}")
            return None

    @staticmethod
    def generate_comparison_plots(
        original_df: pd.DataFrame,
        drifted_df: pd.DataFrame,
        output_dir: str,
        filename: str = "plot_comparison.html",
        columns: Optional[List[str]] = None,
        drift_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Generates comprehensive comparison visualizations (statistical & distribution).

        If drift_config is provided, includes a Drift Configuration Summary.
        Otherwise, acts as a standard fidelity comparison.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping comparison plots.")
            return None

        try:
            from scipy.spatial.distance import jensenshannon
            from scipy.stats import ks_2samp

            # Select numeric columns
            if columns:
                numeric_cols = [
                    c
                    for c in columns
                    if c in original_df.columns
                    and c in drifted_df.columns
                    and pd.api.types.is_numeric_dtype(original_df[c])
                ]
            else:
                numeric_cols = [
                    c
                    for c in original_df.select_dtypes(include=[np.number]).columns
                    if c in drifted_df.columns
                ]

            if not numeric_cols:
                logger.warning("No numeric columns found for comparison.")
                return None

            # Limit to 12 columns
            numeric_cols = numeric_cols[:12]

            # Calculate metrics for each column
            metrics = {}
            for col in numeric_cols:
                try:
                    orig_vals = original_df[col].dropna().values
                    drift_vals = drifted_df[col].dropna().values

                    if len(orig_vals) == 0 or len(drift_vals) == 0:
                        continue

                    # JS Divergence
                    min_val = min(orig_vals.min(), drift_vals.min())
                    max_val = max(orig_vals.max(), drift_vals.max())
                    if min_val == max_val:
                        continue
                    bins = np.linspace(min_val, max_val, 50)
                    hist_orig, _ = np.histogram(orig_vals, bins=bins, density=True)
                    hist_drift, _ = np.histogram(drift_vals, bins=bins, density=True)
                    hist_orig = (hist_orig + 1e-10) / (hist_orig + 1e-10).sum()
                    hist_drift = (hist_drift + 1e-10) / (hist_drift + 1e-10).sum()
                    js_div = jensenshannon(hist_orig, hist_drift)

                    # KS test
                    ks_stat, ks_pval = ks_2samp(orig_vals, drift_vals)

                    # Cohen's d
                    pooled_std = np.sqrt(
                        (orig_vals.std() ** 2 + drift_vals.std() ** 2) / 2
                    )
                    cohens_d = (
                        (drift_vals.mean() - orig_vals.mean()) / pooled_std
                        if pooled_std > 0
                        else 0
                    )

                    # The 'Quality Score' is typically an aggregate metric.JS Divergence
                    if js_div > 0.15:
                        severity = "HIGH"
                        severity_color = "#dc3545"
                    elif js_div > 0.05:
                        severity = "MEDIUM"
                        severity_color = "#ffc107"
                    else:
                        severity = "LOW"
                        severity_color = "#28a745"

                    metrics[col] = {
                        "js_div": js_div,
                        "ks_stat": ks_stat,
                        "ks_pval": ks_pval,
                        "cohens_d": cohens_d,
                        "orig_mean": orig_vals.mean(),
                        "orig_std": orig_vals.std(),
                        "drift_mean": drift_vals.mean(),
                        "drift_std": drift_vals.std(),
                        "pct_change": (
                            (drift_vals.mean() - orig_vals.mean())
                            / abs(orig_vals.mean())
                            * 100
                        )
                        if orig_vals.mean() != 0
                        else 0,
                        "severity": severity,
                        "severity_color": severity_color,
                    }
                except Exception as e:
                    logger.warning(f"Could not compute drift metrics for column '{col}'; using safe defaults. Reason: {e}")
                    metrics[col] = {
                        "js_div": 0,
                        "ks_stat": 0,
                        "ks_pval": 1,
                        "cohens_d": 0,
                        "severity": "LOW",
                        "severity_color": "#28a745",
                    }

            # Calculate duplicates percentage
            try:
                common_cols = list(set(original_df.columns) & set(drifted_df.columns))
                orig_unique = original_df[common_cols].drop_duplicates()
                merged = drifted_df[common_cols].merge(
                    orig_unique, how="left", indicator=True
                )
                cross_dup_count = (merged["_merge"] == "both").sum()
                cross_dup_pct = (
                    cross_dup_count / len(drifted_df) * 100
                    if len(drifted_df) > 0
                    else 0
                )
            except Exception as e:
                logger.warning(f"Could not compute cross-duplicate percentage; defaulting to 0. Reason: {e}")
                cross_dup_pct = 0

            # Sort by JS divergence
            sorted_cols = sorted(
                metrics.keys(), key=lambda x: metrics[x]["js_div"], reverse=True
            )

            # Build HTML report
            html_parts = []

            # 1. Header (Conditional)
            if drift_config:
                drift_cols_str = ", ".join(columns) if columns else "All numeric"
                drift_type = drift_config.get("drift_type", "N/A")
                drift_mag = drift_config.get("drift_magnitude", "N/A")

                html_parts.append(f"""
                <div style="font-family: 'Segoe UI', sans-serif; padding: 20px; max-width: 1200px; margin: auto;">
                    <h1 style="color: #333;">🌊 Drift Analysis Report</h1>

                    <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 30px;">
                        <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 12px; flex: 1; min-width: 200px;">
                            <div style="font-size: 14px; opacity: 0.8;">AFFECTED COLUMNS</div>
                            <div style="font-size: 24px; font-weight: bold;">{drift_cols_str}</div>
                        </div>
                        <div style="background: linear-gradient(135deg, #f093fb, #f5576c); color: white; padding: 20px; border-radius: 12px; flex: 1; min-width: 200px;">
                            <div style="font-size: 14px; opacity: 0.8;">DRIFT TYPE</div>
                            <div style="font-size: 24px; font-weight: bold;">{drift_type}</div>
                        </div>
                        <div style="background: linear-gradient(135deg, #4facfe, #00f2fe); color: white; padding: 20px; border-radius: 12px; flex: 1; min-width: 200px;">
                            <div style="font-size: 14px; opacity: 0.8;">MAGNITUDE</div>
                            <div style="font-size: 24px; font-weight: bold;">{drift_mag}</div>
                        </div>
                        <div style="background: linear-gradient(135deg, #fa709a, #fee140); color: white; padding: 20px; border-radius: 12px; flex: 1; min-width: 200px;">
                            <div style="font-size: 14px; opacity: 0.8;">DUPLICATES WITH ORIGINAL</div>
                            <div style="font-size: 24px; font-weight: bold;">{cross_dup_pct:.1f}%</div>
                        </div>
                    </div>
                """)
            else:
                html_parts.append("""
                <div style="font-family: 'Segoe UI', sans-serif; padding: 20px; max-width: 1200px; margin: auto;">
                    <h1 style="color: #333;">📊 Statistical Distribution Comparison</h1>
                    <p style="color: #666; margin-bottom: 30px;">Comparison of statistical properties between Original and Synthetic datasets.</p>
                """)

            html_parts.append("""
                <h2>Statistical Metrics by Feature</h2>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 30px;">
                    <tr style="background: #f8f9fa;">
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Feature</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">Severity</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">JS Div</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">KS Stat</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">KS p-value</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">Cohen's d</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">Mean Δ%</th>
                    </tr>
            """)

            for col in sorted_cols:
                m = metrics[col]
                pval_color = "#28a745" if m["ks_pval"] > 0.05 else "#dc3545"
                severity = m.get("severity", "LOW")
                severity_color = m.get("severity_color", "#28a745")
                html_parts.append(f"""
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{col}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">
                            <span style="background: {severity_color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: bold;">{severity}</span>
                        </td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6; color: {severity_color};">{m["js_div"]:.4f}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{m["ks_stat"]:.4f}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6; color: {pval_color};">{m["ks_pval"]:.4f}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{m["cohens_d"]:+.3f}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{m.get("pct_change", 0):+.1f}%</td>
                    </tr>
                """)

            html_parts.append("</table>")

            # Add Before/After Statistics Table
            html_parts.append("""
                <h2>📈 Before/After Statistics</h2>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 30px;">
                    <tr style="background: #f8f9fa;">
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Feature</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">Original Mean</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">Original Std</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">Drifted Mean</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">Drifted Std</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">Change</th>
                    </tr>
            """)

            for col in sorted_cols:
                m = metrics[col]
                orig_mean = m.get("orig_mean", 0)
                orig_std = m.get("orig_std", 0)
                drift_mean = m.get("drift_mean", 0)
                drift_std = m.get("drift_std", 0)
                pct_change = m.get("pct_change", 0)
                change_color = (
                    "#dc3545"
                    if abs(pct_change) > 10
                    else "#ffc107"
                    if abs(pct_change) > 5
                    else "#28a745"
                )
                html_parts.append(f"""
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{col}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{orig_mean:.4f}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{orig_std:.4f}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{drift_mean:.4f}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{drift_std:.4f}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6; color: {change_color}; font-weight: bold;">{pct_change:+.1f}%</td>
                    </tr>
                """)

            html_parts.append("</table>")

            # Close HTML wrapper
            html_parts.append("</div>")

            # Save stats HTML ONLY if drift_config was provided
            if drift_config:
                stats_path = os.path.join(output_dir, "drift_stats.html")
                with open(stats_path, "w") as f:
                    f.write("".join(html_parts))

            # Generate Plotly density plots
            n_density_cols = min(6, len(sorted_cols))
            if n_density_cols == 0:
                return
            n_cols = min(3, n_density_cols)
            n_rows = (n_density_cols + n_cols - 1) // n_cols

            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=sorted_cols[:n_density_cols],
                vertical_spacing=0.12,
                horizontal_spacing=0.08,
            )

            for idx, col in enumerate(sorted_cols[:n_density_cols]):
                row = idx // n_cols + 1
                col_pos = idx % n_cols + 1

                fig.add_trace(
                    go.Histogram(
                        x=original_df[col].dropna(),
                        name="Original",
                        opacity=0.6,
                        marker_color="#636EFA",
                        showlegend=(idx == 0),
                        legendgroup="original",
                        histnorm="probability density",
                    ),
                    row=row,
                    col=col_pos,
                )

                fig.add_trace(
                    go.Histogram(
                        x=drifted_df[col].dropna(),
                        name="Comparison",
                        opacity=0.6,
                        marker_color="#EF553B",
                        showlegend=(idx == 0),
                        legendgroup="drifted",
                        histnorm="probability density",
                    ),
                    row=row,
                    col=col_pos,
                )

            fig.update_layout(
                title_text="Distribution Comparison",
                height=300 * n_rows,
                barmode="overlay",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            )

            output_path = os.path.join(output_dir, filename)
            pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate comparison plots: {e}")
            return None

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def generate_spearman_heatmaps(
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        output_dir: str,
        filename: str = "spearman_heatmaps.html",
    ) -> Optional[str]:
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

    @staticmethod
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
