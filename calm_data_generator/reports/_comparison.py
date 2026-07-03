"""Before/after and drift comparison plots for Visualizer."""

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
