"""ML utility (TSTR — Train Synthetic, Test Real) mixin for QualityReporter."""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("QualityReporter")


class _MLUtilityMixin:
    """Mixin providing TSTR (Train Synthetic, Test Real) ML utility metrics for QualityReporter."""

    def _run_tstr(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        target_col: str,
        output_dir: str,
    ) -> Optional[Dict[str, Any]]:
        """Train on Synthetic, Test on Real. RF classifier or regressor depending on target dtype."""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.metrics import (
                balanced_accuracy_score,
                f1_score,
                mean_absolute_percentage_error,
                mean_squared_error,
                r2_score,
                roc_auc_score,
            )
        except ImportError:
            logger.warning("scikit-learn required for TSTR. Install with: pip install scikit-learn")
            return None

        try:
            if target_col not in real_df.columns or target_col not in synthetic_df.columns:
                logger.warning("TSTR skipped: target_col '%s' not in both dataframes.", target_col)
                return None

            # Detect task
            is_classification = (
                real_df[target_col].dtype == object
                or (
                    real_df[target_col].nunique() <= 20
                    and real_df[target_col].dtype in (np.int64, np.int32, np.int8, int)
                )
            )

            # Prepare features — drop target, encode categoricals
            feature_cols = [c for c in real_df.columns if c != target_col]
            synth_X = pd.get_dummies(synthetic_df[feature_cols])
            real_X = pd.get_dummies(real_df[feature_cols])
            real_X = real_X.reindex(columns=synth_X.columns, fill_value=0)

            synth_y = synthetic_df[target_col]
            real_y = real_df[target_col]

            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(synth_X, synth_y)
                preds = model.predict(real_X)
                proba = model.predict_proba(real_X)
                classes = model.classes_
                if len(classes) == 2:
                    auc = roc_auc_score(real_y, proba[:, 1])
                else:
                    auc = roc_auc_score(real_y, proba, multi_class="ovr", average="macro")

                metrics = {
                    "task": "classification",
                    "roc_auc": round(float(auc), 4),
                    "balanced_accuracy": round(float(balanced_accuracy_score(real_y, preds)), 4),
                    "f1_macro": round(float(f1_score(real_y, preds, average="macro")), 4),
                }
                metric_labels = ["ROC AUC", "Balanced Accuracy", "F1 (macro)"]
                metric_values = [metrics["roc_auc"], metrics["balanced_accuracy"], metrics["f1_macro"]]
                metric_colors = [
                    "#3b82f6" if v >= 0.7 else "#f59e0b" if v >= 0.5 else "#ef4444"
                    for v in metric_values
                ]
                task_label = "Classification"
                interpretation = (
                    "ROC AUC near 0.5 means synthetic data preserves little predictive signal. "
                    "Higher values indicate the synthetic data is useful for training."
                )
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(synth_X, synth_y)
                preds = model.predict(real_X)

                metrics = {
                    "task": "regression",
                    "r2": round(float(r2_score(real_y, preds)), 4),
                    "mape": round(float(mean_absolute_percentage_error(real_y, preds)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(real_y, preds))), 4),
                }
                metric_labels = ["R²", "MAPE", "RMSE"]
                metric_values = [metrics["r2"], metrics["mape"], metrics["rmse"]]
                metric_colors = ["#3b82f6", "#f59e0b", "#6366f1"]
                task_label = "Regression"
                interpretation = (
                    "R² close to 1 means synthetic data preserves the target distribution well. "
                    "Lower MAPE and RMSE indicate better predictive utility."
                )

            if self.verbose:
                logger.info("TSTR metrics (%s): %s", task_label, metrics)

            # Generate HTML report
            self._save_tstr_html(
                metrics=metrics,
                metric_labels=metric_labels,
                metric_values=metric_values,
                metric_colors=metric_colors,
                task_label=task_label,
                interpretation=interpretation,
                target_col=target_col,
                n_train=len(synthetic_df),
                n_test=len(real_df),
                output_dir=output_dir,
            )

            return metrics

        except Exception as e:
            logger.error("TSTR failed: %s", e)
            return None

    def _save_tstr_html(
        self,
        metrics: Dict[str, Any],
        metric_labels: List[str],
        metric_values: List[float],
        metric_colors: List[str],
        task_label: str,
        interpretation: str,
        target_col: str,
        n_train: int,
        n_test: int,
        output_dir: str,
    ) -> None:
        """Saves tstr_report.html with a Plotly bar chart + metrics table."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("plotly required to generate TSTR HTML report.")
            return

        fig = go.Figure(go.Bar(
            x=metric_labels,
            y=metric_values,
            marker_color=metric_colors,
            text=[f"{v:.4f}" for v in metric_values],
            textposition="outside",
        ))
        fig.update_layout(
            title=dict(text=f"TSTR — {task_label} ({target_col})", font=dict(size=20)),
            yaxis=dict(range=[0, max(metric_values) * 1.25], title="Score"),
            xaxis=dict(title="Metric"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter, sans-serif", size=13),
            margin=dict(t=80, b=60, l=60, r=40),
        )

        rows_html = "".join(
            f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
            for k, v in metrics.items() if k != "task"
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TSTR Report</title>
<style>
  body {{ font-family: Inter, sans-serif; background: #f8fafc; padding: 32px; color: #1e293b; }}
  .card {{ background: white; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.08); padding: 28px; max-width: 860px; margin: 0 auto 24px; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 4px; }}
  .meta {{ color: #64748b; font-size: .9rem; margin-bottom: 20px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .95rem; }}
  th {{ background: #f1f5f9; text-align: left; padding: 10px 14px; font-weight: 600; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #e2e8f0; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 99px; font-size: .8rem; font-weight: 600; background: #dbeafe; color: #1d4ed8; }}
  .note {{ background: #f8fafc; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 4px; font-size: .9rem; color: #334155; margin-top: 0; }}
</style>
</head>
<body>
<div class="card">
  <h1>TSTR — Train on Synthetic, Test on Real</h1>
  <p class="meta">
    <span class="badge">{task_label}</span>&nbsp;
    Target: <strong>{target_col}</strong> &nbsp;|&nbsp;
    Train (synthetic): <strong>{n_train}</strong> rows &nbsp;|&nbsp;
    Test (real): <strong>{n_test}</strong> rows
  </p>
  {fig.to_html(full_html=False, include_plotlyjs="cdn")}
  <br>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {rows_html}
  </table>
</div>
<div class="card">
  <p class="note">{interpretation}</p>
</div>
</body>
</html>"""

        path = os.path.join(output_dir, "tstr_report.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("TSTR report saved to: %s", path)
