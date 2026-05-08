"""
Discriminator Reporter Module
=============================

This module implements the DiscriminatorReporter class which performs Adversarial Validation.
It trains a classifier (Random Forest) to distinguish between Real and Synthetic data.

Metrics:
- AUC-ROC: ability to distinguish (0.5 = indistinguishable/good quality, 1.0 = distinguishable/bad quality or drift)
- Accuracy, F1, Balanced Accuracy

Explainability:
- Feature Importance (MDI)
- SHAP Values (if available)

"""

import json
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional dependencies
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# SHAP removed
SHAP_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class DiscriminatorReporter:
    """
    Reporter that trains a discriminator to distinguish between two datasets (Real vs Synthetic).
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = logging.getLogger("DiscriminatorReporter")

        if not SKLEARN_AVAILABLE:
            self.logger.warning(
                "scikit-learn not found. DiscriminatorReporter will not work."
            )

    def generate_report(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        output_dir: str,
        label_real: str = "Real",
        label_synthetic: str = "Synthetic",
    ) -> Dict[str, Any]:
        """
        Generates the discriminator report.

        Args:
            real_df: The reference/real dataframe (Label=0)
            synthetic_df: The synthetic/drifted dataframe (Label=1)
            output_dir: Directory to save reports
            label_real: Display label for class 0
            label_synthetic: Display label for class 1

        Returns:
            Dictionary with metrics and paths to generated files.
        """
        if not SKLEARN_AVAILABLE:
            return {}

        self.logger.info("Starting Adversarial Validation (Discriminator)...")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Prepare Data
        X, y, feature_names, encoders = self._prepare_data(real_df, synthetic_df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # 2. Train Model
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            # max_depth=None,  # Allow full depth to avoid underfitting
        )
        clf.fit(X_train, y_train)

        # 3. Compute Metrics

        # Train score (to detect underfitting)
        y_train_proba = clf.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_proba)

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Compute raw metrics
        raw_auc = float(roc_auc_score(y_test, y_pred_proba))
        raw_conf_acc = float(
            accuracy_score(y_test, y_pred)
        )  # Classifier Accuracy (Target: 0.5)

        metrics = {
            "train_auc": float(train_auc),
            # --- Traditional Classifier Metrics (How good is the discriminator?) ---
            # Ideally, these should be poor (random guessing)
            "discriminator_auc": raw_auc,
            "discriminator_accuracy": raw_conf_acc,
            # --- Indistinguishability Scores (Higher is Better for Data Quality) ---
            # 1.0 = Perfect Indistinguishability (Real == Synthetic)
            # 0.0 = Easy to Distinguish (Drift/Bad Quality)
            "similarity_score": 1.0 - (2.0 * abs(raw_auc - 0.5)),  # Based on AUC
            "confusion_score": 1.0
            - (2.0 * abs(raw_conf_acc - 0.5)),  # Based on Accuracy (0.5 is ideal)
        }

        self.logger.info(
            f"Discriminator Metrics: AUC={metrics['discriminator_auc']:.4f}, Similarity={metrics['similarity_score']:.4f}"
        )

        # 4. Generate Reports

        # Metrics & ROC HTML
        metrics_file = self._generate_metrics_report(
            metrics, y_test, y_pred_proba, output_dir, label_real, label_synthetic
        )

        # Explainability HTML
        explain_file = self._generate_explainability_report(
            clf, X_test, feature_names, output_dir
        )

        return {
            "metrics": metrics,
            "metrics_file": metrics_file,
            "explainability_file": explain_file,
        }

    def _prepare_data(
        self, df0: pd.DataFrame, df1: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """
        Concatenates datasets and encodes categorical variables.
        """
        # Ensure common columns
        common_cols = [c for c in df0.columns if c in df1.columns]
        df0 = df0[common_cols].copy()
        df1 = df1[common_cols].copy()

        # Add targets
        df0["__target__"] = 0
        df1["__target__"] = 1

        combined = pd.concat([df0, df1], axis=0, ignore_index=True)

        y = combined.pop("__target__").values

        # Simple encoding for categoricals
        X_df = combined.copy()
        encoders = {}

        for col in X_df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X_df[col] = X_df[col].fillna("__NaN__").astype(str)
            X_df[col] = le.fit_transform(X_df[col])
            encoders[col] = le

        # Handle dates - convert to numeric timestamp or drop?
        # For simplicity, drop or convert to ordinal
        for col in X_df.select_dtypes(include=["datetime"]).columns:
            X_df[col] = X_df[col].astype("int64") // 10**9

        # Fill NaNs
        X_df = X_df.fillna(0)  # Simple imputation for validity

        return X_df.values, y, X_df.columns.tolist(), encoders

    def _generate_metrics_report(
        self, metrics: Dict, y_test, y_scores, output_dir: str, label0: str, label1: str
    ) -> str:
        """Creates an HTML report with metrics table and ROC curve."""
        filename = "discriminator_metrics.html"
        filepath = os.path.join(output_dir, filename)

        if not PLOTLY_AVAILABLE:
            return None

        if len(np.unique(y_test)) < 2:
            self.logger.warning("Only one class in y_test — skipping ROC curve generation.")
            return filepath

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name="ROC Curve (AUC = {:.3f})".format(metrics["discriminator_auc"]),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash"),
                name="Random (AUC=0.5)",
            )
        )
        fig.update_layout(
            title="Adversarial Validation - ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=700,
            height=500,
        )
        roc_div = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

        # Metrics Table (HTML)
        html_content = f"""
        <html>
        <head>
            <title>Adversarial Validation Metrics</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 60%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f8f9fa; }}
                .badge {{ padding: 8px; border-radius: 4px; color: white; font-weight: bold; display: inline-block; text-align: center; }}

                /* New logic: 1.0 = Good (Indistinguishable), 0.0 = Bad (Distinguishable) */
                .score-container {{
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-left: 5px solid #007bff;
                    margin-bottom: 20px;
                }}
                .main-score {{ font-size: 2em; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Adversarial Validation Metrics</h1>

            <div class="score-container">
                <div>Similarity Score (Indistinguishability)</div>
                <div class="main-score" style="color: {self._get_score_color(metrics["similarity_score"])};">
                    {metrics["similarity_score"]:.1%}
                </div>
                <small>
                    100% = Perfect Quality (Indistinguishable)<br>
                    0% = Low Quality (Easily Distinguishable / Drifted)<br>
                    <i>Formula: 1 - 2 * |AUC - 0.5|</i>
                </small>
            </div>

            <p>
                The Discriminator tries to distinguish between <b>{label0}</b> (0) and <b>{label1}</b> (1).
            </p>

            <table>
                <tr><th>Metric</th><th>Value</th><th>Interpretation (1.0 = Perfect Quality)</th></tr>
                <tr>
                    <td><b>Similarity Score (AUC-based)</b></td>
                    <td><b>{metrics["similarity_score"]:.4f}</b></td>
                    <td><b>{self._interpret_score(metrics["similarity_score"])}</b></td>
                </tr>
                <tr>
                    <td>Confusion Score (Acc-based)</td>
                    <td>{metrics["confusion_score"]:.4f}</td>
                    <td>Ability to confuse the discriminator (Target: 1.0)</td>
                </tr>
                <tr>
                    <td><i>Technical Discriminator AUC</i></td>
                    <td><i>{metrics["discriminator_auc"]:.4f}</i></td>
                    <td><i>Raw model performance (Target: 0.5)</i></td>
                </tr>
                <tr><td><i>Technical Train AUC</i></td><td><i>{metrics["train_auc"]:.4f}</i></td><td><i>(Check for underfitting if close to 0.5)</i></td></tr>
            </table>

            <div style="margin-top: 20px;">
                {roc_div}
            </div>
        </body>
        </html>
        """

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return filepath

    def _interpret_score(self, score: float) -> str:
        if score > 0.8:
            return '<span class="badge" style="background-color: #28a745;">Excellent (Indistinguishable)</span>'
        elif score > 0.4:
            return '<span class="badge" style="background-color: #ffc107; color: black;">Fair (Separable)</span>'
        else:
            return '<span class="badge" style="background-color: #dc3545;">Poor (Highly Distinct)</span>'

    def _get_score_color(self, score: float) -> str:
        if score > 0.8:
            return "#28a745"
        elif score > 0.4:
            return "#ffc107"
        else:
            return "#dc3545"

    def _interpret_auc(self, auc: float) -> str:
        # Legacy/Support method
        return self._interpret_score(1.0 - (2.0 * abs(auc - 0.5)))

    def _generate_explainability_report(
        self, clf, X_test, feature_names, output_dir: str
    ) -> str:
        """Creates an HTML report with Feature Importance and SHAP values."""
        filename = "discriminator_explainability.html"
        filepath = os.path.join(output_dir, filename)

        # 1. Feature Importance (MDI)
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Plot top 20
        top_n = min(20, len(feature_names))
        top_indices = indices[:top_n]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[feature_names[i] for i in top_indices],
                y=importances[top_indices],
                marker_color="#17a2b8",
            )
        )
        fig.update_layout(
            title="Discriminator Feature Importance (Top Features)",
            xaxis_title="Feature",
            yaxis_title="Importance (Gini)",
            width=900,
            height=500,
        )
        importance_div = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

        # HTML Structure
        html_content = f"""
        <html>
        <head>
            <title>Discriminator Explainability</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
            </style>
        </head>
        <body>
            <h1>Discriminator Explainability</h1>
            <p>Which features allow the model to distinguish between the datasets?</p>

            <h2>1. Feature Importance (Random Forest)</h2>
            {importance_div}

            <p><i>SHAP values are disabled in this version.</i></p>
        </body>
        </html>
        """

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return filepath
