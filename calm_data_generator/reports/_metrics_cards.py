"""Quality-score card and evolution plots for Visualizer."""

import logging
import os
from typing import Any, Dict, List, Optional

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger("Visualizer")


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
