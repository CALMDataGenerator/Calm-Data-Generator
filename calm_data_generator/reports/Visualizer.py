"""
Visualizer Module
Generates interactive Plotly HTML plots for data visualization.

Implementation lives in themed submodules (_distribution.py, _correlation.py,
_dimensionality.py, _metrics_cards.py, _comparison.py, _sequence.py) — this
file is a thin facade so `Visualizer.generate_x(...)` call sites keep working
unchanged. See ARCHITECTURE.md for the module map.
"""

from ._comparison import generate_comparison_plots
from ._correlation import generate_spearman_heatmaps
from ._dimensionality import generate_dimensionality_plot
from ._distribution import generate_density_plots, generate_qq_plots
from ._metrics_cards import generate_quality_evolution_plot, generate_quality_scores_card
from ._sequence import generate_evolution_plot, generate_sequence_plot


class Visualizer:
    """
    Generates interactive Plotly HTML visualizations for data reports.
    """

    generate_density_plots = staticmethod(generate_density_plots)
    generate_qq_plots = staticmethod(generate_qq_plots)
    generate_spearman_heatmaps = staticmethod(generate_spearman_heatmaps)
    generate_dimensionality_plot = staticmethod(generate_dimensionality_plot)
    generate_quality_scores_card = staticmethod(generate_quality_scores_card)
    generate_quality_evolution_plot = staticmethod(generate_quality_evolution_plot)
    generate_comparison_plots = staticmethod(generate_comparison_plots)
    generate_sequence_plot = staticmethod(generate_sequence_plot)
    generate_evolution_plot = staticmethod(generate_evolution_plot)
