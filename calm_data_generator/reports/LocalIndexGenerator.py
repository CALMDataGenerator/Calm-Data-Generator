"""
Local Index Generator
Generates a static index.html dashboard    - Uses synthetic data generation (e.g. via Synthcity) to augment the dataset.
"""

import logging
import os
import time

logger = logging.getLogger("LocalIndexGenerator")


class LocalIndexGenerator:
    """
    Generates a local index.html to act as a dashboard for reports.
    """

    HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CalmOps Data Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f8f9fa; margin: 0; padding: 0; }
        .container { display: flex; height: 100vh; }
        .sidebar { width: 280px; background-color: #343a40; color: white; padding: 20px; display: flex; flex-direction: column; overflow-y: auto; }
        .sidebar h2 { margin-top: 0; font-size: 1.2rem; margin-bottom: 20px; border-bottom: 1px solid #4b545c; padding-bottom: 10px; }
        .section-title { font-size: 0.75rem; text-transform: uppercase; color: #6c757d; margin: 15px 0 8px 0; letter-spacing: 0.5px; }
        .nav-btn { background: none; border: none; color: #c2c7d0; padding: 10px 15px; text-align: left; cursor: pointer; font-size: 0.95rem; border-radius: 4px; margin-bottom: 3px; transition: background 0.2s; width: 100%; }
        .nav-btn:hover { background-color: #495057; color: white; }
        .nav-btn.active { background-color: #007bff; color: white; }
        .content { flex-grow: 1; padding: 0; background-color: white; overflow: hidden; position: relative; }
        iframe { width: 100%; height: 100%; border: none; display: none; }
        iframe.active { display: block; }
        .tab-row { display: flex; align-items: center; margin-bottom: 3px; }
        .tab-row a { color: #6c757d; margin-left: 8px; text-decoration: none; font-size: 1.1rem; }
        .tab-row a:hover { color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>CalmOps Report</h2>
            <!-- YDATA_SECTION -->
            <!-- PLOTLY_SECTION -->
            <!-- SC_SECTION -->
        </div>

        <div class="content">
            <!-- IFRAMES_PLACEHOLDER -->
        </div>
    </div>

    <script>
        function showTab(id) {
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            const btn = document.getElementById('btn-' + id);
            if(btn) btn.classList.add('active');

            document.querySelectorAll('iframe').forEach(f => f.classList.remove('active'));

            const el = document.getElementById(id);
            if(el) el.classList.add('active');
        }

        document.addEventListener('DOMContentLoaded', () => {
             const priority = ['comparison', 'profile', 'quality_scores', 'density', 'dimensionality', 'quality_evolution'];
             for (const id of priority) {
                 if (document.getElementById(id)) {
                     showTab(id);
                     break;
                 }
             }
        });
    </script>
</body>
</html>
"""

    @staticmethod
    def create_index(output_dir: str) -> str:
        """
        Scans output_dir for artifacts and generates index.html.
        """
        def _scan_reports(report_defs, output_dir, ts, multi_file=False):
            found, nav_html, iframe_html = [], "", ""
            for rid, config in report_defs.items():
                files = config.get("files", [config.get("file")])
                label = config["label"]
                for fname in files:
                    if os.path.exists(os.path.join(output_dir, fname)):
                        found.append((rid, fname, label))
                        break
            for rid, fname, label in found:
                nav_html += f'''
                    <div class="tab-row">
                        <button class="nav-btn" onclick="showTab('{rid}')" id="btn-{rid}">{label}</button>
                        <a href="{fname}" target="_blank" title="Open in New Tab">-></a>
                    </div>
                    '''
                iframe_html += f'<iframe id="{rid}" src="{fname}?v={ts}" scrolling="yes"></iframe>\n'
            return found, nav_html, iframe_html

        try:
            ts = int(time.time())
            iframes_html = ""

            ydata_reports = {
                "comparison": {"files": ["comparison_report.html"], "label": "YData Comparison"},
                "profile": {"files": ["generated_profile.html", "profile_report.html"], "label": "Generated Data Profile"},
            }
            plotly_reports = {
                "quality_scores": {"file": "quality_scores.html", "label": "Quality Scores"},
                "quality_evolution": {"file": "quality_evolution.html", "label": "Quality Evolution"},
                "drift_stats": {"file": "drift_stats.html", "label": "Drift Statistics"},
                "evolution_plot": {"file": "evolution_plot.html", "label": "Feature Evolution (ScenarioInjector)"},
                "plot_comparison": {"file": "plot_comparison.html", "label": "Distribution Comparison"},
                "density": {"file": "density_plots.html", "label": "Density Plots"},
                "dimensionality": {"file": "dimensionality_plot.html", "label": "PCA Visualization"},
                "discriminator_metrics": {"file": "discriminator_metrics.html", "label": "Adversarial Validation"},
                "discriminator_explainability": {"file": "discriminator_explainability.html", "label": "Discriminator Explainability"},
                "tstr": {"file": "tstr_report.html", "label": "TSTR (ML Utility)"},
                "spearman_heatmaps": {"file": "spearman_heatmaps.html", "label": "Spearman Correlations"},
                "qq_plots": {"file": "qq_plots.html", "label": "QQ Plots"},
                "statistical_tests": {"file": "statistical_tests.html", "label": "Statistical Tests (KS · Levene · MMD)"},
            }
            sc_reports = {
                "scgft": {"file": "scgft_report.html", "label": "scGFT Evaluation"},
            }

            found_ydata, ydata_nav, ydata_iframes = _scan_reports(ydata_reports, output_dir, ts)
            found_plotly, plotly_nav, plotly_iframes = _scan_reports(plotly_reports, output_dir, ts)
            found_sc, sc_nav, sc_iframes = _scan_reports(sc_reports, output_dir, ts)

            ydata_section = ('<div class="section-title">YData Reports</div>\n' + ydata_nav) if found_ydata else ""
            plotly_section = ('<div class="section-title">Interactive Plots</div>\n' + plotly_nav) if found_plotly else ""
            sc_section = ('<div class="section-title">Single Cell Analysis</div>\n' + sc_nav) if found_sc else ""
            iframes_html = ydata_iframes + plotly_iframes + sc_iframes

            html = LocalIndexGenerator.HTML_TEMPLATE.replace(
                "<!-- YDATA_SECTION -->", ydata_section
            ).replace(
                "<!-- PLOTLY_SECTION -->", plotly_section
            ).replace(
                "<!-- SC_SECTION -->", sc_section
            ).replace(
                "<!-- IFRAMES_PLACEHOLDER -->", iframes_html
            )

            index_path = os.path.join(output_dir, "index.html")
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(html)

            logger.info(f"Dashboard created at: {index_path}")
            return index_path

        except Exception as e:
            logger.error(f"Failed to create dashboard index: {e}")
            return ""
