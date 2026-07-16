"""
Tutorial 9: Reports Deep Dive
=============================

This tutorial demonstrates how to use the various reporting tools available in
calm_data_generator to assess data quality, privacy and drift.

We will cover:
1. QualityReporter: Comparing Real vs Synthetic Data
2. evaluate(): Lightweight, in-memory fidelity check (no files written)
3. StreamReporter: Analyzing Synthetic Data Streams
4. Privacy Assessment: DCR, NNDR and Singling-Out risk via QualityReporter
"""

import os
import shutil

import numpy as np
import pandas as pd

from calm_data_generator.generators.stream import StreamReporter
from calm_data_generator.generators.tabular import QualityReporter

# Note: PrivacyReporter removed - use QualityReporter with privacy_check=True

# Setup output directory
OUTPUT_DIR = "tutorial_reports_output"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

print(f"Reports will be saved to: {OUTPUT_DIR}")

# ============================================================
# 1. Prepare Data
# ============================================================

# Create a "Real" dataset
np.random.seed(42)
n_samples = 500
real_df = pd.DataFrame(
    {
        "age": np.random.normal(40, 10, n_samples).astype(int),
        "salary": np.random.normal(50000, 15000, n_samples),
        "category": np.random.choice(["A", "B", "C"], n_samples),
        "target": np.random.randint(0, 2, n_samples),
    }
)

# Create a "Synthetic" dataset (slightly different to show contrast)
synthetic_df = pd.DataFrame(
    {
        "age": np.random.normal(42, 12, n_samples).astype(int),  # Drifted mean/std
        "salary": np.random.normal(52000, 18000, n_samples),  # Drifted
        "category": np.random.choice(
            ["A", "B", "C"], n_samples, p=[0.2, 0.5, 0.3]
        ),  # Diff probs
        "target": np.random.randint(0, 2, n_samples),
    }
)

print("Data prepared.")

# ============================================================
# 2. QualityReporter (Real vs Synthetic)
# ============================================================

print("\n--- Running QualityReporter ---")
quality_reporter = QualityReporter(verbose=True)

# Generate comprehensive report
# This creates report_results.json, html profiles, and visualizations.
# Since v2.3.0, generate_comprehensive_report() also returns that same results dict.
quality_report_results = quality_reporter.generate_comprehensive_report(
    real_df=real_df,
    synthetic_df=synthetic_df,
    generator_name="Tutorial_Synthetic_Gen",
    output_dir=os.path.join(OUTPUT_DIR, "quality_report"),
    target_column="target",
    privacy_check=True,  # Calculates DCR, NNDR, and Singling-Out risk (if anonymeter installed)
    tstr=True,  # Also runs TSTR (Train Synthetic, Test Real)
    minimal=False,  # Set to True for faster execution (skips PCA/Correlations)
)

print("Quality report generated.")
print("Statistical metrics (KS/MMD/Wasserstein):", quality_report_results["statistical_metrics"])

# ============================================================
# 2b. evaluate() - Lightweight, In-Memory Fidelity Check
# ============================================================

# For a fast check without writing any file (e.g. inside a training loop or a test),
# use evaluate() instead of generate_comprehensive_report(). It computes the same
# quality/statistical/TSTR metrics, but not privacy metrics (those stay file-report-only).
print("\n--- Running QualityReporter.evaluate() (no files written) ---")
eval_result = quality_reporter.evaluate(real_df, synthetic_df, target_column="target")
print("quality_scores:", eval_result["quality_scores"])
print("tstr_metrics:", eval_result["tstr_metrics"])

# ============================================================
# 3. StreamReporter (Single Synthetic Dataset)
# ============================================================

print("\n--- Running StreamReporter ---")
stream_reporter = StreamReporter(verbose=True)

# Analyze the synthetic dataframe as if it came from a stream
stream_reporter.generate_report(
    synthetic_df=synthetic_df,
    generator_name="Tutorial_Stream_Gen",
    output_dir=os.path.join(OUTPUT_DIR, "stream_report"),
    target_column="target",
    focus_cols=["age", "salary"],
)

print("Stream report generated.")

# ============================================================
# 4. Privacy Assessment (via QualityReporter)
# ============================================================

print("\n--- Privacy Assessment with QualityReporter ---")
print("Privacy features are integrated into QualityReporter via privacy_check=True.")
privacy_metrics = quality_report_results.get("privacy_metrics") or {}
print("DCR  (Distance to Closest Record):", privacy_metrics.get("dcr_mean"), "/",
      privacy_metrics.get("dcr_5th_percentile"), "(mean / 5th percentile)")
print("NNDR (Nearest Neighbor Distance Ratio):", privacy_metrics.get("nndr_mean"), "/",
      privacy_metrics.get("nndr_5th_percentile"), "(mean / 5th percentile)")
singling_out = privacy_metrics.get("singling_out")
if singling_out is not None:
    print("Singling-Out risk:", singling_out["risk"],
          f"(95% CI: {singling_out['ci_low']}-{singling_out['ci_high']})")
else:
    print("Singling-Out risk: not computed (install with `pip install "
          "calm-data-generator[privacy]` to enable)")

print("\nReports tutorial completed!")
print(
    f"Check the '{OUTPUT_DIR}' directory to explore the generated HTML and JSON files."
)
