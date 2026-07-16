"""
Tutorial 1: RealGenerator - Synthetic Data from Real Datasets
==============================================================

This tutorial demonstrates how to use RealGenerator to create
synthetic data that preserves the statistical properties of real data.
"""

import numpy as np
import pandas as pd

from calm_data_generator import RealGenerator
from calm_data_generator.generators.configs import DriftConfig, ReportConfig

# ============================================================
# 1. Basic Usage - Generate synthetic data with CART
# ============================================================

# Create sample dataset
# NOTE: In a real-world scenario, you would load your own dataset here.
# Example: data = pd.read_csv("path/to/your_real_data.csv")
np.random.seed(42)
data = pd.DataFrame(
    {
        "age": np.random.randint(18, 80, 100),
        "income": np.random.normal(50000, 15000, 100),
        "score": np.random.uniform(0, 1, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "target": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
    }
)

print("Original data shape:", data.shape)
print(data.head())

# Initialize generator
gen = RealGenerator()

# Define Configurations (New API)
drift_config = [
    DriftConfig(
        method="inject_feature_drift",
        params={"feature_cols": ["income"], "drift_magnitude": 0.5},
    )
]
report_config = ReportConfig(output_dir="tutorial_output", target_column="target")

# Generate synthetic data using CART (fast, tree-based)
# Passing configs to enable automatic drift injection and reporting
synthetic = gen.generate(
    data=data,
    n_samples=200,
    method="cart",
    target_col="target",
    drift_injection_config=drift_config,
    report_config=report_config,
    auto_report=True,
)

print("\nSynthetic data shape:", synthetic.shape)
print(synthetic.head())

# ============================================================
# 2. Deep Learning Methods - CTGAN/TVAE
# ============================================================

# Generate with CTGAN (requires synthcity/torch)
try:
    synthetic_ctgan = gen.generate(
        data=data,
        n_samples=100,
        method="ctgan",
        epochs=100,  # Passed as kwargs
        batch_size=64,
    )
    print("\nCTGAN synthetic data:", synthetic_ctgan.shape)
except Exception as e:
    print(f"CTGAN not available: {e}")

# Generate with Gaussian Copula
try:
    synthetic_copula = gen.generate(data=data, n_samples=100, method="copula")
    print("\nCopula synthetic data:", synthetic_copula.shape)
except Exception as e:
    print(f"Copula failed: {e}")


# ============================================================
# 3. Constraints - Apply business rules
# ============================================================

# Generate with constraints: age > 20, income > 0
constraints = [
    {"col": "age", "op": ">", "val": 20},
    {"col": "income", "op": ">", "val": 0},
]

synthetic_constrained = gen.generate(
    data=data, n_samples=100, method="cart", constraints=constraints
)

print("\nConstrained data - Min age:", synthetic_constrained["age"].min())
print("Constrained data - Min income:", synthetic_constrained["income"].min())

# ============================================================
# 4. Single-Cell - scVI & GEARS
# ============================================================

# Generate with scVI
try:
    synthetic_scvi = gen.generate(data=data, n_samples=50, method="scvi", epochs=10)
    print("\nscVI synthetic data:", synthetic_scvi.shape)
except Exception as e:
    print(f"scVI not available: {e}")

# Generate with GEARS (Perturbation Prediction)
try:
    synthetic_gears = gen.generate(
        data=data,
        n_samples=50,
        method="gears",
        perturbations=["age", "income"],  # Genes/features to perturb
        epochs=10,
    )
    print("\nGEARS synthetic data:", synthetic_gears.shape)
except Exception as e:
    print(f"GEARS not available: {e}")


# ============================================================
# 5. Oversampling Methods - SMOTE/ADASYN
# ============================================================


# Balance classes with SMOTE
synthetic_smote = gen.generate(
    data=data, n_samples=100, method="smote", target_col="target"
)

print("\nSMOTE class distribution:")
print(synthetic_smote["target"].value_counts())

# ============================================================
# 6. Bayesian Network (bn) - Clinical/structured tabular data
# ============================================================

# BN models conditional dependencies between variables using a directed
# acyclic graph (structure learning). Ideal for clinical/epidemiological data.
try:
    clinical_data = pd.DataFrame(
        {
            "age": np.random.randint(20, 80, 100),
            "gender": np.random.choice(["M", "F"], 100),
            "bmi": np.random.normal(25, 5, 100),
            "diagnosis": np.random.choice([0, 1], 100),
        }
    )
    synthetic_bn = gen.generate(
        data=clinical_data,
        n_samples=100,
        method="bn",
        target_col="diagnosis",
    )
    print("\nBayesian Network synthetic data:", synthetic_bn.shape)
except Exception as e:
    print(f"BN not available: {e}")

# ============================================================
# 7. FourierFlows (fflows) - Periodic time series
# ============================================================

# fflows applies normalizing flows in the frequency domain.
# More stable than TimeGAN, best for periodic/seasonal series.
# Requires: sequence_key (identifies sequences), time_key (timestamps/steps).
# Needs at least ~20 sequences to pass Synthcity's internal cross-validation.
try:
    n_seq, seq_len = 30, 10
    ts_rows = []
    for i in range(n_seq):
        for t in range(seq_len):
            ts_rows.append(
                {
                    "seq_id": i,
                    "time": t,
                    "feature1": np.sin(t / 3.0) + np.random.normal(0, 0.1),
                    "feature2": np.cos(t / 3.0) + np.random.normal(0, 0.1),
                }
            )
    ts_data = pd.DataFrame(ts_rows)

    synthetic_fflows = gen.generate(
        data=ts_data,
        n_samples=10,
        method="fflows",
        sequence_key="seq_id",
        time_key="time",
        n_iter=50,
    )
    print("\nFourierFlows synthetic data:", synthetic_fflows.shape)
except Exception as e:
    print(f"FourierFlows not available: {e}")

# ============================================================
# 8. Simpler API - fit() / sample()
# ============================================================

# For the common case — train once, sample as many times as you want — use the
# sklearn-style wrapper instead of calling generate() repeatedly. fit() trains the
# model (no report/dataset written to disk); sample() reuses the fitted model
# without retraining, however many times you call it.
gen_fit = RealGenerator(auto_report=False, random_state=42)
gen_fit.fit(data, method="cart", target_col="target")

small_batch = gen_fit.sample(50)
large_batch = gen_fit.sample(500)  # no retraining — same fitted model
print("\nfit()/sample() — small batch:", small_batch.shape, "| large batch:", large_batch.shape)

# Chaining works too:
synthetic_chained = RealGenerator(auto_report=False).fit(data, method="cart").sample(100)
print("Chained fit().sample():", synthetic_chained.shape)
