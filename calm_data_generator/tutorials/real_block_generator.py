"""
Tutorial 6: RealBlockGenerator - Generating Data in Blocks
==========================================================

This tutorial demonstrates how to use RealBlockGenerator to synthetize
data that is partitioned into blocks (e.g., by time periods, regions, or customer segments).
"""

import os
import shutil

import numpy as np
import pandas as pd

from calm_data_generator.generators.tabular.RealBlockGenerator import RealBlockGenerator

# Setup output directory
OUTPUT_DIR = "tutorial_output/06_real_block"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Create a Block-Structured Dataset
# ============================================================

print("Creating sample dataset with 'Year' as blocks...")
np.random.seed(42)

# Simulate data for 3 years (2020, 2021, 2022) with slightly different distributions (Drift)
data_blocks = []
years = [2020, 2021, 2022]

for year in years:
    n = 200
    # Drift: Every year income increases slightly
    income_mean = 50000 + (year - 2020) * 5000

    df = pd.DataFrame(
        {
            "Year": [year] * n,
            "Income": np.random.normal(income_mean, 10000, n),
            "Age": np.random.randint(20, 70, n),
            "Employment": np.random.choice(["Full-time", "Part-time", "Unemployed"], n),
            "Churn": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        }
    )
    data_blocks.append(df)

full_data = pd.concat(data_blocks, ignore_index=True)
print(f"Original Data: {len(full_data)} samples across years {years}")
print(full_data.groupby("Year")["Income"].mean().to_dict())


# ============================================================
# 2. Basic Block Generation
# ============================================================

print("\n--- 2. Generating Synthetic Blocks ---")
gen = RealBlockGenerator(verbose=True)

synthetic_data = gen.generate(
    data=full_data,
    output_dir=OUTPUT_DIR,
    method="cart",  # Use CART for fast generation
    block_column="Year",  # Split and learn each block separately
    target_col="Churn",
    n_samples_block=100,  # Generate 100 samples per block (Uniform)
)

print("Synthetic Data Generated:")
print(synthetic_data["Year"].value_counts())
print("Synthetic Income Means by Year:")
print(synthetic_data.groupby("Year")["Income"].mean())


# ============================================================
# 3. Dynamic Chunking (No Existing Block Column)
# ============================================================

print("\n--- 3. Dynamic Chunking (Fixed Size) ---")
# Drop the Year column to simulate a continuous stream
data_continuous = full_data.drop(columns=["Year"])

synthetic_chunked = gen.generate(
    data=data_continuous,
    output_dir=OUTPUT_DIR,
    method="cart",
    chunk_size=150,  # Automatically split into blocks of 150 rows
    target_col="Churn",
)

print("Synthetic Data with Dynamic Chunks:")
print(synthetic_chunked["chunk"].value_counts())


# ============================================================
# 4. Injecting Drift Schedule
# ============================================================

print("\n--- 4. Injecting Drift Schedule ---")
# We can tell the generator to inject specific drift scenarios after generation

drift_schedule = [
    {
        "method": "inject_outliers_global",
        "params": {"cols": ["Income"], "outlier_prob": 0.1, "factor": 5.0},
    }
]

synthetic_drifted = gen.generate(
    data=full_data,
    output_dir=OUTPUT_DIR,
    method="cart",
    block_column="Year",
    drift_config=drift_schedule,  # Apply this drift to the final result
)

print("Drift injection complete. Check output folder for reports.")

print(f"\n✅ Tutorial completed! Outputs saved to {OUTPUT_DIR}")
