"""
Tutorial 4: StreamGenerator - Stream-Based Data Generation
===============================================================

This tutorial demonstrates how to generate synthetic data from
stream-based generators (River library compatible).
"""

import numpy as np

from calm_data_generator import StreamGenerator
from calm_data_generator.generators.configs import DriftConfig, ReportConfig

# ============================================================
# 1. Create a stream generator
# ============================================================


def simple_stream():
    """A simple Python generator that yields (features, label) tuples."""
    rng = np.random.default_rng(42)
    while True:
        x = {
            "feature1": rng.normal(0, 1),
            "feature2": rng.normal(5, 2),
            "feature3": rng.uniform(0, 10),
        }
        y = 1 if x["feature1"] + x["feature2"] > 5 else 0
        yield x, y


# ============================================================
# 2. Basic Generation
# ============================================================

gen = StreamGenerator()

# Define Configurations
drift_config = [
    DriftConfig(
        method="inject_feature_drift",
        params={
            "feature_cols": ["feature1"],
            "drift_magnitude": 0.5,
            "drift_type": "shift",
        },
    )
]
report_config = ReportConfig(output_dir="stream_tutorial_output")

# Generate data from stream with drift and reporting
synthetic = gen.generate(
    generator_instance=simple_stream(),
    n_samples=100,
    drift_config=drift_config,
    report_config=report_config,
    generate_report=True,
)

print("Generated data shape:", synthetic.shape)
print(synthetic.head())
print("\nClass distribution:")
print(synthetic["target"].value_counts())

# ============================================================
# 3. Balanced Generation
# ============================================================

# Generate balanced data (equal classes)
synthetic_balanced = gen.generate(
    generator_instance=simple_stream(), n_samples=100, balance=True
)

print("\n--- Balanced Generation ---")
print("Class distribution:")
print(synthetic_balanced["target"].value_counts())

# ============================================================
# 4. Balanced Generation (Alternative)
# ============================================================

# Generate balanced data using random oversampling of minority class
synthetic_balanced2 = gen.generate(
    generator_instance=simple_stream(),
    n_samples=100,
    balance=True,
)

print("\n--- Balanced (random oversampling) ---")
print("Class distribution:")
print(synthetic_balanced2["target"].value_counts())

# ============================================================
# 5. Sequence Generation (User Sessions)
# ============================================================

# Generate with entity-based sequences
sequence_config = {
    "entity_col": "user_id",
    "events_per_entity": 5,  # ~5 events per user
}

synthetic_seq = gen.generate(
    generator_instance=simple_stream(),
    n_samples=50,
    date_start="2024-01-01",
    sequence_config=sequence_config,
)

print("\n--- Sequence Generation ---")
print(f"Shape: {synthetic_seq.shape}")
print(f"Unique users: {synthetic_seq['user_id'].nunique()}")
print("\nSample user events:")
print(synthetic_seq[synthetic_seq["user_id"] == "User_0"].head())

# ============================================================
# 6. River Library Integration
# ============================================================

try:
    from river import synth

    # Use River's built-in generators
    river_stream = synth.Agrawal(seed=42)

    synthetic_river = gen.generate(generator_instance=river_stream, n_samples=100)

    print("\n--- River Agrawal Dataset ---")
    print(f"Shape: {synthetic_river.shape}")
    print(synthetic_river.head())

except ImportError:
    print("\nRiver library not installed. Install with: pip install river")

print("\n✅ Stream-based generation tutorial completed!")
