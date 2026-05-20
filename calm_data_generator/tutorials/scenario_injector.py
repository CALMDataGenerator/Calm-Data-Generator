"""
Tutorial 6: ScenarioInjector - Feature Evolution and Target Construction
=========================================================================

This tutorial demonstrates how to evolve features over time and
construct target variables based on custom formulas.
"""

import numpy as np
import pandas as pd

from calm_data_generator.generators.dynamics import ScenarioInjector

# ============================================================
# 1. Create sample time series data
# ============================================================

np.random.seed(42)
n_samples = 100

data = pd.DataFrame(
    {
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="D"),
        "temperature": np.random.normal(20, 5, n_samples),
        "humidity": np.random.normal(60, 10, n_samples),
        "pressure": np.random.normal(1013, 10, n_samples),
        "wind_speed": np.random.exponential(5, n_samples),
        "region": np.random.choice(["North", "South", "East", "West"], n_samples),
    }
)

print("Original data:")
print(data.head())
print(f"\nTemperature mean: {data['temperature'].mean():.2f}")

# ============================================================
# 2. Evolve Features Over Time
# ============================================================

injector = ScenarioInjector(seed=42)

# Define evolution configuration
evolution_config = {
    "temperature": {
        "type": "trend",
        "slope": 0.05,  # 0.05 units increase per step
        "noise_std": 0.5,  # Add some noise
    },
    "humidity": {
        "type": "cycle",
        "period": 30,  # 30-day cycle
        "amplitude": 5,  # ±5 units
    },
    "wind_speed": {
        "type": "random_walk",
        "step_std": 0.5,  # Random walk step size
    },
}

evolved_data = injector.evolve_features(
    df=data.copy(), evolution_config=evolution_config, time_col="timestamp"
)

print("\n--- Evolved Features ---")
print(f"Original temp mean: {data['temperature'].mean():.2f}")
print(f"Evolved temp mean:  {evolved_data['temperature'].mean():.2f}")
print(f"Evolved temp (last 10): {evolved_data['temperature'].tail(10).values}")

# ============================================================
# 3. Construct Target Variable (Regression)
# ============================================================

# Create target based on formula: temp * 0.5 + humidity * 0.3 + noise
data_with_target = injector.construct_target(
    df=data.copy(),
    target_col="energy_consumption",
    formula="temperature * 2.5 + humidity * 0.8 + wind_speed * 3",
    noise_std=5.0,
    task_type="regression",
)

print("\n--- Regression Target ---")
print(
    data_with_target[
        ["temperature", "humidity", "wind_speed", "energy_consumption"]
    ].head()
)

# ============================================================
# 4. Construct Target Variable (Classification)
# ============================================================

# Create binary target with threshold
data_with_binary = injector.construct_target(
    df=data.copy(),
    target_col="is_hot",
    formula="temperature + wind_speed * 0.5",
    task_type="classification",
    threshold=22,  # >22 = hot (1), else cold (0)
)

print("\n--- Classification Target ---")
print(data_with_binary[["temperature", "wind_speed", "is_hot"]].head(10))
print("\nClass distribution:")
print(data_with_binary["is_hot"].value_counts())

# ============================================================
# 5. Custom Formula Functions
# ============================================================


# Use lambda for complex formulas
def custom_formula(row):
    if row["region"] in ["North", "South"]:
        return row["temperature"] * 1.2
    else:
        return row["temperature"] * 0.8


data_custom = injector.construct_target(
    df=data.copy(),
    target_col="adjusted_temp",
    formula=custom_formula,
    task_type="regression",
)

print("\n--- Custom Formula (Region-based) ---")
print(data_custom[["region", "temperature", "adjusted_temp"]].head(10))

# ============================================================
# 6. Project to Future Periods
# ============================================================

# Project data to future with trends
future_data = injector.project_to_future_period(
    df=data.copy(),
    periods=3,  # 3 future periods
    period_length=30,  # 30 days each
    trend_config={
        "temperature": 0.02,  # 2% increase per period
        "humidity": -0.01,  # 1% decrease per period
    },
    time_col="timestamp",
)

print("\n--- Future Projection ---")
print(f"Original rows: {len(data)}")
print(f"With future periods: {len(future_data)}")
print(f"Last original date: {data['timestamp'].max()}")
print(f"Last projected date: {future_data['timestamp'].max()}")

# ============================================================
# 7. driven_by Evolution — delta driven by another column
# ============================================================

print("\n--- driven_by: pressure delta = f(temperature) per row ---")

driven_data = injector.evolve_features(
    df=data.copy(),
    evolution_config={
        "pressure": {
            "type":        "driven_by",
            "driver_col":  "temperature",    # each row's delta depends on its temperature
            "func":        "linear",
            "func_params": {"slope": 0.4},   # delta_pressure = 0.4 * temperature
        },
        "humidity": {
            "type":        "driven_by",
            "driver_col":  "temperature",
            "func":        "exponential",
            "func_params": {"scale": 0.002, "rate": 0.03},
        },
    },
)

delta_pressure = driven_data["pressure"] - data["pressure"]
print("Sample driven_by deltas (pressure ~ 0.4 × temperature):")
print(
    data[["temperature"]].assign(
        delta_pressure=delta_pressure,
        expected=data["temperature"] * 0.4,
    ).head(5).to_string(index=False)
)

print("\n✅ ScenarioInjector tutorial completed!")
