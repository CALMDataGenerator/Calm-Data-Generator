import numpy as np
import pandas as pd

from calm_data_generator.generators.drift.DriftInjector import DriftInjector
from calm_data_generator.generators.dynamics.ScenarioInjector import ScenarioInjector


def test_correlation_propagation():
    print("=== Testing Correlation Propagation ===")

    # 1. Create correlated data: Y = 2*X + noise
    # Perfect correlation would be rho=1. With noise it's high but < 1.
    np.random.seed(42)
    n = 1000
    x = np.random.normal(0, 1, n)
    y = 2 * x + np.random.normal(0, 0.1, n)  # Very high correlation
    df = pd.DataFrame({"X": x, "Y": y})

    print(f"Original Correlation:\n{df.corr()}\n")

    print("--- Test 1: DriftInjector (Shift X, should shift Y) ---")
    injector = DriftInjector(auto_report=False)

    # Inject shift in X WITHOUT propagation (correlations=None)
    df_no_prop = injector.inject_feature_drift(
        df, ["X"], drift_type="shift", drift_magnitude=1.0, correlations=None
    )
    print("Without propagation:")
    print(f"Mean Shift X: {df_no_prop['X'].mean() - df['X'].mean():.4f}")
    print(f"Mean Shift Y: {df_no_prop['Y'].mean() - df['Y'].mean():.4f} (Should be ~0)")

    # Inject shift in X WITH propagation (pass the correlation DataFrame)
    # We pass the original correlation to strictly enforce that relationship
    correlations = df.corr()
    df_prop = injector.inject_feature_drift(
        df,
        ["X"],
        drift_type="shift",
        drift_magnitude=1.0,
        correlations=correlations,
    )
    print("\nWith propagation:")
    delta_x = df_prop["X"].mean() - df["X"].mean()
    delta_y = df_prop["Y"].mean() - df["Y"].mean()
    print(f"Mean Shift X: {delta_x:.4f}")
    print(f"Mean Shift Y: {delta_y:.4f}")

    # Expected: Delta_Y ~ rho * (std_y/std_x) * Delta_X
    # Here Y ~ 2X, so std_y ~ 2*std_x, rho ~ 1. So Delta_Y ~ 2 * Delta_X
    expected_y = correlations.loc["X", "Y"] * (df["Y"].std() / df["X"].std()) * delta_x
    print(f"Expected Shift Y: {expected_y:.4f}")

    print("\n--- Test 2: ScenarioInjector (Evolve X, should evolve Y) ---")
    scenario = ScenarioInjector(seed=42)

    evolution_config = {
        "X": {"type": "linear", "slope": 0.01}  # X grows over time
    }

    # Without propagation (correlations=None)
    df_scen_no_prop = scenario.evolve_features(
        df, evolution_config, correlations=None
    )
    print("Without propagation:")
    print(f"Change X: {df_scen_no_prop['X'].mean() - df['X'].mean():.4f}")
    print(
        f"Change Y: {df_scen_no_prop['Y'].mean() - df['Y'].mean():.4f} (Should be ~0)"
    )

    # With propagation (pass the correlation DataFrame)
    df_scen_prop = scenario.evolve_features(
        df,
        evolution_config,
        correlations=correlations,
    )
    print("\nWith propagation:")
    delta_x_scen = df_scen_prop["X"].mean() - df["X"].mean()
    delta_y_scen = df_scen_prop["Y"].mean() - df["Y"].mean()
    print(f"Change X: {delta_x_scen:.4f}")
    print(f"Change Y: {delta_y_scen:.4f}")
    print(
        f"Expected Change Y: {correlations.loc['X', 'Y'] * (df['Y'].std() / df['X'].std()) * delta_x_scen:.4f}"
    )

    if abs(delta_y_scen - expected_y) < 0.1:  # Approximate check
        print("\nSUCCESS: Correlation propagation working as expected.")
    else:
        print("\nWARNING: Propagation values might be off.")


if __name__ == "__main__":
    test_correlation_propagation()
