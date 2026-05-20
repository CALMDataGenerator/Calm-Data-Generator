import numpy as np

from calm_data_generator.generators.clinical.Clinic import ClinicalDataGenerator
from calm_data_generator.generators.configs import DriftConfig


def build_correlation_matrix(n_demo, group_sizes, correlations):
    """
    Helper to build a block correlation matrix.
    Supports fixed values or ranges (min, max) for internal correlations.
    """
    n_omics = sum(group_sizes)
    n_total = n_demo + n_omics
    matrix = np.eye(n_total)

    current_idx = n_demo
    for i, size in enumerate(group_sizes):
        end_idx = current_idx + size
        config = correlations[i]

        # Internal correlation
        if "internal" in config:
            internal_val = config["internal"]
            # Handle range (tuple or list)
            if isinstance(internal_val, (tuple, list)) and len(internal_val) == 2:
                val = np.random.uniform(internal_val[0], internal_val[1])
            else:
                val = float(internal_val)

            if val > 0:
                block = matrix[current_idx:end_idx, current_idx:end_idx]
                block[:] = val
                np.fill_diagonal(block, 1.0)
                matrix[current_idx:end_idx, current_idx:end_idx] = block

        # Demographic correlation
        if "demo_idx" in config and "demo_corr" in config:
            demo_idx = config["demo_idx"]
            corr = config["demo_corr"]
            if demo_idx is not None:
                matrix[demo_idx, current_idx:end_idx] = corr
                matrix[current_idx:end_idx, demo_idx] = corr

        current_idx = end_idx

    return matrix


def run_tutorial():
    print("=== ClinicalDataGenerator Tutorial (Unified API) ===")

    # 1. Initialize
    generator = ClinicalDataGenerator(seed=42)
    n_samples = 1000

    # 2. Configure Scenarios

    # Custom Demographics
    custom_demo_cols = {
        "Age": {
            "distribution": "truncnorm",
            "a": -2.0,
            "b": 2.5,
            "loc": 60,
            "scale": 10,
        },
        "Sex": {"distribution": "binom", "n": 1, "p": 0.5},
    }

    # Correlations Config
    # We need to pre-calculate dimensions to build matrix, this part remains "advanced"
    # But passing it is now cleaner.
    # Note: Using indexes requires knowing column order.
    # For tutorial simplicity, we'll verify this order or assume it.
    col_to_idx = {"Age": 0, "Sex": 1}  # Simplified assumption for tutorial
    n_demo = 2

    gene_group_sizes = [100, 200, 500]
    gene_correlations_config = [
        {"internal": (0.3, 0.5), "demo_idx": col_to_idx.get("Age"), "demo_corr": 0.4},
        {
            "internal": 0.3,
            "demo_idx": None,
            "demo_corr": 0.0,
        },  # Sex mapped to Sex_Binario internally?
        {"internal": 0.0},
    ]

    gene_corr_matrix = build_correlation_matrix(
        n_demo, gene_group_sizes, gene_correlations_config
    )

    # Weights for Target Variable (Diagnosis)
    # Using 'Age', 'Sex_Binario' (which is created internally from Sex)
    # And genes.
    target_weights = {
        "Age": 0.3,
        "Sex_Binario": 0.1,
    }
    # Add weights for first 10 genes (G_0 to G_9)
    for i in range(10):
        target_weights[f"G_{i}"] = 0.05

    # 3. Configure Drift (New API)
    # Drift for demographics (e.g., Age)
    drift_demo = DriftConfig(
        method="inject_feature_drift_gradual",
        params={
            "feature_cols": ["Age"],
            "drift_magnitude": 0.5,
            "drift_type": "shift",
            "start_index": int(n_samples * 0.5),
        },
    )

    # Drift for genes (e.g., G_0)
    drift_genes = DriftConfig(
        method="inject_feature_drift",
        params={
            "feature_cols": ["G_0"],
            "drift_magnitude": 1.0,  # Large shift
            "drift_type": "shift",
            "start_index": int(n_samples * 0.7),
        },
    )

    # 4. GENERATE (One Call)
    print("\nExecuting generation pipeline with Drift...")
    output_dir = "tutorial_output"

    datasets = generator.generate(
        n_samples=n_samples,
        n_genes=sum(gene_group_sizes),
        n_proteins=0,  # Skip proteins for this tutorial
        # Demographics Config
        custom_demographic_columns=custom_demo_cols,
        control_disease_ratio=0.5,
        # Genes Config
        gene_type="Microarray",
        demographic_gene_correlations=gene_corr_matrix,
        # Target Variable Config
        target_variable_config={
            "weights": target_weights,
            "binary_threshold": 0.0,
            "name": "diagnosis",
        },
        output_dir=output_dir,
        save_dataset=True,
        # Pass Drift Configs
        demographics_drift_config=[drift_demo],
        genes_drift_config=[drift_genes],
    )

    # 4. Verify Results
    demo_df = datasets["demographics"]
    genes_df = datasets["genes"]

    print(f"\nData saved to {output_dir}/")
    print("Demographics shape:", demo_df.shape)
    print("Genes shape:", genes_df.shape)

    if "diagnosis" in demo_df.columns:
        print("Diagnosis balance:\n", demo_df["diagnosis"].value_counts(normalize=True))
    else:
        print("Warning: 'diagnosis' column not found in demographics.")

    print("\nDrift Check:")
    print(f"Age mean (first 100): {demo_df['Age'].head(100).mean():.2f}")
    print(f"Age mean (last 100): {demo_df['Age'].tail(100).mean():.2f}")

    # Check gene drift
    if "genes" in datasets and "G_0" in datasets["genes"].columns:
        g0 = datasets["genes"]["G_0"]
        print(f"G_0 mean (first 100): {g0.head(100).mean():.2f}")
        print(f"G_0 mean (last 100): {g0.tail(100).mean():.2f}")


if __name__ == "__main__":
    run_tutorial()
