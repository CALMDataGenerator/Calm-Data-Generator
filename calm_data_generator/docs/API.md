# CalmGenerator API Documentation

## Modules Overview

### generators.tabular - Real Data Synthesis

```python
from calm_data_generator.generators.tabular import RealGenerator, QualityReporter
```

**RealGenerator** - Generate synthetic data from real datasets

| Method | Description |
|--------|-------------|
| `cart` | CART-based iterative synthesis |
| `rf` | Random Forest synthesis |
| `lgbm` | LightGBM synthesis |
| `ctgan` | CTGAN (deep learning) |
| `tvae` | TVAE (variational autoencoder) |
| `bn` | Bayesian Network (causal structure) |
| `smote` | SMOTE oversampling |
| `adasyn` | ADASYN adaptive sampling |
| `timegan` | TimeGAN (time series) |
| `timevae` | TimeVAE (time series VAE) |
| `fflows` | FourierFlows (periodic time series) |
| `scvi` | scVI (Single-Cell VI) |
| `ddpm` | Tabular Diffusion (DDPM) |

**Notable Parameters:**
- `differentiation_factor` (float): Enhances class separation in latent space (TVAE/scVI only).
- `clipping_mode` (str): `'strict'`, `'permissive'`, or `'none'` for handling output ranges.
- `use_latent_sampling` (bool): For scVI, sample from real data latent space.

**Advanced Validation & Processing:**
- `_apply_postprocess_distribution(self, synthetic_df, class_counts, target_col, ...)`: Intelligently resamples rows in a synthetic DataFrame to meet target class distributions while preserving correlations. Emits warnings if classes are missing. Used automatically by many models when `custom_distributions` is set.

---

### generators.complex - Abstract Mathematical Layer

```python
from calm_data_generator.generators.complex.ComplexGenerator import ComplexGenerator
```

**ComplexGenerator** - Abstract base for any generator that needs correlated or conditional synthesis. Inherit from this instead of `BaseGenerator` when you need the three built-in engines:

| Engine | Method | Description |
|--------|--------|-------------|
| Gaussian Copula (unconditional) | `_generate_correlated_module(n, marginals, sigma)` | Generates correlated samples from arbitrary marginals |
| Gaussian Copula (conditional) | `_generate_conditional_data(n, cond_data, cond_marginals, tgt_marginals, cov)` | Generates target variables conditioned on observed data |
| Stochastic effects | `apply_stochastic_effects(df, entity_ids, effect_config)` | Applies one of 7 effect types in-place |

See [COMPLEX_GENERATOR_REFERENCE.md](COMPLEX_GENERATOR_REFERENCE.md) for full documentation.

---

### generators.clinical - Clinical Data

```python
from calm_data_generator.generators.clinical import ClinicalDataGenerator
```

`ClinicalDataGenerator` inherits from `ComplexGenerator` and uses all three engines internally.

**Methods:**
- `generate()` - Generate demographics + omics
- `generate_longitudinal_data()` - Multi-visit patient data

---

### generators.stream - Stream-Based

```python
from calm_data_generator.generators.stream import StreamGenerator
```

**Features:**
- River library compatible
- Balanced generation
- SMOTE post-hoc
- Sequence generation

---

### generators.drift - Drift Injection

```python
from calm_data_generator.generators.drift import DriftInjector
```

**Drift Types:**
- `inject_drift()` **(Unified)**
- `inject_feature_drift_gradual()`
- `inject_feature_drift_abrupt()`
- `inject_feature_drift_recurrent()`
- `inject_label_drift_gradual()`
- `inject_label_drift_abrupt()`
- `inject_label_drift_incremental()`
- `inject_concept_drift()`
- `inject_conditional_drift()`
- `inject_outliers_global()`
- `inject_new_category_drift()`
- `inject_correlation_matrix_drift()`
- `inject_binary_probabilistic_drift()`
- `inject_multiple_types_of_drift()`
- `inject_functional_drift()` — drift magnitude as f(driver_col) per row (Pilar 5)
- `inject_causal_cascade()` — DAG-based non-linear causal propagation (Pilar 5)

---

### generators.dynamics - Scenario Evolution & Causal Engine

```python
from calm_data_generator.generators.dynamics import ScenarioInjector
from calm_data_generator.generators.dynamics import CausalEngine
```

**ScenarioInjector** — evolves features over time. Evolution types: `linear`, `exponential_growth`, `decay`, `seasonal`, `step`, `noise`, `random_walk`, `driven_by` (Pilar 5).

**CausalEngine** — DAG-based causal cascade propagation. Define parent→child relationships with transfer functions and propagate perturbations through the graph. See [CAUSAL_ENGINE_REFERENCE.md](CAUSAL_ENGINE_REFERENCE.md).

---

### generators.utils - Shared Utilities

```python
from calm_data_generator.generators.utils.propagation import propagate_numeric_drift, apply_func
```

- `propagate_numeric_drift(df, rows, driver_col, delta, correlations)` — correlation-based delta propagation used by DriftInjector and ScenarioInjector
- `apply_func(func_name, params, x)` — evaluate named transfer functions (`linear`, `exponential`, `power`, `polynomial`, callable)

---

### privacy - Privacy Transformations (Integrated)

Privacy features are integrated into the `QualityReporter`. You can assess quality and privacy using:

```python
# Comprehensive Quality Report (including ARI metrics for class separability)
reporter.generate_comprehensive_report(..., privacy_check=True)

# Standalone ARI calculation
ari_scores = reporter.calculate_ari(real_df, synthetic_df, target_col="label")
```

For differential privacy guarantees at generation time, use the `dpgan` or `pategan` synthesis methods via `RealGenerator.generate(method="dpgan", ...)` or `RealGenerator.generate(method="pategan", ...)`.

---

## Installation

```bash
# Basic
pip install calm-data-generator

# Stream (River)
pip install calm-data-generator[stream]

# Full
pip install calm-data-generator[full]
```

> [!NOTE]
> **Privacy Features**: Privacy assessment (DCR metrics) is now integrated into `QualityReporter`. Use `privacy_check=True` when generating reports.
