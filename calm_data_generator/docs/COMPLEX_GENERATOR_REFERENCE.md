# ComplexGenerator — Reference

**Location:** `calm_data_generator.generators.complex.ComplexGenerator`

---

## What is ComplexGenerator?

`ComplexGenerator` is the **mathematical core** of the library. It is an abstract intermediate class that sits between `BaseGenerator` (pure interface) and concrete domain generators (`ClinicalDataGenerator`, etc.).

Its purpose is to provide three reusable mathematical engines — **unconditional Gaussian Copula**, **conditional Gaussian Copula**, and **stochastic effects** — so any new domain generator can inherit them without reimplementing the math.

```
BaseGenerator (ABC)
└── ComplexGenerator                  ← mathematical engines (this module)
    ├── ClinicalDataGenerator         ← clinical domain (genes, proteins, demographics)
    ├── (your own generator)          ← inherits all three engines for free
    └── ...
```

If you want to build a generator for a new domain (IoT sensors, financial data, insurance claims...), **inheriting from `ComplexGenerator` is the right starting point**.

---

## The Gaussian Copula — foundation of the first two engines

Both generation engines are built on the same idea: the **Gaussian Copula**.

The problem it solves: generate `n` correlated variables where each variable has its own marginal distribution (NegBinomial, Gamma, Normal, Exponential...).

The algorithm in three steps:

```
1.  Z ~ N(0, Σ)          → Gaussian vector with the desired correlation
2.  U = Φ(Z)             → transform to uniform [0,1] via the normal CDF
                           (preserves rank correlation structure)
3.  X_i = F_i⁻¹(U_i)    → apply each variable's marginal distribution
                           via its quantile function (PPF)
```

Result: `X` has exactly marginal distribution `F_i` in each column, and the Spearman correlation specified in `Σ`. Pearson correlation is very close for symmetric distributions, and slightly attenuated for highly skewed ones (NegBinomial, Exponential).

**Important:** `Σ` must be positive semidefinite (PSD). If it is not, both engines repair it automatically by clipping negative eigenvalues to `1e-6` — no exception is raised.

---

## Complete example: IoT sensor generator

This example shows how to build a full domain generator from scratch using the three `ComplexGenerator` engines.

**Domain:** industrial sensor network. Each row is a time reading. Variables are temperature per sensor (Normal distribution). Sensors are correlated with each other. Anomalies can be injected into any sensor subset.

```python
import numpy as np
import pandas as pd
import scipy.stats as stats

from calm_data_generator.generators.complex.ComplexGenerator import ComplexGenerator


class IoTSensorGenerator(ComplexGenerator):
    """
    Generator for correlated industrial sensor readings.

    Each sensor produces temperature readings (Normal distribution).
    Sensors are block-correlated (same plant floor).
    Anomalies can be injected into any subset of sensors.

    Usage:
        gen = IoTSensorGenerator(random_state=42)
        df = gen.generate(n_readings=500, n_sensors=6)

        df_anomaly = gen.generate(
            n_readings=500, n_sensors=6,
            anomaly_sensors=[0, 2],
            anomaly_type="additive_shift",
            anomaly_value=[10.0, 20.0],
        )
    """

    def generate(
        self,
        n_readings: int,
        n_sensors: int,
        sensor_correlation: float = 0.4,
        anomaly_sensors: list = None,
        anomaly_type: str = "additive_shift",
        anomaly_value = 5.0,
    ) -> pd.DataFrame:

        # ── 1. Define per-sensor marginals ───────────────────────────────────
        rng = np.random.default_rng(self.rng.integers(2**31))
        marginals = []
        for i in range(n_sensors):
            temp_mean = rng.uniform(60, 80)   # each sensor has its own baseline
            marginals.append(stats.norm(loc=temp_mean, scale=rng.uniform(2, 5)))

        # ── 2. Define correlation structure ──────────────────────────────────
        sigma = np.full((n_sensors, n_sensors), sensor_correlation)
        np.fill_diagonal(sigma, 1.0)

        # ── 3. Generate correlated readings (Engine 1) ───────────────────────
        X = self._generate_correlated_module(n_readings, marginals, sigma)
        columns = [f"sensor_{i:02d}_temp" for i in range(n_sensors)]
        df = pd.DataFrame(X, columns=columns)

        # ── 4. Inject anomalies if requested (Engine 3) ──────────────────────
        if anomaly_sensors:
            anomaly_rows = df.index[n_readings // 2:]   # second half of readings
            effect_config = {
                "index":        anomaly_sensors,
                "effect_type":  anomaly_type,
                "effect_value": anomaly_value,
            }
            self.apply_stochastic_effects(df, anomaly_rows, effect_config)
            df["anomaly"] = 0
            df.loc[anomaly_rows, "anomaly"] = 1

        return df


# ── Usage ─────────────────────────────────────────────────────────────────────

gen = IoTSensorGenerator(random_state=42, auto_report=False)

# Normal generation
df_normal = gen.generate(n_readings=1000, n_sensors=6, sensor_correlation=0.5)
print(df_normal.head())
print(f"\nMean inter-sensor correlation: "
      f"{df_normal.corr().values[~np.eye(6, dtype=bool)].mean():.3f}")

# Generation with anomaly on sensors 0 and 2
df_anomaly = gen.generate(
    n_readings=1000,
    n_sensors=6,
    sensor_correlation=0.5,
    anomaly_sensors=[0, 2],
    anomaly_type="additive_shift",
    anomaly_value=[10.0, 20.0],   # random +10 to +20°C spike
)
print(f"\nAnomaly readings: {df_anomaly['anomaly'].sum()}/1000")
```

---

## Engine 1: `_generate_correlated_module`

Unconditional Gaussian Copula. Generates data where each column has its own marginal distribution and all columns are correlated according to `sigma_module`.

### Signature

```python
def _generate_correlated_module(
    self,
    n_samples: int,
    marginals_list: list,      # scipy frozen rv objects, one per variable
    sigma_module: np.ndarray,  # correlation matrix (n_vars × n_vars)
) -> np.ndarray:               # shape: (n_samples, n_vars)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_samples` | int | Number of rows to generate |
| `marginals_list` | list | Scipy `frozen rv` distributions, one per column. They can all be different |
| `sigma_module` | np.ndarray | Correlation matrix. Non-PSD matrices are repaired automatically |

### When to use it

Whenever you need to generate multiple correlated variables with heterogeneous distributions. This is the most common case.

### Example

```python
import numpy as np
import scipy.stats as stats
from calm_data_generator.generators.complex.ComplexGenerator import ComplexGenerator

class MyGenerator(ComplexGenerator):
    def generate(self, n):
        marginals = [
            stats.norm(loc=100, scale=15),      # age-like variable
            stats.lognorm(s=0.8, scale=50000),  # income (right-skewed)
            stats.nbinom(n=5, p=0.3),           # visit count (discrete)
        ]
        sigma = np.array([
            [1.0,  0.3, -0.2],
            [0.3,  1.0,  0.1],
            [-0.2, 0.1,  1.0],
        ])
        X = self._generate_correlated_module(n, marginals, sigma)
        return pd.DataFrame(X, columns=["age", "income", "visits"])

gen = MyGenerator(random_state=42, auto_report=False)
df = gen.generate(1000)
```

---

## Engine 2: `_generate_conditional_data`

Conditional Gaussian Copula. Generates variables **given that we already know others**. Useful for modelling causality or observed dependencies: genes conditioned on demographics, prices conditioned on macroeconomic indicators, etc.

### Signature

```python
def _generate_conditional_data(
    self,
    n_samples: int,
    conditioning_data: np.ndarray,    # observed data (n_samples, n_cond)
    conditioning_marginals: list,      # marginals for the known variables
    target_marginals: list,            # marginals for the variables to generate
    full_covariance: np.ndarray,       # joint covariance (n_cond + n_target, n_cond + n_target)
) -> np.ndarray:                       # shape: (n_samples, n_target)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_samples` | int | Number of samples |
| `conditioning_data` | np.ndarray | Already-observed variables, shape `(n_samples, n_cond)` |
| `conditioning_marginals` | list | Marginals for the known variables. Discrete distributions are handled automatically via Randomized Quantile Residuals (RQR) |
| `target_marginals` | list | Marginals for the variables to generate |
| `full_covariance` | np.ndarray | Joint covariance matrix over all variables |

### How it works internally

```
Z_cond = Φ⁻¹(F_cond(conditioning_data))    ← latent Gaussian space
μ_t|c  = S_tc · S_cc⁻¹ · Z_cond            ← conditional mean per sample
Σ_t|c  = S_tt - S_tc · S_cc⁻¹ · S_ct      ← conditional covariance (fixed)
Z_t    ~ N(μ_t|c, Σ_t|c)                   ← conditional sample
X_t    = F_target⁻¹(Φ(Z_t))               ← apply target marginals
```

### When to use it

When you have real data for some variables and want to generate synthetic data for others that is **coherent** with the observed values. Example: you have real demographic data for 100 patients and want to generate synthetic gene expression that is realistic given their profile.

### Example

```python
# Real demographic data for 100 patients
age_bmi = np.column_stack([
    np.random.normal(55, 12, 100),   # age
    np.random.normal(27,  4, 100),   # BMI
])

demographic_marginals = [stats.norm(55, 12), stats.norm(27, 4)]
gene_marginals = [stats.lognorm(s=0.5, scale=200) for _ in range(5)]

# Joint 7×7 covariance: first 2 = demographics, last 5 = genes
joint_cov = np.eye(7)
joint_cov[0, 2] = joint_cov[2, 0] = 0.4   # age correlated with gene 0
joint_cov[1, 3] = joint_cov[3, 1] = -0.3  # BMI correlated with gene 1

X_genes = gen._generate_conditional_data(
    n_samples=100,
    conditioning_data=age_bmi,
    conditioning_marginals=demographic_marginals,
    target_marginals=gene_marginals,
    full_covariance=joint_cov,
)
# X_genes.shape == (100, 5)
```

---

## Engine 3: `apply_stochastic_effects`

Applies a stochastic effect to a subset of entities **in-place**. Used to inject disease signals, market shocks, sensor anomalies, temporal drift, etc.

### Signature

```python
def apply_stochastic_effects(
    self,
    df: pd.DataFrame,     # modified in-place, no return value
    entity_ids,           # index labels of the affected entities
    effect_config: dict,
) -> None:
```

### Effect configuration

```python
effect_config = {
    "index":        [0, 1, 5],       # column indices to affect
    "effect_type":  "fold_change",   # one of the 7 types
    "effect_value": [1.5, 3.0],      # scalar or [min, max] to sample from
}
```

When `effect_value` is `[min, max]`, each entity receives an independent value sampled from `Uniform(min, max)`. When it is a scalar, values are sampled from `Normal(value, |value|·0.1)`.

### Supported effect types

| Type | Formula | Typical use case |
|------|---------|-----------------|
| `additive_shift` | `x += offset` | Sensor bias, background signal |
| `fold_change` | `x *= factor` | Gene overexpression, price multiplier |
| `power_transform` | `x **= exponent` | Non-linear distortion |
| `variance_scale` | Rescales around mean | Heteroscedasticity, volatility regime |
| `log_transform` | `x = log(x + ε)` | Log-normalisation of counts |
| `polynomial_transform` | `x = P(x)` | Arbitrary polynomial mapping |
| `sigmoid_transform` | `x = 1/(1+e^{-k(x-x₀)})` | Saturation, soft clipping |

`simple_additive_shift` is accepted as an alias for `additive_shift`.

### Example: anomaly in the last 100 readings

```python
effect = {
    "index":        [0, 3],
    "effect_type":  "additive_shift",
    "effect_value": [15.0, 25.0],   # random +15 to +25°C per row
}
gen.apply_stochastic_effects(df, df.index[-100:], effect)
```

### Example: disease signal (fold change) in sick patients

```python
sick_patients = demo_df[demo_df["Group"] == "Disease"].index
effect = {
    "index":        list(range(10, 20)),  # genes 10-19 overexpressed
    "effect_type":  "fold_change",
    "effect_value": [2.0, 5.0],           # 2x to 5x overexpression
}
gen.apply_stochastic_effects(genes_df, sick_patients, effect)
```

---

## When to use each engine

| Situation | Engine |
|-----------|--------|
| Generate correlated variables from scratch | `_generate_correlated_module` |
| Generate variables conditioned on existing observed data | `_generate_conditional_data` |
| Add signals, anomalies or effects to already-generated data | `apply_stochastic_effects` |
| All three together | Combine them inside your `generate()` |

---

## Error handling

| Situation | Behaviour |
|-----------|-----------|
| Non-PSD `sigma_module` | Repaired automatically (eigenvalue clipping, no exception) |
| Singular `S_cc` in conditional | Regularised with `+1e-6·I` |
| Non-PSD conditional covariance | Repaired automatically |
| Shape mismatch in `_generate_conditional_data` | Descriptive `ValueError` |
| Unknown `effect_type` | `ValueError` |
| Empty `entity_ids` | Safe no-op, returns immediately |
