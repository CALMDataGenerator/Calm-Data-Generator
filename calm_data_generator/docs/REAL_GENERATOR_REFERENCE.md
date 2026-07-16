# RealGenerator - Complete Reference

**Location:** `calm_data_generator.generators.tabular.RealGenerator`

The main generator for tabular data synthesis from real datasets.

---

## Initialization

```python
from calm_data_generator.generators.tabular import RealGenerator

gen = RealGenerator(
    auto_report=True,       # Automatically generate report after synthesis
    minimal_report=False,   # If True, faster report without correlations/PCA
    random_state=42,        # Seed for reproducibility
    logger=None,            # Optional custom Python logger
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_report` | bool | `True` | Automatically generate a quality report |
| `minimal_report` | bool | `False` | Simplified report (faster) |
| `random_state` | int | `None` | Seed for reproducibility |
| `logger` | Logger | `None` | Custom Python Logger instance |
| `verbose_training` | bool | `False` | Show Synthcity epoch-by-epoch loss in console during training. Useful for models like TVAE or CTGAN where `get_training_history()` is not available. |

> [!TIP]
> This library acts as a high-level wrapper. For advanced hyperparameter tuning and deep architectural details, we highly recommend consulting the documentation of the original engines:
> - **Synthcity** (CTGAN, TVAE, DDPM, TimeGAN, FF): [Synthcity Docs](https://github.com/vanderschaarlab/synthcity)
> - **scvi-tools** (scVI, scANVI): [scvi-tools Docs](https://docs.scvi-tools.org/)
> - **GEARS**: [GEARS GitHub](https://github.com/snap-stanford/GEARS)



---

## Main Method: `generate()`

```python
# New Config Imports
from calm_data_generator.generators.configs import DriftConfig, ReportConfig, DateConfig

synthetic_df = gen.generate(
    data=df,                          # Original DataFrame (required)
    n_samples=1000,                   # Number of samples to generate (required)
    method="ctgan",                   # Synthesis method

    # Configuration Objects
    report_config=ReportConfig(       # Reporting configuration
        output_dir="./output",
        target_column="target"
    ),

    # Drift Injection
    drift_injection_config=[
        DriftConfig(
            method="inject_feature_drift",
            params={"feature_cols": ["age"], "drift_magnitude": 0.5}
        )
    ],

    # Legacy arguments are still supported but Config objects are recommended
    # target_col="target",
    # output_dir="./output"
)
```

### `generate()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | - | Original and reference dataset (required) |
| `n_samples` | int | - | Number of samples to generate (required) |
| `method` | str | `"cart"` | Synthesis method to use |
| `target_col` | str | `None` | Target variable column name for balancing |
| `output_dir` | str | `None` | Directory to save reports and datasets |
| `generator_name` | str | `"RealGenerator"` | Base name for output files |
| `save_dataset` | bool | `False` | Whether to save the generated dataset to a CSV file |
| `custom_distributions` | Dict | `None` | Force specific distributions for columns |
| `custom_distribution` | Dict | `None` | ⚠️ Legacy singular alias for `custom_distributions`. Logs a warning when used — prefer the plural form. |
| `date_col` | str | `None` | Name of the date column to inject |
| `date_start` | str | `None` | ⚠️ Legacy. Start date ("YYYY-MM-DD"). Logs a warning when used without `date_config` — prefer `date_config=DateConfig(...)`. |
| `date_step` | Dict | `None` | ⚠️ Legacy — same note as `date_start`. Time increment (e.g., `{"days": 1}`) |
| `date_every` | int | `1` | ⚠️ Legacy — same note as `date_start`. Increment date every N rows |
| `drift_injection_config` | List[Union[Dict, DriftConfig]] | `None` | Configuration for post-generation drift injection |
| `dynamics_config` | Dict | `None` | Configuration for dynamic feature evolution |
| `model_params` | Dict | `None` | Model-specific hyperparameters (passed as `**kwargs`) |
| `constraints` | List[Dict] | `None` | Integrity constraints |
| `adversarial_validation` | bool | `False` | Activate Discriminator Report (Real vs Synthetic) |
| `report_config` | ReportConfig | `None` | Advanced reporting configuration object |
| `date_config` | DateConfig | `None` | Advanced date injection configuration object |
| `balance` | bool | `False` | Automatically balance class distribution in `target_col` |
| `**kwargs` | Any | - | Method-specific parameters (e.g., `epochs`, `n_latent`, `lr`) |

---

## Simpler API: `fit()` / `sample()`

For the common case — train once, sample as many times as you want — `RealGenerator` offers a
thin, sklearn-style wrapper around `generate()`:

```python
gen = RealGenerator(auto_report=False, random_state=42)

# Trains the model. Does not write any report/dataset to disk.
gen.fit(df, method="cart", target_col="target")

# Draws synthetic rows from the fitted model, as many times as you like — no retraining.
synth_small = gen.sample(100)
synth_large = gen.sample(10_000)

# Chaining works too:
synth = RealGenerator().fit(df, method="ctgan").sample(1000)
```

- `fit(data, method="cart", target_col=None, **kwargs)` accepts the same keyword arguments as
  `generate()` (minus `n_samples`, `output_dir`, `save_dataset`) and returns `self`.
- `sample(n_samples)` raises a clear `ValueError` if called before `fit()` (or `.load()`).
- Internally, `sample()` calls `generate(data=None, n_samples=...)`, which was already able to
  reuse a previously-trained `self.synthesizer` without retraining — `fit()`/`sample()` simply
  give that existing capability a more discoverable, conventional name.
- For one-shot calls (train + generate + report in a single call), `generate()` is still the
  right choice.

---

## Full `model_params` Reference

The `model_params` dictionary allows fine-tuning internal parameters for each synthesis method.

### Deep Learning (Synthcity)

| Parameter | Methods | Description |
|-----------|---------|-------------|
| `epochs` | `ctgan`, `tvae`, `rtvae`, `great`, `dpgan`, `pategan` | Number of training epochs (default: 300) |
| `batch_size` | `ctgan`, `tvae`, `rtvae`, `great` | Training batch size (default: 500) |
| `lr` | `ctgan`, `tvae`, `rtvae` | Learning rate |
| `differentiation_factor` | `tvae`, `rtvae`, `scvi` | Shift class centroids apart in latent space to force separability. |
| `clipping_mode` | `tvae`, `rtvae`, `scvi` | Clipping strategy: `'strict'`, `'permissive'`, or `'none'`. (Default: `'strict'`) |
| `clipping_factor` | `tvae`, `rtvae`, `scvi` | Percentage of range margin for `'permissive'` mode (Default: `0.1`). |

**Example:**
```python
gen.generate(
    df, 1000,
    method="ctgan",
    epochs=500,
    batch_size=256
)
```

### RTVAE (Regularized Tabular VAE)

| Parameter | Description |
|-----------|-------------|
| `epochs` | Training epochs (default: 300) |
| `batch_size` | Training batch size (default: 500) |
| `differentiation_factor` | Latent space class separation factor |
| `clipping_mode` | `'strict'`, `'permissive'`, or `'none'` |

**Example:**
```python
gen.generate(df, 1000, method="rtvae", epochs=200, differentiation_factor=0.3, target_col="label")
```

### GReaT (LLM-based Tabular Synthesis)

| Parameter | Description |
|-----------|-------------|
| `epochs` | Training epochs (default: 100) |
| `batch_size` | Training batch size (default: 32) |

**Example:**
```python
gen.generate(df, 500, method="great", epochs=50)
```

> **Note:** GReaT uses a language model internally and may be slow. It excels at preserving complex semantic correlations.

### CART (Decision Trees)

| Parameter | Description |
|-----------|-------------|
| `iterations` | Number of FCS iterations (default: 10) |
| `min_samples_leaf` | Minimum samples per leaf (auto if None) |
| `**kwargs` | Any parameter supported by sklearn's DecisionTree |

**Example:**
```python
model_params={"iterations": 20, "min_samples_leaf": 5, "max_depth": 10}
```

### Random Forest

| Parameter | Description |
|-----------|-------------|
| `iterations` | Number of FCS iterations (default: 10) |
| `n_estimators` | Number of trees |
| `min_samples_leaf` | Minimum samples per leaf |
| `**kwargs` | Any parameter supported by sklearn's RandomForest |

**Example:**
```python
model_params={"n_estimators": 100, "min_samples_leaf": 3, "max_depth": 15}
```

### LightGBM

| Parameter | Description |
|-----------|-------------|
| `iterations` | Number of FCS iterations (default: 10) |
| `n_estimators` | Number of boosting rounds |
| `learning_rate` | Learning rate |
| `**kwargs` | Any parameter supported by LightGBM |

**Example:**
```python
model_params={"n_estimators": 200, "learning_rate": 0.05, "max_depth": 8}
```

### XGBoost

| Parameter | Description |
|-----------|-------------|
| `iterations` | Number of FCS iterations (default: 10) |
| `n_estimators` | Number of boosting rounds |
| `learning_rate` | Learning rate |
| `**kwargs` | Any parameter supported by XGBoost |

**Example:**
```python
gen.generate(df, 1000, method="xgboost", n_estimators=100, learning_rate=0.1, target_col="target")
```

### Gaussian Mixture Models

| Parameter | Description |
|-----------|-------------|
| `n_components` | Number of Gaussian components (default: 5) |
| `covariance_type` | Covariance type: "full", "tied", "diag", "spherical" (default: "full") |
| `**kwargs` | Any parameter supported by sklearn's GaussianMixture |

**Example:**
```python
model_params={"n_components": 10, "covariance_type": "diag"}
```

### KDE (Kernel Density Estimation)

| Parameter | Description |
|-----------|-------------|
| `bandwidth` | Bandwidth for the kernel (default: `'scott'`) |
| `kernel` | Kernel type: `'gaussian'`, `'tophat'`, etc. (default: `'gaussian'`) |

**Example:**
```python
gen.generate(df, 500, method="kde", bandwidth=0.5)
```

> **Note:** KDE works best with purely numeric data. Categorical columns are not supported natively.

### SMOTE / ADASYN

| Parameter | Description |
|-----------|-------------|
| `k_neighbors` | Number of k-NN neighbors for SMOTE (default: 5) |
| `n_neighbors` | Number of k-NN neighbors for ADASYN (default: 5) |
| `**kwargs` | Any parameter supported by imbalanced-learn's SMOTE/ADASYN |

**Example:**
```python
model_params={"k_neighbors": 7}  # For SMOTE
model_params={"n_neighbors": 5}  # For ADASYN
```

### Single-Cell Methods

**GEARS** (Graph-based Perturbation Prediction)
```python
synthetic = gen.generate(
    expression_df, 500,
    method='gears',
    perturbations=['GENE1', 'GENE2'],  # Required: genes to perturb
    epochs=20,
    batch_size=32,
    device='cpu'
)
```
> **IMPORTANT:** GEARS requires installation from source via:
> `pip install "git+https://github.com/snap-stanford/GEARS.git@f374e43"`
> And PyTorch >= 2.4.0.

> **Note:** Valid perturbations must be present in the GEARS Gene Ontology graph.

### Privacy-Preserving Methods

#### DPGAN

| Parameter | Description |
|-----------|-------------|
| `epsilon` | Privacy ε budget — lower = more private (default: 1.0) |
| `delta` | Privacy δ parameter (default: 1e-5) |
| `epochs` | Training epochs (default: 300) |

**Example:**
```python
gen.generate(df, 1000, method="dpgan", epsilon=0.5, delta=1e-5)
```

#### PATE-GAN

| Parameter | Description |
|-----------|-------------|
| `epsilon` | Privacy ε budget — lower = more private (default: 1.0) |
| `delta` | Privacy δ parameter (default: 1e-5) |
| `epochs` | Training epochs (default: 300) |
| `teacher_iters` | Teacher training iterations |
| `student_iters` | Student training iterations |

**Example:**
```python
gen.generate(df, 1000, method="pategan", epsilon=1.0, delta=1e-5)
```

### DataSynthesizer

| Parameter | Description |
|-----------|-------------|
| `k` | Bayesian network degree (default: 5, k=1 is independent) |

**Example:**
```python
model_params={"k": 3}  # Higher k captures more correlations
```

### Drift-Aware Methods

#### Conditional Drift

| Parameter | Default | Description |
|-----------|---------|-------------|
| `time_col` | `None` | Column used to order rows by time. If `None`, row index is used |
| `n_stages` | `5` | Number of discrete drift stages to discretize the time axis into |
| `base_method` | `"tvae"` | Underlying Synthcity model: `"tvae"` or `"ctgan"` |
| `general_stages` | `None` | List of stage indices to generate (e.g. `[3, 4]` for end drift only). If `None`, generates all stages evenly |

**Example:**
```python
gen.generate(
    df, 1000,
    method="conditional_drift",
    time_col="date",
    n_stages=5,
    base_method="tvae",
    general_stages=[3, 4],   # Only generate from late drift stages
)
```

#### Windowed Copula

| Parameter | Default | Description |
|-----------|---------|-------------|
| `time_col` | `None` | Column used to sort data chronologically before windowing |
| `n_windows` | `5` | Number of time windows to fit independent copulas on |
| `generate_at` | `None` | List of interpolation points in `[0.0, 1.0]`. `0.0` = first window, `1.0` = last window. If `None`, generates evenly across all windows |

> **Note:** Only numeric columns are supported. Categorical columns are ignored.

**Example:**
```python
gen.generate(
    df, 1000,
    method="windowed_copula",
    time_col="timestamp",
    n_windows=4,
    generate_at=[0.0, 0.5, 1.0],   # Start, middle, end of drift
)
```

#### HMM (Hidden Markov Model)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | `4` | Number of hidden regimes (more = more diverse drift patterns) |
| `covariance_type` | `"full"` | Covariance matrix type per regime: `"full"`, `"diag"`, `"tied"`, `"spherical"` |
| `n_iter` | `100` | EM algorithm iterations for fitting |

> **Note:** Only numeric columns are supported. Drift emerges naturally from regime transitions.

**Example:**
```python
gen.generate(
    df, 1000,
    method="hmm",
    n_components=3,
    covariance_type="diag",
)
```

---

### Diffusion Models

| Parameter | Default | Description |
|-----------|---------|-------------|
| `diffusion_steps` | 50 | Number of diffusion steps (higher = better quality, slower) |

**Example:**
```python
model_params={"diffusion_steps": 100}
```

---

## Available Synthesis Methods

### Deep Learning

| Method | Description | Key Parameters | Dependencies |
|--------|-------------|----------------|--------------|
| `ctgan` | Conditional Tabular GAN | `epochs`, `batch_size`, `lr` | Synthcity |
| `tvae` | Tabular Variational Autoencoder | `epochs`, `batch_size` | Synthcity |
| `rtvae` | Regularized Tabular VAE | `epochs`, `batch_size`, `differentiation_factor` | Synthcity |
| `great` | GReaT (LLM-based tabular synthesis) | `epochs`, `batch_size` | Synthcity |
| `ddpm` | Tabular Diffusion (TabDDPM) | `n_iter`, `num_timesteps`, `scheduler` | Synthcity |
| `diffusion` | Tabular Diffusion Models | `steps` | PyTorch |

### Statistical Models

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `gmm` | Gaussian Mixture Model | `n_components`, `covariance_type` |
| `copula` | Copula-based synthesis | - |
| `kde` | Kernel Density Estimation | `bandwidth`, `kernel` |

### Drift-Aware Generation

| Method | Description | Key Parameters | Dependencies |
|--------|-------------|----------------|--------------|
| `conditional_drift` | Temporal conditioning via TVAE/CTGAN — learns stage-dependent distributions | `time_col`, `n_stages`, `base_method`, `general_stages` | Synthcity |
| `windowed_copula` | Gaussian Copula interpolated across time windows | `time_col`, `n_windows`, `generate_at` | copulae |
| `hmm` | Hidden Markov Model — drift emerges from regime transitions | `n_components`, `covariance_type`, `n_iter` | hmmlearn |

### Fully Conditional Specification (FCS)

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `cart` | Decision Trees (FCS) | `min_samples_leaf`, `iterations` |
| `rf` | Random Forest (FCS) | `n_estimators`, `min_samples_leaf`, `iterations` |
| `lgbm` | LightGBM (FCS) | `n_estimators`, `learning_rate`, `iterations` |
| `xgboost` | XGBoost (FCS) | `n_estimators`, `learning_rate`, `iterations` |

### Oversampling

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `smote` | Classic SMOTE | `n_neighbors` |
| `adasyn` | Adaptive SMOTE | `n_neighbors` |
| `resample` | Simple Bootstrap | - |

### Privacy-Preserving

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `dpgan` | Differentially Private GAN | `epsilon`, `delta`, `epochs` |
| `pategan` | PATE-GAN | `epsilon`, `delta`, `teacher_iters`, `student_iters` |

### Single-Cell / High-Dimensional

| Method | Description | Key Parameters | Dependencies |
|--------|-------------|----------------|--------------|
| `scvi` | scVI Variational Autoencoder | `epochs`, `n_latent`, `n_layers` | scvi-tools |
| `scanvi` | scANVI (semi-supervised scVI) | `epochs`, `n_latent`, `target_col` | scvi-tools |
| `gears` | GEARS Perturbation Prediction | `perturbations`, `epochs`, `batch_size` | gears |

---

## Method Selection Guide

Choose the right method based on your data and requirements:

| Use Case | Recommended Methods | Why |
|----------|---------------------|-----|
| **General tabular data** | `ctgan`, `tvae`, `rtvae` | Best balance of quality and speed |
| **Small datasets (< 1000 rows)** | `cart`, `rf`, `gmm`, `kde` | Don't overfit, fast |
| **Large datasets (> 100k rows)** | `lgbm`, `xgboost`, `ctgan` | Scalable |
| **Preserve correlations** | `ctgan`, `rtvae` | Capture feature relationships |
| **Class imbalance** | `smote`, `adasyn` | Designed for oversampling |
| **Fast prototyping** | `resample`, `cart` | Instant results |
| **Numeric-only data** | `gmm`, `kde`, `diffusion` | Simple distributions |
| **Differential privacy** | `dpgan`, `pategan` | Formal DP guarantees |
| **Privatize existing data** | `privatize()` | Laplace/Gaussian/RandomResponse |
| **Custom model** | `generate_custom()` | Any sklearn/synthcity/copulae model |
| **Single-cell (labeled)** | `scanvi` | Semi-supervised, label-conditioned |
| **Semantic tabular** | `great` | LLM-based, best for complex correlations |
| **Synthetic data with drift** | `conditional_drift`, `windowed_copula`, `hmm` | Native drift in generated data |

### Single-Cell Methods Details

#### scVI (Single-cell Variational Inference)

Best for generating new single-cell-like observations from scratch. These methods are specifically designed for high-dimensional **transcriptomic data (RNA-seq)**. They use deep generative models to represent biological variation while handling the heavy sparsity and technical noise (dropout) typical of single-cell datasets. They are excellent for correcting "batch effects" and synthesizing coherent gene expression profiles.

**Input Format:** Accepts `pd.DataFrame`, `AnnData` objects, or strings (paths to `.h5`, `.h5ad`, or `.csv` files) directly.

**Using File Paths (H5/H5AD/CSV):**
```python
# Pass the file path directly - the generator loads it for you!
synthetic = gen.generate(
    data="path/to/data.csv",  # Or .h5ad, .h5
    n_samples=1000,
    method="scvi",
    target_col="cell_type"
)
```


**DataFrame Input:**
```python
synthetic = gen.generate(
    data=expression_df,      # Rows=cells, Columns=genes
    n_samples=1000,
    method="scvi",
    target_col="cell_type",  # Optional metadata column
    model_params={
        "epochs": 200,
        "n_latent": 30,      # Latent space dimensions
        "n_layers": 1,       # Encoder/decoder depth
    }
)
```

**AnnData Input (Recommended for single-cell data):**
```python
import anndata

# Create or load AnnData object
adata = anndata.read_h5ad("single_cell_data.h5ad")

synthetic = gen.generate(
    data=adata,              # Pass AnnData directly
    n_samples=1000,
    method="scvi",
    target_col="cell_type",
    epochs=200,
    n_latent=30,
    n_layers=1,
)
# Returns pd.DataFrame with gene columns + metadata
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 200 | Training epochs |
| `n_latent` | 30 | Latent space dimensionality |
| `n_layers` | 1 | Number of hidden layers |
| `differentiation_factor` | 0.0 | Latent separation factor. Uses unified 5-step process to push classes apart. |
| `clipping_mode` | `'strict'` | Clipping strategy: `'strict'`, `'permissive'`, or `'none'`. |
| `use_latent_sampling` | True | Controls generation fidelity. If True, it samples from real data "anchors" to preserve patient-specific textures. |
| `preserve_library_size` | True | If True, the generated cells will have total count (library size) distributions similar to the original data. |
| `latent_noise_std` | 0.05 | Noise magnitude for latent space sampling (higher = more diversity). |

> [!TIP]
> **AnnData Support:** When passing `AnnData`, the object is used directly without conversion, preserving the original structure. The output is always a `pd.DataFrame` containing both the gene expression and the observations metadata.

#### scANVI (Semi-supervised Single-Cell Inference)

scANVI extends scVI by conditioning generation on cell type labels (semi-supervised). It requires `target_col` to identify the label column.

**When to use scANVI:**
- You have annotated cell type labels and want label-conditioned generation.
- You need to generate specific proportions of each cell type (`custom_distributions`).
- You want better biological separability between classes than unsupervised scVI.

```python
synthetic = gen.generate(
    data=adata,              # AnnData or DataFrame
    n_samples=2000,
    method="scanvi",
    target_col="cell_type",  # Required: column with cell type labels
    epochs=200,
    n_latent=30,
    custom_distributions={"cell_type": {"T cell": 0.5, "B cell": 0.3, "NK cell": 0.2}}
)
```

| `custom_distributions` | `None` | Per-class proportions for generation |

#### Single-Cell Workflow Utilities

For users working with single-cell transcriptomics, `RealGenerator` provides a utility to convert the generated synthetic DataFrames back into `AnnData` objects, which are the standard format for analysis libraries like `scanpy` or `squidpy`.

**`to_anndata(df, target_col=None, obs_cols=None)`** (Static Method)

Converts a synthetic DataFrame (the output of `generate()`) into an `AnnData` object.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | **required** | The synthetic DataFrame generated by `calm_data_generator`. |
| `target_col` | `str` | `None` | The column to be used as `cell_type` in `adata.obs`. |
| `obs_cols` | `List[str]` | `None` | List of additional columns to move from the feature matrix (`X`) to the metadata (`obs`). |

**Example:**
```python
from calm_data_generator.generators.tabular import RealGenerator

# 1. Generate synthetic data (e.g., using scVI)
synthetic_df = gen.generate(
    data=real_adata,
    n_samples=2000,
    method="scvi",
    target_col="cell_type"
)

# 2. Convert back to AnnData for scanpy analysis
synthetic_adata = RealGenerator.to_anndata(
    synthetic_df,
    target_col="cell_type"
)

# 3. Standard scanpy analysis
import scanpy as sc
sc.pp.pca(synthetic_adata)
sc.pl.pca(synthetic_adata, color="cell_type")
```

**Validating single-cell quality with scGFT:**

After generating synthetic single-cell data, validate its fidelity using [scgft-evaluator](https://github.com/nasim23ea/scgft-evaluator):

```python
from calm_data_generator.reports.QualityReporter import QualityReporter
from calm_data_generator.generators.configs import ReportConfig

reporter = QualityReporter(verbose=True)
reporter.generate_comprehensive_report(
    real_df=real_df,
    synthetic_df=synthetic_df,
    generator_name="scVI_SingleCell",
    report_config=ReportConfig(
        output_dir="./sc_report",
        use_scgft=True,
        target_column="cell_type",
    ),
)
# Generates scgft_report.html with ARI, MMD, Jaccard, Kendall Tau metrics
```

> See [REPORTS_REFERENCE.md](REPORTS_REFERENCE.md#single-cell-evaluation-scgft) for full details.

---

---

## Advanced Features

### Custom Distributions

You can force a specific marginal distribution for any categorical column using `custom_distributions`. The exact behavior depends on the chosen synthesis method:

- **CTGAN / Deep Learning (Conditional):** Performs *real conditional generation*. It generates exact proportional counts per class directly from the synthesized model, without relying on post-generation resampling.
- **SMOTE / ADASYN:** Translates the requested distribution into absolute counts and applies them natively as the `sampling_strategy` in `imbalanced-learn`.
- **GMM, TVAE, Copula, BN, scVI, DDPM:** Uses the internal `_apply_postprocess_distribution` method. After generating bulk synthetic data, it intelligently resamples rows to fulfill the requested class proportions while preserving column correlations.
- **Time Series (TimeGAN, TimeVAE, fflows):** `custom_distributions` and `balance` are not applicable to temporal sequences. A warning will be emitted and the argument will be ignored.

```python
synthetic = gen.generate(
    data=df,
    n_samples=5000,
    method="ctgan",
    target_col="Churn",
    custom_distributions={
        "Churn": {0: 0.5, 1: 0.5},
        "Region": {"North": 0.4, "South": 0.6},
    }
)
```

### Date Injection

You can inject a datetime column into the generated data using `DateConfig`.

```python
from calm_data_generator.generators.configs import DateConfig

synthetic = gen.generate(
    data=df,
    n_samples=1000,
    method="cart",
    date_config=DateConfig(
        date_col="timestamp",
        start_date="2024-06-01",
        step={"hours": 1},  # Increment
        frequency=1         # Every 1 row
    )
)
```

### Post-Generation Drift

```python
synthetic = gen.generate(
    data=df,
    n_samples=1000,
    method="tvae",
    drift_injection_config=[
        {
            "method": "inject_feature_drift",
            "feature_cols": ["price"],
            "drift_type": "shift",
            "drift_magnitude": 0.3,
            "start_index": 500,
        }
    ],
)
```

---

## Automatic Reports

When `auto_report=True`, the following are generated:

- `quality_report.html`: Data profiling report
- `comparison_report.html`: Real vs. Synthetic comparison
- `quality_scores.json`: Detailed quality metrics
- `distribution_plots.png`: Distribution visualizations
- `correlation_heatmap.png`: Correlation maps

---

## Saving and Loading Models

`RealGenerator` allows you to save trained generator models and load them later for inference without retraining. This is useful for production pipelines where training is expensive.

### Saving a Model

After generating data (which trains the underlying model), you can save the generator:

```python
# 1. Train and Generate
gen.generate(data, n_samples=1000, method="ctgan", batch_size=500)

# 2. Save the trained generator
gen.save("models/my_ctgan_model.pkl")
```
> **Note:** The saved file is a zip archive containing the `RealGenerator` configuration and the underlying model (e.g., Synthcity plugin state).

### Loading a Model

You can load a saved model using the `load()` class method. Once loaded, you can generate more samples without providing the original training data.

```python
from calm_data_generator.generators.tabular import RealGenerator

# 1. Load the generator
loaded_gen = RealGenerator.load("models/my_ctgan_model.pkl")

# 2. Generate new samples (No 'data' argument needed!)
new_samples = loaded_gen.generate(n_samples=500)
```

> **Warning:** When generating from a loaded model, you **must not** pass `data` to `generate()`, but you **must** pass `n_samples`.

> **Note:** scVI and scANVI models use a directory-based save format internally. These are packaged inside the zip file transparently — save/load works identically to all other methods.

---

## Privacy Methods

### `privatize()` — Apply Differential Privacy to Existing Data

Applies differential privacy mechanisms directly to a real DataFrame, returning a privatized version. Unlike `dpgan`/`pategan` which train a generative model, `privatize` is a direct transformation of the input data.

- **Numeric columns:** Laplace or Gaussian noise is added.
- **Categorical columns:** Randomized Response is applied.

```python
# Laplace mechanism (default)
private_df = gen.privatize(df, epsilon=1.0)

# Gaussian mechanism
private_df = gen.privatize(df, epsilon=1.0, mechanism="gaussian", delta=1e-5)

# Custom randomized response probability
private_df = gen.privatize(df, epsilon=0.5, categorical_p=0.8)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | - | Input DataFrame to privatize |
| `epsilon` | float | `1.0` | Privacy budget ε. Lower = more private. |
| `delta` | float | `None` | Required for Gaussian mechanism. |
| `numeric_sensitivity` | float | `1.0` | Global sensitivity for numeric columns. |
| `mechanism` | str | `'laplace'` | `'laplace'` or `'gaussian'` for numeric columns. |
| `categorical_p` | float | `None` | Probability of keeping true category. If None, derived from ε. |

---

## Custom Models: `generate_custom()` and `CustomPluginAdapter`

### `generate_custom()` — Use Any External Model

Wraps any external model (sklearn, synthcity, copulae, etc.) for use with `RealGenerator`. The adapter auto-detects the model's interface (`fit`/`train`, `generate`/`sample`/`random`) using duck typing. You can override with explicit lambdas for full control.

```python
from sklearn.neighbors import KernelDensity

kde_model = KernelDensity(kernel='gaussian', bandwidth=0.5)

synthetic_df = gen.generate_custom(
    data=df,
    model=kde_model,
    n_samples=500,
    # Optional: explicit fit/generate functions if auto-detection fails
    fit_fn=lambda m, data: m.fit(data.values),
    generate_fn=lambda m, n: pd.DataFrame(m.sample(n), columns=df.columns),
    method_name="my_kde",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | - | Training data |
| `model` | any | - | Any model object with fit/train and generate/sample/random |
| `n_samples` | int | - | Number of synthetic samples to generate |
| `fit_fn` | callable | `None` | `lambda model, data: ...` — override auto-detected fit method |
| `generate_fn` | callable | `None` | `lambda model, n: ...` — override auto-detected generate method |
| `postprocess_fn` | callable | `None` | Post-process generated DataFrame |
| `method_name` | str | `"custom"` | Label for logging and metadata |
| `columns` | List[str] | `None` | Column names (defaults to `data.columns`) |

### `CustomPluginAdapter` — Standalone Adapter

You can also use `CustomPluginAdapter` directly for more advanced scenarios:

```python
from calm_data_generator.generators.tabular.CustomPluginAdapter import CustomPluginAdapter
from synthcity.plugins import Plugins

# Use any Synthcity plugin
plugin = Plugins().get("adsgan")

adapter = CustomPluginAdapter(
    model=plugin,
    method_name="adsgan",
    columns=df.columns.tolist()
)
adapter.fit(df)
synthetic_df = adapter.generate(n_samples=500)
```

**Auto-detected interfaces (in priority order):**

| Interface | Detected when model has | Return format |
|-----------|------------------------|---------------|
| Synthcity | `.generate()` | `.dataframe()` called automatically |
| sklearn-like | `.sample()` | Wrapped in `pd.DataFrame` |
| Copulae-like | `.random()` | Wrapped in `pd.DataFrame` |

---

## Best Practices

1. **Severe imbalance:** Use `smote` or `adasyn` with `target_col`.

---

## Comprehensive Use Cases

### Case 1: Fraud Detection (Imbalanced Data)

**Scenario:** You have a dataset of credit card transactions where only 0.1% are fraudulent. You want to train a model but need more fraud cases.

**Solution:** Use `smote` or `ctgan` with forced distribution.

```python
from calm_data_generator.generators.tabular import RealGenerator

gen = RealGenerator()

# Option A: SMOTE (Oversampling)
synthetic_fraud = gen.generate(
    data=df,
    n_samples=5000,
    method="smote",
    target_col="is_fraud",
    model_params={"smote_neighbors": 5}
)

# Option B: CTGAN with Controlled Sampling
synthetic_balanced = gen.generate(
    data=df,
    n_samples=5000,
    method="ctgan",
    target_col="is_fraud",
    custom_distributions={"is_fraud": {0: 0.5, 1: 0.5}}, # Force 50/50 split
    model_params={"epochs": 500}
)
```




### Case 2: High-Performance Upsampling for Large Datasets

**Scenario:** You have a 1M row dataset and need 5M rows for stress testing a database. Deep learning methods are too slow.

**Solution:** Use `lgbm` (LightGBM) or `rf` (Random Forest) which are faster and scalable.

```python
synthetic_large = gen.generate(
    data=large_df,
    n_samples=5_000_000,
    method="lgbm",
    model_params={
        "n_estimators": 100,
        "learning_rate": 0.1
    },
    minimal_report=True  # Skip heavy reporting to save time
)
```

---

## Common Usage Scenarios (Quick Guide)

### 1. Time Series (Sequences)
For time series data, use standard tabular methods (CTGAN, TVAE, etc.) on properly structured temporal data.
*   **Future Forecasting:** Use `StreamGenerator` for infinite flows or manual date injection.

### 2. Classification and Regression (Supervised)
If you have a `target` column (e.g., price, churn) and the relationship $X \rightarrow Y$ is critical:
*   Use `method="lgbm"` (LightGBM) or `method="rf"` (Random Forest).
*   Always specify `target_col="column_name"`.
    ```python
    # The generator automatically detects if it's Regression or Classification
    gen.generate(data, target_col="price", method="lgbm")
    ```

### 3. Clustering (Unsupervised)
If there's no clear target and you want to preserve natural data clusters:
*   Use `method="gmm"` (Gaussian Mixture Models) or `method="tvae"` (Variational Autoencoder).
    ```python
    gen.generate(data, method="tvae")
    ```

### 4. Multi-Label (Multiple Labels)
If a cell contains multiple values (e.g., `["A", "B", "C"]`) or string format `"A,B,C"`:
*   **Limitation:** Standard models don't handle lists within cells well.
*   **Solution:** Transform the column to **One-Hot Encoding** (multiple binary columns `is_A`, `is_B`) before passing to the generator. Tree-based models (`lgbm`, `cart`) will learn correlations between labels (e.g., if `is_A=1` often implies `is_B=1`).

### 5. Block Data (Partitioned Data)
If your data is logically fragmented (e.g., by Stores, Countries, Patients) and you want independent models for each:
*   Use **`RealBlockGenerator`** instead of `RealGenerator`.
    ```python
    block_gen = RealBlockGenerator()
    block_gen.generate(data, block_column="StoreID", method="cart")
    ```
    *This trains a different model for each StoreID.*

### 6. Handling Imbalanced Data
If your target column has minority classes that you want to amplify:
*   **Automatic Balancing:** Use `balance=True`. The generator applies internal oversampling (SMOTE/RandomOverSampler) so the model learns equally from all classes.
    ```python
    gen.generate(data, target_col="fraud", balance=True, method="cart")
    ```
*   **Custom Distribution:** If you want a specific ratio (e.g., 70% Class A, 30% Class B):
    ```python
    gen.generate(data, target_col="status", custom_distributions={"status": {"Low": 0.7, "High": 0.3}})
    ```
    *Note: `balance=True` is a shortcut for `custom_distributions={"col": "balanced"}`. For extreme imbalances, Deep Learning methods like `method="ctgan"` usually provide better stability than tree-based methods.*

---

### `ddpm` - Synthcity TabDDPM (Advanced Tabular Diffusion)

**Type:** Deep Learning (Diffusion Model)
**Best For:** High-quality tabular synthesis, production environments, large datasets
**Requirements:** `synthcity` (included in base installation)

#### Description

TabDDPM (Tabular Denoising Diffusion Probabilistic Model) is Synthcity's advanced implementation of diffusion models for tabular data. It offers multiple architectures, advanced schedulers, and superior quality compared to the custom `diffusion` method.

#### When to Use

✅ **Use `ddpm` when:**
- You need **maximum quality** synthetic data
- Working with **large datasets** (>100k rows)
- In **production environments** requiring robust, maintained code
- You need **advanced architectures** (ResNet, TabNet)
- You want **cosine scheduling** for better diffusion
- You have **time for longer training** (1000 epochs default)

❌ **Don't use `ddpm` when:**
- You need **quick prototyping** (use `diffusion` instead)
- Working with **very small datasets** (<1k rows)
- You have **limited computational resources**
- You need **custom modifications** to the algorithm

#### Parameters

```python
synth = gen.generate(
    data,
    method='ddpm',
    n_samples=1000,

    # Training parameters
    n_iter=1000,                    # Training epochs (default: 1000)
    lr=0.002,                       # Learning rate (default: 0.002)
    batch_size=1024,                # Batch size (default: 1024)

    # Diffusion parameters
    num_timesteps=1000,             # Diffusion timesteps (default: 1000)
    scheduler='cosine',             # 'cosine' or 'linear' (default: 'cosine')
    gaussian_loss_type='mse',       # 'mse' or 'kl' (default: 'mse')

    # Model architecture
    model_type='mlp',               # 'mlp', 'resnet', or 'tabnet' (default: 'mlp')
    model_params={                  # Architecture-specific parameters
        'n_layers_hidden': 3,
        'n_units_hidden': 256,
        'dropout': 0.0
    },

    # Task type
    is_classification=False,        # True for classification tasks
)
```

#### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_iter` | int | 1000 | Number of training epochs |
| `lr` | float | 0.002 | Learning rate for optimizer |
| `batch_size` | int | 1024 | Training batch size |
| `num_timesteps` | int | 1000 | Number of diffusion timesteps |
| `scheduler` | str | `'cosine'` | Beta scheduler: `'cosine'` (recommended) or `'linear'` |
| `gaussian_loss_type` | str | `'mse'` | Loss function: `'mse'` or `'kl'` |
| `model_type` | str | `'mlp'` | Architecture: `'mlp'`, `'resnet'`, or `'tabnet'` |
| `model_params` | dict | See above | Architecture-specific parameters |
| `is_classification` | bool | False | Set to True for classification tasks |

#### Model Types

**MLP (Multi-Layer Perceptron)**
- Best for: General tabular data
- Speed: Fast
- Parameters: `n_layers_hidden`, `n_units_hidden`, `dropout`

**ResNet (Residual Network)**
- Best for: Complex feature relationships
- Speed: Medium
- Parameters: `n_layers_hidden`, `n_units_hidden`, `dropout`

**TabNet**
- Best for: Tabular data with feature importance
- Speed: Slower
- Parameters: Specific to TabNet architecture

#### Comparison: `diffusion` vs `ddpm`

| Aspect | `diffusion` (custom) | `ddpm` (Synthcity) |
|--------|---------------------|-------------------|
| **Speed** | ⚡ Fast (100 epochs) | 🐢 Slower (1000 epochs) |
| **Quality** | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Excellent |
| **Architectures** | MLP only | MLP/ResNet/TabNet |
| **Scheduler** | Linear | Cosine/Linear |
| **Batch Size** | 64 | 1024 |
| **Use Case** | Quick prototyping | Production quality |
| **Customization** | Easy to modify | Black box |
| **Maintenance** | Your responsibility | Synthcity team |

#### Usage Examples

**Basic Usage:**
```python
from calm_data_generator import RealGenerator
import pandas as pd

gen = RealGenerator()
synth = gen.generate(
    data,
    method='ddpm',
    n_samples=1000,
    n_iter=500  # Reduce for faster training
)
```

**Classification Task:**
```python
synth = gen.generate(
    data,
    method='ddpm',
    n_samples=1000,
    is_classification=True,
    target_col='label'
)
```

**Advanced Architecture:**
```python
synth = gen.generate(
    data,
    method='ddpm',
    n_samples=1000,
    model_type='resnet',
    model_params={
        'n_layers_hidden': 5,
        'n_units_hidden': 512,
        'dropout': 0.1
    },
    scheduler='cosine',
    n_iter=2000
)
```

---

### `timegan` - TimeGAN (Time Series GAN)

**Type:** Deep Learning (GAN for Time Series)
**Best For:** Complex temporal patterns, multi-entity time series
**Requirements:** `synthcity` (included in base installation)

#### Description

TimeGAN (Time-series Generative Adversarial Network) is designed specifically for sequential/temporal data. It learns both temporal dynamics and feature distributions, making it ideal for time series with complex patterns.

#### When to Use

✅ **Use `timegan` when:**
- You have **time series data** with temporal dependencies
- Working with **multi-entity sequences** (e.g., multiple users/sensors)
- You need to preserve **temporal dynamics**
- You have **complex temporal patterns** to learn
- You need **high-quality** time series synthesis

❌ **Don't use `timegan` when:**
- You have **simple tabular data** (use `ctgan` or `ddpm` instead)
- Working with **very short sequences** (<10 timesteps)
- You need **fast generation** (use `timevae` instead)
- You have **limited computational resources**

#### Data Requirements

TimeGAN expects data in a specific temporal format:
- **Temporal ordering**: Data must be sorted by time
- **Entity grouping**: If multi-entity, group by entity ID
- **Consistent timesteps**: Regular time intervals preferred

#### Parameters

```python
synth = gen.generate(
    data,
    method='timegan',
    n_samples=100,  # Number of sequences to generate

    # Training parameters
    n_iter=1000,                    # Training epochs (default: 1000)
    n_units_hidden=100,             # Hidden units in RNN (default: 100)
    batch_size=128,                 # Batch size (default: 128)
    lr=0.001,                       # Learning rate (default: 0.001)
)
```

#### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_iter` | int | 1000 | Number of training epochs |
| `n_units_hidden` | int | 100 | Number of hidden units in RNN layers |
| `batch_size` | int | 128 | Training batch size |
| `lr` | float | 0.001 | Learning rate for optimizer |

#### Usage Examples

**Basic Time Series:**
```python
from calm_data_generator import RealGenerator
import pandas as pd

# Data must have temporal structure
# Example: sensor readings over time
gen = RealGenerator()
synth = gen.generate(
    time_series_data,
    method='timegan',
    n_samples=100,  # Generate 100 sequences
    n_iter=1000,
    n_units_hidden=100
)
```

**Multi-Entity Time Series:**
```python
# Data with multiple entities (e.g., users, sensors)
# Ensure data is sorted by entity_id and timestamp
synth = gen.generate(
    multi_entity_data,
    method='timegan',
    n_samples=50,  # Generate 50 entity sequences
    n_iter=2000,
    n_units_hidden=150,
    batch_size=64
)
```

---

### `timevae` - TimeVAE (Time Series VAE)

**Type:** Deep Learning (VAE for Time Series)
**Best For:** Regular time series, faster training than TimeGAN
**Requirements:** `synthcity` (included in base installation)

#### Description

TimeVAE is a variational autoencoder designed for temporal data. It's generally faster than TimeGAN and works well for regular time series with consistent patterns.

#### When to Use

✅ **Use `timevae` when:**
- You have **regular time series** data
- You need **faster training** than TimeGAN
- Working with **consistent temporal patterns**
- You want **good quality** with **less computation**
- You have **moderate-length sequences**

❌ **Don't use `timevae` when:**
- You have **highly irregular** time series
- You need **maximum quality** (use `timegan` instead)
- Working with **very complex** temporal dynamics
- You have **simple tabular data** (use `ctgan` or `ddpm`)

#### Data Requirements

Similar to TimeGAN:
- **Temporal ordering**: Data sorted by time
- **Regular intervals**: Works best with consistent timesteps
- **Entity grouping**: If multi-entity, group by entity ID

#### Parameters

```python
synth = gen.generate(
    data,
    method='timevae',
    n_samples=100,  # Number of sequences to generate

    # Training parameters
    n_iter=1000,                    # Training epochs (default: 1000)
    decoder_n_layers_hidden=2,      # Decoder layers (default: 2)
    decoder_n_units_hidden=100,     # Decoder units (default: 100)
    batch_size=128,                 # Batch size (default: 128)
    lr=0.001,                       # Learning rate (default: 0.001)
)
```

#### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_iter` | int | 1000 | Number of training epochs |
| `decoder_n_layers_hidden` | int | 2 | Number of hidden layers in decoder |
| `decoder_n_units_hidden` | int | 100 | Number of hidden units in decoder |
| `batch_size` | int | 128 | Training batch size |
| `lr` | float | 0.001 | Learning rate for optimizer |

#### Comparison: `timegan` vs `timevae`

| Aspect | `timegan` | `timevae` |
|--------|-----------|-----------|
| **Speed** | 🐢 Slower | ⚡ Faster |
| **Quality** | ⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
| **Complexity** | Handles complex patterns | Best for regular patterns |
| **Training Time** | Longer | Shorter |
| **Use Case** | Complex temporal dynamics | Regular time series |

#### Usage Examples

**Basic Time Series:**
```python
from calm_data_generator import RealGenerator
import pandas as pd

gen = RealGenerator()
synth = gen.generate(
    time_series_data,
    method='timevae',
    n_samples=100,
    n_iter=500,  # Faster than TimeGAN
    decoder_n_units_hidden=100
)
```

**Faster Training:**
```python
# Reduce parameters for quick prototyping
synth = gen.generate(
    time_series_data,
    method='timevae',
    n_samples=50,
    n_iter=300,
    decoder_n_layers_hidden=1,
    decoder_n_units_hidden=50,
    batch_size=64
)
```

---

### `fflows` - FourierFlows (Frequency-Domain Normalizing Flows for Time Series)

**Type:** Deep Learning (Normalizing Flows for Time Series)
**Best For:** Periodic/quasi-periodic time series, stable alternative to TimeGAN
**Requirements:** `synthcity`

#### Description

FourierFlows (`fflows`) applies normalizing flows in the frequency domain to generate temporal sequences. It is generally more stable than TimeGAN and excels at series with periodic patterns (sinusoidal, seasonal).

#### Parameters

```python
synth = gen.generate(
    data,
    method='fflows',
    n_samples=100,
    sequence_key='seq_id',   # Column identifying each sequence
    time_key='timestamp',    # Column with timestamps
    n_iter=1000,             # Training epochs (default: 1000)
    batch_size=128,          # Batch size (default: 128)
    lr=0.001,                # Learning rate (default: 0.001)
)
```

#### Comparison: `timegan` vs `timevae` vs `fflows`

| Aspect | `timegan` | `timevae` | `fflows` |
|--------|-----------|-----------|----------|
| **Speed** | 🐢 Slow | ⚡ Fast | ⚡ Fast |
| **Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Stability** | Low | Medium | High |
| **Best for** | Complex patterns | Regular series | Periodic series |

---

### `bn` - Bayesian Network

**Type:** Probabilistic Graphical Model
**Best For:** Clinical/structured tabular data with causal dependencies between variables
**Requirements:** `synthcity`

#### Description

A Bayesian Network (BN) models the conditional dependencies between variables using a directed acyclic graph. Structure learning discovers which variables causally influence others. Especially useful for healthcare and clinical data where domain causal knowledge matters.

#### Parameters

```python
synth = gen.generate(
    data,
    method='bn',
    n_samples=1000,
    target_col='diagnosis',  # Optional target column
    # Additional parameters passed to Synthcity BN plugin
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_iter` | 1000 | Training iterations |
| `struct_learning_n_iter` | 1000 | Structure learning iterations |
| `struct_learning_search_method` | `'tree_search'` | Structure learning method |

#### When to Use

✅ **Use `bn` when:**
- Data has **causal relationships** between variables (e.g., diagnosis ← symptoms ← lab values)
- Working with **clinical or epidemiological** datasets
- You want an **interpretable model** (the network is inspectable)
- Dataset is **small to medium** size

❌ **Don't use `bn` when:**
- Data is **high-dimensional** (100+ features) — structure learning becomes slow
- You need **time series** data (use `timegan`/`timevae`/`fflows`)

---

## Method Selection Guide

### For Tabular Data

| Scenario | Recommended Method | Alternative |
|----------|-------------------|-------------|
| **Quick prototyping** | `diffusion` | `cart`, `rf` |
| **Production quality** | `ddpm` | `ctgan` |
| **Large datasets (>100k)** | `ddpm`, `lgbm` | `ctgan` |
| **Small datasets (<1k)** | `cart`, `rf` | `diffusion` |
| **Class imbalance** | `smote`, `adasyn` | `ctgan` |
| **Preserve correlations** | `ctgan`, `ddpm` | `copula` |
| **Fast generation** | `cart`, `diffusion` | `rf` |
| **Maximum quality** | `ddpm` (ResNet) | `ctgan` |

### For Time Series Data

| Scenario | Recommended Method | Alternative |
|----------|-------------------|-------------|
| **Complex temporal patterns** | `timegan` | `fflows` |
| **Regular time series** | `timevae` | `timegan` |
| **Periodic/seasonal series** | `fflows` | `timevae` |
| **Fast training** | `timevae` | `fflows` |
| **Multi-entity sequences** | `timegan` | `fflows` |
| **Maximum quality** | `timegan` | `fflows` |

### For Special Cases

| Data Type | Recommended Method |
|-----------|-------------------|
| **Single-cell RNA-seq** | `scvi` |
| **Clinical/Medical tabular** | `bn` or `ClinicalDataGenerator` |
| **Clinical/Medical** | Use `ClinicalDataGenerator` |
| **Streaming data** | Use `StreamGenerator` |
| **Block/Batch data** | Use `RealBlockGenerator` |

---

## v1.2.0 Features

### Latent Space Differentiation (`differentiation_factor`)

Available for `tvae` and `scvi`. Controls how far apart the class centroids are pushed in the latent space during synthesis.

```python
synth = gen.generate(
    data=df,
    n_samples=500,
    method="tvae",
    target_col="group",
    differentiation_factor=1.5  # Push classes further apart
)
```

| Value | Effect |
|-------|--------|
| `0.0` | No shift (default behaviour) |
| `0.5–1.0` | Subtle separation |
| `1.5–2.0` | Moderate/strong class separation |
| `> 2.0` | Risk of out-of-distribution samples – use with care |

> **TVAE:** Shift is applied directly in the neural network latent space (mu vectors).
> **scVI:** Shift applied in the `z` latent space before decoding.

---

### Training Visibility (`verbose_training`)

Pass `verbose_training=True` at construction time to let Synthcity print the epoch-by-epoch loss:

```python
gen = RealGenerator(verbose_training=True)
gen.generate(data=df, n_samples=500, method="tvae", epochs=200)
# → 2024-03-06 14:01:12 | INFO | tvae | epoch 1/200 | loss: 1.2341
# → 2024-03-06 14:01:15 | INFO | tvae | epoch 2/200 | loss: 1.1872
# → ...
```

For **scVI**, the PyTorch Lightning progress bar is always shown. After training, the final loss is also logged via the Python logger regardless of `verbose_training`.

---

### Introspection Accessor Methods

After calling `generate()`, these methods expose internal model state:

#### `get_encoder()`

Returns the encoder network of the last trained model.

```python
synth = gen.generate(df, 500, method="tvae", target_col="label")
encoder = gen.get_encoder()
# Returns: nn.Module (inner VAE encoder) for tvae
# Returns: scvi Module z_encoder for scvi
# Returns: None for methods without explicit encoders (ctgan, cart...)
```

#### `get_decoder()`

Returns the decoder network.

```python
decoder = gen.get_decoder()
# Returns: nn.Module for tvae and scvi
```

#### `get_latest_embeddings()`

Returns the latent space embeddings computed during the last synthesis that applied `differentiation_factor`.

```python
embeddings = gen.get_latest_embeddings()  # np.ndarray or None
if embeddings is not None:
    print(f"Embedding shape: {embeddings.shape}")  # (n_samples, n_latent)
    # E.g. for UMAP visualization:
    import umap
    reducer = umap.UMAP()
    projection = reducer.fit_transform(embeddings)
```

> Returns `None` if no differentiation was applied or if the model uses feature-space fallback.

#### `get_training_history()`

Returns the training history dict (**scVI/scANVI only**).

```python
synth = gen.generate(df, 500, method="scvi", epochs=100)
history = gen.get_training_history()

if history:
    import matplotlib.pyplot as plt
    elbo = history["elbo_train"]
    plt.plot(elbo.values)
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.title("scVI Training Evolution")
    plt.show()
```

| Key | Description |
|-----|-------------|
| `train_loss_epoch` | Total training loss per epoch |
| `elbo_train` | Evidence Lower Bound |
| `reconstruction_loss_train` | Reconstruction (expression) loss |
| `kl_local_train` | KL divergence – local per-cell term |
| `kl_global_train` | KL divergence – global term |

> Returns `None` for Synthcity-based models (TVAE, CTGAN). Use `verbose_training=True` for those.

#### `get_synthesizer_model()`

Returns the raw underlying model object (Synthcity plugin, scVI model, sklearn model, etc.).

```python
raw = gen.get_synthesizer_model()
# tvae/ctgan: Synthcity plugin object
# scvi:       scvi.model.SCVI instance
# cart/rf:    FCSModel instance
```
