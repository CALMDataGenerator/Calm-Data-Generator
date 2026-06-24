# CALM-Data-Generator


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/calm-data-generator.svg)](https://pypi.org/project/calm-data-generator/)
[![Downloads](https://img.shields.io/pypi/dm/calm-data-generator)](https://pypi.org/project/calm-data-generator/)

> **Now available on PyPI!** Install with: `pip install calm-data-generator`

> **[Versión en Español](README_ES.md)**

**CALM-Data-Generator** is a comprehensive Python library for synthetic data generation with advanced features for:
- **Clinical/Medical Data** - Generate realistic patient demographics, genes, proteins
- **Tabular Data Synthesis** - CTGAN, TVAE, CART, and more
- **Time Series** - TimeGAN, TimeVAE, FourierFlows
- **Single-Cell** - scVI, GEARS (Perturbation Prediction)
- **Latent Differentiation** - Force class separability in TVAE & scVI
- **Flexible Clipping** - Strict or permissive data range enforcement
- **Drift Injection** - Test ML model robustness with controlled drift
- **Privacy Assessment** - DCR metrics for re-identification risk
- **Scenario Evolution** - Feature evolution and target construction

## Scope & Capabilities

**Calm-Data-Generator** is optimized for **structured tabular data**. It is designed to handle:
- **Classification** (Binary & Multi-class)
- **Regression** (Continuous variables)
- **Multi-label** problems
- **Clustering** (Preserving natural groupings)
- **Time Series** (Temporal correlations and patterns)
- **Single-Cell / Genomics** (scRNA-seq expression data)

> [!IMPORTANT]
> This library is **NOT** designed for unstructured data such as **Images**, **Videos**, or **Audio**. It does not include Computer Vision or Signal Processing models.

---

## What Makes This Library Unique?

**CALM-Data-Generator** is not just another synthetic data tool—it's a **unified ecosystem** that brings together the best open-source libraries under a single, consistent API:

### Unified Multi-Library Integration
Instead of learning and managing multiple complex libraries separately, CALM-Data-Generator provides:
- **One API** for 15+ synthesis methods from different sources (Synthcity, scvi-tools, GEARS, imbalanced-learn, etc.)
- **Seamless interoperability** between tabular, time-series, streaming, and genomic data generators
- **Consistent configuration** across all methods with automatic parameter validation
- **Integrated reporting** with YData Profiling for all generation methods
- **Extensible generator hierarchy**: `BaseGenerator` -> `ComplexGenerator` -> domain generators. New domains (Finance, IoT, Insurance) inherit three reusable mathematical engines (Gaussian Copula unconditional, Gaussian Copula conditional, stochastic effects) without duplicating code.

### Advanced Drift Injection (Industry-Leading)
The **DriftInjector** module is one of the most comprehensive drift simulation tools available:
- **14+ drift types**: Feature drift (gradual, abrupt, incremental, recurrent), label drift, concept drift, correlation drift, outlier injection, and more
- **Correlation-aware drift**: Propagate realistic drift across correlated features (e.g., increase income → increase spending)
- **Multi-modal drift profiles**: Sigmoid, linear, cosine transitions for gradual drift
- **Conditional drift**: Apply drift only to specific data subsets based on business rules
- **Functional drift** *(Pilar 5)*: Drift magnitude varies per row as a function of another column (e.g., sensor noise that scales exponentially with temperature)
- **Causal cascades** *(Pilar 5)*: Define a causal DAG and propagate perturbations through non-linear transfer functions (`CausalEngine`)
- **Integrated with generators**: Inject drift directly during synthesis or post-hoc on existing data
- **Perfect for MLOps**: Test data drift monitoring, concept drift detection, and model robustness before production

### Scenario Evolution
The **ScenarioInjector** creates deterministic or stochastic temporal patterns:
- **7 evolution types**: `linear`, `exponential_growth`, `decay`, `seasonal`, `step`, `noise`, `random_walk`
- **driven_by** *(Pilar 5)*: Feature delta per row = f(current value of another column) — couples variables without a full DAG
- **Target construction**: Build synthetic ground-truth target variables from feature formulas or callables
- **Future projection**: Extend historical datasets into future time periods

> **In summary**: While other tools focus on a single approach (e.g., just GANs, just statistical methods), CALM-Data-Generator **unifies the ecosystem** and adds **production-grade drift simulation** that most libraries don't offer.

---

## Core Technologies

This library leverages and unifies best-in-class open-source tools to provide a seamless data generation experience:

### Synthesis Engines

| Library | Methods / Features | Docs |
|---------|-------------------|------|
| [Synthcity](https://github.com/vanderschaarlab/synthcity) | CTGAN, TVAE, DDPM, TimeGAN, TimeVAE, FFlows, GREAT, RTVAE, BN, DPGAN, PATEGAN, Conditional Drift | [synthcity docs](https://github.com/vanderschaarlab/synthcity) |
| [scikit-learn](https://scikit-learn.org/) | CART, RF, KDE, GMM, metrics, preprocessing | [sklearn docs](https://scikit-learn.org/stable/user_guide.html) |
| [LightGBM](https://lightgbm.readthedocs.io/) | LGBM synthesis (FCS-style) | [lgbm docs](https://lightgbm.readthedocs.io/en/stable/) |
| [XGBoost](https://xgboost.readthedocs.io/) | XGBoost synthesis (FCS-style) | [xgb docs](https://xgboost.readthedocs.io/en/stable/) |
| [Copulae](https://github.com/DanielBok/copulae) | Copula, Windowed Copula | [copulae docs](https://copulae.readthedocs.io/) |
| [imbalanced-learn](https://imbalanced-learn.org/) | SMOTE, ADASYN | [imblearn docs](https://imbalanced-learn.org/stable/) |
| [hmmlearn](https://hmmlearn.readthedocs.io/) | HMM synthesis | [hmmlearn docs](https://hmmlearn.readthedocs.io/) |
| [PyTorch](https://pytorch.org/) | Diffusion (custom), backend for all deep learning methods | [torch docs](https://pytorch.org/docs/) |

### Single-Cell / Omics

| Library | Methods / Features | Docs |
|---------|-------------------|------|
| [scvi-tools](https://docs.scvi-tools.org/) | scVI, scANVI | [scvi docs](https://docs.scvi-tools.org/) |
| [GEARS](https://github.com/snap-stanford/GEARS) | Perturbation prediction (GEARS method) | [gears docs](https://github.com/snap-stanford/GEARS) |
| [AnnData](https://anndata.readthedocs.io/) | Single-cell data structures for scVI/scANVI/GEARS | [anndata docs](https://anndata.readthedocs.io/) |

### Quality & Reporting

| Library | Methods / Features | Docs |
|---------|-------------------|------|
| [SDMetrics](https://docs.sdv.dev/sdmetrics/) | QualityReporter — statistical quality scores | [sdmetrics docs](https://docs.sdv.dev/sdmetrics/) |
| [YData Profiling](https://ydata-profiling.ydata.ai/) | Automated data profiling reports | [ydata docs](https://ydata-profiling.ydata.ai/docs/) |
| [Plotly](https://plotly.com/python/) | Interactive visualizations and dashboards | [plotly docs](https://plotly.com/python/) |

### Mathematical Foundations

| Library | Methods / Features | Docs |
|---------|-------------------|------|
| [SciPy](https://docs.scipy.org/) | Gaussian Copula (ComplexGenerator), PSD matrix repair, statistics | [scipy docs](https://docs.scipy.org/) |
| [NumPy](https://numpy.org/) | Numerical arrays, all generators | [numpy docs](https://numpy.org/doc/) |
| [Pandas](https://pandas.pydata.org/) | DataFrames, all generators | [pandas docs](https://pandas.pydata.org/docs/) |

### Streaming

| Library | Methods / Features | Docs |
|---------|-------------------|------|
| [River](https://riverml.xyz/) | StreamGenerator — Agrawal, SEA, Hyperplane, Sine, etc. (`[stream]` extra) | [river docs](https://riverml.xyz/latest/) |

## Presets (Templates)

**Calm-Data-Generator** includes **19 ready-to-use Presets** covering the most common synthetic data scenarios. Each preset encapsulates a generator configuration (method, hyperparameters, reporting) and exposes a single `.generate()` call.

> [!TIP]
> **Presets are starting points**: instantiate a preset, call `.generate()`, and override any parameter you need via `__init__` arguments (`random_state`, `verbose`, `fast_dev_run`).

### Available Presets

**Speed & Prototyping**

| Preset | Method | Use Case |
|--------|--------|----------|
| `FastPreset` | LightGBM | Fastest generation, configurable via kwargs |
| `FastPrototypePreset` | LightGBM | CI/CD pipelines, integration tests (fixed 10 iterations) |

**Quality & Fidelity**

| Preset | Method | Use Case |
|--------|--------|----------|
| `HighFidelityPreset` | CTGAN (1000 epochs) | Production-quality tabular data |
| `DiffusionPreset` | TabDDPM (1000 steps) | Complex multi-modal distributions |
| `CopulaPreset` | Gaussian Copula | Fast statistical baseline, dependency modeling |
| `DataQualityAuditPreset` | TVAE + full report | Quality audit with comprehensive automated reporting |

**Class Distribution**

| Preset | Method | Use Case |
|--------|--------|----------|
| `ImbalancedGeneratorPreset` | CTGAN | Force a specific minority/majority ratio (binary) |
| `BalancedDataGeneratorPreset` | SMOTE | Oversample minority classes to balance a skewed dataset |

**Time Series**

| Preset | Method | Use Case |
|--------|--------|----------|
| `TimeSeriesPreset` | TimeGAN / TimeVAE / FourierFlows | Sequential data with temporal dynamics |
| `SeasonalTimeSeriesPreset` | TimeGAN + ScenarioInjector | Time series with injected sinusoidal seasonality |

**Drift & Scenarios**

| Preset | Method | Use Case |
|--------|--------|----------|
| `DriftScenarioPreset` | CTGAN + DriftConfig | Stress-test drift detection systems |
| `GradualDriftPreset` | CTGAN + linear drift | Simulate slow linear feature drift |
| `ConceptDriftPreset` | CTGAN + concept drift | Alter P(y\|x) relationships |
| `ScenarioInjectorPreset` | ScenarioInjector | Apply custom evolution scenarios to existing data |

**Clinical & Omics**

| Preset | Method | Use Case |
|--------|--------|----------|
| `LongitudinalHealthPreset` | ClinicalDataGenerator | Multi-visit patient records |
| `RareDiseasePreset` | ClinicalDataGenerator | Cohorts with rare condition (default 1% prevalence) |
| `OmicsIntegrationPreset` | ClinicalDataGenerator | Multi-omics (clinical + gene expression + proteomics) |
| `SingleCellQualityPreset` | scVI (400 epochs) | High-quality single-cell RNA-seq data |

> Full parameter reference: [`calm_data_generator/docs/PRESETS_REFERENCE.md`](calm_data_generator/docs/PRESETS_REFERENCE.md)

### Quick-Start Examples

```python
from calm_data_generator.presets import FastPreset, HighFidelityPreset, ImbalancedGeneratorPreset

# --- Fast generation ---
preset = FastPreset(random_state=42)
synthetic_df = preset.generate(data=real_df, n_samples=1000)

# --- High-fidelity production data ---
preset = HighFidelityPreset(random_state=42)
synthetic_df = preset.generate(data=real_df, n_samples=5000)

# --- Imbalanced dataset (5% minority) ---
preset = ImbalancedGeneratorPreset(random_state=42)
synthetic_df = preset.generate(
    data=real_df, n_samples=2000,
    target_col="label", imbalance_ratio=0.05
)
```

```python
from calm_data_generator.presets import TimeSeriesPreset, SeasonalTimeSeriesPreset

# --- Time series with TimeGAN ---
preset = TimeSeriesPreset(random_state=42)
synthetic_df = preset.generate(
    data=ts_df, n_samples=500, sequence_key="patient_id", time_key="visit_date"
)

# --- Seasonal time series (monthly pattern) ---
preset = SeasonalTimeSeriesPreset(random_state=42)
synthetic_df = preset.generate(
    data=ts_df, n_samples=500,
    time_col="date", seasonal_cols=["sales", "demand"],
    period=12, amplitude=2.0
)
```

```python
from calm_data_generator.presets import LongitudinalHealthPreset, SingleCellQualityPreset

# --- Clinical longitudinal data (multi-visit) ---
preset = LongitudinalHealthPreset(random_state=42)
result = preset.generate(n_samples=200, n_visits=6)

# --- Single-cell RNA-seq ---
preset = SingleCellQualityPreset(random_state=42)
synthetic_df = preset.generate(data=adata_df, n_samples=500)
```

## Key Libraries & Ecosystem

 | Library | Role | Usage in Calm-Data-Generator |
 | :--- | :--- | :--- |
 | **Synthcity** | Deep Learning Engine | Powers `CTGAN`, `TVAE`, `DDPM`, `TimeGAN`. Handling privacy & fidelity. |
 | **scvi-tools** | Single-Cell Analysis | Powers `scvi` method for high-dimensional genomic/transcriptomic data. |
 | **GEARS** | Graph Perturbation | Powers `gears` method for predicting single-cell perturbation effects. |
 | **River** | Streaming ML | Powers `StreamGenerator` for concept drift simulation and real-time data flow. |
 | **YData Profiling**| Reporting | Generates automated quality reports (`QualityReporter`). |
 | **Pydantic** | Validation | Ensures strict type checking and configuration management. |
 | **PyTorch** | Backend | Underlying tensor computation for all deep learning models. |
 | **Copulae** | Statistical Modeling | Powers the `copula` and `windowed_copula` methods for multivariate dependence modeling and drift-aware generation. |
 | **hmmlearn** | Statistical Modeling | Powers the `hmm` method for drift-aware generation via Hidden Markov Model regime transitions. |
 | **SciPy** | Mathematical Core | Powers the Gaussian Copula engines inside `ComplexGenerator` (unconditional and conditional) for correlated multi-variate generation across domains. |

## Safe Data Sharing

A key advantage of **Calm-Data-Generator** is enabling the use of private data in public or collaborative environments:

1.  **Private Origin**: You start with sensitive data (e.g., GDPR/HIPAA restricted) that cannot leave your secure environment.
2.  **Synthetic Twin**: The library generates a synthetic dataset that statistically mirrors the original but contains **no real individuals**.
3.  **Safe Distribution**: Once validated (using `QualityReporter`'s privacy checks), this synthetic dataset allows for **risk-free sharing**, model training, and testing without exposing confidential information.

## Key Use Cases

- **MLOps Monitoring Validation**: Use **StreamGenerator** and **DriftInjector** to simulate data drift (gradual, abrupt) and verify if your monitoring alerts trigger correctly before deployment.
- **Biomedical Research (HealthTech)**: Generate synthetic patient cohorts with **ClinicalDataGenerator** that preserve complex biological correlations (e.g., gene-age relationships) for collaborative studies without compromising patient privacy.
- **Stress Testing ("What-If" Analysis)**: Use **ScenarioInjector** to simulate future scenarios (e.g., "What if the customer age base increases by 10 years?") and measure model performance degradation under stress.
- **Correlation-Aware Drift**: Inject drift that realistically propagates to correlated features (e.g., increasing income also proportionally increases spending) using the `correlations=True` parameter.
- **Development Data**: Provide developers with high-fidelity synthetic replicas of production databases, allowing them to build and test features safely without accessing sensitive real-world data.

---

## Architecture & Design

### Technical Architecture
Minimalist view of the system's core components and data flow.

![Architecture Diagram](calm_data_generator/docs/assets/architecture.png)



---

## Installation

 > [!WARNING]
 > The installation is **heavy (~2-3 GB)** and may take several minutes. Use a fresh virtual environment.

 ### Versioning Strategy

 - **GitHub (Recommended for latest features)**: The `main` branch contains the most up-to-date version with the latest bug fixes and features.
 - **PyPI (Stable)**: Releases on PyPI are stable versions updated less frequently for major changes.

### Step 1 — Install PyTorch for your hardware

PyTorch must be installed **before** the library so pip resolves the correct wheel (CPU, CUDA, or ROCm).

```bash
# CPU — Mac Intel, CI environments, no GPU
pip install "torch>=2.2.0" "torchvision>=0.17.0"

# CUDA 12.4 — Linux / Windows with NVIDIA GPU
pip install "torch>=2.2.0" "torchvision>=0.17.0" --index-url https://download.pytorch.org/whl/cu124

# ROCm 6.1 — Linux with AMD GPU
pip install "torch>=2.2.0" "torchvision>=0.17.0" --index-url https://download.pytorch.org/whl/rocm6.1

# Mac Apple Silicon (M1/M2/M3) — MPS is auto-detected, no special wheel needed
pip install "torch>=2.2.0" "torchvision>=0.17.0"
```

> Replace `cu124` with your installed CUDA version (e.g. `cu118`, `cu121`). Find yours with `nvidia-smi`.

### Step 2 — Install the library

```bash
pip install --upgrade pip setuptools wheel
pip install calm-data-generator
```

### Optional extras

```bash
# River streaming support
pip install "calm-data-generator[stream]"

# GEARS perturbation prediction (requires PyG wheels — see below)
pip install "calm-data-generator[gears]"

# Full suite (stream + gears)
pip install "calm-data-generator[full]"
```

> **GEARS / PyTorch Geometric** requires platform-specific wheels from the PyG index.
> Install them **before** `calm-data-generator[gears]`:
> ```bash
> # Get your torch version first
> python -c "import torch; print(torch.__version__)"
>
> # CPU example (replace torch-2.2.0 and +cpu with your version and platform)
> pip install torch-geometric --find-links https://data.pyg.org/whl/torch-2.2.0+cpu.html
>
> # CUDA 12.4 example
> pip install torch-geometric --find-links https://data.pyg.org/whl/torch-2.2.0+cu124.html
>
> pip install "calm-data-generator[gears]"
> ```

**From source (GitHub - Latest Updates):**

```bash
# Option A: Install directly from GitHub
pip install git+https://github.com/AlejandroBeldaFernandez/Calm-Data_Generator.git

# Option B: Clone and install (for development)
git clone https://github.com/AlejandroBeldaFernandez/Calm-Data_Generator.git
cd Calm-Data_Generator
pip install .
```

### Troubleshooting

**Zsh shell (macOS/Linux):** If brackets cause errors, use quotes:
```bash
pip install "calm-data-generator[stream]"
```

**River compilation errors (Linux/macOS):**
```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev

# macOS
xcode-select --install

# Then retry
pip install calm-data-generator
```

**Windows users:** Install Visual Studio Build Tools first:
1. Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install "Desktop development with C++"
3. Then retry installation

**Windows — Long Path error during installation:**

Some packages (e.g. `orbax-checkpoint`) contain very deep directory structures that exceed the Windows default 260-character path limit. If you see an error like:
```
OSError: [Errno 2] No such file or directory: 'C:\...\very\long\path'
HINT: This error might have occurred since this system does not have Windows Long Path support enabled.
```
Enable long paths via PowerShell (run as Administrator):
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
Then restart your machine and retry installation. Alternatively, install your virtual environment in a short root path (e.g. `C:\venv\`) to reduce total path length.

**Dependency conflicts:** Use a clean virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows
pip install calm-data-generator
```

---

## Quick Start

### Generate Synthetic Data from Real Dataset

```python
from calm_data_generator import RealGenerator
import pandas as pd

# Your real dataset (can be a DataFrame, path to .csv, .h5, or .h5ad)
data = pd.read_csv("your_data.csv")  # or "your_data.h5ad"

# Initialize generator
gen = RealGenerator()

# Generate 1000 synthetic samples using CTGAN

synthetic = gen.generate(
    data=data,
    n_samples=1000,
    method='ctgan',
    target_col='label',
    differentiation_factor=2.0, # NEW: Enhance class separability
    clipping_mode='permissive', # NEW: Manage data ranges
    epochs=300,
    batch_size=500,
    discriminator_steps=1,
)

print(f"Generated {len(synthetic)} samples")
```

### GPU Acceleration

GPU is **auto-detected** at runtime in the order CUDA → MPS → CPU. No parameter is needed for most methods.

| Method | Device support | Notes |
| ------ | -------------- | ----- |
| `scvi`, `scanvi` | CUDA · MPS · CPU | Auto-detected via Lightning accelerator |
| `gears` | CUDA · MPS · CPU | Auto-detected via PyTorch device |
| `ctgan`, `tvae`, `rtvae`, `ddpm`, `timegan`, `timevae`, `fflows`, `dpgan`, `pategan`, `great` | CUDA · CPU | Auto-detected internally by Synthcity |
| `cart`, `rf`, `lgbm`, `xgboost`, `gmm`, `copula`, `windowed_copula`, `kde`, `smote`, `adasyn`, `hmm`, `bn` | CPU only | Statistical / tree-based, no GPU path |

**Mac Apple Silicon (M1/M2/M3):** MPS is used automatically for `scvi`, `scanvi`, and `gears` when `torch.backends.mps.is_available()` returns `True`. Install the standard CPU wheel — no special MPS wheel is required.

To override the device explicitly:

```python
# Force CPU for scVI/scANVI
gen.generate(data=data, n_samples=1000, method='scvi', accelerator='cpu')

# Force a specific device for GEARS
gen.generate(data=data, n_samples=1000, method='gears', device='cuda')
gen.generate(data=data, n_samples=1000, method='gears', device='mps')
```

### Generate Clinical Data

```python
from calm_data_generator import ClinicalDataGenerator
from calm_data_generator.generators.configs import DateConfig

gen = ClinicalDataGenerator()

# Generate patient data with genes and proteins
result = gen.generate(
    n_samples=100,
    n_genes=500,
    n_proteins=200,
    date_config=DateConfig(start_date="2024-01-01")
)

demographics = result['demographics']
genes = result['genes']
proteins = result['proteins']
```

### Inject Drift for ML Testing

**Option 1: Directly from `generate()` (recommended)**

```python
from calm_data_generator import RealGenerator

gen = RealGenerator()

# Generate synthetic data WITH drift in one call
synthetic = gen.generate(
    data=real_data,
    n_samples=1000,
    method='ctgan',
    target_col='label',
    drift_injection_config=[
        {
            "method": "inject_drift",
            "params": {
                "columns": ["age", "income", "label"],
                "drift_mode": "gradual", # Auto-detects column types
                "drift_magnitude": 0.3,
                "center": 500,
                "width": 200
            }
        }
    ]
)
```

**Option 2: Standalone DriftInjector**

```python
from calm_data_generator import DriftInjector

injector = DriftInjector()

# Unified drift injection (auto-detects types)
drifted_data = injector.inject_drift(
    df=data,
    columns=['feature1', 'feature2', 'status'],
    drift_mode='gradual',
    drift_magnitude=0.5,
    # Optional specific configs
    numeric_operation='shift',
    categorical_operation='frequency',
    boolean_operation='flip'
)
```
```

**Available drift methods:** `inject_feature_drift`, `inject_feature_drift_gradual`, `inject_feature_drift_incremental`, `inject_feature_drift_recurrent`, `inject_label_drift`, `inject_concept_drift`, `inject_categorical_frequency_drift`, and more. See [DRIFT_INJECTOR_REFERENCE.md](calm_data_generator/docs/DRIFT_INJECTOR_REFERENCE.md).

### Single-Cell / Gene Expression Data

Generate synthetic single-cell RNA-seq-like data using specialized VAE models:

```python
from calm_data_generator import RealGenerator

gen = RealGenerator()

# scVI: Generate new cells from scratch
synthetic = gen.generate(
    data="expression_data.h5ad", # Paths to .h5 or .h5ad are supported directly
    n_samples=1000,
    method='scvi',
    target_col='cell_type',
    epochs=100,
    n_latent=10,

)


```

| Method | Use Case |
|--------|----------|
| `scvi` | Generate new cells from learned distribution |

Once you have synthetic single-cell data, validate its quality using [scGFT Evaluator](https://github.com/nasim23ea/scgft-evaluator) via `QualityReporter`:

```python
from calm_data_generator.reports.QualityReporter import QualityReporter
from calm_data_generator.generators.configs import ReportConfig

reporter = QualityReporter(verbose=True)
reporter.generate_comprehensive_report(
    real_df=real_df,
    synthetic_df=synthetic_df,
    generator_name="SingleCell_Example",
    report_config=ReportConfig(
        output_dir="./sc_report",
        use_scgft=True,
        target_column="cell_type",
    ),
)
```

> See [REPORTS_REFERENCE.md](calm_data_generator/docs/REPORTS_REFERENCE.md#single-cell-evaluation-scgft) for full details.

### Stream Data Generation

```python
from calm_data_generator import StreamGenerator

stream_gen = StreamGenerator()

# Generate a data stream with Concept Drift
stream_data = stream_gen.generate(
    n_chunks=10,
    chunk_size=1000,
    concept_drift=True,  # Simulate concept drift over time
    n_features=10
)

print(f"Generated stream with {len(stream_data)} total samples")
```

### Quality Reporting

```python
from calm_data_generator import QualityReporter

# Generate a quality report comparing real vs synthetic data
reporter = QualityReporter()

reporter.generate_report(
    real_data=data,
    synthetic_data=synthetic,
    output_dir="./quality_report",
    target_col="target"
)
# Report saved to ./quality_report/report.html
# Results JSON (including compared_data_files) saved to ./quality_report/report_results.json
```

For **single-cell data**, enable [scGFT](https://github.com/nasim23ea/scgft-evaluator) evaluation (Graph Fourier Transform-based manifold preservation metrics):

```python
from calm_data_generator.generators.configs import ReportConfig

reporter.generate_comprehensive_report(
    real_df=real_df,
    synthetic_df=synthetic_df,
    generator_name="MyGen",
    report_config=ReportConfig(
        output_dir="./sc_report",
        use_scgft=True,
        target_column="cell_type",
    ),
)
# Generates scgft_report.html with ARI, MMD, Jaccard, Kendall Tau metrics
```

---

## Modules

| Module | Import | Description |
|--------|--------|-------------|
| **Tabular** | `generators.tabular` | RealGenerator, QualityReporter |
| **Clinical** | `generators.clinical` | ClinicalDataGenerator, ClinicalDataGeneratorBlock |
| **Stream** | \`generators.stream\` | StreamGenerator, StreamBlockGenerator |
| **Blocks** | `generators.tabular` | RealBlockGenerator |
| **Drift** | `generators.drift` | DriftInjector |
| **Dynamics** | `generators.dynamics` | ScenarioInjector |
| **Reports** | `reports` | Visualizer |

---

## Synthesis Methods

| Method | Type | Description | Requirements |
| ------ | ---- | ----------- | ------------ |
| `cart` | ML | CART-based iterative synthesis (fast) | Base |
| `rf` | ML | Random Forest synthesis | Base |
| `lgbm` | ML | LightGBM-based synthesis | Base |
| `xgboost` | ML | XGBoost-based synthesis | Base |
| `gmm` | Statistical | Gaussian Mixture Models | Base |
| `kde` | Statistical | Kernel Density Estimation (numeric only) | Base |
| `copula` | Statistical | Gaussian Copula synthesis | Base |
| `windowed_copula` | Drift-Aware | Gaussian Copula interpolated across time windows | Base |
| `hmm` | Drift-Aware | Hidden Markov Model — drift via regime transitions | Base |
| `smote` | Augmentation | SMOTE oversampling | Base |
| `adasyn` | Augmentation | ADASYN adaptive sampling | Base |
| `resample` | Augmentation | Weighted resampling from original data | Base |
| `ctgan` | DL | Conditional GAN for tabular data | `synthcity` |
| `tvae` | DL | Variational Autoencoder | `synthcity` |
| `rtvae` | DL | Robust TVAE | `synthcity` |
| `great` | DL | GReaT — LLM-based tabular generation | `synthcity` |
| `bn` | Probabilistic | Bayesian Network — causal dependencies | `synthcity` |
| `diffusion` | DL | Tabular Diffusion (custom, lightweight) | Base (PyTorch) |
| `ddpm` | DL | TabDDPM — advanced diffusion | `synthcity` |
| `dpgan` | DL + Privacy | Differentially private GAN | `synthcity` |
| `pategan` | DL + Privacy | PATE-GAN — teacher-ensemble privacy | `synthcity` |
| `timegan` | Time Series | TimeGAN for sequential data | `synthcity` |
| `timevae` | Time Series | TimeVAE for sequential data | `synthcity` |
| `fflows` | Time Series | FourierFlows — normalizing flows, stable for periodic series | `synthcity` |
| `conditional_drift` | Drift-Aware | Temporal stage conditioning via TVAE/CTGAN | `synthcity` |
| `scvi` | Single-Cell | scVI (VAE) for RNA-seq expression data | `scvi-tools` |
| `scanvi` | Single-Cell | scANVI — semi-supervised, label-aware scVI | `scvi-tools` |
| `gears` | Single-Cell | GEARS — perturbation prediction via GNN | `[gears]` extra |


---

## CLI Access

```bash
# List all tutorials
calm-data-generator tutorials

# Show a specific tutorial
calm-data-generator tutorials show 1

# Run a tutorial
calm-data-generator tutorials run 1

# Show version
calm-data-generator version
```

---

## Tutorials

| # | Tutorial | Description |
|---|----------|-------------|
| 1 | Real Generator | Tabular data synthesis |
| 2 | Clinical Generator | Clinical/medical data |
| 3 | Drift Injector | Drift injection for ML |
| 4 | Stream Generator | Stream-based generation |
| 5 | Scenario Injector | Feature evolution |

---

## Documentation Index

Explore the full documentation in the `calm_data_generator/docs/` directory:

| Document | Description |
|----------|-------------|
| **[DOCUMENTATION.md](calm_data_generator/docs/DOCUMENTATION.md)** | **Main User Guide**. Comprehensive manual covering all modules, concepts, and advanced usage. |
| **[REAL_GENERATOR_REFERENCE.md](calm_data_generator/docs/REAL_GENERATOR_REFERENCE.md)** | **API Reference for `RealGenerator`**. Detailed parameters for all synthesis methods (`ctgan`, `lgbm`, `scvi`, etc.). |
| **[DRIFT_INJECTOR_REFERENCE.md](calm_data_generator/docs/DRIFT_INJECTOR_REFERENCE.md)** | **API Reference for `DriftInjector`**. Guide to using `inject_drift` and specialized drift capabilities. |
| **[STREAM_GENERATOR_REFERENCE.md](calm_data_generator/docs/STREAM_GENERATOR_REFERENCE.md)** | **API Reference for `StreamGenerator`**. Details on stream simulation and drift integration. |
| **[CLINICAL_GENERATOR_REFERENCE.md](calm_data_generator/docs/CLINICAL_GENERATOR_REFERENCE.md)** | **API Reference for `ClinicalGenerator`**. Configuration for genes, proteins, and patient data. |
| **[API.md](calm_data_generator/docs/API.md)** | **Technical API Index**. High-level index of classes and functions. |

---

## License

MIT License - see [LICENSE](LICENSE) file

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full history. Summary of recent releases:

### v2.0.0 — 2026-03-27
- **ComplexGenerator**: new abstract layer with 3 reusable mathematical engines (Gaussian Copula unconditional/conditional + stochastic effects). `ClinicalDataGenerator` now inherits from it.
- **CausalEngine**: DAG-based causal cascade with topological sort (Kahn's algorithm) and non-linear transfer functions.
- **`inject_functional_drift()`**: per-row drift magnitude = f(driver column value).
- **`inject_causal_cascade()`**: full causal propagation integrated into `DriftInjector`.
- **`driven_by` evolve type**: `ScenarioInjector` feature delta driven by another column per row.
- **Bug fixes**: CART datetime handling, `bn` method dispatch, `conditional_drift` Synthcity API, `windowed_copula` 1D array reshape.
- **Tests**: 186 passed, 0 failed — all `unittest.TestCase` files converted to pytest.

### v1.2.0
- `differentiation_factor`, `clipping_mode`, `use_latent_sampling` parameters.
- Windowed Copula, Conditional Drift, DPGAN, PATEGAN synthesis methods.

---

## Acknowledgements & Credits

We stand on the shoulders of giants. This library is possible thanks to these amazing open-source projects:

- **[Synthcity](https://github.com/vanderschaarlab/synthcity)** (Apache 2.0) - The engine behind our deep learning models.
- **[River](https://github.com/online-ml/river)** (BSD-3-Clause) - Powering our streaming capabilities.
- **[YData Profiling](https://github.com/ydataai/ydata-profiling)** (MIT) - Providing comprehensive data reporting.
- **[scvi-tools](https://github.com/scverse/scvi-tools)** (BSD-3-Clause) - Enabling single-cell analysis.
- **[GEARS](https://github.com/snap-stanford/GEARS)** (MIT) - Supporting graph-based perturbation prediction.
- **[Imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn)** (MIT) - Providing SMOTE and ADASYN implementations.
- **[SDMetrics](https://github.com/sdv-dev/SDMetrics)** (MIT) - Powering the standardized metrics in our QualityReporter.
- **[Copulae](https://github.com/DanielBok/copulae)** (MIT) - Enabling multivariate dependence modeling via Gaussian Copulas.
- **[AnnData](https://github.com/scverse/anndata)** (BSD-3-Clause) - Providing the core data structure for single-cell and omics integration.
- **[LightGBM](https://github.com/microsoft/LightGBM)** (MIT) - Powering our gradient boosting synthesis method.
- **[PyTorch](https://github.com/pytorch/pytorch)** (BSD-3-Clause) - The deep learning framework powering our generative models.
- **[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)** (MIT) - Enabling Graph Neural Network operations for relational data.
- **[XGBoost](https://github.com/dmlc/xgboost)** (Apache-2.0) - optimized distributed gradient boosting library.
- **[Hugging Face Hub](https://github.com/huggingface/huggingface_hub)** (Apache-2.0) - Facilitating model sharing and versioning.
- **[Plotly](https://github.com/plotly/plotly.py)** (MIT) - Enabling interactive data visualizations.
- **[hmmlearn](https://github.com/hmmlearn/hmmlearn)** (BSD-3-Clause) - Powering the `hmm` method for drift-aware generation via Hidden Markov Models.
- **[scgft-evaluator](https://github.com/nasim23ea/scgft-evaluator)** - Providing Graph Fourier Transform-based evaluation for single-cell synthetic data quality assessment.
