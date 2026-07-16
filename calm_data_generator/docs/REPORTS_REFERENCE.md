# Reports Reference

The `calm_data_generator` library includes a suite of reporting tools designed to assess the quality, privacy, and characteristics of generated data.

---

## ReportConfig Class Reference

**Import:** `from calm_data_generator.generators.configs import ReportConfig`

`ReportConfig` is a Pydantic model that provides type-safe configuration for report generation across all reporter classes.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | `"output"` | Directory to save generated reports |
| `auto_report` | bool | `True` | Automatically generate reports after data generation |
| `minimal` | bool | `False` | Generate minimal reports (faster, less detail) |
| `target_column` | str | `None` | Target/label column for classification/regression analysis |
| `time_col` | str | `None` | Time column for time-series analysis |
| `block_column` | str | `None` | Block identifier column for block-based data |
| `resample_rule` | str/int | `None` | Resampling rule for time-series (e.g., `"1D"`, `"1H"`) |
| `privacy_check` | bool | `False` | Enable privacy assessment — DCR, NNDR, and Singling-Out risk (if `anonymeter` is installed) |
| `discriminator` | bool | `False` | Enable discriminator-based (adversarial validation) reporting |
| `tstr` | bool | `False` | Run TSTR (Train Synthetic, Test Real) — requires `target_column` |
| `spearman` | bool | `False` | Generate Spearman correlation heatmaps (real vs. synthetic) |
| `focus_columns` | List[str] | `None` | Specific columns to focus analysis on |
| `constraints_stats` | Dict[str, int] | `None` | Constraint violation statistics |
| `sequence_config` | Dict | `None` | Configuration for sequence-based analysis |
| `per_block_external_reports` | bool | `False` | Generate separate reports per block |
| `use_scgft` | bool | `False` | Enable specialized scGFT single-cell evaluation |

> [!NOTE]
> When both `report_config` and individual keyword arguments are passed to
> `generate_comprehensive_report()`, `discriminator`/`privacy_check`/`tstr`/`spearman` use OR
> semantics — an explicitly-set argument always wins, even if `report_config` leaves it at its
> default. Other fields (e.g. `target_column`, `minimal`) come from `report_config` when it is
> provided.

### Usage Examples

**Basic Report Configuration:**
```python
from calm_data_generator.generators.configs import ReportConfig

report_config = ReportConfig(
    output_dir="./my_reports",
    target_column="target",
    privacy_check=True,
    adversarial_validation=True
)
```

**Time-Series Report:**
```python
report_config = ReportConfig(
    output_dir="./timeseries_report",
    time_col="timestamp",
    resample_rule="1D",  # Daily aggregation
    target_column="sales"
)
```

**Block-Based Report:**
```python
report_config = ReportConfig(
    output_dir="./block_report",
    block_column="patient_id",
    per_block_external_reports=True,
    target_column="diagnosis"
)
```

**Minimal Report (Fast):**
```python
report_config = ReportConfig(
    output_dir="./quick_report",
    minimal=True,
    focus_columns=["age", "income", "target"]
)
```

---

## Quality Reporter (`Tabular`)
**Module:** `calm_data_generator.generators.tabular.QualityReporter`

Generates comprehensive reports comparing real and synthetic tabular data.

### `generate_comprehensive_report`
Generates a static, file-based report (HTML + `report_results.json`) including:
- **Overall Quality Scores**: Overall and column-wise similarity metrics (SDMetrics).
- **Statistical Similarity**: KS test, Levene test, Wasserstein distance (per numeric column),
  and MMD (global), saved to `statistical_tests.html`.
- **Privacy Assessment** (`privacy_check=True`): Distance to Closest Record (DCR), Nearest
  Neighbor Distance Ratio (NNDR), and Singling-Out risk (if the optional `anonymeter` dependency
  is installed — see the [Privacy Metrics](#privacy-metrics-dcr-nndr-singling-out) section below).
- **ML Utility (TSTR)** (`tstr=True`, requires `target_column`): Train-Synthetic-Test-Real
  metrics, saved to `tstr_report.html`.
- **Visualizations**: Histograms, density plots, QQ plots, PCA/UMAP projections, Spearman
  correlation heatmaps (`spearman=True`).
- **ARI Metrics (Class Separability)**: Adjusted Rand Index (ARI) using K-Means (k=2) to quantify how well classes (Cases vs Controles) are separated in both real and synthetic data.
- **Drift Analysis**: Visual comparison of feature distributions.

Since v2.3.0, this method **returns the results dict** (the same one written to
`report_results.json`) instead of `None`.

```python
from calm_data_generator.generators.configs import ReportConfig

reporter = QualityReporter(verbose=True)
results = reporter.generate_comprehensive_report(
    real_df=original_df,
    synthetic_df=synthetic_df,
    generator_name="MyGenerator",
    report_config=ReportConfig(
        output_dir="./report_output",
        target_column="target_col",
        privacy_check=True,
        tstr=True,
    )
)
print(results["quality_scores"], results["statistical_metrics"])
```

### `evaluate()` — lightweight, in-memory check

For a fast fidelity check without writing anything to disk (e.g. inside a training loop, a
test, or a quick sanity check), use `evaluate()` instead:

```python
reporter = QualityReporter(verbose=False)
result = reporter.evaluate(real_df, synthetic_df, target_column="target")
# {
#   "quality_scores": {"overall_quality_score": 0.94, "weighted_quality_score": 0.93},
#   "statistical_metrics": {"mmd": 0.012, "per_column": {...}},   # includes Wasserstein per column
#   "tstr_metrics": {"task": "classification", "roc_auc": 0.91, ...},  # None if no target_column
# }
```

`evaluate()` computes SDMetrics quality scores, statistical similarity tests (KS/Levene/MMD/
Wasserstein), and TSTR (if `target_column` is given) — the same underlying computations as
`generate_comprehensive_report()`, but it writes no HTML/JSON and doesn't require `output_dir`.
It does **not** compute privacy metrics (DCR/NNDR/Singling-Out) — those remain file-report-only
via `privacy_check=True`, since Singling-Out risk in particular can take several seconds.

### `calculate_quality_metrics`
Calculates quality metrics (SDMetrics) for two datasets without generating a full report.

```python
reporter = QualityReporter(verbose=False)
metrics = reporter.calculate_quality_metrics(
    real_df=df1,
    synthetic_df=df2
)
# Returns: {'overall_quality_score': 0.85, 'weighted_quality_score': 0.82}
```

### `calculate_ari`
Standalone calculation of Adjusted Rand Index (ARI) to quantify class separability.

```python
ari_metrics = reporter.calculate_ari(
    real_df=df1,
    synthetic_df=df2,
    target_col="label"
)
# Returns: {'ari_original': 0.95, 'ari_synthetic': 0.98, 'ari_improvement': 0.03}
```

## Privacy Metrics (DCR, NNDR, Singling-Out)

Enabled by `privacy_check=True` in `generate_comprehensive_report()`. Results land under the
`privacy_metrics` key of the returned/saved results dict.

- **DCR (Distance to Closest Record)**: Euclidean distance (on normalized numeric columns) from
  each synthetic record to its closest real record. Lower values mean the synthetic record sits
  closer to a real one — a raw, scale-dependent proximity signal.
  - `dcr_mean`, `dcr_5th_percentile`
- **NNDR (Nearest Neighbor Distance Ratio)**: ratio of the distance to the closest real record
  over the distance to the second-closest one. A ratio near 0 means a synthetic record is much
  closer to one specific real record than to any other (higher re-identification risk); near 1
  means it's roughly equidistant to several real records (lower risk). Scale-invariant,
  complements DCR.
  - `nndr_mean`, `nndr_5th_percentile`
- **Singling-Out risk**: via the optional [anonymeter](https://github.com/statice/anonymeter)
  (MIT) dependency — `pip install calm-data-generator[privacy]`. Estimates the risk that an
  attacker can isolate one real record using a combination of attributes learned from the
  synthetic data. Nested under `privacy_metrics["singling_out"]`; `None` (with an info-level
  log, not an error) if `anonymeter` isn't installed.
  - `risk` (0–1, higher = riskier), `ci_low`/`ci_high` (95% confidence interval), `used_control`

```python
results = reporter.generate_comprehensive_report(
    real_df, synthetic_df, "MyGenerator", "./out", privacy_check=True,
)
pm = results["privacy_metrics"]
print(pm["dcr_mean"], pm["nndr_mean"])
print(pm.get("singling_out"))  # None if anonymeter isn't installed
```

> [!NOTE]
> anonymeter's own retry limit for finding unique attack queries defaults to 10,000,000, which
> can take minutes on low-cardinality/duplicate-heavy data. `calm-data-generator` bounds this to
> a much lower default internally, so a report run never blocks on it — at the cost of a wider
> confidence interval on pathological data. Call `reporter._calculate_singling_out_risk(...)`
> directly if you need to tune `n_attacks`/`n_cols`/`max_attempts`.

## Discriminator Reporter (Adversarial Validation)
**Module:** `calm_data_generator.reports.DiscriminatorReporter`

This reporter trains a classifier model (Random Forest) to attempt to distinguish between real and synthetic data. It is used to detect drift or assess general fidelity.

### Key Metrics
- **Similarity Score (Indistinguishability)**: (0.0 - 1.0).
    - **Formula**: `1 - 2 * |AUC - 0.5|`
    - `1.0`: Indistinguishable data (AUC = 0.5). Excellent Quality.
    - `0.0`: Easily distinguishable data (AUC = 1.0 or 0.0). Drift detected or poor quality.
- **Confusion Score**: Ability of the data to "confuse" the discriminator (based on inverted Accuracy).
- **Explainability**:
    - **Feature Importance**: Which variables allowed the model to distinguish the data.
    - **SHAP Values**: Detailed explanation of the impact of each feature.

### Usage
This reporter is automatically integrated into `QualityReporter` if the optional parameter is activated:
```python
reporter.generate_comprehensive_report(
    ...,
    report_config=ReportConfig(
        output_dir="./report_output",
        adversarial_validation=True  # Activate Discriminator
    )
)
```

## Stream Reporter (`Stream`)
**Module:** `calm_data_generator.generators.stream.StreamReporter`

Designed for analyzing synthetic data streams without a direct "real" reference dataset (though it can compare against expectations).

### `generate_report`
Generates a report for a synthetic dataset:
- **Data Profiling**: YData Profiling integration.
- **Visualizations**: Density plots and dimensionality reduction.
- **Block-wise Analysis**: Can generate separate reports for each data block.

```python
reporter = StreamReporter()
reporter.generate_report(
    synthetic_df=stream_df,
    generator_name="StreamGen",
    report_config=ReportConfig(output_dir="./stream_report")
)
```


## Single-Cell Evaluation (scGFT)
**Module:** `calm_data_generator.reports.QualityReporter`

The library integrates [`scgft-evaluator`](https://github.com/nasim23ea/scgft-evaluator) to provide specialized validation for single-cell RNA sequencing (scRNA-seq) data. This method uses Graph Fourier Transforms (GFT) to assess if the synthetic data preserves the underlying manifold and biological structure of the original cells.

### Installation

```bash
pip install scgft-evaluator @ git+https://github.com/nasim23ea/scgft-evaluator.git
```

Or via `requirements.txt` (already included in calm-data-generator):

```
scgft-evaluator @ git+https://github.com/nasim23ea/scgft-evaluator.git
```

### Key Features
- **Manifold Preservation**: Evaluates if the cell-to-cell relationships are maintained.
- **Cluster/Population Integrity**: Metrics on how well synthetic cells represent real populations (ARI, MMD, Jaccard, Kendall Tau).
- **Limma-based DE comparison**: Differential expression concordance between real and synthetic via `limma`.
- **Dashboard Integration**: Generates a dedicated `scgft_report.html` tab in the HTML dashboard with a results table.

### Usage
Set `use_scgft=True` in your `ReportConfig` and specify the cell-type column:

```python
from calm_data_generator.generators.configs import ReportConfig

reporter.generate_comprehensive_report(
    ...,
    report_config=ReportConfig(
        output_dir="./sc_report",
        use_scgft=True,
        target_column="cell_type"  # column with cell type labels
    )
)
```

The evaluator runs `ScGFT_Evaluator.run_all()` comparing the two most prevalent cell populations and prints a metrics table when `verbose=True`.

> [!IMPORTANT]
> **Data Format**: This method is specifically designed for single-cell data where columns represent genes and rows represent cells. It is **not recommended** for standard bulk or tabular data.

## Clinic Reporter (`Clinical`)
**Module:** `calm_data_generator.generators.clinical.ClinicReporter`

A specialized version of `StreamReporter` for clinical data. It inherits standard reporting capabilities but is tailored to handle clinical feature sets and may include domain-specific checks in the future.

```python
reporter = ClinicReporter()
reporter.generate_report(...)
```

### Report Results JSON (`report_results.json`)
Every report generates a `report_results.json` file containing the raw metrics:

```json
{
  "generator_name": "MyGenerator",
  "generation_timestamp": "2024-01-01T12:00:00",
  "real_rows": 1000,
  "synthetic_rows": 1000,
  "quality_scores": {
    "overall_quality_score": 0.85,
    "weighted_quality_score": 0.82
  },
  "compared_data_files": {
    "original": "real_data",
    "generated": "synthetic_data"
  }
}
```

> [!NOTE]
> **Privacy Reporting**: Privacy features (DCR metrics) are now integrated into `QualityReporter`. Use `privacy_check=True` when generating reports.
