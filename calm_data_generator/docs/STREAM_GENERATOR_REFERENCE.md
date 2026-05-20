# Stream Generator Reference

The `calm_data_generator.generators.stream.StreamGenerator` class provides functionality for generating synthetic data streams, built on top of the `River` library. It supports concept drift injection, data balancing, and dynamics simulation.

## Class: `StreamGenerator`

### Usage
```python
from calm_data_generator.generators.stream.StreamGenerator import StreamGenerator
from river import synth

# Initialize
generator = StreamGenerator(random_state=42)

# Create a River generator instance (e.g., SEA)
river_gen = synth.SEA()

# Generate data
df = generator.generate(
    generator_instance=river_gen,
    n_samples=1000,
    filename="stream_data.csv",
    output_dir="./output"
)
```

### `__init__`
**Signature:** `__init__(random_state: Optional[int] = None, auto_report: bool = True, minimal_report: bool = False)`

- **Args:**
    - `random_state`: Seed for reproducibility.
    - `auto_report`: If True, automatically generates a quality report.
    - `minimal_report`: If True, skips expensive report calculations (e.g. correlations).

### `generate`
**Signature:** `generate(...)`

Main method to generate a synthetic dataset.

- **Args:**
    - `generator_instance`: An instantiated River generator (or compatible iterator).
    - `n_samples` (int): Number of samples to generate.
    - `filename` (str): Output filename (CSV).
    - `output_dir` (str): Directory to save output.
    - `target_col` (str): Name of the target variable column (default: "target").
    - `balance` (bool): If True, balances class distribution (default: False).
    - `date_config` (DateConfig): Configuration object for date injection.
    - `drift_type` (str): Type of drift to inject ('none', 'virtual_drift', 'gradual', 'abrupt', 'incremental').
    - `drift_options` (dict): Options for drift injection (e.g. `missing_fraction` for virtual drift).
    - `drift_config` (list): List of `DriftConfig` objects for `DriftInjector` post-generation.
    - `report_config` (ReportConfig): Configuration for report generation.
    - `dynamics_config` (dict): Configuration for `ScenarioInjector` (e.g. feature evolution, target construction).
    - `save_dataset` (bool): Whether to save the CSV file (default: False).

- **Returns:** `pd.DataFrame`: The generated dataset.

### `generate_longitudinal_data`
Generates multi-visit clinical-style data based on a base generation step.

- **Args:**
    - `n_samples`: Number of base entities/patients.
    - `longitudinal_config`: Dictionary with keys like `n_visits`, `time_step_days`, `evolution_config`.
    - `date_config`: Base date configuration.

- **Returns:** Dict containing 'longitudinal', 'base_demographics', and 'base_omics' DataFrames.
