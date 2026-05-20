# Referencia del Stream Generator

La clase `calm_data_generator.generators.stream.StreamGenerator` proporciona funcionalidad para generar flujos de datos sintéticos, construido sobre la biblioteca `River`. Soporta inyección de concept drift, balanceo de datos y simulación de dinámicas.

## Clase: `StreamGenerator`

### Uso
```python
from calm_data_generator import StreamGenerator
try:
    from river import synth
except ImportError:
    from river.datasets import synth

# Inicializar
generator = StreamGenerator(random_state=42)

# Crear instancia de generador River (ej. SEA)
river_gen = synth.SEA()

# Configuración de Drift
drift_conf = [DriftConfig(method="inject_feature_drift", params={"feature_cols": ["col1"], "drift_magnitude": 0.5})]
report_conf = ReportConfig(output_dir="./output", target_column="target")

# Generar datos
df = generator.generate(
    generator_instance=river_gen,
    n_samples=1000,
    filename="stream_data.csv",
    drift_config=drift_conf,
    report_config=report_conf
)
```

### `__init__`
**Firma:** `__init__(random_state: Optional[int] = None, auto_report: bool = True, minimal_report: bool = False)`

- **Argumentos:**
    - `random_state`: Semilla para reproducibilidad.
    - `auto_report`: Si es True, genera automáticamente un informe de calidad.
    - `minimal_report`: Si es True, omite cálculos costosos (ej. correlaciones).

### `generate`
**Firma:** `generate(...)`

Método principal para generar un dataset sintético.

- **Argumentos:**
    - `generator_instance`: Un generador de River instanciado (o iterador compatible).
    - `n_samples` (int): Número de muestras a generar.
    - `filename` (str): Nombre del archivo de salida (CSV).
    - `output_dir` (str): Directorio donde guardar.
    - `target_col` (str): Nombre de la columna objetivo (defecto: "target").
    - `balance` (bool): Si es True, balancea la distribución de clases (defecto: False).
    - `date_config` (DateConfig): Objeto de configuración para inyección de fechas.
    - `drift_type` (str): Tipo de drift a inyectar ('none', 'virtual_drift', 'gradual', 'abrupt', 'incremental').
    - `drift_options` (dict): Opciones para inyección de drift (ej. `missing_fraction` para virtual drift).
    - `drift_config` (List[DriftConfig]): Lista de objetos `DriftConfig` para inyección de drift post-generación con `DriftInjector`.
    - `report_config` (ReportConfig): Configuración para la generación de informes.
    - `dynamics_config` (dict): Configuración para `ScenarioInjector` (ej. evolución de features, construcción de targets).
    - `save_dataset` (bool): Si se debe guardar el archivo CSV (defecto: False).

- **Retorna:** `pd.DataFrame`: El dataset generado.

### `generate_longitudinal_data`
Genera datos de estilo clínico multi-visita basados en un paso base de generación.

- **Argumentos:**
    - `n_samples`: Número de entidades/pacientes base.
    - `longitudinal_config`: Diccionario con claves como `n_visits`, `time_step_days`, `evolution_config`.
    - `date_config`: Configuración de fecha base.

- **Retorna:** Diccionario conteniendo DataFrames 'longitudinal', 'base_demographics' y 'base_omics'.
