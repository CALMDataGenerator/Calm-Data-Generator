# Referencia de Reportes

La biblioteca `calm_data_generator` incluye un conjunto de herramientas de reporte diseñadas para evaluar la calidad, privacidad y características de los datos generados.

---

## Referencia de la Clase ReportConfig

**Importar:** `from calm_data_generator.generators.configs import ReportConfig`

`ReportConfig` es un modelo Pydantic que proporciona configuración con tipos seguros para la generación de reportes en todas las clases de reportes.

### Parámetros

| Parámetro | Tipo | Por Defecto | Descripción |
|-----------|------|-------------|-------------|
| `output_dir` | str | `"output"` | Directorio para guardar reportes generados |
| `auto_report` | bool | `True` | Generar reportes automáticamente después de la generación de datos |
| `minimal` | bool | `False` | Generar reportes mínimos (más rápido, menos detalle) |
| `target_column` | str | `None` | Columna objetivo/etiqueta para análisis de clasificación/regresión |
| `time_col` | str | `None` | Columna de tiempo para análisis de series temporales |
| `block_column` | str | `None` | Columna identificadora de bloques para datos basados en bloques |
| `resample_rule` | str/int | `None` | Regla de remuestreo para series temporales (ej., `"1D"`, `"1H"`) |
| `privacy_check` | bool | `False` | Habilitar evaluación de privacidad — DCR, NNDR y riesgo de Singling-Out (si `anonymeter` está instalado) |
| `discriminator` | bool | `False` | Habilitar reporte basado en discriminador (validación adversarial) |
| `tstr` | bool | `False` | Ejecutar TSTR (Train Synthetic, Test Real) — requiere `target_column` |
| `spearman` | bool | `False` | Generar heatmaps de correlación de Spearman (real vs. sintético) |
| `focus_columns` | List[str] | `None` | Columnas específicas en las que enfocar el análisis |
| `constraints_stats` | Dict[str, int] | `None` | Estadísticas de violación de restricciones |
| `sequence_config` | Dict | `None` | Configuración para análisis basado en secuencias |
| `per_block_external_reports` | bool | `False` | Generar reportes separados por bloque |
| `use_scgft` | bool | `False` | Habilitar evaluación especializada scGFT para single-cell |

> [!NOTE]
> Cuando se pasan tanto `report_config` como argumentos individuales a
> `generate_comprehensive_report()`, `discriminator`/`privacy_check`/`tstr`/`spearman` usan
> semántica OR — un argumento fijado explícitamente siempre gana, incluso si `report_config` lo
> deja en su valor por defecto. El resto de campos (p. ej. `target_column`, `minimal`) vienen de
> `report_config` cuando se proporciona.

### Ejemplos de Uso

**Configuración Básica de Reporte:**
```python
from calm_data_generator.generators.configs import ReportConfig

report_config = ReportConfig(
    output_dir="./mis_reportes",
    target_column="target",
    privacy_check=True,
    adversarial_validation=True
)
```

**Reporte de Series Temporales:**
```python
report_config = ReportConfig(
    output_dir="./reporte_series_temporales",
    time_col="timestamp",
    resample_rule="1D",  # Agregación diaria
    target_column="ventas"
)
```

**Reporte Basado en Bloques:**
```python
report_config = ReportConfig(
    output_dir="./reporte_bloques",
    block_column="paciente_id",
    per_block_external_reports=True,
    target_column="diagnostico"
)
```

**Reporte Mínimo (Rápido):**
```python
report_config = ReportConfig(
    output_dir="./reporte_rapido",
    minimal=True,
    focus_columns=["edad", "ingresos", "target"]
)
```

---

## Reporter de Calidad (`Tabular`)
**Módulo:** `calm_data_generator.generators.tabular.QualityReporter`

Genera informes completos comparando datos tabulares reales y sintéticos.

### `generate_comprehensive_report`
Genera un informe estático basado en archivos (HTML + `report_results.json`) incluyendo:
- **Puntuaciones de Calidad Global**: Métricas de similitud generales y por columna (SDMetrics).
- **Similitud Estadística**: test KS, test de Levene, distancia de Wasserstein (por columna
  numérica) y MMD (global), guardado en `statistical_tests.html`.
- **Evaluación de Privacidad** (`privacy_check=True`): Distancia al Registro Más Cercano (DCR),
  Nearest Neighbor Distance Ratio (NNDR) y riesgo de Singling-Out (si la dependencia opcional
  `anonymeter` está instalada — ver la sección [Métricas de Privacidad](#métricas-de-privacidad-dcr-nndr-singling-out) más abajo).
- **ML Utility (TSTR)** (`tstr=True`, requiere `target_column`): métricas Train-Synthetic-Test-Real,
  guardadas en `tstr_report.html`.
- **Visualizaciones**: Histogramas, gráficos de densidad, QQ plots, proyecciones PCA/UMAP,
  heatmaps de correlación de Spearman (`spearman=True`).
- **Métricas ARI (Separabilidad de Clases)**: Índice de Rand Ajustado (ARI) mediante K-Means (k=2) para cuantificar qué tan bien se separan las clases (Casos vs Controles) tanto en datos reales como sintéticos.
- **Análisis de Drift**: Comparación visual de distribuciones de features.

Desde v2.3.0, este método **devuelve el dict de resultados** (el mismo que se escribe en
`report_results.json`) en vez de `None`.

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

### `evaluate()` — comprobación ligera en memoria

Para una comprobación rápida de fidelidad sin escribir nada a disco (p. ej. dentro de un bucle
de entrenamiento, un test, o una comprobación rápida), usa `evaluate()`:

```python
reporter = QualityReporter(verbose=False)
result = reporter.evaluate(real_df, synthetic_df, target_column="target")
# {
#   "quality_scores": {"overall_quality_score": 0.94, "weighted_quality_score": 0.93},
#   "statistical_metrics": {"mmd": 0.012, "per_column": {...}},   # incluye Wasserstein por columna
#   "tstr_metrics": {"task": "classification", "roc_auc": 0.91, ...},  # None si no hay target_column
# }
```

`evaluate()` calcula los scores de calidad SDMetrics, los tests de similitud estadística
(KS/Levene/MMD/Wasserstein) y el TSTR (si se da `target_column`) — los mismos cálculos internos
que `generate_comprehensive_report()`, pero sin escribir HTML/JSON ni requerir `output_dir`.
**No** calcula métricas de privacidad (DCR/NNDR/Singling-Out) — esas siguen siendo exclusivas
del reporte de archivo vía `privacy_check=True`, ya que el riesgo de Singling-Out en particular
puede tardar varios segundos.

### `calculate_quality_metrics`
Calcula métricas de calidad (SDMetrics) para dos conjuntos de datos sin generar un informe completo.

```python
reporter = QualityReporter(verbose=False)
metrics = reporter.calculate_quality_metrics(
    real_df=df1,
    synthetic_df=df2
)
# Devuelve: {'overall_quality_score': 0.85, 'weighted_quality_score': 0.82}
```

### `calculate_ari`
Cálculo independiente del Índice de Rand Ajustado (ARI) para cuantificar la separabilidad de clases.

```python
ari_metrics = reporter.calculate_ari(
    real_df=df1,
    synthetic_df=df2,
    target_col="label"
)
# Devuelve: {'ari_original': 0.95, 'ari_synthetic': 0.98, 'ari_improvement': 0.03}
```

## Métricas de Privacidad (DCR, NNDR, Singling-Out)

Se habilitan con `privacy_check=True` en `generate_comprehensive_report()`. Los resultados
quedan bajo la clave `privacy_metrics` del dict de resultados devuelto/guardado.

- **DCR (Distance to Closest Record)**: distancia euclídea (sobre columnas numéricas
  normalizadas) de cada registro sintético a su registro real más cercano. Valores más bajos
  significan que el registro sintético está más pegado a uno real — señal de proximidad bruta,
  dependiente de escala.
  - `dcr_mean`, `dcr_5th_percentile`
- **NNDR (Nearest Neighbor Distance Ratio)**: ratio entre la distancia al registro real más
  cercano y la distancia al segundo más cercano. Un ratio próximo a 0 significa que un registro
  sintético está mucho más cerca de un registro real concreto que de cualquier otro (mayor
  riesgo de reidentificación); próximo a 1 significa que está a distancia similar de varios
  (menor riesgo). Invariante a escala, complementa al DCR.
  - `nndr_mean`, `nndr_5th_percentile`
- **Riesgo de Singling-Out**: vía la dependencia opcional
  [anonymeter](https://github.com/statice/anonymeter) (MIT) — `pip install
  calm-data-generator[privacy]`. Estima el riesgo de que un atacante aísle un registro real
  concreto combinando atributos aprendidos del sintético. Anidado bajo
  `privacy_metrics["singling_out"]`; `None` (con un log informativo, no un error) si
  `anonymeter` no está instalado.
  - `risk` (0–1, mayor = más riesgo), `ci_low`/`ci_high` (intervalo de confianza del 95%), `used_control`

```python
results = reporter.generate_comprehensive_report(
    real_df, synthetic_df, "MyGenerator", "./out", privacy_check=True,
)
pm = results["privacy_metrics"]
print(pm["dcr_mean"], pm["nndr_mean"])
print(pm.get("singling_out"))  # None si anonymeter no está instalado
```

> [!NOTE]
> El límite propio de anonymeter para buscar ataques únicos es por defecto 10.000.000, lo que
> puede tardar minutos con datos de baja cardinalidad o muy duplicados. `calm-data-generator`
> lo acota internamente a un valor mucho más bajo por defecto, así que una ejecución de reporte
> nunca se bloquea por esto — a costa de un intervalo de confianza algo más ancho en datos
> patológicos. Llama directamente a `reporter._calculate_singling_out_risk(...)` si necesitas
> ajustar `n_attacks`/`n_cols`/`max_attempts`.

## Reporter Discriminador (Adversarial Validation)
**Módulo:** `calm_data_generator.reports.DiscriminatorReporter`

Este reporter entrena un modelo clasificador (Random Forest) para intentar distinguir entre datos reales y sintéticos. Se utiliza para detectar drift o evaluar la fidelidad general.

### Métricas Clave
- **Similarity Score (Indistinguishability)**: (0.0 - 1.0).
    - **Fórmula**: `1 - 2 * |AUC - 0.5|`
    - `1.0`: Datos indistinguibles (AUC = 0.5). Excelente Calidad.
    - `0.0`: Datos fácilmente distinguibles (AUC = 1.0 o 0.0). Drift detectado o baja calidad.
- **Confusion Score**: Capacidad de los datos para "confundir" al discriminador (basado en Accuracy inversamente).
- **Explicabilidad**:
    - **Feature Importance**: Qué variables permitieron al modelo distinguir los datos.
    - **SHAP Values**: Explicación detallada del impacto de cada feature.

### Uso
Este reporter se integra automáticamente en `QualityReporter` si se activa el parámetro opcional:
```python
reporter.generate_comprehensive_report(
    ...,
    report_config=ReportConfig(
        output_dir="./report_output",
        adversarial_validation=True  # Activar Discriminator
    )
)
```

## Reporter de Stream (`Stream`)
**Módulo:** `calm_data_generator.generators.stream.StreamReporter`

Diseñado para analizar flujos de datos sintéticos sin un dataset de referencia "real" directo (aunque puede comparar contra expectativas).

### `generate_report`
Genera un informe para un dataset sintético:
- **Perfilado de Datos**: Integración con YData Profiling.
- **Visualizaciones**: Gráficos de densidad y reducción de dimensionalidad.
- **Análisis por Bloques**: Puede generar informes separados para cada bloque de datos.

```python
reporter = StreamReporter()
reporter.generate_report(
    synthetic_df=stream_df,
    generator_name="StreamGen",
    report_config=ReportConfig(output_dir="./stream_report")
)
```


## Evaluación Single-Cell (scGFT)
**Módulo:** `calm_data_generator.reports.QualityReporter`

La biblioteca integra [`scgft-evaluator`](https://github.com/nasim23ea/scgft-evaluator) para proporcionar validación especializada para datos de secuenciación de ARN de célula única (scRNA-seq). Este método utiliza Transformadas de Fourier en Grafos (GFT) para evaluar si los datos sintéticos preservan el manifold subyacente y la estructura biológica de las células originales.

### Instalación

```bash
pip install scgft-evaluator @ git+https://github.com/nasim23ea/scgft-evaluator.git
```

O mediante `requirements.txt` (ya incluido en calm-data-generator):

```
scgft-evaluator @ git+https://github.com/nasim23ea/scgft-evaluator.git
```

### Características Clave
- **Preservación del Manifold**: Evalúa si se mantienen las relaciones célula a célula.
- **Integridad de Clusters/Poblaciones**: Métricas sobre qué tan bien las células sintéticas representan las poblaciones reales (ARI, MMD, Jaccard, Tau de Kendall).
- **Comparación DE basada en limma**: Concordancia de expresión diferencial entre real y sintético vía `limma`.
- **Integración en el Dashboard**: Genera una pestaña dedicada `scgft_report.html` en el panel HTML con tabla de resultados.

### Uso
Establezca `use_scgft=True` en su `ReportConfig` e indique la columna de tipo celular:

```python
from calm_data_generator.generators.configs import ReportConfig

reporter.generate_comprehensive_report(
    ...,
    report_config=ReportConfig(
        output_dir="./sc_report",
        use_scgft=True,
        target_column="cell_type"  # columna con las etiquetas de tipo celular
    )
)
```

El evaluador ejecuta `ScGFT_Evaluator.run_all()` comparando las dos poblaciones celulares más frecuentes e imprime una tabla de métricas cuando `verbose=True`.

> [!IMPORTANT]
> **Formato de Datos**: Este método está diseñado específicamente para datos de célula única donde las columnas representan genes y las filas representan células. **No se recomienda** para datos estándar de bulk o tabulares.

## Reporter Clínico (`Clinical`)
**Módulo:** `calm_data_generator.generators.clinical.ClinicReporter`

Una versión especializada de `StreamReporter` para datos clínicos. Hereda capacidades de reporte estándar pero está adaptado para manejar conjuntos de características clínicas y puede incluir verificaciones específicas de dominio en el futuro.

```python
reporter = ClinicReporter()
reporter.generate_report(...)
```

### JSON de Resultados del Reporte (`report_results.json`)
Cada reporte genera un archivo `report_results.json` que contiene las métricas en bruto:

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
> **Informes de Privacidad**: Las características de privacidad (métricas DCR) ahora están integradas en `QualityReporter`. Usa `privacy_check=True` al generar informes.
