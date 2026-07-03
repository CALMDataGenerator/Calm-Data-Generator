# Architecture

Mapa de módulos y flujo de datos de `calm_data_generator`. Complementa el
diagrama `docs/assets/architecture.png` con texto navegable — léelo antes de
tocar código si eres nuevo en el proyecto.

## Capas

```
BaseGenerator (ABC)                      calm_data_generator/generators/base.py
  └── ComplexGenerator                   generators/complex/ComplexGenerator.py
        └── ClinicalDataGenerator        generators/clinical/Clinic.py
  RealGenerator                          generators/tabular/RealGenerator.py
        └── RealBlockGenerator           generators/tabular/RealBlockGenerator.py
  StreamGenerator                        generators/stream/StreamGenerator.py
        └── StreamBlockGenerator         generators/stream/StreamBlockGenerator.py
```

- **`BaseGenerator`**: interfaz común (seed, logging, auto_report) que heredan
  todos los generadores.
- **`ComplexGenerator`**: capa intermedia abstracta, no se usa directamente.
  Da 3 motores reutilizables (Gaussian Copula incondicional, Gaussian Copula
  condicional, efectos estocásticos por entidad) para que generadores de
  dominio (hoy solo `ClinicalDataGenerator`; pensado para futuros dominios
  tipo Finance/IoT) no dupliquen matemática.
- **`RealGenerator`**: sintetiza datos a partir de un dataset real usando
  modelos de deep learning vía Synthcity (TVAE, RTVAE, CTGAN, DDPM, TimeGAN,
  BN, DPGAN, PATEGAN...) y SCVI/SCANVI para datos single-cell. Ver sección
  `generators/tabular/` más abajo para el desglose en mixins.
- **`StreamGenerator`**: generación online con River (evolución fila a fila,
  no todo el dataset a la vez). Usa los mismos `DriftInjector` /
  `ScenarioInjector` que el resto.

## Flujo de datos típico

```
datos reales (DataFrame)
      │
      ▼
RealGenerator.generate() ──► modelo entrenado (Synthcity/SCVI) ──► DataFrame sintético
      │
      ▼ (opcional)
DriftInjector.inject_*_drift()  ──► DataFrame con drift controlado
      │
      ▼ (opcional, orquesta secuencias de eventos)
ScenarioInjector / CausalEngine ──► DataFrame con cascada causal (DAG, Kahn topological sort)
      │
      ▼
QualityReporter.generate_report() ──► HTML/JSON con métricas real-vs-sintético
```

`ClinicalDataGenerator` sigue el mismo flujo pero generando desde cero (no
parte de datos reales): usa los motores de `ComplexGenerator` para
demográficos/expresión génica/proteínas correlacionados, luego opcionalmente
pasa por `DriftInjector` y `ScenarioInjector` igual que `RealGenerator`.

## `generators/drift/`

`DriftInjector` es una clase "gorda" compuesta de mixins, uno por familia de
drift — así cada tipo de drift vive en su propio archivo en vez de un único
archivo de 3000+ líneas:

- `_feature_drift.py` — ruido, shift, scale sobre columnas.
- `_label_drift.py` — label flipping, label shift, nuevas categorías.
- `_structural_drift.py` — deriva de la matriz de correlación.
- `_data_quality_drift.py` — missing values, duplicados.
- `_drift_utils.py` — perfiles de ventana compartidos (sigmoid/linear/cosine
  para drift gradual/abrupto/recurrente).

## `generators/tabular/`

`RealGenerator` combina 7 mixins por herencia múltiple (evita una única
clase de 5000+ líneas):

- `_synth_tabular.py` — métodos FCS/árboles y deep learning tabular (cart,
  rf, lgbm, xgboost, ctgan, tvae, rtvae, great, ddpm, bn, gmm, kde, copula,
  resample, smote, adasyn...).
- `_synth_scvi.py` — single-cell (scVI, scANVI, GEARS).
- `_synth_timeseries.py` — series temporales (timegan, timevae, fflows).
- `_synth_latent.py` — `encode_to_latent`/`decode_from_latent`, embeddings,
  soporte AnnData.
- `_synth_privacy.py` — métodos con privacidad diferencial (dpgan, pategan).
- `_synth_utils.py` — utilidades compartidas (detección de dispositivo,
  validación de método, tensores condicionales, inyección de fechas,
  `_synthesize_fcs_generic` — el helper genérico de FCS usado por
  cart/rf/lgbm/xgboost, vive aquí y no en `_synth_scvi.py` porque no tiene
  nada que ver con single-cell).
- `_generate_pipeline.py` (`_GeneratePipelineMixin`) — la lógica interna de
  `RealGenerator.generate()`. El método público en sí se quedó en ~110
  líneas (antes 740); cada paso se llama en el mismo orden en que antes
  ejecutaba inline, sin cambios de comportamiento:
  - `_validate_generate_call` — valida epsilon/método DP, nombre de método.
  - `_resolve_generate_config` — resuelve `ReportConfig`, `output_dir`
    efectivo, `DateConfig` legacy.
  - `_resolve_generate_distributions` — alias `custom_distribution`, valida
    distribuciones, aplica `balance`.
  - `_dispatch_synthesis` — el switch por `method` que llama al
    `_synthesize_*` correspondiente, con sugerencia de métodos alternativos
    si falla el entrenamiento.
  - `_apply_generate_postprocess` — resampling post-generación para
    métodos sin soporte nativo de `custom_distributions`.
  - `_apply_generate_constraints` — filtra por constraints, reintenta
    generación para rellenar filas descartadas.
  - `_finalize_generate_output` — dynamics injection, date injection,
    drift injection, reporting, guardado del dataset.

## `generators/clinical/`

`ClinicalDataGenerator` (`Clinic.py`) sigue el mismo patrón mixin que
`DriftInjector`/`RealGenerator`: la clase principal solo tiene el
constructor y el orquestador `generate()`; el resto vive en mixins por
concern:

- `_demographic_mixin.py` — generación de demográficos, filtrado por
  constraints, contexto demográfico compartido por genes/proteínas.
- `_omics_params_mixin.py` — diseño de parámetros de distribución
  (actualmente sin call sites en el repo — código muerto conservado tal
  cual, ver nota en el propio archivo).
- `_gene_expression_mixin.py` / `_protein_expression_mixin.py` — síntesis de
  expresión génica y proteica, con efectos de enfermedad por subgrupo.
- `_target_omics_mixin.py` — variable objetivo `Y` como combinación lineal
  de features, y regeneración de ómicas correlacionadas para un subconjunto.
- `_longitudinal_mixin.py` — datos multi-visita y drift de transición de
  grupo/módulo entre timepoints.
- `_reporting_mixin.py` — helpers de reporting y resumen de texto.

## `generators/dynamics/`

- `ScenarioInjector` — orquesta secuencias de eventos/escenarios sobre un
  dataset (usa `seed=`, no `random_state=` — inconsistencia histórica, ver
  README "Known quirks").
- `CausalEngine` — cascada causal basada en DAG (orden topológico de Kahn):
  cambios en una columna se propagan a columnas dependientes según el grafo
  definido por el usuario.

## `reports/`

- `QualityReporter` — orquestador principal (`generate_comprehensive_report`):
  coordina resampling, TSTR, tests estadísticos, quality assessment y
  visualizaciones. El resto de métricas vive en mixins:
  - `_statistical_similarity_mixin.py` — KS, Levene, MMD.
  - `_quality_scoring_mixin.py` — SDMetrics, quality por bloque, resampling.
  - `_privacy_metrics_mixin.py` — DCR, ARI (separabilidad de clases).
  - `_single_cell_mixin.py` — evaluación scGFT.
  - `_ml_utility_mixin.py` — TSTR (Train Synthetic, Test Real).
  - `DiscriminatorReporter` — adversarial validation (¿un clasificador
    distingue real de sintético?).
  - `ExternalReporter` — wrapper sobre `ydata-profiling` para HTML.
  - `LocalIndexGenerator` — genera el índice HTML que enlaza todos los
    reports de una ejecución.
- `Visualizer` — fachada estática sobre funciones de gráficos Plotly
  agrupadas por tema en `_distribution.py`, `_correlation.py`,
  `_dimensionality.py`, `_metrics_cards.py`, `_comparison.py`,
  `_sequence.py`. Todas sin estado (`@staticmethod`), por eso son funciones
  sueltas por módulo en vez de mixins — `Visualizer.generate_x(...)` sigue
  funcionando igual para quien la llame.

## `presets/`

Cada preset es una configuración empaquetada (parámetros de generador +
drift + reporting) para un caso de uso concreto (`FastPreset`,
`HighFidelityPreset`, `TimeSeriesPreset`, `RareDiseasePreset`...). Son la
puerta de entrada recomendada para usuarios nuevos de la librería — antes de
tocar `RealGenerator`/`ClinicalDataGenerator` directamente, comprueba si ya
existe un preset que cubra el caso. Ver
`calm_data_generator/docs/PRESETS_REFERENCE.md`.

## Lazy imports

`calm_data_generator/__init__.py` y `reports/__init__.py` usan `__getattr__`
para importar perezosamente (`RealGenerator`, `QualityReporter`,
`ClinicalDataGenerator`, etc.) — evita cargar Synthcity/SCVI/River al hacer
`import calm_data_generator` si el usuario solo necesita una parte. Al añadir
una clase pública nueva, hay que registrarla en `_lazy_map`, no solo
importarla arriba del archivo.

## Dónde tocar según la tarea

| Quiero... | Miro en... |
|---|---|
| Nuevo tipo de drift | `generators/drift/_*_drift.py` + mixin correspondiente |
| Nuevo método de síntesis basado en modelo real | `generators/tabular/_synth_*.py` |
| Nueva métrica de calidad | `reports/QualityReporter.py` o nuevo Reporter en `reports/` |
| Nuevo caso de uso empaquetado | `presets/` (heredar de `presets/base.py:GeneratorPreset`) |
| Nueva variable clínica correlacionada | `generators/clinical/Clinic.py` + motores de `ComplexGenerator` |
| Orquestar secuencia de eventos causales | `generators/dynamics/CausalEngine.py` / `ScenarioInjector.py` |
