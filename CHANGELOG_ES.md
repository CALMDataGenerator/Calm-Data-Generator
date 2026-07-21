# Changelog

Todos los cambios notables de CALM-Data-Generator están documentados aquí.

---

## [2.3.1] — 2026-07-21

### Documentación

- **Autoría y atribución documentadas en los ficheros del proyecto** (con independencia de
  quién administre el repositorio de GitHub):
  - `pyproject.toml`: autor original en `authors`, mantenedor actual y contacto del
    proyecto en el nuevo campo `maintainers`.
  - `CITATION.cff` (nuevo): metadatos de cita para que la librería se cite correctamente en
    trabajos académicos — GitHub lo expone mediante el botón "Cite this repository".
  - `AUTHORS` (nuevo): atribución completa, incluyendo el crédito al autor de
    [scGFT Evaluator](https://github.com/nasim23ea/scgft-evaluator) por la evaluación
    single-cell integrada mediante `use_scgft`.
  - `README.md` / `README_ES.md`: sección de autoría y email de contacto del proyecto.
- URLs del repositorio actualizadas al nombre actual (`Calm-Data-Generator`), que había
  cambiado desde la última vez que se fijaron.

---

## [2.3.0] — 2026-07-15

### Nuevas funcionalidades

- **`RealGenerator.fit(data)` / `.sample(n_samples)`**: API estilo sklearn. `fit()` entrena el
  modelo una sola vez (sin escribir reporte ni dataset a disco); `sample()` genera filas
  sintéticas del modelo ya entrenado tantas veces como haga falta, sin reentrenar. Es una
  envoltura fina sobre la capacidad ya existente de reutilizar el modelo vía
  `generate(data=None, n_samples=N)` — sin lógica de síntesis nueva. Soporta encadenado:
  `RealGenerator().fit(df).sample(1000)`.
- **`QualityReporter.evaluate(real_df, synthetic_df, target_column=None)`**: comprobación de
  fidelidad ligera, en memoria — scores de calidad SDMetrics, tests de similitud estadística
  (KS/MMD/Wasserstein) y TSTR (si se da `target_column`). No escribe nada a disco, a diferencia
  de `generate_comprehensive_report()`.
- **`generate_comprehensive_report()` ahora devuelve el dict de resultados** en vez de `None` —
  el mismo dict que se escribe en `report_results.json` (scores de calidad, tests estadísticos,
  TSTR, privacidad, ARI, etc.) ya está disponible directamente para quien la llama.
- **Distancia de Wasserstein** añadida por columna numérica en los tests de similitud
  estadística, junto al test KS, el test de Levene y el MMD ya existentes.
- **NNDR (Nearest Neighbor Distance Ratio)** añadido a las métricas de privacidad, junto al
  DCR. Extiende la comprobación de distancia al registro más cercano con un ratio invariante a
  escala (distancia al más cercano / distancia al 2º más cercano de un registro real) — señal
  más fuerte de riesgo de reidentificación.
- **Riesgo de Singling-Out** vía [anonymeter](https://github.com/statice/anonymeter) (MIT),
  nueva dependencia **opcional** (`pip install calm-data-generator[privacy]`). Estima el riesgo
  de que un atacante aísle un registro real concreto a partir de combinaciones de atributos
  aprendidas del sintético. Se activa automáticamente con `privacy_check=True`; degrada a `None`
  con un log informativo (no un error) si `anonymeter` no está instalado. Su límite interno de
  reintentos está acotado para que una sola evaluación sea rápida incluso con datos de baja
  cardinalidad o muy duplicados (el propio default de anonymeter puede colgarse minutos en ese
  caso).

### Corrección de errores

- **`RealGenerator.generate()` no tenía docstring en runtime**: el docstring estaba escrito
  *después* de código ejecutable, convirtiéndose en un string muerto en vez de un docstring real
  (`generate.__doc__` era `None`). Mismo problema corregido en `generate_custom` y en los 18
  métodos `.generate()` de los presets.
- **`from calm_data_generator import presets` lanzaba `RecursionError`**: el mapa de imports
  perezosos tenía una entrada auto-referente para el subpaquete `presets`. Ahora se importa
  directamente.
- **`generate_comprehensive_report(report_config=..., discriminator=True/tstr=True/...)`
  descartaba en silencio los flags booleanos explícitos** cuando también se pasaba
  `report_config`. Ahora usa semántica OR para `privacy_check`/`discriminator`/`tstr`/`spearman`:
  un flag pedido explícitamente siempre gana, incluso junto a un `report_config` que lo deja en
  su valor por defecto. `tstr`/`spearman` son ahora campos propios de `ReportConfig` (antes solo
  se podían fijar como argumentos sueltos del método).
- **Excepción silenciosa en la generación scVI/scANVI con gene-label**
  (`_synth_latent.py`): un `except Exception: pass` sin registro alrededor de la construcción
  del tensor de etiquetas para `dispersion="gene-label"` ocultaba cualquier fallo y caía a
  generación incondicional sin ninguna visibilidad. Ahora registra un warning con el motivo.

### Experiencia de desarrollo

- **`QualityReporter.generate_custom` y los 18 métodos `.generate()` de los presets tienen ahora
  docstrings correctos** (antes indocumentados a nivel de `help()`/IDE).
- **Los parámetros legacy de `generate()`** (`custom_distribution` alias singular,
  `date_start`/`date_every`/`date_step`) ahora registran un warning apuntando al equivalente
  moderno (`custom_distributions`, `date_config=DateConfig(...)`) cuando se usan. Nota: un
  `warnings.warn()` normal habría quedado silenciado — `RealGenerator.py` suprime
  `DeprecationWarning`/`UserWarning`/`FutureWarning` a nivel de módulo — así que se usa
  `logger.warning()` en su lugar. El docstring de `generate()` también se reescribió, agrupando
  sus ~25 parámetros por bloque (core, drift & dynamics, reporting, distribuciones, alias
  legacy) en vez de una lista plana.
- **Eliminado el árbol duplicado de la raíz del paquete**: `generators/`, `reports/`,
  `presets/`, `docs/`, `tutorials/`, `cli.py`, `logger.py`, `__init__.py` existían tanto en la
  raíz del repo como dentro de `calm_data_generator/` (el paquete instalable real) como copia
  exacta y sin trackear — editar la copia de la raíz no tenía ningún efecto sobre el paquete
  instalado. Eliminado.
- **Eliminado el módulo `QualityReporter` duplicado**: `generators/tabular/QualityReporter.py`
  era un shim de re-exportación de 5 líneas que colisionaba con la clase homónima ya
  re-exportada por `generators/tabular/__init__.py`. La ruta de importación canónica es ahora
  únicamente `calm_data_generator.reports.QualityReporter.QualityReporter`.
- **Los reportes HTML son ahora totalmente autónomos**: 5 páginas de reporte
  (`statistical_tests.html`, `spearman_heatmaps.html`, `qq_plots.html`, `tstr_report.html`, y 2
  fragmentos en `DiscriminatorReporter`) cargaban Plotly desde un CDN y se renderizaban en
  blanco sin conexión a internet. Ahora se empaquetan en línea, igual que el resto del
  dashboard.
- **El CI ahora avisa (sin bloquear) cuando un PR edita un lado de un par de docs EN/ES** (p.
  ej. `README.md`/`README_ES.md`) sin el otro, vía un nuevo job `check-doc-pairs`.

---

## [2.2.1] — 2026-04-29

### Nuevas Funcionalidades

- **`RealGenerator.encode_to_latent()` / `decode_from_latent()`**: Ida y vuelta de un dataset por el espacio latente del modelo entrenado para `tvae`, `rtvae`, `scvi` y `scanvi`. Gestiona internamente el preprocesado (TabularEncoder, tensores de condicionamiento, tamaño de librería de SCVI) para que los análisis de drift externos no tengan que reimplementar el pipeline de encode/decode por cada método.

### Corrección de Errores

- **`QualityReporter` — detección de duplicados cruzados**: corregido un fallo/cero silencioso cuando las columnas de real y sintético tenían dtypes distintos; ahora el merge se restringe a columnas compartidas y castea los dtypes del sintético antes de comparar.
- **`RealBlockGenerator.generate_block()`**: `model_params` ahora se propaga como `**kwargs` a la llamada subyacente `RealGenerator.generate()` en vez de pasarse como argumento posicional sin usar, así los parámetros de modelo por bloque sí tienen efecto.
- **`ExternalReporter`**: el flag `minimal` ahora se pasa correctamente a `ProfileReport` de `ydata-profiling` (antes se aceptaba pero se ignoraba en silencio).

### Dependencias

- Pin de `scgft-evaluator` cambiado de URL git (`git+https://github.com/nasim23ea/scgft-evaluator.git`) a rango de versión de PyPI (`>=0.1.0,<0.2.0`).

### Experiencia de Desarrollo

- Limpieza de ordenación de imports y espacios en blanco (ruff) en `RealBlockGenerator.py`, `QualityReporter.py`, `ExternalReporter.py`.

---

## [2.2.0] — 2026-04-23

### Nuevas Funcionalidades

- **`RealGenerator.generate()` — parámetro `cond`**: Pasa un array/DataFrame de condicionamiento directamente a los plugins de Synthcity (`tvae`, `rtvae`, `ctgan`, `dpgan`, `pategan`). Se propaga tanto a `fit()` como a `generate()` para que el modelo entrene y muestree bajo el mismo condicional.
- **`RealGenerator.generate()` — retry loop en `constraints`**: Cuando el filtrado por restricciones elimina filas, el generador regenera automáticamente (`needed × 2` muestras, hasta 5 reintentos) para devolver siempre `n_samples` filas. Antes el déficit se registraba en el log pero no se recuperaba.
- **Reproducibilidad completa en todos los métodos**: `random_state` ahora se propaga consistentemente en todos los lugares donde faltaba:
  - `_get_synthesizer()` inyecta `random_state` en los constructores de todos los plugins de Synthcity al inicializar.
  - Las llamadas a `generate()` de `tvae`, `rtvae`, `ctgan`, `bn`, `dpgan`, `pategan`, `ddpm`, `timegan`, `timevae`, `fflows`, `conditional_drift` y el path de fallback de diferenciación latente ahora pasan `random_state`.
  - Todas las llamadas a `np.random.*` sin semilla en `scvi`, `scanvi`, `gears`, `fcs_generic` y `privatize()` reemplazadas por `np.random.default_rng(self.random_state)`.

### Documentación

- **scGFT Evaluator** — añadido a `Agradecimientos y Créditos` en `README.md` y `README_ES.md`.
- **scGFT Evaluator** — añadidos ejemplos de uso y referencias cruzadas en las secciones `Single-Cell / Gene Expression` e `Informe de Calidad` de ambos READMEs.
- **`PRESETS_REFERENCE.md` / `PRESETS_REFERENCE_ES.md`** — añadido ejemplo de validación scGFT tras `SingleCellQualityPreset`.
- **`REAL_GENERATOR_REFERENCE.md` / `REAL_GENERATOR_REFERENCE_ES.md`** — añadida sección de validación scGFT tras el workflow `to_anndata`.
- **`README_ES.md`** — añadida fila `scgft-evaluator` en la tabla del ecosistema Single-Cell / Omics.

### Dependencias

- `scgft-evaluator>=0.1.0,<0.2.0` promovido de dependencia opcional (solo `requirements.txt`) a dependencia obligatoria en `pyproject.toml`.

### Experiencia de Desarrollo

- Añadido `.pre-commit-config.yaml` con hooks de `ruff` (linter + ordenación de imports), `trailing-whitespace`, `end-of-file-fixer` y `check-merge-conflict`.
- Añadida configuración `[tool.ruff]` en `pyproject.toml` (`line-length = 120`, selecciona `E`, `F`, `W`, `I`, ignora `E501` y `F401`).

---

## [2.1.0] — 2026-04-17

### Rendimiento

- **Clinic.py** — bucle de transición de grupos vectorizado con máscaras booleanas; elimina llamadas `.loc` O(n) por paciente
- **DriftInjector** — eliminado `df.copy()` redundante en `inject_composite_drift`; cada método drift ya copia internamente
- **RealGenerator — codificación FCS** — reemplazado `copy()` + mutación in-place por `assign()` por iteración; evita dos copias completas del DataFrame por (iteración × columna)
- **DriftInjector — `_apply_cat_drift`** — vectorizado por grupo de valor con `rng.choice(size=n)`; O(cats) llamadas en lugar de O(n)
- **RealGenerator — privatización** — reemplazado closure `apply(randomize)` por máscara numpy + `np.random.choice` agrupado; elimina overhead Python por elemento
- **QualityReporter** — evita `df.copy()` incondicional cuando no hay resampling; copia diferida al bloque condicional
- **persistence_models** — reemplazado `copy()` + mutación in-place por `assign()`; skip de copia para modelos nativos de categorías (LGBM/XGB)
- **RealGenerator** — cacheo del resultado de `select_dtypes`; `encoding_info` colapsado a dict comprehension

### Correcciones de Bugs

- **FCSModel RNG** — reemplazadas llamadas globales `np.random` por `numpy.random.default_rng(random_state)` con semilla para reproducibilidad
- **Integración ScGFT** — corregida firma de `ScGFT_Evaluator.run_all()` (`genes_top`, `col_grupo`, `grupo_a`, `grupo_b`); eliminado parámetro inválido `label_col`
- **RealGenerator** — limpieza de imports duplicados y no utilizados

### Dependencias

- Añadidos `statsmodels>=0.14.0,<0.15.0` y `tqdm>=4.60.0,<5.0.0` (faltaban en requirements)
- Migrado scGFT de módulo vendorizado `scGFT_Evaluator.py` a paquete instalable `scgft-evaluator`

---

## [2.0.0] — 2026-03-27

### Nuevas Funcionalidades

#### ComplexGenerator — Capa Matemática Abstracta
- Nueva clase abstracta `ComplexGenerator(BaseGenerator)` como capa intermedia entre `BaseGenerator` y los generadores de dominio.
- Proporciona tres motores matemáticos reutilizables sin duplicar código:
  - `_generate_correlated_module(n, marginals, sigma)` — Cópula Gaussiana (incondicional) con reparación de matrices PSD vía `scipy.linalg.eigh`.
  - `_generate_conditional_data(n, cond_data, cond_marginals, tgt_marginals, cov)` — Cópula Gaussiana condicional con RQR para marginales discretas.
  - `apply_stochastic_effects(df, entity_ids, effect_config)` — 7 tipos de efectos estocásticos + alias `simple_additive_shift`.
- `ClinicalDataGenerator` ahora hereda de `ComplexGenerator` en lugar de `BaseGenerator`.

#### Dinámica Causal (DriftInjector + ScenarioInjector)
- **`CausalEngine`** — propagación causal basada en DAG (`generators/dynamics/CausalEngine.py`):
  - Orden topológico via algoritmo de Kahn con detección de ciclos.
  - Propagación diferencial: `delta_hijo = f(v_padre + delta) - f(v_padre)`.
  - Funciones de transferencia: `linear`, `exponential`, `power`, `polynomial`, o cualquier callable.
- **`DriftInjector.inject_functional_drift()`** — magnitud del drift por fila = f(valor actual de `driver_col`). Soporta modo aditivo y multiplicativo.
- **`DriftInjector.inject_causal_cascade()`** — aplica una cascada `CausalEngine` con el sistema de selección de filas e informes del `DriftInjector`.
- **`ScenarioInjector` tipo de evolución `driven_by`** — delta de una feature por fila = f(valor de otra columna). Desacoplado del índice temporal.
- **`generators/utils/propagation.py`** — módulo de utilidades compartidas:
  - `propagate_numeric_drift(df, rows, driver_col, delta_driver, correlations)` — extraído de `DriftInjector` y `ScenarioInjector` para eliminar duplicación.
  - `apply_func(func_name, params, x)` — evalúa funciones de transferencia por nombre sobre arrays.
- **`EvolutionFeatureConfig`** extendido con los campos `driver_col`, `func`, `func_params`.

### Correcciones de Bugs

- **RealGenerator — columnas datetime en CART/RF**: `_synthesize_fcs_generic` ahora convierte columnas datetime a `int64` antes del bucle FCS, corrigiendo errores `DType DateTime64DType cannot be promoted`.
- **RealGenerator — dispatch del método `bn`**: `elif method == "bayesian_network"` extendido a `elif method in ("bayesian_network", "bn")`, corrigiendo la síntesis que devolvía `None` al usar `method="bn"`.
- **RealGenerator — API Synthcity en `conditional_drift`**: eliminado el parámetro inválido `cond=` de `syn.generate()` — TVAE/CTGAN son generadores incondicionales y no soportan condicionamiento en inferencia.
- **RealGenerator — array 1D en `windowed_copula`**: `copula.random(n)` puede devolver un array 1D cuando `n=1`; ahora se redimensiona a 2D antes de `scaler.inverse_transform()`.
- **`ClinicalDataGenerator` — dos llamadas restantes a `_generate_module_data`**: actualizadas a `_generate_correlated_module` tras la refactorización de ComplexGenerator.
- **`test_disease_effects_fix.py`**: convertido de script a nivel de módulo a función pytest correcta.

### Tests

- Todos los archivos de test con `unittest.TestCase` convertidos a pytest puro (9 archivos, 41 tests).
- Nuevo `tests/test_causal_engine.py` — 10 tests cubriendo propagación DAG, detección de ciclos, filas parciales y orden topológico.
- Nuevo `tests/test_functional_drift.py` — 8 tests cubriendo drift funcional, cascada causal, `driven_by` y `propagate_numeric_drift`.
- Suite completa: **186 passed, 8 skipped, 0 failed**.

### Documentación

- Nuevo `CAUSAL_ENGINE_REFERENCE.md` / `_ES.md` — referencia completa del DAG con ejemplos de IoT, Finanzas y Clínica.
- Nuevo `COMPLEX_GENERATOR_REFERENCE.md` / `_ES.md` — referencia de los tres motores matemáticos.
- Nueva sección `Referencia de Librerías` en `DOCUMENTATION.md` / `_ES.md` — mapea cada método de síntesis a su librería subyacente con enlaces a la documentación oficial.
- Actualizado `DRIFT_INJECTOR_REFERENCE.md` / `_ES.md` — añadidos `inject_functional_drift` e `inject_causal_cascade`.
- Actualizado `SCENARIO_INJECTOR_REFERENCE.md` / `_ES.md` — añadido tipo de evolución `driven_by`.
- Actualizado `API.md` / `API_ES.md` — añadidas secciones `generators.dynamics` (CausalEngine) y `generators.utils`.
- Actualizado `CLINICAL_GENERATOR_REFERENCE.md` / `_ES.md` — herencia de ComplexGenerator, advertencia `additive_shift` para proteínas.
- Actualizado `README.md` / `README_ES.md` — "Tecnologías Principales" expandido con tablas completas de librerías y enlaces; añadida sección Evolución de Escenarios.
- Tutoriales actualizados: `advanced_drifts.py` y `scenario_injector.py` incluyen ejemplos de `inject_functional_drift`, `inject_causal_cascade` y `driven_by`.

---

## [1.2.0] — Versión anterior

- Parámetro `differentiation_factor` para TVAE y scVI (aumenta la separabilidad de clases en el espacio latente).
- Parámetro `clipping_mode`: `'strict'`, `'permissive'` o `'none'`.
- `use_latent_sampling` para scVI.
- `_apply_postprocess_distribution` para remuestreo inteligente respetando la distribución de clases.
- Método de síntesis Windowed Copula.
- Método de síntesis Conditional Drift.
- Métodos de Privacidad Diferencial: DPGAN, PATEGAN.
