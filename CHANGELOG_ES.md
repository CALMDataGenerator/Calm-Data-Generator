# Changelog

Todos los cambios notables de CALM-Data-Generator están documentados aquí.

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
