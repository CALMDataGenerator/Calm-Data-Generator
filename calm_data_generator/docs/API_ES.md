# Documentación API de CalmGenerador

## Descripción General de Módulos

### generators.tabular - Síntesis de Datos Reales

```python
from calm_data_generator.generators.tabular import RealGenerator, QualityReporter
```

**RealGenerator** - Genera datos sintéticos a partir de datasets reales

| Método | Descripción |
|--------|-------------|
| `cart` | Síntesis iterativa basada en CART |
| `rf` | Síntesis con Random Forest |
| `lgbm` | Síntesis con LightGBM |
| `ctgan` | CTGAN (deep learning) |
| `tvae` | TVAE (autoencoder variacional) |
| `bn` | Red Bayesiana (estructura causal) |
| `smote` | Sobremuestreo SMOTE |
| `adasyn` | Muestreo adaptativo ADASYN |
| `diffusion` | Difusión Tabular (DDPM) |
| `ddpm` | Synthcity TabDDPM (avanzado) |
| `timegan` | TimeGAN (series temporales) |
| `timevae` | TimeVAE (series temporales) |
| `fflows` | FourierFlows (series periódicas) |
| `scvi` | scVI (Single-Cell VI) |

**Parámetros destacados:**
- `differentiation_factor` (float): Aumenta la separación de clases en el espacio latente (solo TVAE/scVI).
- `clipping_mode` (str): `'strict'`, `'permissive'`, o `'none'` para manejar los rangos de salida.
- `use_latent_sampling` (bool): Para scVI, muestrea desde el espacio latente de datos reales.

**Validación y Procesamiento Avanzado:**
- `_apply_postprocess_distribution(self, synthetic_df, class_counts, target_col, ...)`: Remuestrea filas inteligentemente en un DataFrame sintético para cumplir la distribución de clases objetivo preservando las correlaciones. Emite warnings si faltan clases. Usado automáticamente por muchos modelos cuando se define `custom_distributions`.




---

### generators.complex - Capa Matematica Abstracta

```python
from calm_data_generator.generators.complex.ComplexGenerator import ComplexGenerator
```

**ComplexGenerator** - Base abstracta para cualquier generador que necesite sintesis correlacionada o condicional. Hereda de esta clase en lugar de `BaseGenerator` cuando necesitas los tres motores integrados:

| Motor | Metodo | Descripcion |
|-------|--------|-------------|
| Copula Gaussiana (incondicional) | `_generate_correlated_module(n, marginals, sigma)` | Genera muestras correlacionadas con marginales arbitrarias |
| Copula Gaussiana (condicional) | `_generate_conditional_data(n, cond_data, cond_marginals, tgt_marginals, cov)` | Genera variables objetivo condicionadas a datos observados |
| Efectos estocasticos | `apply_stochastic_effects(df, entity_ids, effect_config)` | Aplica uno de 7 tipos de efecto en-lugar |

Consulta [COMPLEX_GENERATOR_REFERENCE_ES.md](COMPLEX_GENERATOR_REFERENCE_ES.md) para documentacion completa.

---

### generators.clinical - Datos Clinicos

```python
from calm_data_generator.generators.clinical import ClinicalDataGenerator
```

`ClinicalDataGenerator` hereda de `ComplexGenerator` y usa los tres motores internamente.

**Metodos:**
- `generate()` - Genera demografia + omicas
- `generate_longitudinal_data()` - Datos de paciente multi-visita

---

### generators.stream - Basado en Stream

```python
from calm_data_generator.generators.stream import StreamGenerator
```

**Características:**
- Compatible con librería River
- Generación balanceada
- SMOTE post-hoc
- Generación de secuencias

---

### generators.drift - Inyección de Drift

```python
from calm_data_generator.generators.drift import DriftInjector
```

**Tipos de Drift:**
- `inject_drift()` **(Unificado)**
- `inject_feature_drift_gradual()`
- `inject_feature_drift_abrupt()`
- `inject_feature_drift_recurrent()`
- `inject_label_drift_gradual()`
- `inject_label_drift_abrupt()`
- `inject_label_drift_incremental()`
- `inject_concept_drift()`
- `inject_conditional_drift()`
- `inject_outliers_global()`
- `inject_new_category_drift()`
- `inject_correlation_matrix_drift()`
- `inject_binary_probabilistic_drift()`
- `inject_multiple_types_of_drift()`
- `inject_functional_drift()` — magnitud del drift como f(driver_col) por fila (Pilar 5)
- `inject_causal_cascade()` — propagación causal no lineal basada en DAG (Pilar 5)

---

### generators.dynamics - Evolución de Escenarios y Motor Causal

```python
from calm_data_generator.generators.dynamics import ScenarioInjector
from calm_data_generator.generators.dynamics import CausalEngine
```

**ScenarioInjector** — evoluciona features en el tiempo. Tipos de evolución: `linear` (`trend`), `exponential_growth`, `exponential_decay` (`decay`), `cycle` (`sinusoidal`, `seasonal`), `sigmoid`, `step`, `noise`, `random_walk`, `driven_by`.

**CausalEngine** — propagación causal basada en DAG. Define relaciones padre→hijo con funciones de transferencia y propaga perturbaciones por el grafo. Ver [CAUSAL_ENGINE_REFERENCE_ES.md](CAUSAL_ENGINE_REFERENCE_ES.md).

---

### generators.utils - Utilidades Compartidas

```python
from calm_data_generator.generators.utils.propagation import propagate_numeric_drift, apply_func
```

- `propagate_numeric_drift(df, rows, driver_col, delta, correlations)` — propagación de delta basada en correlaciones, usado por DriftInjector y ScenarioInjector
- `apply_func(func_name, params, x)` — evalúa funciones de transferencia por nombre (`linear`, `exponential`, `power`, `polynomial`, callable)

---

### privacy - Transformaciones de Privacidad (Integrado)

Las funciones de privacidad están integradas en el `QualityReporter`. Puedes evaluar calidad y privacidad usando:

```python
# Reporte Completo de Calidad (incluyendo métricas ARI para separabilidad)
reporter.generate_comprehensive_report(..., privacy_check=True)

# Cálculo de ARI independiente
ari_scores = reporter.calculate_ari(real_df, synthetic_df, target_col="label")
```

Para garantías de privacidad diferencial en la generación, usa los métodos `dpgan` o `pategan` via `RealGenerator.generate(method="dpgan", ...)` o `RealGenerator.generate(method="pategan", ...)`.

## Instalación

```bash
# Básica
pip install calm-data-generator

# Stream (River)
pip install "calm-data-generator[stream]"

# Completa
pip install "calm-data-generator[full]"
```
