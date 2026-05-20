# Documentación de Calm Data Generator

## Referencia de Librerías

Cada método de síntesis está respaldado por una o más librerías de código abierto. Las tablas siguientes muestran exactamente qué librería usa cada método, con enlace a su documentación oficial.

### Métodos de Síntesis → Librería

| Método | Librería Principal | Descripción | Docs |
|--------|-------------------|-------------|------|
| `cart` | [scikit-learn](https://scikit-learn.org/) | Especificación Completamente Condicional con Árboles de Decisión | [sklearn.tree](https://scikit-learn.org/stable/modules/tree.html) |
| `rf` | [scikit-learn](https://scikit-learn.org/) | FCS con Random Forests | [sklearn.ensemble](https://scikit-learn.org/stable/modules/ensemble.html) |
| `lgbm` | [LightGBM](https://lightgbm.readthedocs.io/) | FCS con gradient boosting; maneja categóricas nativamente | [docs lgbm](https://lightgbm.readthedocs.io/en/stable/) |
| `xgboost` | [XGBoost](https://xgboost.readthedocs.io/) | FCS con extreme gradient boosting | [docs xgb](https://xgboost.readthedocs.io/en/stable/) |
| `ctgan` | [Synthcity](https://github.com/vanderschaarlab/synthcity) | GAN Condicional para datos tabulares | [synthcity](https://github.com/vanderschaarlab/synthcity) |
| `tvae` | [Synthcity](https://github.com/vanderschaarlab/synthcity) | Autoencoder Variacional para datos tabulares | [synthcity](https://github.com/vanderschaarlab/synthcity) |
| `ddpm` | [Synthcity](https://github.com/vanderschaarlab/synthcity) | Modelo de Difusión Probabilístico (Denoising DDPM) | [synthcity](https://github.com/vanderschaarlab/synthcity) |
| `diffusion` | [PyTorch](https://pytorch.org/) | Difusión custom ligera (sin Synthcity) | [docs torch](https://pytorch.org/docs/) |
| `timegan` | [Synthcity](https://github.com/vanderschaarlab/synthcity) | GAN para series temporales | [synthcity](https://github.com/vanderschaarlab/synthcity) |
| `timevae` | [Synthcity](https://github.com/vanderschaarlab/synthcity) | VAE para series temporales | [synthcity](https://github.com/vanderschaarlab/synthcity) |
| `fflows` | [Synthcity](https://github.com/vanderschaarlab/synthcity) | Flujos normalizantes para series periódicas | [synthcity](https://github.com/vanderschaarlab/synthcity) |
| `great` | [Synthcity](https://github.com/vanderschaarlab/synthcity) | Método tabular basado en grafos | [synthcity](https://github.com/vanderschaarlab/synthcity) |
| `rtvae` | [Synthcity](https://github.com/vanderschaarlab/synthcity) | TVAE recurrente para datos secuenciales | [synthcity](https://github.com/vanderschaarlab/synthcity) |
| `bn` | [Synthcity](https://github.com/vanderschaarlab/synthcity) | Redes Bayesianas con aprendizaje de estructura | [synthcity](https://github.com/vanderschaarlab/synthcity) |
| `dpgan` | [Synthcity](https://github.com/vanderschaarlab/synthcity) | GAN con Privacidad Diferencial | [synthcity](https://github.com/vanderschaarlab/synthcity) |
| `pategan` | [Synthcity](https://github.com/vanderschaarlab/synthcity) | Private Aggregation of Teacher Ensembles GAN | [synthcity](https://github.com/vanderschaarlab/synthcity) |
| `copula` | [Copulae](https://copulae.readthedocs.io/) | Cópula Gaussiana con escalado Min-Max | [docs copulae](https://copulae.readthedocs.io/) |
| `windowed_copula` | [Copulae](https://copulae.readthedocs.io/) | Múltiples cópulas para ventanas temporales no estacionarias | [docs copulae](https://copulae.readthedocs.io/) |
| `gmm` | [scikit-learn](https://scikit-learn.org/) | Modelos de Mezcla Gaussiana | [sklearn.mixture](https://scikit-learn.org/stable/modules/mixture.html) |
| `kde` | [scikit-learn](https://scikit-learn.org/) | Estimación de Densidad por Kernel | [sklearn.neighbors](https://scikit-learn.org/stable/modules/density.html) |
| `hmm` | [hmmlearn](https://hmmlearn.readthedocs.io/) | Modelos Ocultos de Markov | [docs hmmlearn](https://hmmlearn.readthedocs.io/) |
| `smote` | [imbalanced-learn](https://imbalanced-learn.org/) | Sobremuestreo SMOTE | [docs imblearn](https://imbalanced-learn.org/stable/) |
| `adasyn` | [imbalanced-learn](https://imbalanced-learn.org/) | Muestreo adaptativo ADASYN | [docs imblearn](https://imbalanced-learn.org/stable/) |
| `scvi` | [scvi-tools](https://docs.scvi-tools.org/) | VAE para scRNA-seq | [docs scvi](https://docs.scvi-tools.org/) |
| `scanvi` | [scvi-tools](https://docs.scvi-tools.org/) | scVI semi-supervisado con etiquetas de tipo celular | [docs scvi](https://docs.scvi-tools.org/) |
| `gears` | [GEARS](https://github.com/snap-stanford/GEARS) | Predicción de perturbaciones con GNN | [docs gears](https://github.com/snap-stanford/GEARS) |
| `conditional_drift` | [Synthcity](https://github.com/vanderschaarlab/synthcity) | Generación por etapas para simular drift de distribución | [synthcity](https://github.com/vanderschaarlab/synthcity) |
| `resample` | [scikit-learn](https://scikit-learn.org/) | Bootstrap resampling | [sklearn.utils](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html) |

### Generadores → Librerías

| Generador | Librerías Utilizadas |
|-----------|---------------------|
| **RealGenerator** | scikit-learn, Synthcity, LightGBM, XGBoost, Copulae, imbalanced-learn, hmmlearn, scvi-tools, GEARS, PyTorch |
| **ClinicalDataGenerator** | NumPy, Pandas, SciPy (`stats`, `linalg`) — vía ComplexGenerator |
| **ComplexGenerator** | SciPy (`linalg.eigh`, `stats`), NumPy — motores de Cópula Gaussiana |
| **StreamGenerator** | NumPy, Pandas, DriftInjector |
| **GeneratorFactory** | [River](https://riverml.xyz/) — extra opcional `[stream]` |
| **DriftInjector** | NumPy, Pandas, QualityReporter |
| **ScenarioInjector** | NumPy, Pandas |
| **CausalEngine** | NumPy, Pandas |
| **QualityReporter** | scikit-learn, [SDMetrics](https://docs.sdv.dev/sdmetrics/), [Plotly](https://plotly.com/python/), [YData Profiling](https://ydata-profiling.ydata.ai/) |

### Dependencias Matemáticas Base

| Librería | Versión | Rol | Docs |
|----------|---------|-----|------|
| [NumPy](https://numpy.org/) | `>=1.26, <2.0` | Arrays numéricos, álgebra lineal | [docs numpy](https://numpy.org/doc/) |
| [Pandas](https://pandas.pydata.org/) | `>=2.3` | DataFrames, manipulación de datos | [docs pandas](https://pandas.pydata.org/docs/) |
| [SciPy](https://docs.scipy.org/) | `>=1.10, <1.15` | Distribuciones, reparación de matrices PSD, estadísticas | [docs scipy](https://docs.scipy.org/) |
| [PyTorch](https://pytorch.org/) | `>=2.2, <2.4` | Infraestructura de deep learning | [docs torch](https://pytorch.org/docs/) |

---

Bienvenido a la documentación completa de **Calm Data Generator**. Esta guía cubre la instalación, configuración y uso avanzado de todos los módulos.

> **Nota:** Para documentos de referencia de API específicos, ver:
> - [RealGenerator API](./REAL_GENERATOR_REFERENCE_ES.md)
> - [DriftInjector API](./DRIFT_INJECTOR_REFERENCE_ES.md)
> - [StreamGenerator API](./STREAM_GENERATOR_REFERENCE_ES.md)
> - [ClinicalGenerator API](./CLINICAL_GENERATOR_REFERENCE_ES.md)
> - [Índice API](./API_ES.md)

---

## Tabla de Contenidos

1. [Instalación](#instalación)
2. [Inicio Rápido](#inicio-rápido)
3. [Generador Real (Tabular)](#realgenerator)
4. [Generador Clínico](#clinicalgenerator)
5. [Generador de Stream](#streamgenerator)
6. [Inyector de Drift](#driftinjector)
7. [Privacidad y Anonimización](#privacidad-y-anonimización)
8. [Generadores de Bloques](#generadores-de-bloques)
9. [Informes de Calidad](#informes-de-calidad)

---

## Instalación

### Instalación Estándar
La librería está disponible en PyPI. Para una instalación estable y rápida, recomendamos usar un entorno virtual:

```bash
# 1. Crear y activar el entorno virtual
python3 -m venv venv
source venv/bin/activate

# 2. Actualizar pip, setuptools y wheel (Crucial para una instalación exitosa)
pip install --upgrade pip setuptools wheel

# 3. Instalar la librería (optimizada para velocidad)
pip install calm-data-generator
```

### Dependencias Opcionales

| Extra | Comando | Incluye |
|-------|---------|---------|
| stream | `pip install "calm-data-generator[stream]"` | River (streaming ML) |
| full | `pip install "calm-data-generator[full]"` | Todas las dependencias anteriores |

> [!NOTE]
> **Velocidad de Instalación**: En la versión 1.0.0, hemos bloqueado dependencias clave (`pydantic`, `xgboost`, `cloudpickle`) para evitar el bucle de resolución de ~40 minutos causado por los requisitos complejos de `synthcity`. La instalación ahora es mucho más rápida.

---

## Inicio Rápido

Ver [README_ES.md](../../README_ES.md) para ejemplos básicos de código.

---

## RealGenerator

**Clase:** `calm_data_generator.generators.tabular.RealGenerator`

El motor principal para generar datos sintéticos que imitan datasets tabulares reales.

### Uso Básico

```python
from calm_data_generator.generators.tabular import RealGenerator

gen = RealGenerator()
synthetic_data = gen.generate(real_data, n_samples=1000, method='lgbm')
```

### Métodos Soportados

| Método | Descripción | Caso de Uso |
|--------|-------------|-------------|
| `cart` | Árboles de Clasificación y Regresión | Iteración rápida, captura estructura básica. |
| `rf` | Random Forest | Mejor calidad que CART, más lento. |
| `copula` | Copula | Copula-based synthesis | Base installation |
| `lgbm` | LightGBM | Alta eficiencia y rendimiento para tablas grandes. |
| `ctgan` | Conditional GAN (Synthcity) | Deep learning para distribuciones complejas multi-modales. |
| `tvae` | Variational Autoencoder (Synthcity) | A menudo más rápido y robusto que GANs para datos tabulares. |
| `copula` | Gaussian Copula | Modela correlaciones multivariadas usando la librería `copulae`. |
| `diffusion` | Difusión Tabular (DDPM) | Estado del arte experimental. Lento pero alta fidelidad. |
| `scvi` | Single-Cell (Genómica) | Modelado biológico especializado para RNA-Seq (scVI/scANVI). |


### Configuración Avanzada (`**kwargs`)

Puedes pasar parámetros específicos al modelo subyacente a través de `**kwargs`.

**Para métodos de Deep Learning (CTGAN, TVAE) vía Synthcity:**
- `epochs`: Número de épocas de entrenamiento (defecto: 300).
- `batch_size`: Tamaño del lote (defecto: 500).
- `n_units_conditional`: Parámetros específicos de Synthcity.
- `cuda`: `True`/`False` para forzar uso de GPU.

**Para métodos basados en ML (LGBM):**
- `n_estimators`: Número de árboles.
- `max_depth`: Profundidad máxima.
- `balance`: `True` para reequilibrar clases antes de entrenar.
- `differentiation_factor`: Factor de separación latente (v1.2.0).
- `clipping_mode`: Estrategia de recorte (`'strict'`, `'permissive'`, `'none'`).
- `use_latent_sampling`: `True` para mayor fidelidad biológica.

---

## ClinicalGenerator

**Clase:** `calm_data_generator.generators.clinical.ClinicalDataGenerator`

Diseñado para simular datos sanitarios complejos incluyendo datos demográficos, genómicos (genes) y proteómicos (proteínas).

### Características Clave
- **Correlaciones Biológicas:** Simula dependencias realistas entre edad, género y expresión de biomarcadores.
- **Efectos de Enfermedad:** Permite inyectar señales específicas de enfermedad (ej. sobreexpresión de un gen).
- **Longitudinal:** Genera trayectorias de pacientes a lo largo del tiempo.

Ver [CLINICAL_GENERATOR_REFERENCE_ES.md](./CLINICAL_GENERATOR_REFERENCE_ES.md) para detalles completos de configuración.

---

## StreamGenerator

**Clase:** `calm_data_generator.generators.stream.StreamGenerator`

Un wrapper alrededor de la biblioteca `River` para generar flujos de datos infinitos con concept drift evolutivo.

### Flujo de Trabajo
1. Instanciar un generador de River (ej. `SEA`, `Agrawal`).
2. Pasarlo a `StreamGenerator.generate()`.
3. Aplicar drift, balanceo o inyección de fechas.

```python
from river import synth
from calm_data_generator.generators.stream import StreamGenerator

river_gen = synth.SEA()
gen = StreamGenerator()
df = gen.generate(river_gen, n_samples=5000)
```

Ver [STREAM_GENERATOR_REFERENCE_ES.md](./STREAM_GENERATOR_REFERENCE_ES.md).

---

## DriftInjector

**Clase:** `calm_data_generator.generators.drift.DriftInjector`

Permite modificar datasets existentes para introducir cambios estadísticos controlados (drift), útiles para probar sistemas de monitorización de ML.

### Tipos de Drift
- **Feature Drift:** Cambios en la distribución de las variables de entrada $P(X)$.
- **Label Drift:** Cambios en la distribución de la variable objetivo $P(y)$.
- **Concept Drift:** Cambios en la relación entre entrada y objetivo $P(y|X)$.

### Inyección Unificada

Usa `inject_drift()` para aplicar drift fácilmente a múltiples columnas sin preocuparte por sus tipos de datos.

```python
injector.inject_drift(df, columns=['salary'], drift_mode='gradual', drift_magnitude=0.5)
```

Ver [DRIFT_INJECTOR_REFERENCE_ES.md](./DRIFT_INJECTOR_REFERENCE_ES.md).

---

## Privacidad y Anonimización

> [!NOTE]
> **Módulo de Privacidad Eliminado**: El módulo `anonymizer` independiente ha sido eliminado en favor de características de privacidad integradas.

Las características de privacidad ahora están disponibles a través de:

1. **QualityReporter con métricas DCR**: Usa `privacy_check=True` para calcular métricas de Distance to Closest Record (DCR), que miden el riesgo de re-identificación.

```python
from calm_data_generator.generators.tabular import QualityReporter

reporter = QualityReporter()
reporter.generate_report(real_df, synthetic_df, privacy_check=True)
```

2. **Modelos de Privacidad Diferencial de Synthcity**: Algunos plugins de Synthcity soportan privacidad diferencial de forma nativa. Consulta la documentación de Synthcity para más detalles.

---


## Generadores de Bloques

Permiten crear datasets compuestos de múltiples partes ("bloques"), donde cada bloque puede representar un periodo de tiempo, ubicación o concepto diferente.

### Cómo Funciona

1.  **Partición**: Los datos de entrada se dividen en trozos basados en `block_column` (ej. Año, Región).
2.  **Modelado Independiente**: Se entrena un modelo generativo separado para **cada bloque**. Esto captura las propiedades estadísticas locales.
3.  **Generación**: Se generan datos sintéticos para cada bloque independientemente.
4.  **Ensamblaje**: Los bloques sintéticos se concatenan.

### Clases Soportadas

| Generador | Descripción |
|-----------|-------------|
| `RealBlockGenerator` | Divide un dataset real en bloques y aprende de cada uno. |
| `StreamBlockGenerator` | Concatena generadores de stream para simular drift sintético puro. |
| `ClinicalDataGeneratorBlock` | Genera datos clínicos multi-centro (ej. varios hospitales). |

### Ejemplo: RealBlockGenerator

```python
from calm_data_generator.generators.tabular.RealBlockGenerator import RealBlockGenerator

gen = RealBlockGenerator()

# Generar datos divididos por "Año"
synthetic_blocks = gen.generate(
    data=data,
    output_dir="./output",
    block_column="Year",
    target_col="Churn"
)
```

---

## Informes de Calidad

**Clase:** `calm_data_generator.generators.tabular.QualityReporter`

Genera informes HTML interactivos comparando los datos reales y sintéticos.

```python
from calm_data_generator.generators.tabular import QualityReporter

reporter = QualityReporter()
# Genera un reporte HTML completo incluyendo métricas ARI para separabilidad de clases
reporter.generate_report(real_df, synthetic_df, target_col='target')

# Cálculo de ARI independiente para cuantificar la mejora en separación de clases
ari_metrics = reporter.calculate_ari(real_df, synthetic_df, target_col='target')
# Devuelve: {'ari_original': 0.95, 'ari_synthetic': 0.98, 'ari_improvement': 0.03}
```

**Métricas Incluidas:**
- **Estadísticas Descriptivas:** Comparación de media, std, min, max.
- **Distribuciones:** Histogramas superpuestos.
- **Correlaciones:** Mapas de calor de Pearson/Spearman.
- **PCA/TSNE:** Visualización de la variedad de datos en 2D.
- **Privacidad:** (Opcional) Tests de riesgo de reidentificación.

## Síntesis de Series Temporales

CALM-Data-Generator ahora soporta métodos avanzados de síntesis de series temporales mediante integración con Synthcity.

### Métodos Disponibles para Series Temporales

| Método | Tipo | Mejor Para |
|--------|------|-----------|
| `timegan` | GAN | Patrones temporales complejos, secuencias multi-entidad |
| `timevae` | VAE | Series temporales regulares, entrenamiento más rápido |
| `fflows` | Normalizing Flows | Series periódicas/estacionales, más estable que TimeGAN |
| `bn` | Red Bayesiana | Datos tabulares clínicos con dependencias causales |

### Uso Básico

```python
from calm_data_generator import RealGenerator

gen = RealGenerator()

# TimeGAN para patrones complejos
synth = gen.generate(
    datos_series_temporales,
    method='timegan',
    n_samples=100,
    n_iter=1000
)

# FourierFlows - estable para series periódicas
synth = gen.generate(
    datos_series_temporales,
    method='fflows',
    n_samples=100,
    sequence_key='seq_id',
    time_key='timestamp',
    n_iter=500
)

# Red Bayesiana - para datos tabulares con dependencias causales
synth = gen.generate(
    datos_clinicos,
    method='bn',
    n_samples=500,
    target_col='diagnostico'
)
```

Para parámetros detallados y escenarios de uso, ver [REAL_GENERATOR_REFERENCE_ES.md](REAL_GENERATOR_REFERENCE_ES.md).

## Modelos de Difusión Avanzados

### DDPM vs Difusión Custom

| Característica | `diffusion` (custom) | `ddpm` (Synthcity) |
|----------------|---------------------|-------------------|
| Velocidad | ⚡ Rápido | 🐢 Más lento |
| Calidad | ⭐⭐⭐ Buena | ⭐⭐⭐⭐ Excelente |
| Arquitecturas | MLP | MLP/ResNet/TabNet |
| Caso de Uso | Prototipado | Producción |

```python
# Prototipado rápido
synth = gen.generate(data, method='diffusion', n_samples=1000)

# Calidad de producción
synth = gen.generate(
    data,
    method='ddpm',
    n_samples=1000,
    model_type='resnet',
    scheduler='cosine'
)
```

---

## Presets

Los presets son configuraciones de generador listas para usar que encapsulan selección de método, hiperparámetros y reportes para los escenarios más comunes. Cada preset expone una única llamada `.generate()`.

Todos los presets comparten estos parámetros de constructor:

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `random_state` | `42` | Semilla aleatoria |
| `verbose` | `True` | Mostrar mensajes de progreso |
| `fast_dev_run` | `False` | Iteraciones mínimas — para testing de pipelines |

### Categorías de Presets

| Categoría | Preset | Método | Configuración Clave |
|-----------|--------|--------|---------------------|
| **Velocidad** | `FastPreset` | LightGBM | 10 iteraciones, reenvía kwargs |
| **Velocidad** | `FastPrototypePreset` | LightGBM | 10 iteraciones fijas, sin kwargs |
| **Calidad** | `HighFidelityPreset` | CTGAN | 1000 épocas, batch 250, validación adversarial |
| **Calidad** | `DiffusionPreset` | TabDDPM | 1000 pasos de difusión |
| **Calidad** | `CopulaPreset` | Cópula Gaussiana | Línea base estadística rápida |
| **Calidad** | `DataQualityAuditPreset` | TVAE | 300 épocas, reporte completo forzado |
| **Distribución** | `ImbalancedGeneratorPreset` | CTGAN | Proporción minoría/mayoría personalizada |
| **Distribución** | `BalancedDataGeneratorPreset` | SMOTE | Sobremuestreo para balancear |
| **Series Temp.** | `TimeSeriesPreset` | TimeGAN/TimeVAE/FourierFlows | 500 épocas, modelos temporales |
| **Series Temp.** | `SeasonalTimeSeriesPreset` | TimeGAN + ScenarioInjector | Estacionalidad sinusoidal |
| **Drift** | `DriftScenarioPreset` | CTGAN + DriftConfig | Prueba de estrés de drift |
| **Drift** | `GradualDriftPreset` | CTGAN + drift lineal | Drift lineal lento |
| **Drift** | `ConceptDriftPreset` | CTGAN + concept drift | Cambios en P(y\|x) |
| **Drift** | `ScenarioInjectorPreset` | ScenarioInjector | Transformar datos existentes |
| **Clínico** | `LongitudinalHealthPreset` | ClinicalDataGenerator | Pacientes multi-visita |
| **Clínico** | `RareDiseasePreset` | ClinicalDataGenerator | Prevalencia 1% de enfermedad |
| **Clínico** | `OmicsIntegrationPreset` | ClinicalDataGenerator | Clínico + genes + proteínas |
| **Clínico** | `SingleCellQualityPreset` | scVI | 400 épocas, n_latent=10 |

### Uso

```python
from calm_data_generator.presets import FastPreset, HighFidelityPreset, ImbalancedGeneratorPreset

# Generación rápida
preset = FastPreset(random_state=42)
synthetic_df = preset.generate(data=real_df, n_samples=1000)

# Datos de producción de alta fidelidad
preset = HighFidelityPreset(random_state=42)
synthetic_df = preset.generate(data=real_df, n_samples=5000)

# Dataset desbalanceado (5% minoría)
preset = ImbalancedGeneratorPreset(random_state=42)
synthetic_df = preset.generate(
    data=real_df, n_samples=2000,
    target_col="etiqueta", imbalance_ratio=0.05
)
```

Referencia completa de parámetros por preset: [PRESETS_REFERENCE_ES.md](PRESETS_REFERENCE_ES.md)
