# CALM-Data-Generator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/calm-data-generator.svg)](https://pypi.org/project/calm-data-generator/)
[![Downloads](https://img.shields.io/pypi/dm/calm-data-generator)](https://pypi.org/project/calm-data-generator/)

> **¡Ya disponible en PyPI!** Instalar con: `pip install calm-data-generator`

> **[English README](README.md)**

**CALM-Data-Generator** es una biblioteca completa en Python para la generación de datos sintéticos con características avanzadas para:
- **Datos Clínicos/Médicos** - Genera demografía de pacientes, genes y proteínas realistas.
- **Síntesis Tabular** - CTGAN, TVAE, CART y más.
- **Series Temporales** - TimeGAN, TimeVAE, FourierFlows
- **Single-Cell** - scVI, GEARS (Predicción de Perturbaciones)
- **Diferenciación Latente** - Fuerza la separabilidad de clases en TVAE y scVI
- **Clipping Flexible** - Control estricto o permisivo de los rangos de datos
- **Inyección de Drift (Desviación)** - Prueba la robustez de modelos ML con drift controlado.
- **Privacidad** - Métricas DCR (Distance to Closest Record) y métodos de privacidad diferencial (DPGAN, PATEGAN).
- **Evolución de Escenarios** - Evolución de features y construcción de targets.

![Arquitectura CALM](assets/architecture.png)

![Flujo de Trabajo CALM](assets/ecosystem.png)

## Alcance y Capacidades

**Calm-Data-Generator** está optimizado para **datos tabulares estructurados**. Está diseñado para manejar:
- **Clasificación** (Binaria y Multiclase)
- **Regresión** (Variables continuas)
- **Multi-label** (Múltiples objetivos)
- **Clustering** (Preservación de agrupamientos naturales)
- **Series Temporales** (Correlaciones y patrones temporales)
- **Single-Cell / Genómica** (Datos de expresión RNA-seq)

> [!IMPORTANT]
> Esta biblioteca **NO** está diseñada para datos no estructurados como **Imágenes**, **Vídeos** o **Audio**. No incluye modelos de Visión Artificial o Procesamiento de Señales.

---

## ¿Qué hace única a esta librería?

**CALM-Data-Generator** no es solo otra herramienta de datos sintéticos, es un **ecosistema unificado** que reúne las mejores librerías de código abierto bajo una API única y consistente:

### Integración Unificada Multi-Librería
En lugar de aprender y gestionar múltiples librerías complejas por separado, CALM-Data-Generator proporciona:
- **Una sola API** para 15+ métodos de síntesis de diferentes fuentes (Synthcity, scvi-tools, GEARS, imbalanced-learn, etc.)
- **Interoperabilidad fluida** entre generadores tabulares, series temporales, streaming y datos genómicos
- **Configuración consistente** en todos los métodos con validación automática de parámetros
- **Reportes integrados** con YData Profiling para todos los métodos de generación
- **Jerarquía de generadores extensible**: `BaseGenerator` -> `ComplexGenerator` -> generadores de dominio. Los nuevos dominios (Finanzas, IoT, Seguros) heredan tres motores matemáticos reutilizables (Cópula Gaussiana incondicional, Cópula Gaussiana condicional, efectos estocásticos) sin duplicar código.

### Inyección Avanzada de Drift (Líder en la Industria)
El módulo **DriftInjector** es una de las herramientas de simulación de drift más completas disponibles:
- **14+ tipos de drift**: Drift de características (gradual, abrupto, incremental, recurrente), drift de etiquetas, concept drift, correlation drift, inyección de outliers, y más
- **Drift consciente de correlaciones**: Propaga drift realista a través de características correlacionadas (ej. aumentar ingresos → aumentar gastos)
- **Perfiles de drift multi-modales**: Transiciones sigmoid, lineales, coseno para drift gradual
- **Drift condicional**: Aplica drift solo a subconjuntos específicos de datos basándose en reglas de negocio
- **Drift funcional** *(Pilar 5)*: La magnitud del drift varía por fila en función de otra columna (ej. ruido del sensor que escala exponencialmente con la temperatura)
- **Cascadas causales** *(Pilar 5)*: Define un DAG causal y propaga perturbaciones a través de funciones de transferencia no lineales (`CausalEngine`)
- **Integrado con generadores**: Inyecta drift directamente durante la síntesis o post-hoc sobre datos existentes
- **Perfecto para MLOps**: Prueba monitorización de data drift, detección de concept drift, y robustez de modelos antes de producción

### Evolución de Escenarios
El **ScenarioInjector** crea patrones temporales deterministas o estocásticos:
- **7 tipos de evolución**: `linear`, `exponential_growth`, `decay`, `seasonal`, `step`, `noise`, `random_walk`
- **driven_by** *(Pilar 5)*: Delta de una feature por fila = f(valor actual de otra columna) — acopla variables sin necesitar un DAG completo
- **Construcción de target**: Construye variables objetivo de ground-truth sintéticas desde fórmulas de features o callables
- **Proyección futura**: Extiende datasets históricos hacia períodos futuros

> **En resumen**: Mientras otras herramientas se enfocan en un solo enfoque (ej. solo GANs, solo métodos estadísticos), CALM-Data-Generator **unifica el ecosistema** y añade **simulación de drift de grado de producción** que la mayoría de librerías no ofrecen.

---

## Tecnologías Principales

Esta biblioteca aprovecha y unifica las mejores herramientas de código abierto para proporcionar una experiencia de generación de datos fluida:

### Motores de Síntesis

| Librería | Métodos / Funcionalidad | Docs |
|----------|------------------------|------|
| [Synthcity](https://github.com/vanderschaarlab/synthcity) | CTGAN, TVAE, DDPM, TimeGAN, TimeVAE, FFlows, GREAT, RTVAE, BN, DPGAN, PATEGAN, Drift Condicional | [docs synthcity](https://github.com/vanderschaarlab/synthcity) |
| [scikit-learn](https://scikit-learn.org/) | CART, RF, KDE, GMM, métricas, preprocesamiento | [docs sklearn](https://scikit-learn.org/stable/user_guide.html) |
| [LightGBM](https://lightgbm.readthedocs.io/) | Síntesis LGBM (estilo FCS) | [docs lgbm](https://lightgbm.readthedocs.io/en/stable/) |
| [XGBoost](https://xgboost.readthedocs.io/) | Síntesis XGBoost (estilo FCS) | [docs xgb](https://xgboost.readthedocs.io/en/stable/) |
| [Copulae](https://github.com/DanielBok/copulae) | Copula, Windowed Copula | [docs copulae](https://copulae.readthedocs.io/) |
| [imbalanced-learn](https://imbalanced-learn.org/) | SMOTE, ADASYN | [docs imblearn](https://imbalanced-learn.org/stable/) |
| [hmmlearn](https://hmmlearn.readthedocs.io/) | Síntesis HMM | [docs hmmlearn](https://hmmlearn.readthedocs.io/) |
| [PyTorch](https://pytorch.org/) | Difusión (custom), backend para todos los métodos de deep learning | [docs torch](https://pytorch.org/docs/) |

### Single-Cell / Omics

| Librería | Métodos / Funcionalidad | Docs |
|----------|------------------------|------|
| [scvi-tools](https://docs.scvi-tools.org/) | scVI, scANVI | [docs scvi](https://docs.scvi-tools.org/) |
| [GEARS](https://github.com/snap-stanford/GEARS) | Predicción de perturbaciones (método GEARS) | [docs gears](https://github.com/snap-stanford/GEARS) |
| [AnnData](https://anndata.readthedocs.io/) | Estructuras de datos single-cell para scVI/scANVI/GEARS | [docs anndata](https://anndata.readthedocs.io/) |
| [scgft-evaluator](https://github.com/nasim23ea/scgft-evaluator) | Evaluación de calidad single-cell via Graph Fourier Transforms (`use_scgft=True`) | [ver REPORTS_REFERENCE_ES.md](calm_data_generator/docs/REPORTS_REFERENCE_ES.md) |

### Calidad e Informes

| Librería | Métodos / Funcionalidad | Docs |
|----------|------------------------|------|
| [SDMetrics](https://docs.sdv.dev/sdmetrics/) | QualityReporter — puntuaciones de calidad estadística | [docs sdmetrics](https://docs.sdv.dev/sdmetrics/) |
| [YData Profiling](https://ydata-profiling.ydata.ai/) | Informes de perfilado de datos automatizados | [docs ydata](https://ydata-profiling.ydata.ai/docs/) |
| [Plotly](https://plotly.com/python/) | Visualizaciones interactivas y dashboards | [docs plotly](https://plotly.com/python/) |

### Fundamentos Matemáticos

| Librería | Métodos / Funcionalidad | Docs |
|----------|------------------------|------|
| [SciPy](https://docs.scipy.org/) | Cópula Gaussiana (ComplexGenerator), reparación de matrices PSD, estadísticas | [docs scipy](https://docs.scipy.org/) |
| [NumPy](https://numpy.org/) | Arrays numéricos, todos los generadores | [docs numpy](https://numpy.org/doc/) |
| [Pandas](https://pandas.pydata.org/) | DataFrames, todos los generadores | [docs pandas](https://pandas.pydata.org/docs/) |

### Streaming

| Librería | Métodos / Funcionalidad | Docs |
|----------|------------------------|------|
| [River](https://riverml.xyz/) | StreamGenerator — Agrawal, SEA, Hyperplane, Sine, etc. (extra `[stream]`) | [docs river](https://riverml.xyz/latest/) |

## Presets (Plantillas)

**Calm-Data-Generator** incluye **19 Presets** listos para usar que cubren los escenarios de datos sintéticos más comunes. Cada preset encapsula una configuración de generador (método, hiperparámetros, reportes) y expone una única llamada `.generate()`.

> [!TIP]
> **Los Presets son puntos de partida**: instancia un preset, llama a `.generate()` y sobreescribe cualquier parámetro que necesites mediante los argumentos de `__init__` (`random_state`, `verbose`, `fast_dev_run`).

### Presets Disponibles

**Velocidad y Prototipado**

| Preset | Método | Caso de Uso |
|--------|--------|-------------|
| `FastPreset` | LightGBM | Generación más rápida, configurable via kwargs |
| `FastPrototypePreset` | LightGBM | Pipelines CI/CD, tests de integración (10 iteraciones fijas) |

**Calidad y Fidelidad**

| Preset | Método | Caso de Uso |
|--------|--------|-------------|
| `HighFidelityPreset` | CTGAN (1000 épocas) | Datos tabulares de calidad de producción |
| `DiffusionPreset` | TabDDPM (1000 pasos) | Distribuciones complejas multimodales |
| `CopulaPreset` | Cópula Gaussiana | Línea base estadística rápida, modelado de dependencias |
| `DataQualityAuditPreset` | TVAE + reporte completo | Auditoría de calidad con reporte completo automatizado |

**Distribución de Clases**

| Preset | Método | Caso de Uso |
|--------|--------|-------------|
| `ImbalancedGeneratorPreset` | CTGAN | Forzar proporción minoría/mayoría específica (binario) |
| `BalancedDataGeneratorPreset` | SMOTE | Sobremuestrear la clase minoritaria para balancear el dataset |

**Series Temporales**

| Preset | Método | Caso de Uso |
|--------|--------|-------------|
| `TimeSeriesPreset` | TimeGAN / TimeVAE / FourierFlows | Datos secuenciales con dinámicas temporales |
| `SeasonalTimeSeriesPreset` | TimeGAN + ScenarioInjector | Series temporales con estacionalidad sinusoidal inyectada |

**Drift y Escenarios**

| Preset | Método | Caso de Uso |
|--------|--------|-------------|
| `DriftScenarioPreset` | CTGAN + DriftConfig | Prueba de estrés de sistemas de detección de drift |
| `GradualDriftPreset` | CTGAN + drift lineal | Simular drift lineal lento de features |
| `ConceptDriftPreset` | CTGAN + concept drift | Alterar relaciones P(y\|x) |
| `ScenarioInjectorPreset` | ScenarioInjector | Aplicar escenarios de evolución a datos existentes |

**Clínico y Omics**

| Preset | Método | Caso de Uso |
|--------|--------|-------------|
| `LongitudinalHealthPreset` | ClinicalDataGenerator | Registros de pacientes multi-visita |
| `RareDiseasePreset` | ClinicalDataGenerator | Cohortes con enfermedad rara (1% de prevalencia por defecto) |
| `OmicsIntegrationPreset` | ClinicalDataGenerator | Multi-omics (clínico + expresión génica + proteómica) |
| `SingleCellQualityPreset` | scVI (400 épocas) | Datos de RNA-seq de célula única de alta calidad |

> Referencia completa de parámetros: [`calm_data_generator/docs/PRESETS_REFERENCE_ES.md`](calm_data_generator/docs/PRESETS_REFERENCE_ES.md)

### Ejemplos de Inicio Rápido

```python
from calm_data_generator.presets import FastPreset, HighFidelityPreset, ImbalancedGeneratorPreset

# --- Generación rápida ---
preset = FastPreset(random_state=42)
synthetic_df = preset.generate(data=real_df, n_samples=1000)

# --- Datos de producción de alta fidelidad ---
preset = HighFidelityPreset(random_state=42)
synthetic_df = preset.generate(data=real_df, n_samples=5000)

# --- Dataset desbalanceado (5% minoría) ---
preset = ImbalancedGeneratorPreset(random_state=42)
synthetic_df = preset.generate(
    data=real_df, n_samples=2000,
    target_col="etiqueta", imbalance_ratio=0.05
)
```

```python
from calm_data_generator.presets import TimeSeriesPreset, SeasonalTimeSeriesPreset

# --- Series temporales con TimeGAN ---
preset = TimeSeriesPreset(random_state=42)
synthetic_df = preset.generate(
    data=ts_df, n_samples=500, sequence_key="paciente_id", time_key="fecha_visita"
)

# --- Series temporales estacionales (patrón mensual) ---
preset = SeasonalTimeSeriesPreset(random_state=42)
synthetic_df = preset.generate(
    data=ts_df, n_samples=500,
    time_col="fecha", seasonal_cols=["ventas", "demanda"],
    period=12, amplitude=2.0
)
```

```python
from calm_data_generator.presets import LongitudinalHealthPreset, SingleCellQualityPreset

# --- Datos clínicos longitudinales (multi-visita) ---
preset = LongitudinalHealthPreset(random_state=42)
result = preset.generate(n_samples=200, n_visits=6)

# --- RNA-seq de célula única ---
preset = SingleCellQualityPreset(random_state=42)
synthetic_df = preset.generate(data=adata_df, n_samples=500)
```

## Librerías Clave y Ecosistema

 | Librería | Rol | Uso en Calm-Data-Generator |
 | :--- | :--- | :--- |
 | **Synthcity** | Motor de Deep Learning | Potencia `CTGAN`, `TVAE`, `DDPM`, `TimeGAN`. Manejo de privacidad y fidelidad. |
 | **scvi-tools** | Análisis Single-Cell | Potencia el método `scvi` para datos genómicos/transcriptómicos de alta dimensión. |
 | **River** | Streaming ML | Potencia `StreamGenerator` para simulación de concept drift y flujo de datos en tiempo real. |
 | **YData Profiling**| Reportes | Genera reportes de calidad automatizados (`QualityReporter`). |
 | **Pydantic** | Validación | Asegura chequeo de tipos estricto y gestión de configuración. |
 | **PyTorch** | Backend | Computación tensorial subyacente para todos los modelos de deep learning. |
 | **Copulae** | Modelado Estadístico | Potencia los métodos `copula` y `windowed_copula` para modelado de dependencia multivariante y generación con drift. |
 | **hmmlearn** | Modelado Estadístico | Potencia el método `hmm` para generación con drift mediante transiciones entre regímenes de Modelos Ocultos de Markov. |
 | **SciPy** | Núcleo Matemático | Potencia los motores de Cópula Gaussiana dentro de `ComplexGenerator` (incondicional y condicional) para generación multivariante correlacionada entre dominios. |

## Intercambio Seguro de Datos

Una ventaja clave de **Calm-Data-Generator** es permitir el uso de datos privados en entornos públicos o colaborativos:

1.  **Origen Privado**: Empiezas con datos sensibles (ej. restringidos por GDPR/HIPAA) que no pueden salir de tu entorno seguro.
2.  **Gemelo Sintético**: La biblioteca genera un conjunto de datos sintético que refleja estadísticamente el original pero **no contiene individuos reales**.
3.  **Distribución Segura**: Una vez validado (usando los chequeos de privacidad de `QualityReporter`), este dataset sintético permite **compartir sin riesgos**, entrenar modelos y realizar pruebas sin exponer información confidencial.

## Casos de Uso Clave

- **Validación de Monitorización MLOps**: Usa **StreamGenerator** y **DriftInjector** para simular drift de datos (gradual, abrupto) y verificar si tus alertas de monitorización se activan correctamente antes del despliegue.
- **Investigación Biomédica (HealthTech)**: Genera cohortes de pacientes sintéticos con **ClinicalDataGenerator** que preservan correlaciones biológicas complejas (ej. relaciones gen-edad) para estudios colaborativos sin comprometer la privacidad del paciente.
- **Pruebas de Estrés (Análisis "What-If")**: Usa **ScenarioInjector** para simular escenarios futuros (ej. "¿Qué pasa si la base de clientes envejece 10 años?") y medir la degradación del rendimiento del modelo bajo estrés.
- **Drift con Correlaciones**: Inyecta drift que se propaga realisticamente a características correlacionadas (ej. aumentar ingresos también aumenta gastos proporcionalmente) usando el parámetro `correlations=True`.
- **Datos de Desarrollo**: Proporciona a los desarrolladores réplicas sintéticas de alta fidelidad de bases de datos de producción, permitiéndoles construir y probar funcionalidades de forma segura sin acceder a datos reales sensibles.

---

## Instalación

 > [!WARNING]
 > **Aviso Importante**: Esta librería depende de frameworks de Deep Learning pesados como `PyTorch`, `Synthcity` y librerías `CUDA`.
 > La instalación puede ser **pesada (~2-3 GB)** y tardar unos minutos dependiendo de tu conexión. Recomendamos encarecidamente usar un entorno virtual limpio.

 ### Estrategia de Versiones

 - **GitHub (Recomendado para últimas novedades)**: La rama `main` contiene la versión más actualizada con los últimos arreglos y funcionalidades.
 - **PyPI (Estable)**: Las versiones en PyPI son estables y se actualizan con menor frecuencia para cambios mayores.

 ### Instalación Estándar (PyPI - Estable)
 La librería está disponible en PyPI. Para una experiencia estable, recomendamos usar un entorno virtual:

```bash
# 1. Crear y activar el entorno virtual
python3 -m venv venv
source venv/bin/activate

# 2. Actualizar pip, setuptools y wheel (Crucial para una instalación exitosa)
pip install --upgrade pip setuptools wheel

# 3. Instalar la librería (optimizada para velocidad)
pip install calm-data-generator
```

### Extras de Instalación
Puedes añadir capacidades específicas según tu caso de uso:
```bash
# Para Stream Generator (River)
pip install "calm-data-generator[stream]"


# Instalación completa
pip install "calm-data-generator[full]"
```

> [!NOTE]
> **Nota de Rendimiento y Estabilidad**: Hemos optimizado el árbol de dependencias desde la versión 1.0.0 bloqueando versiones específicas como `pydantic`, `xgboost` y `cloudpickle`. Esto mejora la compatibilidad y reduce problemas de instalación.

**Desde fuente (GitHub - Últimas Actualizaciones):**
Usa este método para obtener los últimos arreglos y funcionalidades aún no disponibles en PyPI.

```bash
# Opción A: Instalar directamente desde GitHub
pip install git+https://github.com/AlejandroBeldaFernandez/Calm-Data_Generator.git

# Opción B: Clonar e instalar (para desarrollo)
git clone https://github.com/AlejandroBeldaFernandez/Calm-Data_Generator.git
cd Calm-Data_Generator
pip install .
```

### Solución de Problemas

**Zsh shell (macOS/Linux):** Si los corchetes dan error, usa comillas:
```bash
pip install "calm-data-generator[stream]"
```

**Errores de compilación de River (Linux/macOS):**
```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev

# macOS
xcode-select --install

# Luego reintenta
pip install calm-data-generator
```

**Usuarios de Windows:** Instala Visual Studio Build Tools primero:
1. Descarga [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Instala "Desktop development with C++"
3. Luego reintenta la instalación

**Windows — Error de ruta larga durante la instalación:**

Algunos paquetes (p. ej. `orbax-checkpoint`) contienen estructuras de directorios muy profundas que superan el límite predeterminado de 260 caracteres de ruta en Windows. Si ves un error como:
```
OSError: [Errno 2] No such file or directory: 'C:\...\ruta\muy\larga'
HINT: This error might have occurred since this system does not have Windows Long Path support enabled.
```
Activa las rutas largas desde PowerShell (como Administrador):
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
Reinicia el equipo y vuelve a intentar la instalación. Alternativamente, instala tu entorno virtual en una ruta corta (p. ej. `C:\venv\`) para reducir la longitud total de la ruta.

**PyTorch solo-CPU (sin GPU):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install calm-data-generator
```

---

## Inicio Rápido

### Generar Datos Sintéticos desde un Dataset Real

```python
from calm_data_generator import RealGenerator
import pandas as pd

# Tu dataset real (puede ser un DataFrame, ruta a .csv, .h5 o .h5ad)
data = pd.read_csv("your_data.csv")  # o "your_data.h5ad"

# Inicializar generador
gen = RealGenerator()

# Generar 1000 muestras sintéticas usando CTGAN

synthetic = gen.generate(
    data=data,
    n_samples=1000,
    method='ctgan',
    target_col='label',
    differentiation_factor=2.0, # NUEVO: Mejora la separabilidad de clases
    clipping_mode='permissive', # NUEVO: Gestión de rangos de datos
    epochs=300,
    batch_size=500,
    discriminator_steps=1

)

print(f"Generadas {len(synthetic)} muestras")
```

### Aceleración por GPU

**Métodos con soporte GPU:**

| Método | Soporte GPU | Parámetro |
|--------|-------------|-----------|
| `ctgan`, `tvae` | Sí — CUDA/MPS | `enable_gpu=True` |
| `diffusion` | Sí — PyTorch | Auto-detectado |
| `ddpm` | Sí — PyTorch + Synthcity | Auto-detectado |
| `timegan` | Sí — PyTorch + Synthcity | Auto-detectado |
| `timevae` | Sí — PyTorch + Synthcity | Auto-detectado |
| `fflows` | Sí — PyTorch + Synthcity | Auto-detectado |

| `smote`, `adasyn`, `cart`, `rf`, `lgbm`, `gmm`, `copula` | No — Solo CPU | - |

```python
synthetic = gen.generate(
    data=data,
    n_samples=1000,
    method='ctgan',
    epochs=300,
    enable_gpu=True,

)
```

### Generar Datos Clínicos

```python
from calm_data_generator import ClinicalDataGenerator
from calm_data_generator.generators.configs import DateConfig

gen = ClinicalDataGenerator()

# Generar datos de pacientes con genes y proteínas
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

### Inyectar Drift para Pruebas de ML

**Opción 1: Directamente desde `generate()` (recomendado)**

```python
from calm_data_generator import RealGenerator

gen = RealGenerator()

# Generar datos sintéticos CON drift en una sola llamada
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
                "drift_mode": "gradual", # Auto-detecta tipos de columna
                "drift_magnitude": 0.3,
                "center": 500,
                "width": 200
            }
        }
    ]
)
```

**Opción 2: DriftInjector Independiente**

```python
from calm_data_generator import DriftInjector

injector = DriftInjector()

# Inyección unificada de drift (auto-detecta tipos)
drifted_data = injector.inject_drift(
    df=data,
    columns=['feature1', 'feature2', 'status'],
    drift_mode='gradual',
    drift_magnitude=0.5,
    # Configuración específica opcional
    numeric_operation='shift',
    categorical_operation='frequency',
    boolean_operation='flip'
)
```

**Métodos de drift disponibles:** `inject_drift` (unificado), `inject_feature_drift_gradual`, `inject_label_drift`, `inject_categorical_frequency_drift`, y más. Ver [DRIFT_INJECTOR_REFERENCE.md](calm_data_generator/docs/DRIFT_INJECTOR_REFERENCE.md).

### Simulación de Streaming

```python
from calm_data_generator import StreamGenerator

# Simular un stream de datos basándose en el dataset real
stream_gen = StreamGenerator()

stream_data = stream_gen.generate(
    data=data,
    n_samples=5000,
    chunk_size=1000,
    concept_drift=True,  # Simular concept drift en el tiempo
    n_features=10
)

print(f"Generado stream con {len(stream_data)} muestras totales")
```

### Informe de Calidad

```python
from calm_data_generator import QualityReporter

# Generar informe comparando datos reales vs sintéticos
reporter = QualityReporter()

reporter.generate_report(
    real_data=data,
    synthetic_data=synthetic,
    output_dir="./quality_report",
    target_col="target"
)
# Informe guardado en ./quality_report/report.html
# JSON de resultados (incluyendo compared_data_files) guardado en ./quality_report/report_results.json
```

Para **datos single-cell**, activa la evaluación [scGFT](https://github.com/nasim23ea/scgft-evaluator) (métricas de preservación de manifold basadas en Transformadas de Fourier en Grafos):

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
# Genera scgft_report.html con métricas ARI, MMD, Jaccard, Kendall Tau
```

---

## Módulos

| Módulo | Importación | Descripción |
|--------|-------------|-------------|
| **Tabular** | `generators.tabular` | RealGenerator, QualityReporter |
| **Clinical** | `generators.clinical` | ClinicalDataGenerator, ClinicalDataGeneratorBlock |
| **Stream** | \`generators.stream\` | StreamGenerator, StreamBlockGenerator |
| **Blocks** | `generators.tabular` | RealBlockGenerator |
| **Drift** | `generators.drift` | DriftInjector |
| **Dynamics** | `generators.dynamics` | ScenarioInjector |
| **Reports** | `reports` | Visualizer |

---

## Métodos de Síntesis

| Método | Tipo | Descripción | Requisitos / Notas |
|--------|------|-------------|-------------------|
| `cart` | ML | Síntesis iterativa basada en CART (rápido) | Instalación base |
| `rf` | ML | Síntesis con Random Forest | Instalación base |
| `lgbm` | ML | Síntesis basada en LightGBM | Instalación base (Requiere `lightgbm`) |
| `ctgan` | DL | GAN Condicional para datos tabulares | Requiere `synthcity` |
| `tvae` | DL | Autoencoder Variacional | Requiere `synthcity` |
| `diffusion` | DL | Difusión Tabular (custom, rápida) | Instalación base (PyTorch) |
| `ddpm` | DL | Synthcity TabDDPM (avanzado) | Requiere `synthcity` |
| `timegan` | Series Temp. | TimeGAN para datos secuenciales | Requiere `synthcity` |
| `timevae` | Series Temp. | TimeVAE para datos secuenciales | Requiere `synthcity` |
| `fflows` | Series Temp. | FourierFlows — flujos normalizantes en dominio de frecuencia, estable para series periódicas | Requiere `synthcity` |
| `bn` | DL / Probabilístico | Red Bayesiana — modela dependencias causales entre variables | Requiere `synthcity` |
| `smote` | Aumento | Sobremuestreo SMOTE | Instalación base |
| `adasyn` | Aumento | Muestreo adaptativo ADASYN | Instalación base |
| `copula` | Copula | Síntesis basada en Copulas | Instalación base |
| `gmm` | Estadístico | Modelos de Mezcla Gaussiana | Instalación base |
| `scvi` | Single-Cell | scVI (Variational Inference) para RNA-seq | Requiere `scvi-tools` |
| `conditional_drift` | Drift Nativo | Condicionamiento temporal con TVAE/CTGAN | Requiere `synthcity` |
| `windowed_copula` | Drift Nativo | Cópula Gaussiana interpolada por ventanas | Instalación base |
| `hmm` | Drift Nativo | Modelo Oculto de Markov — drift por transición de regímenes | Requiere `hmmlearn` |

---

## Documentación e Índice

Explora la documentación completa en el directorio `calm_data_generator/docs/`:

| Documento | Descripción |
|-----------|-------------|
| **[DOCUMENTATION.md](calm_data_generator/docs/DOCUMENTATION.md)** | **Guía Principal**. Manual completo cubriendo todos los módulos, conceptos y uso avanzado. |
| **[REAL_GENERATOR_REFERENCE.md](calm_data_generator/docs/REAL_GENERATOR_REFERENCE.md)** | **Referencia API para `RealGenerator`**. Parámetros detallados para todos los métodos de síntesis (`ctgan`, `lgbm`, `scvi`, etc.). |
| **[DRIFT_INJECTOR_REFERENCE.md](calm_data_generator/docs/DRIFT_INJECTOR_REFERENCE.md)** | **Referencia API para `DriftInjector`**. Guía para usar `inject_drift` y capacidades especializadas de drift. |
| **[STREAM_GENERATOR_REFERENCE.md](calm_data_generator/docs/STREAM_GENERATOR_REFERENCE.md)** | **Referencia API para `StreamGenerator`**. Detalles sobre simulación de stream e integración de drift. |
| **[CLINICAL_GENERATOR_REFERENCE.md](calm_data_generator/docs/CLINICAL_GENERATOR_REFERENCE.md)** | **Referencia API para `ClinicalGenerator`**. Configuración para genes, proteínas y datos de pacientes. |
| **[API.md](calm_data_generator/docs/API.md)** | **Índice Técnico de API**. Índice de alto nivel de clases y funciones. |

---

## Licencia

Licencia MIT - ver archivo [LICENSE](LICENSE)
## Changelog

Consulta [CHANGELOG_ES.md](CHANGELOG_ES.md) para el historial completo. Resumen de versiones recientes:

### v2.0.0 — 2026-03-27
- **ComplexGenerator**: nueva capa abstracta con 3 motores matemáticos reutilizables (Cópula Gaussiana incondicional/condicional + efectos estocásticos). `ClinicalDataGenerator` ahora hereda de ella.
- **CausalEngine**: cascada causal basada en DAG con ordenación topológica (algoritmo de Kahn) y funciones de transferencia no lineales.
- **`inject_functional_drift()`**: magnitud del drift por fila = f(valor de la columna driver).
- **`inject_causal_cascade()`**: propagación causal completa integrada en `DriftInjector`.
- **Tipo de evolución `driven_by`**: delta de una feature en `ScenarioInjector` impulsada por otra columna por fila.
- **Correcciones de bugs**: manejo de datetime en CART, dispatch del método `bn`, API Synthcity en `conditional_drift`, reshape 1D en `windowed_copula`.
- **Tests**: 186 passed, 0 failed — todos los archivos `unittest.TestCase` convertidos a pytest.

### v1.2.0
- Parámetros `differentiation_factor`, `clipping_mode`, `use_latent_sampling`.
- Métodos de síntesis Windowed Copula, Conditional Drift, DPGAN, PATEGAN.

---

## Agradecimientos y Créditos

Nos apoyamos en hombros de gigantes. Esta librería es posible gracias a estos increíbles proyectos de código abierto:

- **[Synthcity](https://github.com/vanderschaarlab/synthcity)** (Apache 2.0) - El motor detrás de nuestros modelos de deep learning.
- **[River](https://github.com/online-ml/river)** (BSD-3-Clause) - Potenciando nuestras capacidades de streaming.
- **[YData Profiling](https://github.com/ydataai/ydata-profiling)** (MIT) - Proporcionando reportes de datos exhaustivos.
- **[scvi-tools](https://github.com/scverse/scvi-tools)** (BSD-3-Clause) - Habilitando análisis single-cell.
- **[GEARS](https://github.com/snap-stanford/GEARS)** (MIT) - Soportando la predicción de perturbaciones basada en grafos.
- **[Imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn)** (MIT) - Proporcionando implementaciones de SMOTE y ADASYN.
- **[SDMetrics](https://github.com/sdv-dev/SDMetrics)** (MIT) - Potenciando las métricas estandarizadas en nuestro QualityReporter.
- **[Copulae](https://github.com/DanielBok/copulae)** (MIT) - Habilitando modelado de dependencia multivariante vía Cópulas Gaussianas.
- **[AnnData](https://github.com/scverse/anndata)** (BSD-3-Clause) - Proporcionando la estructura de datos central para integración de single-cell y ómicas.
- **[LightGBM](https://github.com/microsoft/LightGBM)** (MIT) - Potenciando nuestro método de síntesis basado en gradient boosting.
- **[PyTorch](https://github.com/pytorch/pytorch)** (BSD-3-Clause) - El framework de deep learning que potencia nuestros modelos generativos.
- **[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)** (MIT) - Habilitando operaciones de Graph Neural Networks para datos relacionales.
- **[XGBoost](https://github.com/dmlc/xgboost)** (Apache-2.0) - Librería optimizada de gradient boosting distribuido.
- **[Hugging Face Hub](https://github.com/huggingface/huggingface_hub)** (Apache-2.0) - Facilitando el intercambio y versionado de modelos.
- **[Plotly](https://github.com/plotly/plotly.py)** (MIT) - Habilitando visualizaciones de datos interactivas.
- **[hmmlearn](https://github.com/hmmlearn/hmmlearn)** (BSD-3-Clause) - Potenciando el método `hmm` para generación con drift mediante Modelos Ocultos de Markov.
- **[scgft-evaluator](https://github.com/nasim23ea/scgft-evaluator)** - Proporcionando evaluación basada en Transformadas de Fourier en Grafos para la valoración de calidad de datos sintéticos single-cell.
