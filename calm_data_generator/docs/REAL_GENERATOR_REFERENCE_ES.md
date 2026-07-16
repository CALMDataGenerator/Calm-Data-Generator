# RealGenerator - Referencia Completa

**Ubicación:** `calm_data_generator.generators.tabular.RealGenerator`

El generador principal para la síntesis de datos tabulares a partir de datasets reales.

---

## Inicialización

```python
from calm_data_generator import RealGenerator

gen = RealGenerator(
    auto_report=True,       # Generar informe automáticamente tras síntesis
    minimal_report=False,   # Si es True, informe más rápido sin correlaciones/PCA
    random_state=42,        # Semilla para reproducibilidad
    logger=None,            # Logger de Python personalizado opcional
)
```

### Parámetros del Constructor

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `auto_report` | bool | `True` | Generar informe de calidad automáticamente |
| `minimal_report` | bool | `False` | Informe simplificado (más rápido) |
| `random_state` | int | `None` | Semilla para reproducibilidad |
| `logger` | Logger | `None` | Instancia de Logger de Python personalizada |
| `verbose_training` | bool | `False` | Muestra la pérdida por época de Synthcity en consola durante el entrenamiento. Útil para modelos como TVAE o CTGAN donde `get_training_history()` no está disponible. |

> [!TIP]
> Esta librería actúa como un wrapper de alto nivel. Para ajustes de hiperparámetros avanzados y detalles arquitectónicos profundos, recomendamos encarecidamente consultar la documentación de los motores originales:
> - **Synthcity** (CTGAN, TVAE, DDPM, TimeGAN, FF): [Docs de Synthcity](https://github.com/vanderschaarlab/synthcity)
> - **scvi-tools** (scVI, scANVI): [Docs de scvi-tools](https://docs.scvi-tools.org/)
> - **GEARS**: [GitHub de GEARS](https://github.com/snap-stanford/GEARS)


---

## Método Principal: `generate()`

```python
# Nuevos Imports de Configuración
from calm_data_generator.generators.configs import DriftConfig, ReportConfig, DateConfig

synthetic_df = gen.generate(
    data=df,                          # DataFrame original (requerido)
    n_samples=1000,                   # Número de muestras a generar (requerido)
    method="ctgan",                   # Método de síntesis

    # Objetos de Configuración
    report_config=ReportConfig(       # Configuración de informes
        output_dir="./output",
        target_column="target"
    ),

    # Inyección de Drift
    drift_injection_config=[
        DriftConfig(
            method="inject_feature_drift",
            feature_cols=["age"],
            drift_type="shift",
            magnitude=0.5
        )
    ],

    # Los argumentos legacy aún son soportados pero se recomiendan los objetos Config
    # target_col="target",
    # output_dir="./output"
)
```

### Parámetros de `generate()`

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `data` | DataFrame | - | Dataset original (requerido) |
| `n_samples` | int | - | Número de muestras a generar (requerido) |
| `method` | str | `"cart"` | Método de síntesis |
| `target_col` | str | `None` | Columna objetivo para balanceo |
| `output_dir` | str | `None` | Directorio para archivos de salida |
| `generator_name` | str | `"RealGenerator"` | Nombre base para archivos de salida |
| `save_dataset` | bool | `False` | Guardar dataset generado como CSV |
| `custom_distributions` | Dict | `None` | Distribución forzada por columna |
| `custom_distribution` | Dict | `None` | ⚠️ Alias legacy singular de `custom_distributions`. Registra un warning si se usa — se recomienda la forma plural. |
| `date_col` | str | `None` | Nombre de columna de fecha a inyectar |
| `date_start` | str | `None` | ⚠️ Legacy. Fecha de inicio ("YYYY-MM-DD"). Registra un warning si se usa sin `date_config` — se recomienda `date_config=DateConfig(...)`. |
| `date_step` | Dict | `None` | ⚠️ Legacy — misma nota que `date_start`. Incremento temporal (ej., `{"days": 1}`) |
| `date_every` | int | `1` | ⚠️ Legacy — misma nota que `date_start`. Incrementar fecha cada N filas |
| `drift_injection_config` | List[Union[Dict, DriftConfig]] | `None` | Configuración de drift post-generación |
| `dynamics_config` | Dict | `None` | Configuración de evolución dinámica |
| `**kwargs` | Dict | `None` | Hiperparámetros específicos  |
| `constraints` | List[Dict] | `None` | Restricciones de integridad |
| `adversarial_validation` | bool | `False` | Activar reporte de discriminador (Real vs Sintético) |
| `report_config` | ReportConfig | `None` | Objeto de configuración avanzada de informes |
| `date_config` | DateConfig | `None` | Objeto de configuración avanzada de inyección de fechas |
| `balance` | bool | `False` | Balancear automáticamente la distribución de clases en `target_col` |
| `**kwargs` | Any | - | Parámetros específicos del método (ej., `epochs`, `n_latent`, `lr`) |

---

## API más simple: `fit()` / `sample()`

Para el caso más común — entrenar una vez, muestrear tantas veces como haga falta —
`RealGenerator` ofrece una envoltura fina, estilo sklearn, sobre `generate()`:

```python
gen = RealGenerator(auto_report=False, random_state=42)

# Entrena el modelo. No escribe ningún reporte/dataset a disco.
gen.fit(df, method="cart", target_col="target")

# Genera filas sintéticas del modelo ya entrenado, tantas veces como se quiera — sin reentrenar.
synth_small = gen.sample(100)
synth_large = gen.sample(10_000)

# El encadenado también funciona:
synth = RealGenerator().fit(df, method="ctgan").sample(1000)
```

- `fit(data, method="cart", target_col=None, **kwargs)` acepta los mismos argumentos de palabra
  clave que `generate()` (menos `n_samples`, `output_dir`, `save_dataset`) y devuelve `self`.
- `sample(n_samples)` lanza un `ValueError` claro si se llama antes de `fit()` (o `.load()`).
- Internamente, `sample()` llama a `generate(data=None, n_samples=...)`, que ya podía reutilizar
  un `self.synthesizer` previamente entrenado sin reentrenar — `fit()`/`sample()` simplemente
  dan a esa capacidad ya existente un nombre más descubrible y convencional.
- Para llamadas de un solo paso (entrenar + generar + reportar en una sola llamada), `generate()`
  sigue siendo la opción correcta.

---

## Referencia Completa de `**kwargs`

El diccionario `**kwargs` permite el ajuste fino de parámetros internos para cada método de síntesis.

### Deep Learning (Synthcity)

| Parámetro | Métodos | Descripción |
|-----------|---------|-------------|
| `epochs` | `ctgan`, `tvae`, `rtvae`, `great`, `dpgan`, `pategan` | Número de épocas de entrenamiento (defecto: 300) |
| `batch_size` | `ctgan`, `tvae`, `rtvae`, `great` | Tamaño del batch de entrenamiento (defecto: 500) |
| `lr` | `ctgan`, `tvae`, `rtvae` | Tasa de aprendizaje (Learning rate) |
| `differentiation_factor` | `tvae`, `rtvae`, `scvi` | Desplaza los centroides de clase en el espacio latente para forzar la separabilidad. |
| `clipping_mode` | `tvae`, `rtvae`, `scvi` | Estrategia de recorte: `'strict'`, `'permissive'`, o `'none'`. (Por defecto: `'strict'`) |
| `clipping_factor` | `tvae`, `rtvae`, `scvi` | Porcentaje de margen para el modo `'permissive'` (Por defecto: `0.1`). |
| `epsilon` | `dpgan`, `pategan` | Presupuesto de privacidad ε — menor = más privado (defecto: 1.0) |
| `delta` | `dpgan`, `pategan` | Parámetro δ de privacidad diferencial (defecto: 1e-5) |


**Ejemplo:**
```python
gen.generate(
    df, 1000,
    method="ctgan",
    epochs=500,
    batch_size=256
)
```


### Machine Learning Clásico (CART, RF, LGBM)

| Parámetro | Métodos | Descripción |
|-----------|---------|-------------|
| `balance` | Todos ML | Si es True y `target_col` existe, balancea clases antes de entrenar |
| `n_estimators` | RF, LGBM | Número de árboles |
| `max_depth` | CART, RF | Profundidad máxima |

**Ejemplo:**
```python
method="rf",
target_col="churn",
balance=True,
n_estimators=100,

```

### Genómica Single-Cell

Estos métodos están diseñados específicamente para **datos transcriptómicos (RNA-seq)**. Utilizan modelos generativos profundos para manejar la dispersión (sparsity) y el ruido técnico característico de los datos biológicos. Son ideales para corregir "efectos de lote" (batch effects) y generar perfiles de expresión genética sintéticos coherentes.

#### scVI (Single-cell Variational Inference)
**Formato de Entrada:** Acepta objetos `pd.DataFrame`, `AnnData` o rutas de archivo (`.h5`, `.h5ad` o `.csv`) directamente.

**Uso de Rutas de Archivo (H5/H5AD/CSV):**
```python
# ¡El generador carga el archivo automáticamente por ti!
synthetic = gen.generate(
    data="datos_single_cell.csv",  # O .h5ad, .h5
    n_samples=1000,
    method="scvi",
    target_col="cell_type"
)
```

**Entrada AnnData (Recomendado para datos single-cell):**
```python
import anndata

# Crear o cargar objeto AnnData
adata = anndata.read_h5ad("single_cell_data.h5ad")

synthetic = gen.generate(
    data=adata,              # Pasar AnnData directamente
    n_samples=1000,
    method="scvi",
    target_col="cell_type",  # Debe estar en adata.obs
    epochs=100,
    n_latent=10,
    n_layers=1
)
# Retorna pd.DataFrame con columnas de genes + metadatos
```

| Parámetro | Descripción |
|-----------|-------------|
| `epochs` | Épocas de entrenamiento (default: 200) |
| `n_latent` | Dimensionalidad del espacio latente (default: 30) |
| `n_layers` | Número de capas ocultas (default: 1) |
| `differentiation_factor` | 0.0 | Factor de separación latente. Utiliza el proceso unificado de 5 pasos para empujar las clases. |
| `clipping_mode` | `'strict'` | Estrategia de recorte: `'strict'`, `'permissive'`, o `'none'`. |
| `use_latent_sampling` | True | Controla la fidelidad de generación. Si es True, muestrea desde "anclas" de datos reales. |
| `preserve_library_size` | True | Si es True, mantiene la distribución de conteos totales (library size) original. |
| `latent_noise_std` | 0.05 | Magnitud del ruido para el muestreo del espacio latente (mayor = más diversidad). |



| `custom_distributions` | `None` | Proporciones por clase para la generación |

#### Utilidades de Flujo de Trabajo para Single-Cell

Para los usuarios que trabajan con transcriptómica de célula única (single-cell), `RealGenerator` proporciona una utilidad para convertir los DataFrames sintéticos generados de vuelta a objetos `AnnData`, que es el formato estándar para librerías de análisis como `scanpy` o `squidpy`.

**`to_anndata(df, target_col=None, obs_cols=None)`** (Método Estático)

Convierte un DataFrame sintético (la salida de `generate()`) en un objeto `AnnData`.

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | **requerido** | El DataFrame sintético generado por `calm_data_generator`. |
| `target_col` | `str` | `None` | La columna que se usará como `cell_type` en `adata.obs`. |
| `obs_cols` | `List[str]` | `None` | Lista de columnas adicionales que se moverán de la matriz de características (`X`) a los metadatos (`obs`). |

**Ejemplo:**
```python
from calm_data_generator.generators.tabular import RealGenerator

# 1. Generar datos sintéticos (ej. usando scANVI)
synthetic_df = gen.generate(
    data=real_adata,
    n_samples=2000,
    method="scanvi",
    target_col="cell_type"
)

# 2. Convertir de vuelta a AnnData para análisis con scanpy
synthetic_adata = RealGenerator.to_anndata(
    synthetic_df,
    target_col="cell_type"
)

# 3. Análisis estándar con scanpy
import scanpy as sc
sc.pp.pca(synthetic_adata)
sc.pl.pca(synthetic_adata, color="cell_type")
```

**Validación de calidad single-cell con scGFT:**

Tras generar datos sintéticos de célula única, valida su fidelidad usando [scgft-evaluator](https://github.com/nasim23ea/scgft-evaluator):

```python
from calm_data_generator.reports.QualityReporter import QualityReporter
from calm_data_generator.generators.configs import ReportConfig

reporter = QualityReporter(verbose=True)
reporter.generate_comprehensive_report(
    real_df=real_df,
    synthetic_df=synthetic_df,
    generator_name="scVI_SingleCell",
    report_config=ReportConfig(
        output_dir="./sc_report",
        use_scgft=True,
        target_column="cell_type",
    ),
)
# Genera scgft_report.html con métricas ARI, MMD, Jaccard, Kendall Tau
```

> Ver [REPORTS_REFERENCE_ES.md](REPORTS_REFERENCE_ES.md#evaluación-single-cell-scgft) para documentación completa.

---

## Características Avanzadas

### Inyección de Fechas (DateConfig)

Puedes inyectar una columna de fecha/hora en los datos generados usando `DateConfig`.

```python
from calm_data_generator.generators.configs import DateConfig

synthetic = gen.generate(
    data=df,
    n_samples=1000,
    method="cart",
    date_config=DateConfig(
        date_col="timestamp",
        start_date="2024-06-01",
        step={"hours": 1},  # Incremento temporal
        frequency=1         # Incrementar cada N filas
    )
)
```

## Manejo de Datos Desbalanceados

`RealGenerator` ofrece varias estrategias para trabajar con datasets fuertemente desbalanceados (ej. detección de fraude, diagnósticos raros):

### 1. Re-balanceo Automático (`balance=True`)
Utiliza técnicas de re-muestreo antes o durante el entrenamiento para generar un dataset sintético equilibrado.
*   **Ideal para:** Entrenar modelos de clasificación robustos que requieren clases balanceadas.
*   **Comportamiento:** Si el original es 99% clase A y 1% clase B, el resultado será aprox. 50% A y 50% B.
*   **Métodos compatibles:** `cart`, `rf`, `lgbm`.

### 2. Control Manual de Distribución
Puede forzar la distribución de la clase objetivo usando `DriftInjector`.
*   **Ideal para:** Escenarios "What-If" (ej. "¿Qué pasa si el fraude aumenta al 10%?").
*   **Método:** `DriftInjector.inject_label_shift` post-generación.

### 3. Técnicas de Oversampling
Métodos clásicos para aumentar la clase minoritaria mediante interpolación.
*   **Métodos:** `smote` (Synthetic Minority Over-sampling Technique), `adasyn` (Adaptive Synthetic Sampling).
*   **Ideal para:** Datasets numéricos pequeños donde se necesita aumentar la representación de casos raros.

### 4. Fidelidad Estadística (Por defecto)
Si no se especifica ninguna opción, los modelos generativos avanzados (`ctgan`, `tvae`) aprenderán y replicarán la distribución original, preservando el desbalance real.
*   **Ideal para:** Análisis exploratorio fiel a la realidad o validación de sistemas en condiciones reales.

---

## Métodos Soportados

| Método | Tipo | Descripción |
|--------|------|-------------|
| `cart` | ML | Árboles de Clasificación y Regresión (FCS) |
| `rf` | ML | Random Forest (FCS) |
| `lgbm` | ML | LightGBM (FCS) |
| `xgboost` | ML | XGBoost (FCS) |
| `ctgan` | DL | Conditional GAN para tablas (Vía Synthcity) |
| `tvae` | DL | Variational Autoencoder tabular (Vía Synthcity) |
| `rtvae` | DL | VAE Tabular Regularizado con soporte de diferenciación latente |
| `great` | DL | Síntesis tabular basada en LLM (GReaT, Vía Synthcity) |
| `ddpm` | DL | Difusión Tabular TabDDPM (Vía Synthcity) |
| `diffusion` | DL | Difusión Tabular (Experimental, PyTorch) |
| `copula` | Estadístico | Síntesis basada en Copulas Gaussianas |
| `gmm` | Estadístico | Modelos de Mezcla Gaussiana |
| `kde` | Estadístico | Estimación de Densidad por Kernel |
| `smote` | Aug. | Sobremuestreo SMOTE |
| `adasyn` | Aug. | Muestreo adaptativo ADASYN |
| `resample` | Aug. | Bootstrap Simple |
| `dpgan` | Privacidad | GAN con Privacidad Diferencial (Vía Synthcity) |
| `pategan` | Privacidad | PATE-GAN con garantías DP (Vía Synthcity) |
| `scvi` | Single-Cell | scVI (Variational Inference) para RNA-seq |
| `scanvi` | Single-Cell | scANVI (semi-supervisado, condicionado por clase) |
| `gears` | Single-Cell | GEARS (Predicción de Perturbaciones) |
| `conditional_drift` | Drift | Condicionamiento temporal con TVAE/CTGAN — aprende distribuciones por etapa |
| `windowed_copula` | Drift | Cópula Gaussiana interpolada entre ventanas temporales |
| `hmm` | Drift | Modelo Oculto de Markov — el drift emerge de transiciones entre regímenes |

---

## Métodos de Generación con Drift Nativo

### Drift Condicional (`conditional_drift`)

Discretiza el eje temporal en `n_stages` etapas, entrena un modelo TVAE o CTGAN incluyendo la etapa como columna categórica y genera datos condicionados a cada etapa. El drift emerge de las diferencias de distribución entre etapas.

| Parámetro | Defecto | Descripción |
|-----------|---------|-------------|
| `time_col` | `None` | Columna temporal para ordenar los datos. Si es `None`, se usa el índice |
| `n_stages` | `5` | Número de etapas de drift |
| `base_method` | `"tvae"` | Modelo base: `"tvae"` o `"ctgan"` |
| `general_stages` | `None` | Etapas a generar (ej. `[3, 4]` para solo el final del drift). Si `None`, genera todas |

**Ejemplo:**
```python
gen.generate(
    df, 1000,
    method="conditional_drift",
    time_col="fecha",
    n_stages=5,
    base_method="tvae",
    general_stages=[3, 4],
)
```

---

### Cópula por Ventanas (`windowed_copula`)

Entrena una Cópula Gaussiana independiente por cada ventana temporal e interpola sus parámetros entre ventanas. Permite generar datos en cualquier punto intermedio del drift.

| Parámetro | Defecto | Descripción |
|-----------|---------|-------------|
| `time_col` | `None` | Columna para ordenar los datos cronológicamente antes de ventanear |
| `n_windows` | `5` | Número de ventanas temporales |
| `generate_at` | `None` | Puntos de interpolación en `[0.0, 1.0]`. `0.0` = primera ventana, `1.0` = última. Si `None`, genera en todas las ventanas |

> **Nota:** Solo columnas numéricas. Las columnas categóricas se ignoran.

**Ejemplo:**
```python
gen.generate(
    df, 1000,
    method="windowed_copula",
    time_col="timestamp",
    n_windows=4,
    generate_at=[0.0, 0.5, 1.0],
)
```

---

### HMM — Modelo Oculto de Markov (`hmm`)

Modela los datos como K regímenes ocultos con distribuciones gaussianas distintas. El drift emerge naturalmente de las probabilidades de transición entre regímenes que el modelo aprende durante el entrenamiento.

| Parámetro | Defecto | Descripción |
|-----------|---------|-------------|
| `n_components` | `4` | Número de regímenes ocultos |
| `covariance_type` | `"full"` | Tipo de covarianza por régimen: `"full"`, `"diag"`, `"tied"`, `"spherical"` |
| `n_iter` | `100` | Iteraciones del algoritmo EM |

> **Nota:** Solo columnas numéricas. Requiere `hmmlearn>=0.3.0`.

**Ejemplo:**
```python
gen.generate(df, 1000, method="hmm", n_components=3, covariance_type="diag")
```

---

## Escenarios de Uso Comunes (Guía Rápida)

### 1. Series Temporales (Time Series)
Para datos de series temporales, usa métodos tabulares estándar (CTGAN, TVAE, etc.) en datos temporales estructurados adecuadamente.
*   **Proyección de Futuro (Forecasting):** No es el caso de uso principal. Usa `StreamGenerator` para flujos infinitos o inyección de fechas manual.


### 2. Clasificación y Regresión (Supervisado)
Si tienes una columna `target` (ej. precio, churn) y la relación $X \rightarrow Y$ es crítica:
*   Usa `method="lgbm"` (LightGBM) o `method="rf"` (Random Forest).
*   Especifica siempre `target_col="nombre_columna"`.
    ```python
    # El generador detecta automáticamente si es Regresión o Clasificación
    gen.generate(data, target_col="precio", method="lgbm")
    ```

### 3. Clustering (No Supervisado)
Si no hay un target claro y quieres preservar grupos naturales de datos:
*   Usa `method="gmm"` (Gaussian Mixture Models, vía librería externa si disponible) o `method="tvae"` (Variational Autoencoder).
    ```python
    gen.generate(data, method="tvae")
    ```

### 4. Multi-Label (Etiquetas Múltiples)
Si una celda contiene múltiples valores (ej: `["A", "B", "C"]`) o formato string `"A,B,C"`:
*   **Limitación:** Los modelos estándar no manejan bien listas dentro de celdas.
*   **Solución:** Transforma la columna a **One-Hot Encoding** (múltiples columnas binarias `is_A`, `is_B`) antes de pasarla al generador. Los modelos basados en árboles (`lgbm`, `cart`) aprenderán las correlaciones entre etiquetas (ej: si `is_A=1` suele implicar `is_B=1`).

### 5. Datos por Bloques (Blocks)
Si tus datos están fragmentados lógicamente (ej: por Tiendas, Países, Pacientes) y quieres modelos independientes para cada uno:
*   Usa **`RealBlockGenerator`** en lugar de `RealGenerator`.
    ```python
    block_gen = RealBlockGenerator()
    block_gen.generate(data, block_column="TiendaID", method="cart")
    ```
    *Esto entrena un modelo diferente para cada TiendaID.*

### 6. Manejo de Datos Desbalanceados (Imbalance)
Si tu columna objetivo (`target`) tiene clases muy minoritarias que quieres potenciar:
*   **Balanceo Automático:** Usa `balance=True`. El generador aplicará técnicas de sobremuestreo (SMOTE/RandomOverSampler) internamente para que el modelo aprenda por igual de todas las clases.
    ```python
    gen.generate(data, target_col="fraude", balance=True, method="cart")
    ```
*   **Distribución Personalizada:** Puedes forzar una distribución marginal específica para cualquier columna categórica usando `custom_distributions`. El comportamiento exacto depende del método de síntesis:
    - **CTGAN / Deep Learning (Condicional):** Realiza *generación condicional real*. Genera conteos proporcionales exactos por clase directamente desde el modelo sintetizado, sin depender de un remuestreo posterior.
    - **SMOTE / ADASYN:** Traduce la distribución solicitada a conteos absolutos y los aplica nativamente como `sampling_strategy` en `imbalanced-learn`.
    - **GMM, TVAE, Copula, BN, scVI, DDPM:** Utiliza el método interno `_apply_postprocess_distribution`. Tras generar los datos sintéticos, remuestrea filas inteligentemente para cumplir las proporciones solicitadas preservando las correlaciones.
    - **Series Temporales (TimeGAN, TimeVAE, fflows):** `custom_distributions` y `balance` no son aplicables a secuencias temporales. Se emitirá un warning y el argumento será ignorado.

    ```python
    gen.generate(data, target_col="nivel", custom_distributions={"nivel": {"Bajo": 0.7, "Alto": 0.3}})
    ```
    *Nota: `balance=True` es un atajo para `custom_distributions={"col": "balanced"}`.*
---
---

### `ddpm` - Synthcity TabDDPM (Difusión Tabular Avanzada)

**Tipo:** Deep Learning (Modelo de Difusión)
**Mejor para:** Síntesis tabular de alta calidad, entornos de producción, grandes datasets
**Requisitos:** `synthcity` (incluido en la instalación base)

#### Descripción

TabDDPM (Tabular Denoising Diffusion Probabilistic Model) es la implementación avanzada de modelos de difusión para datos tabulares de Synthcity. Ofrece múltiples arquitecturas, schedulers avanzados y calidad superior comparada con el método `diffusion` personalizado.

#### Cuándo usarlo

✅ **Usa `ddpm` cuando:**
- Necesitas **calidad máxima** en datos sintéticos
- Trabajas con **grandes datasets** (>100k filas)
- En **entornos de producción** que requieren código robusto y mantenido
- Necesitas **arquitecturas avanzadas** (ResNet, TabNet)
- Quieres **cosine scheduling** para una mejor difusión
- Tienes **tiempo para entrenamientos largos** (1000 épocas por defecto)

❌ **No uses `ddpm` cuando:**
- Necesitas **prototipado rápido** (usa `diffusion` en su lugar)
- Trabajas con **datasets muy pequeños** (<1k filas)
- Tienes **recursos computacionales limitados**
- Necesitas **modificaciones personalizadas** al algoritmo

#### Parámetros

```python
synth = gen.generate(
    data,
    method='ddpm',
    n_samples=1000,

    # Training parameters
    n_iter=1000,                    # Training epochs (default: 1000)
    lr=0.002,                       # Learning rate (default: 0.002)
    batch_size=1024,                # Batch size (default: 1024)

    # Diffusion parameters
    num_timesteps=1000,             # Diffusion timesteps (default: 1000)
    scheduler='cosine',             # 'cosine' or 'linear' (default: 'cosine')
    gaussian_loss_type='mse',       # 'mse' or 'kl' (default: 'mse')

    # Model architecture
    model_type='mlp',               # 'mlp', 'resnet', or 'tabnet' (default: 'mlp')
    model_params={                  # Architecture-specific parameters
        'n_layers_hidden': 3,
        'n_units_hidden': 256,
        'dropout': 0.0
    },

    # Task type
    is_classification=False,        # True for classification tasks
)
```

#### Detalles de Parámetros

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `n_iter` | int | 1000 | Número de épocas de entrenamiento |
| `lr` | float | 0.002 | Tasa de aprendizaje para el optimizador |
| `batch_size` | int | 1024 | Tamaño de batch de entrenamiento |
| `num_timesteps` | int | 1000 | Número de timesteps de difusión |
| `scheduler` | str | `'cosine'` | Planificador Beta: `'cosine'` (recomendado) o `'linear'` |
| `gaussian_loss_type` | str | `'mse'` | Función de pérdida: `'mse'` o `'kl'` |
| `model_type` | str | `'mlp'` | Arquitectura: `'mlp'`, `'resnet'`, o `'tabnet'` |
| `model_params` | dict | Ver arriba | Parámetros específicos de la arquitectura |
| `is_classification` | bool | False | Establecer en True para tareas de clasificación |

#### Tipos de Modelo

**MLP (Multi-Layer Perceptron)**
- Mejor para: Datos tabulares generales
- Velocidad: Rápida
- Parámetros: `n_layers_hidden`, `n_units_hidden`, `dropout`

**ResNet (Residual Network)**
- Mejor para: Relaciones complejas entre características
- Velocidad: Media
- Parámetros: `n_layers_hidden`, `n_units_hidden`, `dropout`

**TabNet**
- Mejor para: Datos tabulares con importancia de características
- Velocidad: Más lenta
- Parámetros: Específicos de la arquitectura TabNet

#### Comparación: `diffusion` vs `ddpm`

| Aspecto | `diffusion` (personalizado) | `ddpm` (Synthcity) |
|--------|---------------------|-------------------|
| **Velocidad** | ⚡ Rápida (100 épocas) | 🐢 Más lenta (1000 épocas) |
| **Calidad** | ⭐⭐⭐ Buena | ⭐⭐⭐⭐ Excelente |
| **Arquitecturas** | Solo MLP | MLP/ResNet/TabNet |
| **Scheduler** | Lineal | Cosine/Linear |
| **Tamaño de Batch** | 64 | 1024 |
| **Caso de Uso** | Prototipado rápido | Calidad de producción |
| **Personalización** | Fácil de modificar | Caja negra |
| **Mantenimiento** | Tu responsabilidad | Equipo de Synthcity |

#### Usage Examples

**Basic Usage:**
```python
from calm_data_generator import RealGenerator
import pandas as pd

gen = RealGenerator()
synth = gen.generate(
    data,
    method='ddpm',
    n_samples=1000,
    n_iter=500  # Reduce for faster training
)
```

**Classification Task:**
```python
synth = gen.generate(
    data,
    method='ddpm',
    n_samples=1000,
    is_classification=True,
    target_col='label'
)
```

**Advanced Architecture:**
```python
synth = gen.generate(
    data,
    method='ddpm',
    n_samples=1000,
    model_type='resnet',
    model_params={
        'n_layers_hidden': 5,
        'n_units_hidden': 512,
        'dropout': 0.1
    },
    scheduler='cosine',
    n_iter=2000
)
```

---

### `timegan` - TimeGAN (Time Series GAN)

**Tipo:** Deep Learning (GAN para Series Temporales)
**Mejor para:** Patrones temporales complejos, series temporales multi-entidad
**Requisitos:** `synthcity` (incluido en instalación base)

#### Descripción

TimeGAN (Time-series Generative Adversarial Network) está diseñado específicamente para datos secuenciales/temporales. Aprende tanto la dinámica temporal como la distribución de características, haciéndolo ideal para series temporales con patrones complejos.

#### Cuándo usarlo

✅ **Usa `timegan` cuando:**
- Tienes **datos de series temporales** con dependencias temporales
- Trabajas con **secuencias multi-entidad** (ej. múltiples usuarios/sensores)
- Necesitas preservar **dinámicas temporales**
- Tienes **patrones temporales complejos** para aprender
- Necesitas síntesis de series temporales de **alta calidad**

❌ **No uses `timegan` cuando:**
- Tienes **datos tabulares simples** (usa `ctgan` o `ddpm` en su lugar)
- Trabajas con **secuencias muy cortas** (<10 pasos de tiempo)
- Necesitas **generación rápida** (usa `timevae` en su lugar)
- Tienes **recursos computacionales limitados**

#### Requisitos de Datos

TimeGAN espera datos en un formato temporal específico:
- **Orden temporal**: Los datos deben estar ordenados por tiempo
- **Agrupación por entidad**: Si es multi-entidad, agrupa por ID de entidad
- **Pasos consistentes**: Preferible intervalos de tiempo regulares

#### Parámetros

```python
synth = gen.generate(
    data,
    method='timegan',
    n_samples=100,  # Número de secuencias a generar

    # Parámetros de entrenamiento
    n_iter=1000,                    # Épocas de entrenamiento (defecto: 1000)
    n_units_hidden=100,             # Unidades ocultas en RNN (defecto: 100)
    batch_size=128,                 # Tamaño de batch (defecto: 128)
    lr=0.001,                       # Tasa de aprendizaje (defecto: 0.001)
)
```

#### Detalles de Parámetros

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `n_iter` | int | 1000 | Número de épocas de entrenamiento |
| `n_units_hidden` | int | 100 | Número de unidades ocultas en capas RNN |
| `batch_size` | int | 128 | Tamaño de batch de entrenamiento |
| `lr` | float | 0.001 | Tasa de aprendizaje para el optimizador |

#### Ejemplos de Uso

**Basic Time Series:**
```python
from calm_data_generator import RealGenerator
import pandas as pd

# Data must have temporal structure
# Example: sensor readings over time
gen = RealGenerator()
synth = gen.generate(
    time_series_data,
    method='timegan',
    n_samples=100,  # Generate 100 sequences
    n_iter=1000,
    n_units_hidden=100
)
```

**Multi-Entity Time Series:**
```python
# Data with multiple entities (e.g., users, sensors)
# Ensure data is sorted by entity_id and timestamp
synth = gen.generate(
    multi_entity_data,
    method='timegan',
    n_samples=50,  # Generate 50 entity sequences
    n_iter=2000,
    n_units_hidden=150,
    batch_size=64
)
```

---

### `timevae` - TimeVAE (Time Series VAE)

**Tipo:** Deep Learning (VAE para Series Temporales)
**Mejor para:** Series temporales regulares, entrenamiento más rápido que TimeGAN
**Requisitos:** `synthcity` (incluido en la instalación base)

#### Descripción

TimeVAE es un autoencoder variacional diseñado para datos temporales. Generalmente es más rápido que TimeGAN y funciona bien para series temporales regulares con patrones consistentes.

#### Cuándo usarlo

✅ **Usa `timevae` cuando:**
- Tienes datos de **series temporales regulares**
- Necesitas un **entrenamiento más rápido** que TimeGAN
- Trabajas con **patrones temporales consistentes**
- Quieres **buena calidad** con **menos computación**
- Tienes **secuencias de longitud moderada**

#### Requisitos de Datos

Similar a TimeGAN:
- **Orden temporal**: Datos ordenados por tiempo
- **Intervalos regulares**: Funciona mejor con pasos de tiempo consistentes
- **Agrupación por entidad**: Si es multi-entidad, agrupa por ID de entidad

#### Parámetros

```python
synth = gen.generate(
    data,
    method='timevae',
    n_samples=100,  # Number of sequences to generate

    # Training parameters
    n_iter=1000,                    # Training epochs (default: 1000)
    decoder_n_layers_hidden=2,      # Decoder layers (default: 2)
    decoder_n_units_hidden=100,     # Decoder units (default: 100)
    batch_size=128,                 # Batch size (default: 128)
    lr=0.001,                       # Learning rate (default: 0.001)
)
```

#### Parameter Details

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `n_iter` | int | 1000 | Número de épocas de entrenamiento |
| `decoder_n_layers_hidden` | int | 2 | Número de capas ocultas en el decodificador |
| `decoder_n_units_hidden` | int | 100 | Número de unidades ocultas en el decodificador |
| `batch_size` | int | 128 | Tamaño de batch de entrenamiento |
| `lr` | float | 0.001 | Tasa de aprendizaje para el optimizador |

---

## Guardado y Carga de Modelos

`RealGenerator` permite guardar modelos generadores entrenados y cargarlos posteriormente para inferencia sin re-entrenar. Esto es útil para pipelines de producción donde el entrenamiento es costoso.

### Guardar un Modelo

Después de generar datos (lo cual entrena el modelo subyacente), puedes guardar el generador:

```python
# 1. Entrenar y Generar
gen.generate(data, n_samples=1000, method="ctgan", batch_size=500)

# 2. Guardar el generador entrenado
gen.save("models/mi_modelo_ctgan.pkl")
```
> **Nota:** El archivo guardado es un archivo zip que contiene la configuración del `RealGenerator` y el modelo subyacente (ej. estado del plugin de Synthcity).

### Cargar un Modelo

Puedes cargar un modelo guardado usando el método de clase `load()`. Una vez cargado, puedes generar más muestras sin proporcionar los datos de entrenamiento originales.

```python
from calm_data_generator.generators.tabular import RealGenerator

# 1. Cargar el generador
loaded_gen = RealGenerator.load("models/mi_modelo_ctgan.pkl")

# 2. Generar nuevas muestras (¡No se necesita argumento 'data'!)
new_samples = loaded_gen.generate(n_samples=500)
```

> **Advertencia:** Al generar desde un modelo cargado, **no debes** pasar `data` a `generate()`, pero **debes** pasar `n_samples`.

> **Nota:** Los modelos scVI y scANVI usan un formato de guardado basado en directorios internamente. Estos se empaquetan dentro del archivo zip de forma transparente — `save`/`load` funciona igual que con todos los demás métodos.

---

## Métodos de Privacidad

### `privatize()` — Aplicar Privacidad Diferencial a Datos Existentes

Aplica mecanismos de privacidad diferencial directamente sobre un DataFrame real. A diferencia de `dpgan`/`pategan` (que entrenan un modelo generativo), `privatize` es una transformación directa de los datos de entrada.

- **Columnas numéricas:** Se añade ruido Laplace o Gaussiano.
- **Columnas categóricas:** Se aplica Respuesta Aleatorizada (Randomized Response).

```python
# Mecanismo Laplace (por defecto)
private_df = gen.privatize(df, epsilon=1.0)

# Mecanismo Gaussiano
private_df = gen.privatize(df, epsilon=1.0, mechanism="gaussian", delta=1e-5)
```

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `data` | DataFrame | - | DataFrame de entrada a privatizar |
| `epsilon` | float | `1.0` | Presupuesto de privacidad ε. Menor = más privado. |
| `delta` | float | `None` | Requerido para el mecanismo Gaussiano. |
| `numeric_sensitivity` | float | `1.0` | Sensibilidad global para columnas numéricas. |
| `mechanism` | str | `'laplace'` | `'laplace'` o `'gaussian'` para columnas numéricas. |
| `categorical_p` | float | `None` | Probabilidad de mantener la categoría real. Si None, se deriva de ε. |

---

## Modelos Personalizados: `generate_custom()` y `CustomPluginAdapter`

### `generate_custom()` — Usar Cualquier Modelo Externo

Envuelve cualquier modelo externo (sklearn, synthcity, copulae, etc.) para usarlo con `RealGenerator`. El adaptador detecta automáticamente la interfaz del modelo (`fit`/`train`, `generate`/`sample`/`random`). Puedes sobreescribir con lambdas explícitas para control total.

```python
from sklearn.neighbors import KernelDensity

kde_model = KernelDensity(kernel='gaussian', bandwidth=0.5)

synthetic_df = gen.generate_custom(
    data=df,
    model=kde_model,
    n_samples=500,
    fit_fn=lambda m, data: m.fit(data.values),
    generate_fn=lambda m, n: pd.DataFrame(m.sample(n), columns=df.columns),
    method_name="mi_kde",
)
```

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `data` | DataFrame | - | Datos de entrenamiento |
| `model` | any | - | Cualquier objeto modelo con fit/train y generate/sample/random |
| `n_samples` | int | - | Número de muestras sintéticas a generar |
| `fit_fn` | callable | `None` | `lambda model, data: ...` — sobreescribe el método fit auto-detectado |
| `generate_fn` | callable | `None` | `lambda model, n: ...` — sobreescribe el método de generación auto-detectado |
| `postprocess_fn` | callable | `None` | Post-procesa el DataFrame generado |
| `method_name` | str | `"custom"` | Etiqueta para logging y metadatos |

---

## Mejores Prácticas

6. **Desbalance severo:** Usa `smote` o `adasyn` con `target_col`.

#### Comparación: `timegan` vs `timevae`

| Aspecto | `timegan` | `timevae` |
|--------|-----------|-----------|
| **Velocidad** | 🐢 Más lenta | ⚡ Más rápida |
| **Calidad** | ⭐⭐⭐⭐ Excelente | ⭐⭐⭐ Buena |
| **Complejidad** | Maneja patrones complejos | Mejor para patrones regulares |
| **Tiempo Entr.** | Mayor | Menor |
| **Caso de Uso** | Dinámicas temporales complejas | Series temporales regulares |

#### Usage Examples

**Basic Time Series:**
```python
from calm_data_generator import RealGenerator
import pandas as pd

gen = RealGenerator()
synth = gen.generate(
    time_series_data,
    method='timevae',
    n_samples=100,
    n_iter=500,  # Faster than TimeGAN
    decoder_n_units_hidden=100
)
```

**Faster Training:**
```python
# Reduce parameters for quick prototyping
synth = gen.generate(
    time_series_data,
    method='timevae',
    n_samples=50,
    n_iter=300,
    decoder_n_layers_hidden=1,
    decoder_n_units_hidden=50,
    batch_size=64
)
```

---

### `fflows` - FourierFlows (Flujos Normalizantes en Dominio de Frecuencia)

**Tipo:** Deep Learning (Normalizing Flows para Series Temporales)
**Mejor para:** Series temporales periódicas/quasi-periódicas, alternativa estable a TimeGAN
**Requisitos:** `synthcity`

#### Descripción

`fflows` aplica flujos normalizantes en el dominio de la frecuencia para generar secuencias temporales. Es generalmente más estable que TimeGAN y destaca en series con patrones periódicos (sinusoidales, estacionales).

```python
synth = gen.generate(
    data,
    method='fflows',
    n_samples=100,
    sequence_key='seq_id',   # Columna que identifica cada secuencia
    time_key='timestamp',    # Columna con marcas de tiempo
    n_iter=1000,
    batch_size=128,
    lr=0.001,
)
```

#### Comparación: `timegan` vs `timevae` vs `fflows`

| Aspecto | `timegan` | `timevae` | `fflows` |
|---------|-----------|-----------|----------|
| **Velocidad** | 🐢 Lento | ⚡ Rápido | ⚡ Rápido |
| **Calidad** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Estabilidad** | Baja | Media | Alta |
| **Mejor para** | Patrones complejos | Series regulares | Series periódicas |

---

### `bn` - Red Bayesiana (Bayesian Network)

**Tipo:** Modelo Gráfico Probabilístico
**Mejor para:** Datos tabulares clínicos/estructurados con dependencias causales entre variables
**Requisitos:** `synthcity`

#### Descripción

Una Red Bayesiana modela las dependencias condicionales entre variables usando un grafo acíclico dirigido. El aprendizaje de estructura descubre qué variables influyen causalmente en otras. Especialmente útil para datos sanitarios y clínicos.

```python
synth = gen.generate(
    data,
    method='bn',
    n_samples=1000,
    target_col='diagnostico',
)
```

✅ **Usa `bn` cuando:**
- Los datos tienen **relaciones causales** entre variables (ej. diagnóstico ← síntomas ← analíticas)
- Trabajas con datos **clínicos o epidemiológicos**
- Quieres un **modelo interpretable** (la red es inspeccionable)

❌ **No uses `bn` cuando:**
- Los datos son **alta dimensionalidad** (100+ variables) — el aprendizaje de estructura se vuelve lento
- Necesitas datos de **series temporales** (usa `timegan`/`timevae`/`fflows`)

---

## Guía de Selección de Métodos

### Para Datos Tabulares

| Escenario | Método Recomendado | Alternativa |
|----------|-------------------|-------------|
| **Prototipado rápido** | `diffusion` | `cart`, `rf` |
| **Calidad de producción** | `ddpm` | `ctgan` |
| **Datasets grandes (>100k)** | `ddpm`, `lgbm` | `ctgan` |
| **Datasets pequeños (<1k)** | `cart`, `rf` | `diffusion` |
| **Desbalance de clases** | `smote`, `adasyn` | `ctgan` |
| **Preservar correlaciones** | `ctgan`, `ddpm` | `copula` |
| **Generación rápida** | `cart`, `diffusion` | `rf` |
| **Calidad máxima** | `ddpm` (ResNet) | `ctgan` |

### Para Series Temporales

| Escenario | Método Recomendado | Alternativa |
|----------|-------------------|-------------|
| **Patrones temporales complejos** | `timegan` | `fflows` |
| **Series temporales regulares** | `timevae` | `timegan` |
| **Series periódicas/estacionales** | `fflows` | `timevae` |
| **Entrenamiento rápido** | `timevae` | `fflows` |
| **Secuencias multi-entidad** | `timegan` | `fflows` |
| **Calidad máxima** | `timegan` | `fflows` |

### Para Casos Especiales

| Tipo de Dato | Método Recomendado |
|-----------|-------------------|
| **RNA-seq single-cell** | `scvi` |
| **Datos clínicos tabulares** | `bn` o `ClinicalDataGenerator` |
| **Datos clínicos** | Usa `ClinicalDataGenerator` |
| **Datos en streaming** | Usa `StreamGenerator` |
| **Datos por bloques** | Usa `RealBlockGenerator` |

---

## Novedades v1.2.0 — v2.0.0

### Diferenciación en el Espacio Latente (`differentiation_factor`)

Disponible para `tvae` y `scvi`. Controla cuánto se separan los centroides de clase en el espacio latente durante la síntesis. (Para `ctgan`, este parámetro se ignora actualmente).

```python
synth = gen.generate(
    data=df,
    n_samples=500,
    method="tvae",
    target_col="grupo",
    differentiation_factor=1.5  # Empujar clases más separadas
)
```

| Valor | Efecto |
|-------|--------|
| `0.0` | Sin desplazamiento (comportamiento por defecto) |
| `0.5–1.0` | Separación suave |
| `1.5–2.0` | Separación moderada/fuerte |
| `> 2.0` | Riesgo de muestras fuera de distribución |

> **TVAE:** El desplazamiento se aplica directamente en el espacio latente neuronal (vectores mu).
> **scVI:** El desplazamiento se aplica en el espacio latente `z` antes de decodificar.

---

### Visibilidad del Entrenamiento (`verbose_training`)

Pasa `verbose_training=True` al instanciar para dejar que Synthcity imprima la pérdida por época:

```python
gen = RealGenerator(verbose_training=True)
gen.generate(data=df, n_samples=500, method="tvae", epochs=200)
# → 2024-03-06 14:01:12 | INFO | tvae | epoch 1/200 | loss: 1.2341
# → 2024-03-06 14:01:15 | INFO | tvae | epoch 2/200 | loss: 1.1872
# → ...
```

Para **scVI**, la barra de progreso de PyTorch Lightning siempre se muestra. Al terminar, la pérdida final también se registra en el logger de Python.

---

### Métodos de Introspección (Accessors)

Tras llamar a `generate()`, estos métodos exponen el estado interno del modelo:

#### `get_encoder()`

Devuelve la red encodificadora del último modelo entrenado.

```python
synth = gen.generate(df, 500, method="tvae", target_col="etiqueta")
encoder = gen.get_encoder()
# TVAE: nn.Module (encoder del VAE interno)
# scVI: module.z_encoder
# Otros: None
```

#### `get_decoder()`

Devuelve la red descodificadora.

```python
decoder = gen.get_decoder()
# Devuelve nn.Module para tvae y scvi
```

#### `get_latest_embeddings()`

Devuelve los embeddings del espacio latente calculados durante la última síntesis que aplicó `differentiation_factor`.

```python
embeddings = gen.get_latest_embeddings()  # np.ndarray o None
if embeddings is not None:
    print(f"Forma del embedding: {embeddings.shape}")  # (n_muestras, n_latente)
    # Ej. para UMAP:
    import umap
    reducer = umap.UMAP()
    proyeccion = reducer.fit_transform(embeddings)
```

> Devuelve `None` si no se aplicó diferenciación o si el modelo usó el fallback en espacio de características.

#### `get_training_history()`

Devuelve el diccionario de historial de entrenamiento (**solo scVI/scANVI**).

```python
synth = gen.generate(df, 500, method="scvi", epochs=100)
history = gen.get_training_history()

if history:
    import matplotlib.pyplot as plt
    elbo = history["elbo_train"]
    plt.plot(elbo.values)
    plt.xlabel("Época")
    plt.ylabel("ELBO")
    plt.title("Evolución del entrenamiento scVI")
    plt.show()
```

| Clave | Descripción |
|-------|-------------|
| `train_loss_epoch` | Pérdida total de entrenamiento por época |
| `elbo_train` | Cota inferior de la evidencia (ELBO) |
| `reconstruction_loss_train` | Pérdida de reconstrucción (expresión) |
| `kl_local_train` | Divergencia KL – término local por célula |
| `kl_global_train` | Divergencia KL – término global |

> Devuelve `None` para modelos basados en Synthcity (TVAE, CTGAN). Usa `verbose_training=True` para esos.

#### `get_synthesizer_model()`

Devuelve el objeto del modelo subyacente (plugin de Synthcity, modelo scVI, modelo sklearn, etc.).

```python
raw = gen.get_synthesizer_model()
# tvae/ctgan: objeto plugin de Synthcity
# scvi:       instancia scvi.model.SCVI
# cart/rf:    instancia FCSModel
```
