# ComplexGenerator — Referencia

**Ubicación:** `calm_data_generator.generators.complex.ComplexGenerator`

---

## ¿Qué es ComplexGenerator?

`ComplexGenerator` es el **núcleo matemático** de la librería. Es una clase abstracta intermedia que vive entre `BaseGenerator` (interfaz pura) y los generadores de dominio concretos (`ClinicalDataGenerator`, etc.).

Su propósito es proporcionar tres motores matemáticos reutilizables — **Copula Gaussiana incondicional**, **Copula Gaussiana condicional** y **efectos estocásticos** — de forma que cualquier generador de dominio nuevo pueda heredarlos sin reimplementar la matemática.

```
BaseGenerator (ABC)
└── ComplexGenerator                  ← motores matemáticos (este módulo)
    ├── ClinicalDataGenerator         ← dominio clínico (genes, proteínas, demografía)
    ├── (tu propio generador)         ← hereda los tres motores gratis
    └── ...
```

Si quieres crear un generador para un dominio nuevo (sensores IoT, datos financieros, datos de seguros...), **heredar de `ComplexGenerator` es el punto de partida correcto**.

---

## La Copula Gaussiana — base de los dos primeros motores

Ambos motores de generación se basan en la misma idea: la **Copula Gaussiana**.

El problema que resuelve: generar `n` variables correlacionadas donde cada variable tiene su propia distribución marginal (NegBinomial, Gamma, Normal, Exponencial...).

El algoritmo en tres pasos:

```
1.  Z ~ N(0, Σ)          → vector gaussiano con la correlación deseada
2.  U = Φ(Z)             → transforma a uniforme [0,1] mediante CDF normal
                           (preserva la estructura de correlación de rangos)
3.  X_i = F_i⁻¹(U_i)    → aplica la distribución marginal de cada variable
                           mediante su función cuantil (PPF)
```

Resultado: `X` tiene exactamente la distribución marginal `F_i` en cada columna, y la correlación de Spearman especificada en `Σ`. La correlación de Pearson es muy similar para distribuciones simétricas, y ligeramente atenuada para distribuciones muy asimétricas (NegBinomial, Exponencial).

**Importante:** la matriz `Σ` debe ser semidefinida positiva (PSD). Si no lo es, ambos motores la reparan automáticamente recortando autovalores negativos a `1e-6` — sin lanzar excepción.

---

## Ejemplo completo: generador de sensores IoT

Este ejemplo muestra cómo construir un generador de dominio completo desde cero usando los tres motores de `ComplexGenerator`.

**Dominio:** red de sensores industriales. Cada fila es una lectura temporal. Hay tres tipos de variables: temperatura (normal), vibración (lognormal) y conteo de errores (negbinomial). Los sensores están correlacionados entre sí. Podemos inyectar anomalías en cualquier subconjunto de sensores.

```python
import numpy as np
import pandas as pd
import scipy.stats as stats

from calm_data_generator.generators.complex.ComplexGenerator import ComplexGenerator


class IoTSensorGenerator(ComplexGenerator):
    """
    Generador de lecturas de sensores industriales correlacionados.

    Variables por sensor:
      - temperatura:  Normal(μ, σ)
      - vibración:    LogNormal(s, scale)
      - errores:      NegBinomial(n, p)  ← discreta, cuenta de fallos

    Uso:
        gen = IoTSensorGenerator(random_state=42)
        df  = gen.generate(n_readings=500, n_sensors=6)
        df_anomaly = gen.generate(n_readings=500, n_sensors=6,
                                  anomaly_sensors=[0, 2], anomaly_type="fold_change",
                                  anomaly_value=[1.5, 3.0])
    """

    def generate(
        self,
        n_readings: int,
        n_sensors: int,
        sensor_correlation: float = 0.4,
        anomaly_sensors: list = None,
        anomaly_type: str = "additive_shift",
        anomaly_value = 5.0,
    ) -> pd.DataFrame:

        # ── 1. Definir marginales por sensor ──────────────────────────────────
        # Cada sensor tiene parámetros ligeramente distintos (variedad realista)
        rng = np.random.default_rng(self.rng.integers(2**31))
        marginals = []
        for i in range(n_sensors):
            temp_mean = rng.uniform(60, 80)      # temperatura media entre 60-80°C
            marginals.append(stats.norm(loc=temp_mean, scale=rng.uniform(2, 5)))

        # ── 2. Definir estructura de correlación ─────────────────────────────
        # Todos los sensores correlacionados en bloque (misma planta)
        sigma = np.full((n_sensors, n_sensors), sensor_correlation)
        np.fill_diagonal(sigma, 1.0)

        # ── 3. Generar lecturas correlacionadas (Motor 1) ─────────────────────
        X = self._generate_correlated_module(n_readings, marginals, sigma)
        columns = [f"sensor_{i:02d}_temp" for i in range(n_sensors)]
        df = pd.DataFrame(X, columns=columns)

        # ── 4. Inyectar anomalías si se piden (Motor 3) ───────────────────────
        if anomaly_sensors:
            # Anomalía en la segunda mitad de las lecturas
            anomaly_rows = df.index[n_readings // 2:]
            effect_config = {
                "index":        anomaly_sensors,
                "effect_type":  anomaly_type,
                "effect_value": anomaly_value,
            }
            self.apply_stochastic_effects(df, anomaly_rows, effect_config)
            df["anomaly"] = 0
            df.loc[anomaly_rows, "anomaly"] = 1

        return df


# ── Uso ───────────────────────────────────────────────────────────────────────

gen = IoTSensorGenerator(random_state=42, auto_report=False)

# Generación normal
df_normal = gen.generate(n_readings=1000, n_sensors=6, sensor_correlation=0.5)
print(df_normal.head())
print(f"\nCorrelación media entre sensores: {df_normal.corr().values[~np.eye(6, dtype=bool)].mean():.3f}")

# Generación con anomalía en sensores 0 y 2
df_anomaly = gen.generate(
    n_readings=1000,
    n_sensors=6,
    sensor_correlation=0.5,
    anomaly_sensors=[0, 2],
    anomaly_type="additive_shift",
    anomaly_value=[10.0, 20.0],   # subida aleatoria entre +10 y +20°C
)
print(f"\nLecturas con anomalía: {df_anomaly['anomaly'].sum()}/1000")
```

---

## Motor 1: `_generate_correlated_module`

Copula Gaussiana incondicional. Genera datos donde cada columna tiene su propia distribución marginal y todas están correlacionadas según `sigma_module`.

### Firma

```python
def _generate_correlated_module(
    self,
    n_samples: int,
    marginals_list: list,      # distribuciones scipy congeladas, una por variable
    sigma_module: np.ndarray,  # matriz de correlación (n_vars × n_vars)
) -> np.ndarray:               # forma: (n_samples, n_vars)
```

### Parámetros

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `n_samples` | int | Número de filas a generar |
| `marginals_list` | list | Distribuciones scipy `frozen rv`, una por columna. Pueden ser distintas entre sí |
| `sigma_module` | np.ndarray | Matriz de correlación. Si no es PSD se repara automáticamente |

### Cuándo usarlo

Siempre que necesites generar múltiples variables correlacionadas con distribuciones heterogéneas. Es el caso más común.

### Ejemplo

```python
import numpy as np
import scipy.stats as stats
from calm_data_generator.generators.complex.ComplexGenerator import ComplexGenerator

class MiGenerador(ComplexGenerator):
    def generate(self, n):
        marginals = [
            stats.norm(loc=100, scale=15),      # edad normalizada
            stats.lognorm(s=0.8, scale=50000),  # ingresos (sesgados)
            stats.nbinom(n=5, p=0.3),           # número de visitas (discreta)
        ]
        sigma = np.array([
            [1.0,  0.3, -0.2],
            [0.3,  1.0,  0.1],
            [-0.2, 0.1,  1.0],
        ])
        X = self._generate_correlated_module(n, marginals, sigma)
        return pd.DataFrame(X, columns=["edad", "ingresos", "visitas"])

gen = MiGenerador(random_state=42, auto_report=False)
df = gen.generate(1000)
```

---

## Motor 2: `_generate_conditional_data`

Copula Gaussiana condicional. Genera variables **dado que ya conocemos otras**. Útil para modelar causalidad o dependencias observadas: genes condicionados a demografía, precios condicionados a indicadores macroeconómicos, etc.

### Firma

```python
def _generate_conditional_data(
    self,
    n_samples: int,
    conditioning_data: np.ndarray,    # datos observados (n_samples, n_cond)
    conditioning_marginals: list,      # marginales de las variables conocidas
    target_marginals: list,            # marginales de las variables a generar
    full_covariance: np.ndarray,       # covarianza conjunta (n_cond + n_target, n_cond + n_target)
) -> np.ndarray:                       # forma: (n_samples, n_target)
```

### Parámetros

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `n_samples` | int | Número de muestras |
| `conditioning_data` | np.ndarray | Variables ya observadas, forma `(n_samples, n_cond)` |
| `conditioning_marginals` | list | Marginales de las variables conocidas. Las distribuciones discretas se manejan automáticamente con Residuos Cuantílicos Aleatorizados (RQR) |
| `target_marginals` | list | Marginales de las variables a generar |
| `full_covariance` | np.ndarray | Matriz de covarianza conjunta sobre todas las variables |

### Cómo funciona internamente

```
Z_cond = Φ⁻¹(F_cond(conditioning_data))    ← espacio latente Gaussiano
μ_t|c  = S_tc · S_cc⁻¹ · Z_cond            ← media condicional por muestra
Σ_t|c  = S_tt - S_tc · S_cc⁻¹ · S_ct      ← covarianza condicional (fija)
Z_t    ~ N(μ_t|c, Σ_t|c)                   ← muestra condicional
X_t    = F_target⁻¹(Φ(Z_t))               ← aplica marginales objetivo
```

### Cuándo usarlo

Cuando tienes datos reales de algunas variables y quieres generar datos sintéticos de otras que sean **coherentes** con las observadas. Ejemplo: ya tienes datos demográficos reales de 100 pacientes y quieres generar expresión génica sintética que sea realista dado su perfil.

### Ejemplo

```python
# Tenemos datos reales de 2 variables demográficas
edad_imc = np.column_stack([
    np.random.normal(55, 12, 100),   # edad
    np.random.normal(27,  4, 100),   # IMC
])

# Queremos generar 5 genes condicionados a esos datos
demographic_marginals = [stats.norm(55, 12), stats.norm(27, 4)]
gene_marginals = [stats.lognorm(s=0.5, scale=200) for _ in range(5)]

# Covarianza conjunta 7×7: primeras 2 filas/cols = demografía, resto = genes
joint_cov = np.eye(7)
joint_cov[0, 2] = joint_cov[2, 0] = 0.4   # edad correlacionada con gen 0
joint_cov[1, 3] = joint_cov[3, 1] = -0.3  # IMC correlacionado con gen 1

X_genes = gen._generate_conditional_data(
    n_samples=100,
    conditioning_data=edad_imc,
    conditioning_marginals=demographic_marginals,
    target_marginals=gene_marginals,
    full_covariance=joint_cov,
)
# X_genes.shape == (100, 5)
```

---

## Motor 3: `apply_stochastic_effects`

Aplica un efecto estocástico a un subconjunto de entidades **en-lugar**. Sirve para inyectar señales de enfermedad, choques de mercado, anomalías de sensores, drift temporal, etc.

### Firma

```python
def apply_stochastic_effects(
    self,
    df: pd.DataFrame,     # modificado en-lugar, sin valor de retorno
    entity_ids,           # etiquetas del índice de las entidades afectadas
    effect_config: dict,
) -> None:
```

### Configuración del efecto

```python
effect_config = {
    "index":        [0, 1, 5],       # índices de columnas a afectar
    "effect_type":  "fold_change",   # uno de los 7 tipos
    "effect_value": [1.5, 3.0],      # escalar o [mín, máx] para muestrear
}
```

Cuando `effect_value` es `[mín, máx]`, cada entidad recibe un valor independiente muestreado de `Uniform(mín, máx)`. Cuando es un escalar, se muestrea de `Normal(valor, |valor|·0.1)`.

### Tipos de efectos

| Tipo | Fórmula | Caso de uso típico |
|------|---------|-------------------|
| `additive_shift` | `x += offset` | Sesgo de sensor, señal de fondo |
| `fold_change` | `x *= factor` | Sobreexpresión génica, multiplicador de precio |
| `power_transform` | `x **= exponente` | Distorsión no lineal |
| `variance_scale` | Reescala alrededor de la media | Heterocedasticidad, régimen de volatilidad |
| `log_transform` | `x = log(x + ε)` | Logaritmización de conteos |
| `polynomial_transform` | `x = P(x)` | Transformación polinomial arbitraria |
| `sigmoid_transform` | `x = 1/(1+e^{-k(x-x₀)})` | Saturación, recorte suave |

`simple_additive_shift` se acepta como alias de `additive_shift`.

### Ejemplo: anomalía en últimas 100 lecturas

```python
# Subida brusca de temperatura en sensores 0 y 3
effect = {
    "index":        [0, 3],
    "effect_type":  "additive_shift",
    "effect_value": [15.0, 25.0],  # +15 a +25°C aleatorio por fila
}
# Aplicar solo en las últimas 100 filas
gen.apply_stochastic_effects(df, df.index[-100:], effect)
```

### Ejemplo: señal de enfermedad (fold change) en pacientes enfermos

```python
sick_patients = demo_df[demo_df["Group"] == "Disease"].index
effect = {
    "index":        list(range(10, 20)),  # genes 10-19 sobreexpresados
    "effect_type":  "fold_change",
    "effect_value": [2.0, 5.0],           # 2x a 5x sobreexpresión
}
gen.apply_stochastic_effects(genes_df, sick_patients, effect)
```

---

## Cuándo usar cada motor

| Situación | Motor |
|-----------|-------|
| Generar variables correlacionadas desde cero | `_generate_correlated_module` |
| Generar variables condicionadas a datos ya existentes | `_generate_conditional_data` |
| Añadir señales, anomalías o efectos a datos ya generados | `apply_stochastic_effects` |
| Las tres a la vez | Combínalos dentro de tu `generate()` |

---

## Manejo de errores

| Situación | Comportamiento |
|-----------|---------------|
| `sigma_module` no-PSD | Reparado automáticamente (recorte de autovalores, sin excepción) |
| `S_cc` singular en el condicional | Regularizado con `+1e-6·I` |
| Covarianza condicional no-PSD | Reparado automáticamente |
| Incompatibilidad de formas en `_generate_conditional_data` | `ValueError` descriptivo |
| `effect_type` desconocido | `ValueError` |
| `entity_ids` vacío | No-op seguro, retorna inmediatamente |
