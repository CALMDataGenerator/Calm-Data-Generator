# ScenarioInjector - Referencia Completa

**Ubicación:** `calm_data_generator.generators.dynamics.ScenarioInjector`

El `ScenarioInjector` simula **dinámicas temporales** y **patrones evolutivos** en datasets sintéticos. A diferencia del `DriftInjector` que modifica distribuciones, `ScenarioInjector` crea patrones deterministas o estocásticos de evolución (cómo cambian las features en el tiempo) y construye variables objetivo basadas en lógica.

---

## ⚡ Inicio Rápido: Usando Objetos de Configuración

Recomendamos usar `ScenarioConfig` y `EvolutionFeatureConfig` para seguridad de tipos.

```python
from calm_data_generator.generators.configs import ScenarioConfig, EvolutionFeatureConfig

# 1. Definir evolución de features (ej. ingresos crecen, interés decae)
scenario_conf = ScenarioConfig(
    evolve_features={
        "revenue": EvolutionFeatureConfig(type="trend", slope=100.0),
        "interest": EvolutionFeatureConfig(type="exponential_decay", rate=0.01)
    },
    # 2. Construir target basado en features evolucionadas
    construct_target={
        "target_col": "high_value_customer",
        "formula": "0.4 * revenue - 100 * interest",
        "threshold": 0.8
    }
)

# 3. Aplicar al DataFrame (o usar dentro de RealGenerator con dynamics_config)
# Vía RealGenerator:
# gen.generate(..., dynamics_config=scenario_conf)

# Vía Inyección Directa:
from calm_data_generator.generators.dynamics import ScenarioInjector
injector = ScenarioInjector()
df_evolved = injector.evolve_features(df, scenario_config=scenario_conf)
```

---

## 🌲 Árbol de Decisión: Guía de Uso

```text
¿Qué quieres hacer?
├─ ¿Hacer que valores cambien en el tiempo (Crecimiento, Estacionalidad)?
│  └─ → evolve_features() (pasa evolution_config o scenario_config)
├─ ¿Crear una variable Target a partir de Features?
│  └─ → construct_target()
├─ ¿Proyectar datos históricos al futuro?
│  └─ → project_to_future_period()
└─ ¿Cambiar propiedades de distribución (Media, Ruido)?
   └─ → Usa DriftInjector en su lugar.
```

---

## 📚 Tipos de Evolución (`type`)

| Tipo | Alias | Patrón | Caso de Uso | Fórmula |
|------|-------|--------|-------------|---------|
| `trend` | `linear` | Cambio constante | Ventas, inflación | `y = x + slope * t` |
| `exponential_growth` | — | Incremento acelerado | Crecimiento viral | `y = x * (1 + rate)^t` |
| `exponential_decay` | `decay` | Valores decrecientes | Pérdida de retención | `y = x * (1 - rate)^t` |
| `cycle` | `sinusoidal`, `seasonal`, `cyclic` | Patrón cíclico | Vacaciones, clima | `y = x + A * sin(2πt/P)` |
| `sigmoid` | — | Curva en S | Adopción tecnológica | `y = x + A * σ(t)` |
| `step` | — | Salto repentino | Cambio de política, precio | `y = x + valor si t > paso` |
| `noise` | — | Fluctuación aleatoria | Error de sensor, ruido de mercado | `y = x + N(0, escala)` |
| `random_walk` | — | Paseo aleatorio acumulativo | Movimiento browniano, precios | `y = x + cumsum(N(0, step_std))` |
| `driven_by` | — | Dependencia inter-variable | Sensor IoT acoplado, escenarios causales | `delta = f(valor_driver_col)` |

### `driven_by` — Evolución impulsada por otra columna

Hace que el delta de una feature en cada fila dependa del **valor actual** de otra columna (no del índice de tiempo `t`).

```python
evolved_df = injector.evolve_features(df, evolution_config={
    "pressure": {
        "type":        "driven_by",
        "driver_col":  "temperature",   # columna que impulsa el delta
        "func":        "linear",        # "linear"|"exponential"|"power"|"polynomial"|callable
        "func_params": {"slope": 0.8},
    },
    "humidity": {
        "type":        "driven_by",
        "driver_col":  "temperature",
        "func":        "exponential",
        "func_params": {"scale": 0.002, "rate": 0.05},
    },
})
# Cada fila: delta_pressure = 0.8 * temperature_i
# Cada fila: delta_humidity = 0.002 * exp(0.05 * temperature_i)
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `driver_col` | str | Columna cuyo valor actual computa el delta por fila |
| `func` | str | Función de transferencia: `"linear"`, `"exponential"`, `"power"`, `"polynomial"` o callable |
| `func_params` | dict | Parámetros para `func` |

---

## 🛠️ Referencia de Clase `ScenarioInjector`

**Importar:** `from calm_data_generator.generators.dynamics import ScenarioInjector`

### Método: `evolve_features()`

Evoluciona columnas numéricas basado en patrones configurados.

```python
evolved_df = injector.evolve_features(
    df=df,
    evolution_config={
        "price": {"type": "trend", "slope": 0.01},          # Crecimiento lineal
        "demand": {"type": "seasonal", "amplitude": 10, "period": 30} # Ciclo mensual
    },
    time_col="date"  # Opcional: usar columna fecha paso de tiempo
)
```

### Método: `construct_target()`

Crea una variable objetivo basada en lógica de features. Útil para crear "ground truth" (verdad terreno) en escenarios sintéticos.

```python
# Fórmula de Texto
df = injector.construct_target(
    df=df,
    target_col="risk_score",
    formula="0.3 * age + 0.5 * bmi - 0.2 * exercise",
    noise_std=0.1  # Añadir ruido para realismo
)

# Función Python (Callable)
def complex_logic(row):
    return 1 if (row["age"] > 50 and row["income"] > 100000) else 0

df = injector.construct_target(
    df=df,
    target_col="is_vip",
    formula=complex_logic
)
```

### Método: `project_to_future_period()`

Extiende un dataset hacia el futuro generando nuevas muestras y aplicando evolución.

```python
future_df = injector.project_to_future_period(
    df=historical_df,
    periods=12,                   # Generar 12 pasos futuros (ej. meses)
    time_col="month",
    evolution_config={...},       # Aplicar tendencias a datos futuros
    n_samples_per_period=100
)
```

---

## 🌟 Escenarios del Mundo Real

### Caso 1: Crecimiento SaaS (Viral + Churn)
Simular una startup con crecimiento viral de usuarios pero churn creciente.

```python
scenario_conf = ScenarioConfig(
    evolve_features={
        "users": EvolutionFeatureConfig(type="exponential_growth", rate=0.1), # 10% crecimiento diario
        "churn": EvolutionFeatureConfig(type="trend", slope=0.001)           # Churn sube lentamente
    }
)
```

### Caso 2: Estacionalidad Retail
Simular picos de ventas en vacaciones.

```python
# Ciclo anual con pico a final de año
seasonal_conf = EvolutionFeatureConfig(
    type="seasonal",
    amplitude=5000,
    period=365,
    phase=300 # Desplazar pico hacia ~Día 300 (Nov/Dic)
)
```

### Caso 3: Credit Scoring (Generación de Ground Truth)
Crear un dataset donde CONOCES la relación exacta entre inputs y target.

```python
# Definimos el mecanismo de verdad:
# Riesgo = 2 * Deuda - 0.5 * Ingreso + Ruido
injector.construct_target(
    df=data,
    target_col="default_probability",
    formula="2 * debt_ratio - 0.5 * normalized_income",
    noise_std=0.05
)
```
