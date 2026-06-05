# DatasetVAE — Presentación

---

## SLIDE 1 — Título

**DatasetVAE: Un Espacio Latente de Datasets Completos**

*Generación sintética con control estadístico global*

---

## SLIDE 2 — El problema

**Los generadores actuales generan fila a fila**

- CTGAN, TVAE, DDPM → generan una muestra a la vez
- La estructura global del dataset emerge *implícitamente*
- No puedes decirle: *"dame un dataset donde la correlación entre grupo A y B sea 0.15"*

**Lo que no puedes hacer con generadores actuales:**
- Controlar correlaciones globales de forma explícita
- Interpolar entre tipos de dataset
- Navegar un espacio continuo de estructuras estadísticas

---

## SLIDE 3 — Estado del Arte (1/3): Generadores tabulares

| Modelo | Unidad | Genera datos | Control estadístico |
|--------|--------|-------------|-------------------|
| CTGAN / TVAE | fila | ✓ | ✗ |
| GReaT (LLM) | fila | ✓ | ✗ |
| VGAE | dataset (grafo) | ✗ (solo correlaciones) | parcial |

**El gap**: ninguno combina generación de datos reales + control estadístico explícito + espacio latente navegable

---

## SLIDE 4 — Estado del Arte (2/3): Enfoques relacionados

**TabPFN** *(Müller et al., 2021)*
- Trata el dataset entero como contexto de un transformer
- Pesos preentrenados fijos, inferencia sin reentrenar
- ✗ No genera datos, no tiene espacio latente navegable

**LASIUM** *(2020)*
- Interpola en espacio latente de VAE de imágenes para meta-learning
- Misma operación de interpolación que DatasetVAE
- ✗ Solo imágenes, no datos tabulares

**Dataset Distillation** *(2024)*
- Comprime dataset → z para guardar información
- Proceso **inverso** exacto a DatasetVAE
- ✗ Comprime, no genera

---

## SLIDE 5 — Estado del Arte (3/3): Analogías de imágenes

DatasetVAE traslada al dominio tabular lo que ya funciona en imágenes:

| Concepto imágenes | Herramienta | Equivalente DatasetVAE |
|------------------|-------------|----------------------|
| Espacio latente continuo | VAE original | Espacio latente de datasets |
| Dimensiones interpretables | β-VAE | Predictor λ=10 |
| Interpolar entre imágenes | VAE interpolation | Interpolar entre datasets |
| Texto → imagen | DALL-E / Stable Diffusion | Target → DatasetVAE → dataset |

**La novedad no es el mecanismo — es el dominio y la unidad atómica**

---

## SLIDE 6 — La idea de DatasetVAE

> *Tratamos un dataset completo (200×50) como una "imagen" y entrenamos un VAE sobre él*

**El espacio latente ya no representa muestras individuales — representa tipos de dataset**

```
z ∈ ℝ¹⁶  →  dataset completo (200 muestras × 50 genes)
```

**Un punto z codifica:**
- Correlación intra-grupo A (rA)
- Correlación intra-grupo B (rB)
- Correlación entre grupos (cross)

---

## SLIDE 7 — Los datos de entrenamiento

**1000 datasets sintéticos tipo RNA-seq**
- Cada dataset: 200 muestras × 50 genes
- Estructura 2 grupos: genes 0-24 (Grupo A), genes 25-49 (Grupo B)
- Cada dataset tiene correlaciones distintas: rA, rB, cross

**¿Por qué RNA-seq?**
- Distribución NegBinomial — caso difícil, asimétrico, entero
- Estructura de módulos génicas — representativo de datos reales
- Ground truth controlado — podemos evaluar cuantitativamente

**Preprocesamiento:**
```python
arr = np.log1p(df.values)                        # comprime asimetría NegBinomial
arr = (arr - arr.mean(0)) / (arr.std(0) + 1e-8)  # iguala peso de genes en la loss
```

**Metadata empírica** (no de diseño):
- Diseño rA=0.60 → Empírico rA=0.43
- Gap por shuffling de genes + atenuación Pearson en copula

---

## SLIDE 8 — Arquitectura: el fingerprint

**Problema v1**: encoder recibe dataset crudo (10.000 valores ruidosos)
- Con N=200 muestras: SE ≈ 1/√197 ≈ 0.071
- Dataset rA=0.30 y dataset rA=0.50 son indistinguibles para el encoder

**Solución**: fingerprint estructural precalculado

```
upper_tri(corrcoef(arr.T))  →  1225 dims  (correlaciones entre genes)
mean por gen                →    50 dims
std  por gen                →    50 dims
skew por gen                →    50 dims
─────────────────────────────────────────
fingerprint total            →  1375 dims  ← input al encoder
```

Señal agregada sobre 200 muestras → estable, sin ruido

---

## SLIDE 9 — Arquitectura: diagrama completo

```
fingerprint (1375)
      │
 [ENCODER]
 1375→512→256→128
 LayerNorm + GELU
      │
   [μ] [logvar]          [PREDICTOR]
      │                  μ → 16→32→3→Sigmoid
 z = μ + ε·σ             [rA, rB, cross]
 ε ~ N(0,I)                    │
      │                   λ=10 · MSE(pred, target)
 [DECODER]
 16→128→256→1024→10000
      │
 dataset (200×50)
```

**Decisión clave**: el decoder recibe **solo z** — sin conditioning.
Fuerza al encoder a codificar toda la estructura en z.

---

## SLIDE 10 — Arquitectura: por qué cada decisión

| Decisión | Alternativa | Por qué la nuestra |
|----------|------------|-------------------|
| Fingerprint como input | Dataset crudo | SE≈0.071 hace indistinguibles rA=0.3 y rA=0.5 |
| Sin c en decoder | CVAE [z,c] | Shortcut: decoder usa c, ignora z → z queda vacío |
| Predictor λ=10 | Sin predictor | Organiza z explícitamente por correlaciones |
| LayerNorm | BatchNorm | Cada dataset es unidad — normalizar por features, no por batch |
| LATENT_DIM=16 | 4 o 64 | Suficiente para [rA,rB,cross] + redundancia, denso con 1000 datasets |

---

## SLIDE 11 — Función de pérdida

$$\mathcal{L} = \underbrace{\text{MSE}(x, \hat{x})}_{\text{reconstrucción}} + \underbrace{\lambda_{KL} \cdot \text{KL}}_{\text{regularización}} + \underbrace{\alpha \cdot \mathcal{L}_{\text{div}}}_{\text{diversidad}} + \underbrace{\beta \cdot \mathcal{L}_{\text{corr}}}_{\text{fidelidad}} + \underbrace{\lambda \cdot \mathcal{L}_{\text{pred}}}_{\text{supervisión latente}}$$

**α=0.1 · β=0.2 · λ=10.0 · KL_MAX=0.5 · FREE_BITS=0.3**

| Término | Problema que resuelve |
|---------|----------------------|
| MSE | Reconstruir el dataset original |
| KL + annealing | Espacio latente continuo, sin posterior collapse |
| Free bits ≥ 0.3 | Impide dimensiones latentes muertas |
| L_div | Impide que el decoder genere siempre el dataset promedio |
| L_corr | Preserva estructura de correlación en la reconstrucción |
| **L_pred × 10** | **Organiza z explícitamente por [rA, rB, cross]** |

---

## SLIDE 12 — KL Annealing: dos fases

```
λ_KL
0.5 │                    ────────────────
    │                   /
    │                  /
  0 │─────────────────/
    └──────────────────────────────────→ época
    0        100                      400
```

**Fase 1 (épocas 0-100)**: λ_KL sube 0→0.5
- El encoder organiza z libremente guiado por predictor λ=10
- Sin presión de KL → z usa todo el espacio que necesita

**Fase 2 (épocas 100-400)**: λ_KL=0.5 fijo
- KL compacta el espacio ya organizado hacia N(0,I)
- Garantiza que z~N(0,I) en inferencia genere datasets válidos

---

## SLIDE 13 — El truco de reparametrización

**Problema**: no se puede hacer backprop a través de una operación aleatoria

```python
# MAL — no diferenciable:
z ~ N(μ, σ²)

# BIEN — diferenciable:
ε ~ N(0, I)        ← aleatoriedad fuera del grafo
z = μ + ε · σ      ← operación determinista sobre μ, σ
```

El gradiente fluye hacia μ y logvar sin problema.
Sin este truco el encoder nunca aprendería.

---

## SLIDE 14 — Inferencia: 3 modos de uso

**Modo 1 — Generación libre**
```python
z = torch.randn(1, 16)          # z ~ N(0,I)
dataset = model.decode(z)        # (200, 50)
```

**Modo 2 — Encodear dataset real**
```python
fingerprint = dataset_to_enc_input(arr)
mu, logvar  = model.encode(fingerprint)
dataset_syn = model.decode(mu)   # misma estructura, datos nuevos
```

**Modo 3 — Generación controlada**
```python
z_opt = find_z_for_target(rA=0.55, rB=0.38, cross=0.15)
dataset = model.decode(z_opt)    # dataset con esas correlaciones
```

---

## SLIDE 15 — Generación controlada: cómo funciona

**Gradient descent en el espacio latente** (no sobre los pesos del modelo):

```python
for restart in range(5):
    z   = nn.Parameter(torch.randn(1, 16))   # inicialización aleatoria
    opt = Adam([z], lr=0.05)

    for step in range(500):
        pred = model.predictor(z)             # z → [rA_pred, rB_pred, cross_pred]
        loss = MSE(pred, target_normalizado)
        loss.backward()                       # gradiente sobre z, no sobre pesos
        opt.step()

z* = restart con menor loss
dataset = model.decode(z*)
```

5 reinicios × 500 pasos. Los pesos del modelo están **congelados**.

---

## SLIDE 16 — Resultados: espacio latente

**El predictor puede leer exactamente las correlaciones desde z:**

| Métrica | Valor |
|---------|-------|
| Pearson predictor ↔ rA | **r = 0.995** |
| Pearson predictor ↔ rB | **r = 0.995** |
| Pearson predictor ↔ cross | **r = 0.994** |
| Pearson PC2 ↔ cross_corr | **r = −0.95** |

El espacio latente está perfectamente organizado por estructura de correlación.

---

## SLIDE 17 — Resultados: generación libre

**z ~ N(0,I) → dataset válido**

10 datasets generados desde puntos aleatorios del espacio latente:

| Gen | intra_A | intra_B | inter | 2-grupos válido |
|-----|---------|---------|-------|----------------|
| 0 | 0.548 | 0.499 | 0.308 | ✓ |
| 1 | 0.321 | 0.415 | 0.261 | ✓ |
| ... | ... | ... | ... | ... |
| 9 | 0.282 | 0.400 | 0.226 | ✓ |

**10/10** datasets con estructura 2-grupos válida (intra_A > inter < intra_B)

---

## SLIDE 18 — Resultados: interpolación

**Interpolando entre dataset con rA mínimo y rA máximo:**

```
alpha=0.00  →  rA_empirico=0.243   ← dataset con menor rA
alpha=0.14  →  rA_empirico=0.256
alpha=0.29  →  rA_empirico=0.307
alpha=0.43  →  rA_empirico=0.360
alpha=0.57  →  rA_empirico=0.458
alpha=0.71  →  rA_empirico=0.552
alpha=0.86  →  rA_empirico=0.606
alpha=1.00  →  rA_empirico=0.621   ← dataset con mayor rA
```

**rA sube monotónicamente de 0.24 → 0.62**
El espacio latente es continuo y semántico — no una tabla de lookup.

---

## SLIDE 19 — Resultados: generación controlada

**6 targets con propiedades específicas — MAE target vs empírico:**

| target rA | target rB | target cross | actual rA | actual rB | actual cross | válido |
|-----------|-----------|-------------|-----------|-----------|-------------|--------|
| 0.58 | 0.55 | 0.15 | 0.480 | 0.406 | 0.202 | ✓ |
| 0.50 | 0.52 | 0.22 | 0.492 | 0.537 | 0.261 | ✓ |
| 0.40 | 0.38 | 0.30 | 0.411 | 0.384 | 0.274 | ✓ |
| 0.55 | 0.38 | 0.15 | 0.481 | 0.444 | 0.185 | ✓ |
| 0.60 | 0.60 | 0.13 | 0.508 | 0.521 | 0.223 | ✓ |
| 0.32 | 0.58 | 0.25 | 0.396 | 0.486 | 0.211 | ✓ |

**MAE: rA=0.060 · rB=0.067 · cross=0.048 · 6/6 válidos**

---

## SLIDE 20 — Resumen de resultados

| Capacidad | Test | Resultado |
|-----------|------|-----------|
| Espacio latente organizado | Pearson predictor↔metadata | **r > 0.99** |
| Espacio latente organizado | Pearson PC2↔cross | **r = −0.95** |
| Generación libre válida | z~N(0,I) → 2-grupos | **10/10** |
| Interpolación semántica | Monotonía rA | **0.24 → 0.62** |
| Generación controlada | MAE target vs empírico | **~0.06** |

---

## SLIDE 21 — Limitaciones y trabajo futuro

**Limitaciones actuales:**

| Limitación | Impacto | Fix concreto |
|------------|---------|-------------|
| F fijo (50 genes) | No funciona con otros dominios | Fingerprint PCA → 512 dims fijos |
| N fijo (200 muestras) | No genera datasets de otro tamaño | Decoder condicional en N |
| Solo numéricas | No soporta categóricas | Fingerprint mixto + Cramér's V |
| Gradient descent inverso | Lento, no garantiza óptimo global | Prior condicional p(z\|c) |
| 1000 datasets entrenamiento | Espacio poco denso en extremos | 10.000+ datasets |

---

## SLIDE 22 — Casos de uso

**1. Augmentación biomédica**
50 pacientes reales con RNA-seq → 500 sintéticos con misma estructura
→ estudio estadístico potente sin recolectar más datos

**2. Benchmarking controlado**
Generar datasets con rA=0.3, 0.4, 0.5, 0.6 de forma sistemática
→ comparar algoritmos bajo condiciones estadísticas exactas

**3. Privacidad**
Compartir z (16 números) en vez de datos de pacientes reales
→ el receptor regenera dataset estadísticamente equivalente

---

## SLIDE 23 — Conclusión

**DatasetVAE prueba que es posible:**
- Aprender un espacio latente continuo de datasets completos
- Interpolar semánticamente entre tipos de dataset
- Generar datasets con propiedades estadísticas controladas

**La contribución original:**
Primer enfoque que combina dataset tabular como unidad atómica + espacio latente navegable + generación de datos reales + control estadístico explícito.

**El concepto funciona. Las limitaciones son de ingeniería, no de diseño.**

---

*Arquitectura reproducible: 6M parámetros · 400 épocas · 1000 datasets*
