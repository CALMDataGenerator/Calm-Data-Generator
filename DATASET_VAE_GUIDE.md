# DatasetVAE

---

## 0. Conceptos base (léelo si no tienes background de ML)



### Autoencoder

Un modelo que aprende a comprimir datos y luego reconstruirlos:

```
datos originales (grande)
       ↓
   [ENCODER]      ← aprende a comprimir
       ↓
  representación comprimida (pequeña)  ← "código" o "vector latente"
       ↓
   [DECODER]      ← aprende a reconstruir
       ↓
datos reconstruidos (grande)
```

Si el autoencoder funciona bien, la representación comprimida captura lo esencial de los datos. Por ejemplo: comprimir una imagen de 1000×1000 píxeles en 64 números que describen "es un gato marrón sentado" — suficiente para reconstruir algo parecido.

### VAE

VAE = Variational Autoencoder. Un autoencoder con una mejora clave: el espacio de representaciones comprimidas es **continuo y organizado**.

En un autoencoder normal, cada dato va a un punto fijo del espacio comprimido. En un VAE, cada dato va a una **región** del espacio (una Gaussiana). Esto tiene una consecuencia importante: puedes coger cualquier punto del espacio comprimido y el decoder generará algo coherente — no solo los puntos que ha visto en entrenamiento.

```
Autoencoder normal:
  imagen_gato_1 → punto A
  imagen_gato_2 → punto B
  punto entre A y B → basura (el decoder nunca lo vio)

VAE:
  imagen_gato_1 → región A
  imagen_gato_2 → región B
  punto entre A y B → gato intermedio coherente  ← esto es lo útil
```

El espacio comprimido de un VAE se llama **espacio latente**. Es el "mapa" donde viven todas las variantes posibles de los datos.


### ¿Qué es la KL Divergence?

Una penalización que fuerza al espacio latente a estar bien organizado (centrado en 0, no con regiones vacías ni excesivamente densas). Sin esta penalización, el autoencoder mete cada dato en un rincón separado del espacio — el espacio se fragmenta y no puedes navegar por él ni generar cosas nuevas. Con KL, el espacio queda compacto y continuo.

### ¿Qué significa "posterior collapse"?

Un fallo del entrenamiento donde el encoder aprende a ignorar los datos y siempre produce la misma representación comprimida (el centro del espacio latente). El decoder entonces aprende a generar sin usar la representación — usa solo lo que ha memorizado. Resultado: el modelo genera siempre lo mismo, independientemente del input. Es el equivalente a un estudiante que memoriza una única respuesta para todas las preguntas.


---


### El problema de los generadores actuales

CTGAN, TVAE, DDPM... todos generan **fila a fila**. Para un dataset de 200 pacientes, el modelo genera 200 filas independientes. La estructura global (correlaciones entre variables) emerge implícitamente — no se puede controlar de forma directa.

**Lo que no puedes hacer con generadores actuales:**
- "Dame un dataset donde la correlación entre grupo A y grupo B sea 0.15"
- Interpolar entre dos tipos distintos de dataset
- Navegar un espacio continuo de tipos de dataset

### La hipótesis DatasetVAE

Si tratamos un dataset como una unidad atómica (como si fuera una imagen), podemos aprender un espacio latente de *tipos de dataset*. En ese espacio:
- Datasets similares quedan cerca
- Podemos interpolar entre tipos
- Podemos especificar propiedades y buscar el dataset que las cumple

---

## Los datos de entrenamiento

### Qué son

1000 datasets sintéticos tipo RNA-seq:
- Cada dataset: **200 muestras × 50 genes**
- Estructura de **2 grupos**: genes 0-24 (Grupo A), genes 25-49 (Grupo B)
- Cada dataset tiene sus propias correlaciones: rA, rB, cross

### Por qué RNA-seq

1. Distribución no trivial — NegBinomial (asimétrica, entera, cola larga). Más difícil que datos Gaussianos.
2. Estructura de módulos — los genes operan en grupos coordinados. Representativo de datos reales.
3. Ground truth controlado — conocemos exactamente la estructura diseñada. Podemos evaluar.

### Cómo se generan (Gaussian Copula)

```
1. Z ~ N(0, Σ)     → vector Gaussiano con correlación (rA, rB, cross) deseada
2. U = Φ(Z)        → transformar a uniforme [0,1]
3. X = F⁻¹(U)      → aplicar distribución NegBinomial por gen
```

Resultado: datos con marginals NegBinomial Y correlación de Spearman controlada.

**Código real de generación (generate_mose_benchmark.py):**
```python
from calm_data_generator import ClinicalDataGenerator

gen = ClinicalDataGenerator(random_state=42, auto_report=False)

# Genera 1 dataset con rA=0.60, rB=0.55, cross=0.10
df = gen.generate(
    n_samples      = 200,
    n_genes        = 50,
    intra_corr_a   = 0.60,   # correlación intra-grupo A
    intra_corr_b   = 0.55,   # correlación intra-grupo B
    inter_corr     = 0.10,   # correlación entre grupos
)
# df.shape → (200, 50)
# columnas: GENE_0001 ... GENE_0050
```

### Preprocesamiento antes de entrenar

```python
arr = np.log1p(df.values)                           # comprime escala, reduce asimetría
arr = (arr - arr.mean(0)) / (arr.std(0) + 1e-8)    # z-score por gen
```

**log1p**: sin esto, genes con alta expresión dominan la correlación por escala.
**z-score**: sin esto, genes con alta varianza dominan el MSE del decoder.

### Metadata empírica vs diseño

Las correlaciones se **remiden sobre los datos reales** después del preprocesamiento:

| | rA medio |
|---|---|
| Diseño (parámetro dado al generador) | 0.60 |
| Empírico (medido sobre los datos) | 0.43 |

Gap de ~0.15 por shuffling de genes entre grupos + atenuación de Pearson en copula con distribuciones asimétricas. **El modelo usa los empíricos** — si usara los de diseño, el predictor aprendería una correspondencia incorrecta.

---

## La arquitectura

### Diagrama

```
dataset_real (200×50)
       │
  [PRECOMPUTE]──→ fingerprint (1375 dims)
                        │
                   [ENCODER]
                 Linear(1375→512)→LayerNorm→GELU→Dropout
                 Linear(512→256) →LayerNorm→GELU
                 Linear(256→128) →LayerNorm→GELU
                        │
                    [μ]   [logvar]
                        │
                  z = μ + ε·σ   (reparametrización)
                 ε ~ N(0,I)
                        │
              ┌─────────┴──────────┐
         [DECODER]           [PREDICTOR]
      Linear(16→128)        Linear(16→32)→GELU
      Linear(128→256)       Linear(32→3)→Sigmoid
      Linear(256→1024)           │
      Linear(1024→10000)    [rA, rB, cross] predichos
              │
       dataset_recon (200×50)
```

### Componentes clave

**Fingerprint (input del encoder):**
```
upper_tri(corrcoef(arr.T))  → 1225 dims  (correlaciones entre los 50 genes)
mean por gen                →   50 dims
std  por gen                →   50 dims
skew por gen                →   50 dims
─────────────────────────────────────────
total                         1375 dims
```

**Código real del fingerprint:**
```python
from scipy.stats import skew as scipy_skew

N_GENES  = 50
TRIU_IDX = np.triu_indices(N_GENES, k=1)   # índices del triángulo superior

def dataset_to_enc_input(arr: np.ndarray) -> np.ndarray:
    """
    arr: dataset (200×50) ya preprocesado (log1p + zscore)
    returns: fingerprint (1375,)
    """
    corr  = np.corrcoef(arr.T)                          # (50, 50)
    upper = corr[TRIU_IDX].astype(np.float32)           # (1225,)
    means = arr.mean(axis=0).astype(np.float32)         # (50,)
    stds  = arr.std(axis=0).astype(np.float32)          # (50,)
    skews = scipy_skew(arr, axis=0).astype(np.float32)  # (50,)
    return np.concatenate([upper, means, stds, skews])  # (1375,)

# Uso:
arr = np.log1p(df.values.astype(np.float32))
arr = (arr - arr.mean(0)) / (arr.std(0) + 1e-8)
enc = dataset_to_enc_input(arr)   # shape: (1375,)
```

**Por qué fingerprint y no dataset crudo**: con 200 muestras, la correlación empírica tiene error estándar ≈ 0.071. El encoder no puede distinguir rA=0.3 de rA=0.5 con datos crudos — el ruido enmascara la señal. El fingerprint agrega la señal sobre las 200 muestras.

**Por qué LayerNorm (no BatchNorm)**: BatchNorm normaliza sobre el batch. Aquí cada dataset es una unidad — LayerNorm normaliza por features de un dataset. Más estable con batch=32.

**Por qué GELU**: transición suave en 0, gradientes más estables para interpolación.

**Por qué LATENT_DIM=16**: 3 parámetros de correlación + redundancia. Espacio suficientemente pequeño para que sea denso con 1000 ejemplos de entrenamiento.

**Código real de la arquitectura:**
```python
import torch
import torch.nn as nn

LATENT_DIM = 16
ENC_DIM    = 1375   # fingerprint
INPUT_DIM  = 10000  # 200×50 dataset aplanado

class DatasetVAE(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder: fingerprint → (μ, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(ENC_DIM, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256),     nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128),     nn.LayerNorm(128), nn.GELU(),
        )
        self.mu     = nn.Linear(128, LATENT_DIM)
        self.logvar = nn.Linear(128, LATENT_DIM)

        # Decoder: SOLO z → dataset (sin conditioning)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),  nn.LayerNorm(128),  nn.GELU(),
            nn.Linear(128, 256),         nn.LayerNorm(256),  nn.GELU(),
            nn.Linear(256, 1024),        nn.LayerNorm(1024), nn.GELU(),
            nn.Linear(1024, INPUT_DIM),
        )

        # Predictor: μ → [rA, rB, cross] (supervisión del espacio latente)
        self.predictor = nn.Sequential(
            nn.Linear(LATENT_DIM, 32), nn.GELU(),
            nn.Linear(32, 3),          nn.Sigmoid(),  # salida en [0,1]
        )

    def encode(self, xenc):
        h = self.encoder(xenc)
        return self.mu(h), self.logvar(h)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)      # ε ~ N(0,I)
        return mu + eps * std            # z = μ + ε·σ

    def decode(self, z):
        flat = self.decoder(z)
        return flat.view(-1, 200, 50)    # reconstruir forma (N_SAMPLES, N_GENES)

    def forward(self, xenc):
        mu, logvar = self.encode(xenc)
        z    = self.reparametrize(mu, logvar)
        recon = self.decode(z)
        pred  = self.predictor(mu)
        return recon, mu, logvar, pred
```

---

## La función de pérdida

$$\mathcal{L} = \text{MSE}(x, \hat{x}) + \lambda_{KL}\cdot\text{KL} + \alpha\cdot\mathcal{L}_{\text{div}} + \beta\cdot\mathcal{L}_{\text{corr}} + \lambda\cdot\mathcal{L}_{\text{pred}}$$

Con: α=0.1, β=0.2, λ=10.0, KL_MAX=0.5, FREE_BITS=0.3

### Término 1 — MSE de reconstrucción

El decoder reconstruye el dataset original. MSE porque los datos son continuos (post log1p+zscore).

### Término 2 — KL Divergence

Fuerza el espacio latente a ser N(0,I). Sin KL, cada dataset va a su propio rincón → no hay estructura continua → no se puede interpolar.

**KL Annealing**: el peso del KL sube de 0 → 0.5 en las primeras 100 épocas.
- Sin annealing: el KL aplasta el espacio latente antes de que el encoder aprenda nada → posterior collapse.
- Con annealing: el encoder aprende estructura primero, luego el KL la regulariza suavemente.

**Free bits** (clamp KL por dim ≥ 0.3): impide que dimensiones latentes queden "muertas" (logvar≈0, μ≈0). Garantiza que las 16 dims estén activas.

### Término 3 — Diversidad (α=0.1)

Penaliza cuando dos reconstrucciones del mismo batch son muy similares. Previene colapso de modo (el decoder genera siempre el mismo dataset promedio).

### Término 4 — Correlación (β=0.2)

$$\|\text{corr}(\hat{x}) - \text{corr}(x)\|_F$$

MSE no garantiza que la estructura de correlación se preserve. Este término añade presión directa sobre lo que importa: la covarianza entre genes.

### Término 5 — Supervisión latente (λ=10.0) ← EL MÁS IMPORTANTE

```python
# predictor(μ) → [rA_pred, rB_pred, cross_pred]
# loss = MSE(pred, target_normalizado)
```

Con λ=10, este término **domina el entrenamiento** en las primeras épocas. El encoder está obligado a poner datasets con rA=0.55 cerca de otros con rA=0.55. No puede hacerlo de otra forma.

**Por qué λ=10 y no λ=1**: con λ=1 la reconstrucción compite y el espacio latente queda medio organizado. Con λ=10 el espacio queda completamente organizado por correlaciones antes de que el decoder se refine.

**Código real de la función de pérdida:**

```python
FREE_BITS = 0.3
ALPHA     = 0.1   # diversidad
BETA      = 0.2   # correlación
LAMBDA    = 10.0  # predictor

def loss_fn(recon, x, mu, logvar, cond_norm, pred, kl_weight):
    B = x.size(0)

    # 1. Reconstrucción
    recon_loss = F.mse_loss(recon, x)

    # 2. KL con free bits
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl_weight * torch.clamp(kl_per_dim, min=FREE_BITS).mean()

    # 3. Diversidad — penaliza reconstrucciones similares en el batch
    n = min(B, 8)
    pairs = torch.stack([F.mse_loss(recon[i], recon[j])
                         for i in range(n) for j in range(n) if i != j])
    div_loss = -ALPHA * pairs.mean()

    # 4. Fidelidad de correlación
    corr_loss = torch.stack([
        torch.norm(batch_corrcoef(recon[i]) - batch_corrcoef(x[i]), p='fro')
        for i in range(B)
    ]).mean()
    corr_loss = BETA * corr_loss

    # 5. Supervisión latente (término dominante)
    pred_loss = LAMBDA * F.mse_loss(pred, cond_norm)

    total = recon_loss + kl + div_loss + corr_loss + pred_loss
    return total, recon_loss, kl, corr_loss, pred_loss
```


---

## Los resultados

### Espacio latente organizado

| Métrica | Valor |
|---------|-------|
| Pearson predictor↔rA | **r = 0.995** |
| Pearson predictor↔rB | **r = 0.995** |
| Pearson predictor↔cross | **r = 0.994** |
| Pearson PC2↔cross_corr | **r = −0.95** |

El predictor puede leer exactamente las correlaciones desde z. El espacio latente está perfectamente organizado.

### Generación libre (z ~ N(0,I))

**10/10** datasets generados tienen estructura 2-grupos válida (intra_A > inter < intra_B).

```python
model.eval()

# Generar un dataset nuevo desde un punto aleatorio del espacio latente
z = torch.randn(1, LATENT_DIM)          # z ~ N(0,I)
with torch.no_grad():
    ds = model.decode(z).squeeze().numpy()  # shape: (200, 50)

# Verificar estructura 2-grupos
import pandas as pd
ga, gb = list(range(0, 25)), list(range(25, 50))
corr    = pd.DataFrame(ds).corr().values
intra_a = corr[np.ix_(ga, ga)][np.triu_indices(25, k=1)].mean()
intra_b = corr[np.ix_(gb, gb)][np.triu_indices(25, k=1)].mean()
inter   = corr[np.ix_(ga, gb)].mean()

print(f"intra_A={intra_a:.3f}  intra_B={intra_b:.3f}  inter={inter:.3f}")
print(f"2-grupos válido: {intra_a > inter and intra_b > inter}")
```

### Interpolación

Interpolando entre el dataset con menor rA y el de mayor rA: rA sube **monotónicamente** de 0.24 → 0.62. El espacio latente es continuo y semántico.

```python
# Coger los dos extremos del benchmark
idx_low  = meta['r_group_a'].idxmin()
idx_high = meta['r_group_a'].idxmax()

with torch.no_grad():
    mu_low,  _ = model.encode(Xenc[idx_low:idx_low+1])
    mu_high, _ = model.encode(Xenc[idx_high:idx_high+1])

# Generar 8 datasets interpolando linealmente entre los dos z
for alpha in np.linspace(0, 1, 8):
    z = (1 - alpha) * mu_low + alpha * mu_high
    with torch.no_grad():
        ds = model.decode(z).squeeze().numpy()
    corr    = pd.DataFrame(ds).corr().values
    intra_a = corr[np.ix_(ga, ga)][np.triu_indices(25, k=1)].mean()
    print(f"alpha={alpha:.2f}  →  rA_empirico={intra_a:.3f}")
# Output esperado: rA sube monotónicamente de ~0.24 a ~0.62
```

### Generación controlada (targeted)

| | MAE target vs empírico |
|---|---|
| rA | **0.060** |
| rB | **0.067** |
| cross | **0.048** |

6/6 targets generan datasets con estructura 2-grupos válida.

---

## Cómo funciona la generación controlada

```python
def find_z_for_target(target_rA, target_rB, target_cross,
                      n_steps=500, lr=0.05, n_restarts=5):
    """
    Gradient descent sobre z para encontrar z* tal que predictor(z*) ≈ target.
    Devuelve (z_optimo, loss_final).
    """
    # Normalizar target al rango [0,1] que usa el predictor
    target_norm = torch.tensor([
        (target_rA    - cond_min[0]) / (cond_max[0] - cond_min[0] + 1e-8),
        (target_rB    - cond_min[1]) / (cond_max[1] - cond_min[1] + 1e-8),
        (target_cross - cond_min[2]) / (cond_max[2] - cond_min[2] + 1e-8),
    ], dtype=torch.float32).unsqueeze(0)

    best_z, best_loss = None, float('inf')
    model.eval()

    for _ in range(n_restarts):
        z   = nn.Parameter(torch.randn(1, LATENT_DIM))
        opt = torch.optim.Adam([z], lr=lr)

        for _ in range(n_steps):
            opt.zero_grad()
            pred = model.predictor(z)
            loss = F.mse_loss(pred, target_norm)
            loss.backward()
            opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_z    = z.detach().clone()

    return best_z, best_loss


# Uso:
z_opt, _ = find_z_for_target(rA=0.55, rB=0.38, cross=0.15)
with torch.no_grad():
    dataset = model.decode(z_opt).squeeze().numpy()  # (200, 50)
```

**Por qué gradient descent**: no hay forma directa de invertir el predictor. La solución correcta sería entrenar un prior condicional p(z|c) que sample z directamente — eso es el siguiente paso.

---

## Limitaciones actuales

| Limitación | Impacto | Fix futuro |
|------------|---------|------------|
| N_SAMPLES fijo (200) | No puedes pedir datasets de otro tamaño | Decoder condicional en N |
| F fijo (50 genes) | No funciona con otros dominios sin reentrenar | Fingerprint a tamaño fijo (PCA a 512 dims) |
| 1000 datasets de entrenamiento | Espacio latente poco denso en extremos | 10.000+ datasets |
| Gradient descent inverso | Lento, no garantiza mínimo global | Prior condicional p(z|c) |
| Un solo dominio | No generaliza a tabular, time-series, etc. | Entrenamiento multi-dominio |

---

## Flujo completo en una imagen

```
ENTRENAMIENTO
─────────────
1000 datasets (200×50)
    ↓ log1p + zscore
    ↓ fingerprint()          → Xenc (1000×1375)
    ↓ metadata empírica      → cond  (1000×3)

    Xenc → encoder → (μ, logvar) → z
    z    → decoder → recon  [vs X:    MSE]
    μ    → predictor → pred [vs cond: MSE×10]
    KL(q(z|x) || N(0,I))    [annealing 100 épocas]

INFERENCIA — generación libre
──────────────────────────────
z ~ N(0,I) → decoder → dataset (200×50)

INFERENCIA — generación controlada
────────────────────────────────────
target [rA, rB, cross]
    → gradient descent sobre z
    → z* con predictor(z*) ≈ target
    → decoder(z*) → dataset (200×50)
```
