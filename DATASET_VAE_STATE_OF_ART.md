# Estado del Arte — Generación de Datasets Sintéticos Completos

---

## Generadores clásicos a nivel de fila (baseline)

### CTGAN / TVAE (SDV, 2019-2021)

**Qué hacen**: generan fila a fila. CTGAN usa una GAN condicional con modo-specific normalization para distribuciones multimodales. TVAE adapta el VAE al dominio tabular con el mismo preprocesamiento.

**Limitación clave**: la estructura global del dataset (correlaciones entre columnas) emerge implícitamente de generar muchas filas — no se controla. No existe un "espacio latente de datasets".

---

### GReaT — Tabular LLMs *(Borisov et al., 2022)*

**Qué hace**: serializa filas como texto estructurado (`"Edad: 34, Ingresos: 50000, ..."`), luego finetuning de un LLM para generar más filas en el mismo formato.

**Invariancia al orden**: el mecanismo de self-attention procesa variables de forma semántica — el orden posicional de las columnas es irrelevante por diseño.

**Limitación clave**: genera fila a fila (igual que CTGAN). El LLM no tiene noción de "dataset completo como unidad". Requiere muchos datos para finetuning y es computacionalmente costoso.

---

## Representación geométrica de datasets

### VGAE — Variational Graph Autoencoders *(Kipf & Welling, 2016)*

**Enfoque**: trata un dataset como un **grafo completo**:

- Nodos = columnas/features
- Aristas = covarianzas/correlaciones entre columnas (matriz de adyacencia A)

**Encoder**: GNN (Graph Neural Network) + función READOUT colapsa el grafo entero en un único vector latente z.

**Decoder**: reconstruye la matriz de adyacencia mediante producto interno:
$$\hat{A}_{ij} = \sigma(z_i^T z_j)$$

**Ventaja sobre DatasetVAE**: invariancia a filas Y columnas nativa. El grafo no tiene orden.

**Limitación**: el decoder reconstruye la matriz de correlaciones, no el dataset crudo. Para generar datos reales necesitas un segundo paso (e.g., copula sobre la correlación reconstruida).

---

## Dataset Distillation — Latent Video Dataset Distillation (2024)

**Qué es en términos simples**: imagina que tienes 100.000 imágenes de entrenamiento y quieres comprimirlas en solo 100 imágenes sintéticas que contengan la misma información. Un modelo entrenado en esas 100 imágenes sintéticas debería aprender igual de bien que uno entrenado en las 100.000 originales.

**La versión "latente"**: en vez de comprimir en imágenes sintéticas directamente, se comprimen en vectores del espacio latente de un generador (e.g. un VAE de imágenes). Esos vectores son mucho más compactos y el generador puede reconstruir los datos cuando se necesiten.

**Relación con DatasetVAE**: es el **proceso inverso** exacto:
- Dataset Distillation: dataset → z (comprimir para guardar)
- DatasetVAE: z → dataset (expandir para generar)

Ambos comparten la idea de que un dataset completo puede representarse como un punto z en un espacio latente. La diferencia es el objetivo — destilación quiere guardar; DatasetVAE quiere generar y navegar.

---

## TabPFN / Prior-Data Fitted Networks — Müller et al. (2021-2024)

**Qué es en términos simples**: un modelo preentrenado que, cuando le das un dataset de entrenamiento (X, y), aprende a clasificar nuevos ejemplos **sin reentrenar sus pesos**. Lo hace en menos de un segundo. El modelo ya "sabe" cómo funciona la mayoría de datasets del mundo — aprendido durante el preentrenamiento.

**Cómo funciona**: usa un transformer con self-attention que procesa el dataset entero como contexto. Para predecir la clase de un nuevo ejemplo, el modelo "lee" todos los ejemplos de entrenamiento al mismo tiempo y razona sobre el nuevo punto en relación a ellos.

**La idea clave**: el transformer aprende una representación interna del dataset completo en su espacio de atención — no de filas individuales, sino de la distribución global del dataset. Esto es lo más parecido en la literatura a "tratar un dataset como unidad".

**Diferencia con DatasetVAE**:
- TabPFN aprende a hacer predicciones desde un dataset — no genera datos
- No tiene un espacio latente explícito navegable — no puedes interpolar entre tipos de dataset
- No controla propiedades estadísticas del dataset (rA, rB, cross)

**Conexión con DatasetVAE**: ambos comparten la filosofía de preentrenamiento masivo una vez + inferencia por usuario sin reentrenar. TabPFN prueba que el paradigma funciona en tabular — DatasetVAE lo extiende al dominio generativo.

---

## LASIUM — Meta-Learning con Interpolación en Espacio Latente (2020)

**Qué es en términos simples**: un sistema de meta-learning (aprender a aprender) que genera tareas de entrenamiento sintéticas interpolando en el espacio latente de un VAE de imágenes.

**El problema que resuelve**: para entrenar un modelo que aprenda rápido con pocos ejemplos (few-shot learning), necesitas miles de tareas de entrenamiento distintas. LASIUM las genera automáticamente interpolando entre imágenes en el espacio latente.

**Cómo funciona**:
```
VAE entrenado con imágenes reales
    ↓
Coger dos imágenes A y B → encodear → z_A, z_B
    ↓
Interpolar: z(α) = (1-α)·z_A + α·z_B
    ↓
Decodear z(α) → imagen sintética intermedia
    ↓
Usar esas imágenes sintéticas como tareas de meta-learning
```

**Relación con DatasetVAE**: la interpolación en el espacio latente de LASIUM es **idéntica** a la interpolación de DatasetVAE. La diferencia:
- LASIUM interpola imágenes para crear tareas de clasificación
- DatasetVAE interpola datasets para crear datasets con propiedades intermedias

**Por qué importa para el paper**: LASIUM valida que la interpolación semántica en espacio latente funciona y es útil. DatasetVAE aplica exactamente el mismo principio pero en el dominio de datasets tabulares completos — un dominio donde nadie lo había hecho antes.

---

## Tabla comparativa

| Enfoque | Unidad atómica | Espacio latente navegable | Genera datos reales | Control estadístico |
|---------|---------------|--------------------------|--------------------|--------------------|
| CTGAN / TVAE | fila | ✗ | ✓ | ✗ |
| GReaT | fila | ✗ | ✓ | ✗ |
| VGAE | dataset (grafo) | ✓ | ✗ (solo correlaciones) | parcial |
| TabPFN | dataset (contexto) | ✗ | ✗ | ✗ |
| LASIUM | imagen | ✓ (interpolación) | ✓ (imágenes) | ✗ |
| Dataset Distillation | dataset | ✓ | ✓ | ✗ |
| **DatasetVAE** | **dataset tabular** | **✓** | **✓** | **✓** |

---

## Posición de DatasetVAE

DatasetVAE ocupa un **nicho no cubierto**: es el único enfoque que trata el dataset completo como unidad atómica, aprende un espacio latente continuo de datasets, y permite generación controlada por propiedades estadísticas sin reentrenamiento.


**Lo que aporta de nuevo**:

- Fingerprint estructural como input del encoder (no datos crudos, no GNN)
- Decoder que genera el dataset crudo completo desde z (no la matriz de correlación)
- Predictor auxiliar con λ=10 que organiza el espacio latente explícitamente
- Generación controlada mediante gradient descent en el espacio latente

---

## Analogías con generadores de imágenes

DatasetVAE trata un dataset como una "imagen" — una entidad 2D con estructura global que hay que aprender a comprimir y generar. La literatura de generación de imágenes lleva años resolviendo exactamente los mismos problemas. Estas son las analogías más directas:

---

### VAE original para imágenes — Kingma & Welling (2013)

**Qué hace**: el VAE original aprende a comprimir imágenes (e.g. caras 64×64) en un vector z de 128 dims y reconstruirlas. El espacio latente es continuo — puedes coger cualquier z aleatorio y el decoder genera una cara coherente.

**Analogía exacta con DatasetVAE**:

| Imágenes (VAE original) | DatasetVAE |
|------------------------|------------|
| Imagen 64×64 = 4096 píxeles | Dataset 200×50 = 10.000 valores |
| z ∈ ℝ¹²⁸ representa "tipo de cara" | z ∈ ℝ¹⁶ representa "tipo de dataset" |
| Interpolar z entre cara A y cara B | Interpolar z entre dataset A y dataset B |
| z~N(0,I) → cara nueva coherente | z~N(0,I) → dataset nuevo válido |

DatasetVAE **es** un VAE de imágenes donde la "imagen" es un dataset tabular.

---

### β-VAE — Representaciones Disentangled *(Higgins et al., DeepMind, 2017)*

**Qué hace**: añade un peso β>1 al término KL para forzar que el espacio latente sea **disentangled** — cada dimensión z_i corresponde a un factor de variación interpretable e independiente. Para caras: z_1 controla la edad, z_2 el color de pelo, z_3 la orientación, etc.

**Ejemplo**: en β-VAE entrenado con caras CelebA, si mueves solo z_3 de -2 a +2, la cara generada rota de izquierda a derecha sin cambiar nada más.

**Analogía exacta con DatasetVAE**: el predictor con λ=10 hace exactamente lo mismo que β-VAE — fuerza que determinadas dimensiones de z codifiquen factores interpretables específicos (rA, rB, cross). La diferencia es el mecanismo: β-VAE usa solo la presión del KL; DatasetVAE usa supervisión directa con un predictor.

```
β-VAE:     KL fuerte → z disentangled (sin supervisión)
DatasetVAE: predictor λ=10 → z organizado por correlaciones (con supervisión)
```

DatasetVAE es conceptualmente un **β-VAE supervisado** aplicado a datasets.

---

### Resumen de analogías

| Concepto en imágenes | Herramienta | Equivalente en DatasetVAE |
|----------------------|-------------|--------------------------|
| Espacio latente continuo de imágenes | VAE original | Espacio latente de datasets |
| Dimensiones interpretables en z | β-VAE | Predictor λ=10 organizando z |
| Interpolación entre imágenes | VAE interpolation | Interpolación entre datasets |
| Generación sin ver el input | z~N(0,I) → imagen | z~N(0,I) → dataset |

**Conclusión**: DatasetVAE traslada al dominio tabular exactamente los mismos conceptos que llevan años funcionando en generación de imágenes. La novedad no es el mecanismo — es el dominio y la unidad atómica (dataset vs píxel).

---

## Gaps abiertos en la literatura (oportunidades para el paper)

| Gap | Impacto | Solución propuesta en DatasetVAE |
|-----|---------|----------------------------------|
| No existe espacio latente continuo de datasets | No se puede interpolar ni navegar entre tipos de dataset | ✓ Resuelto |
| Control explícito de correlaciones globales | Correlaciones emergen implícitamente en generadores fila a fila | ✓ Resuelto (predictor + targeted generation) |
| Escalado a múltiples dominios sin reentrenamiento | Cada modelo es dominio-específico | Pendiente (fingerprint de tamaño fijo) |
| Interfaz de lenguaje natural para generación tabular controlada | GReaT genera fila a fila sin control macro | Pendiente (LLM como capa sobre DatasetVAE) |
