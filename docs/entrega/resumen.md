# VoxelScribe: Detección Probabilística de Tinta en Papiros Carbonizados mediante Relational Probability Models y Gibbs Sampling

## VoxelScribe: Probabilistic Ink Detection on Carbonized Papyri via Relational Probability Models and Gibbs Sampling

**Tipo de documento / Document type:** Brief — Resumen extendido bilingüe.
**Curso / Course:** Modelos Estocásticos — Universidad Nacional de Colombia.
**Capítulo de referencia / Reference chapter:** Russell & Norvig, *AIMA* (4th ed.), Chapter 18 — *Probabilistic Programming*.
**Repositorio / Repository:** `docs/Implementation_Guide.md`, `src/`, `run_scenario.py`.

---

# Versión en español

## Resumen

Presentamos **VoxelScribe**, un sistema de inferencia probabilística para
detectar tinta en imágenes de tomografía computarizada (CT) de fragmentos
de papiro carbonizados del **Vesuvius Challenge**. El sistema instancia un
*Relational Probability Model* (RPM) sobre una grilla bidimensional de
píxeles y estima el marginal posterior por píxel mediante **Gibbs Sampling**
sobre un Markov Random Field (MRF) con potencial unario gaussiano y
potencial pairwise de Potts. Toda la matemática se implementa desde cero en
NumPy: no se utilizan librerías de muestreo, modelado gráfico o métricas.
Se evalúan tres regímenes del parámetro de acoplamiento espacial β
(`0.0`, `1.5`, `5.0`) sobre el fragmento público Frag1 (EduceLab-Scrolls),
sin reescalamiento ni preprocesamiento adicional. El esquema de
actualización en *tablero de ajedrez* reduce el tiempo por escenario a
~8 s en CPU portátil. Los resultados reproducen tres regímenes
cualitativos distintos: predicción ruidosa sin coherencia espacial,
predicción regularizada con coherencia moderada, y colapso modal con
acoplamiento excesivo. La cobertura del Capítulo 18 de AIMA se estima en
**~80 %**.

**Palabras clave:** Relational Probability Model · Gibbs Sampling · Markov
Random Field · modelo de Potts · Vesuvius Challenge · inferencia
bayesiana aproximada.

## 1. Introducción

La erupción del Vesubio del año 79 d. C. carbonizó la biblioteca de la
Villa de los Papiros en Herculano. Los rollos resultantes son físicamente
indescifrables, pero recientes campañas de tomografía computarizada de alta
resolución (Vesuvius Challenge, 2023–presente) han hecho posible su lectura
virtual. El problema central de esta lectura es la detección de tinta: la
tinta utilizada es de base de carbono, igual que el sustrato carbonizado, y
la diferencia de densidad CT entre ambos es marginal. Cualquier
clasificador basado únicamente en intensidad es necesariamente ruidoso, y
la información discriminante proviene en gran parte de la **coherencia
espacial** de los trazos.

Este trabajo aplica el formalismo del Capítulo 18 de AIMA al problema. Cada
píxel se modela como una variable aleatoria binaria `HasInk(i, j)` cuyo
valor se infiere mediante Gibbs Sampling. La contribución del trabajo es
**pedagógica**: una implementación íntegramente *from scratch* del aparato
probabilístico —RPM, grounding, MRF, potenciales, sampler, energía,
métricas— sobre datos reales del concurso, con los tres regímenes
canónicos del parámetro β documentados de forma reproducible.

## 2. Métodos

### 2.1 Modelo

Sea `I_{ij} ∈ [0, 1]` la intensidad CT normalizada del píxel `(i, j)` y
`s_{ij} ∈ {0, 1}` la etiqueta latente de tinta. El RPM declara una
verosimilitud unaria gaussiana y un acoplamiento pairwise de Potts:

```
P(I_{ij} | s_{ij} = k) ~ N(μ_k, σ²_k),   k ∈ {0, 1}
ψ_β(s_{ij}, s_{kl})    = exp( β · 1[s_{ij} = s_{kl}] )
```

El grounding sobre un recorte de 300 × 300 produce **90 000** variables
ocultas conectadas en una grilla de 4-vecindad.

### 2.2 Estimación de parámetros

Los parámetros gaussianos por clase se estiman por máxima verosimilitud
sobre la mitad superior del ROI (entrenamiento), con corrección de Bessel
(`n − 1`). La mitad inferior se reserva como conjunto de prueba.

### 2.3 Inferencia

El marginal posterior `P(HasInk_{ij} = 1 | I)` se estima por Gibbs Sampling
con `N_burn = 200` y `N_sample = 500`. Cada sweep recorre toda la grilla en
esquema de **tablero de ajedrez**: los píxeles con `(i + j)` par y los de
`(i + j)` impar son condicionalmente independientes entre sí dado el otro
color, lo que permite actualizar cada color en paralelo con NumPy. La
distribución condicional por píxel se reduce a una sigmoide:

```
δ_{ij} = log P(I | 1) − log P(I | 0) + β · (2·n_1 − n_total)
P(s_{ij} = 1 | blanket) = σ(δ_{ij})
```

donde `n_1` es el conteo de vecinos con etiqueta 1. La sigmoide se evalúa
en dos ramas para tolerar `|δ| > 700` que aparecen con β alto.

### 2.4 Datos

Se utilizan dos archivos del Frag1 (EduceLab-Scrolls): la capa CT
`ctlayer_32.tif` (uint16, 8181 × 6330) y la máscara de tinta
`inklabels.png` (palette PNG, valores `{0, 1}`). El ROI de 300 × 300 se
ubica en `(top = 1600, left = 2200)`, dentro del cuerpo del papiro y con
30 % de fracción de tinta global.

### 2.5 Métricas

Matriz de confusión y métricas binarias (accuracy, precision, recall, F1)
calculadas sobre tres vistas: ROI completo, entrenamiento, prueba.

## 3. Resultados

La Tabla 1 resume las métricas sobre el conjunto de prueba para los tres
escenarios.

**Tabla 1.** Métricas sobre el conjunto de prueba (mitad inferior del ROI).

| Escenario      | β   | Accuracy | Precision | Recall | F1     |
|----------------|-----|----------|-----------|--------|--------|
| `base`         | 0.0 | 0.522    | 0.160     | 0.455  | 0.236  |
| `normal`       | 1.5 | 0.648    | 0.160     | 0.276  | 0.203  |
| `oversmoothed` | 5.0 | 0.838    | 0.000     | 0.000  | 0.000  |

La accuracy del escenario `oversmoothed` es engañosamente alta porque
refleja la proporción mayoritaria del conjunto de prueba (84 % no-tinta);
el F1 nulo confirma el colapso modal: la cadena se atrapa en la
configuración de todo cero. La traza de energía del MRF disminuye
monotónicamente durante el burn-in y se estabiliza en la fase de muestreo
para β ∈ {0, 1.5}, mientras que para β = 5.0 queda fija en el valor
correspondiente a la configuración degenerada (≈ −9.3 × 10⁵).

El parámetro β = 1.5 produce la mejor regularización en el conjunto de
entrenamiento (F1 = 0.689) y mejora la precisión del clasificador unario
en la región del ROI con tinta densa. La degradación en el conjunto de
prueba se atribuye a la heterogeneidad espacial del fragmento (la mitad
inferior tiene apenas 16 % de tinta frente al 44 % de la mitad superior).

El tiempo total por escenario es de aproximadamente **8 s** en una CPU
portátil moderna. El esquema de tablero de ajedrez es esencial: una
implementación con barrido secuencial puro en Python tomaría varios
minutos por escenario.

## 4. Discusión

Los tres regímenes observados corresponden, de forma reproducible, a los
tres comportamientos teóricos del modelo de Potts: independencia con
β = 0, regularización con β moderado y colapso modal con β alto. La
inicialización del sampler en estado todo-cero contribuye al colapso
observado con β = 5.0; un warm-start desde el MAP unario reduciría este
efecto.

La señal unaria es estructuralmente débil (`μ_ink = 0.326`,
`σ_ink = 0.154`; `μ_noink = 0.409`, `σ_noink = 0.157`). Las medias
distan poco más de media desviación estándar, lo que explica los valores
moderados de F1 incluso en el régimen mejor regularizado. Este es un
artefacto del problema físico subyacente —tinta de carbono sobre papiro
carbonizado— no de la implementación.

La arquitectura modular permite extensiones inmediatas: prior de clase
explícito, generalización 3D a múltiples capas CT, sustitución del
sampler por ICM o Belief Propagation. Cualquiera de esas extensiones se
acomoda sin reescribir el modelo.

## 5. Conclusiones

VoxelScribe demuestra que el aparato del Capítulo 18 de AIMA —RPMs,
grounding, Gibbs Sampling sobre MRFs— es **suficiente y didácticamente
adecuado** para abordar el problema real de detección de tinta en
papiros del Vesuvius Challenge, a escala de un proyecto de curso. La
implementación íntegra desde cero, sin librerías de modelado
probabilístico, hace explícita cada decisión matemática y algorítmica.
Los resultados sobre el Frag1 reproducen los tres regímenes esperados del
parámetro β, con tiempos de ejecución compatibles con un ciclo iterativo
de experimentación.

## Disponibilidad de código y datos

El código está disponible en el directorio del proyecto. Los datos son de
descarga pública desde `dl.ash2txt.org`. La especificación técnica
completa está en `docs/Implementation_Guide.md`; el resumen
post-implementación con todos los hallazgos numéricos en
`docs/implementation_summary.md`.

---

# English version

## Abstract

We present **VoxelScribe**, a probabilistic inference system for detecting
ink on computed-tomography (CT) images of carbonized papyrus fragments
from the **Vesuvius Challenge**. The system instantiates a *Relational
Probability Model* (RPM) over a two-dimensional grid of pixels and estimates
the per-pixel posterior marginal via **Gibbs sampling** on a Markov Random
Field (MRF) with a Gaussian unary potential and a Potts pairwise potential.
All mathematics is implemented from scratch in NumPy: no sampling,
graphical modelling, or metrics libraries are used. We evaluate three
regimes of the spatial coupling parameter β (`0.0`, `1.5`, `5.0`) on the
public Frag1 fragment of EduceLab-Scrolls, without re-scaling or additional
preprocessing. A **checkerboard** update schedule brings the per-scenario
runtime to ~8 s on a laptop CPU. The results reproduce three qualitatively
distinct regimes: noisy prediction without spatial coherence, regularized
prediction under moderate coupling, and modal collapse under excessive
coupling. Estimated coverage of AIMA Chapter 18: **~80 %**.

**Keywords:** Relational Probability Model · Gibbs sampling · Markov Random
Field · Potts model · Vesuvius Challenge · approximate Bayesian inference.

## 1. Introduction

The eruption of Mount Vesuvius in 79 AD carbonized the library of the
Villa of the Papyri at Herculaneum. The resulting scrolls are physically
unreadable, but recent high-resolution computed-tomography campaigns
(Vesuvius Challenge, 2023–present) have made their virtual reading
possible. The core problem in this reading is **ink detection**: the ink
is carbon-based, like the carbonized substrate itself, and the CT-density
difference between the two materials is marginal. Any intensity-only
classifier is necessarily noisy, and the discriminative information comes
in large part from the **spatial coherence** of ink strokes.

This work applies the formalism of AIMA Chapter 18 to the problem. Each
pixel is modelled as a binary random variable `HasInk(i, j)` whose value
is inferred via Gibbs sampling. The contribution is **pedagogical**: an
entirely *from-scratch* implementation of the probabilistic machinery —
RPM, grounding, MRF, potentials, sampler, energy diagnostic, metrics — on
real challenge data, with the three canonical regimes of β documented in
a reproducible way.

## 2. Methods

### 2.1 Model

Let `I_{ij} ∈ [0, 1]` be the normalized CT intensity at pixel `(i, j)`
and `s_{ij} ∈ {0, 1}` the latent ink label. The RPM declares a Gaussian
unary likelihood and a Potts pairwise coupling:

```
P(I_{ij} | s_{ij} = k) ~ N(μ_k, σ²_k),   k ∈ {0, 1}
ψ_β(s_{ij}, s_{kl})    = exp( β · 1[s_{ij} = s_{kl}] )
```

Grounding the template over a 300 × 300 crop yields **90,000** hidden
variables connected through a 4-neighbor grid.

### 2.2 Parameter estimation

Per-class Gaussian parameters are estimated by maximum likelihood on the
top half of the ROI (training), using the sample standard deviation with
Bessel's correction (`n − 1`). The bottom half is held out as the test set.

### 2.3 Inference

The posterior marginal `P(HasInk_{ij} = 1 | I)` is estimated by Gibbs
sampling with `N_burn = 200` and `N_sample = 500`. Each sweep walks the
grid in a **checkerboard** schedule: pixels with `(i + j)` even and those
with `(i + j)` odd are conditionally independent given the other color,
allowing each color to be updated in parallel with NumPy. The per-pixel
conditional reduces to a sigmoid:

```
δ_{ij} = log P(I | 1) − log P(I | 0) + β · (2·n_1 − n_total)
P(s_{ij} = 1 | blanket) = σ(δ_{ij})
```

where `n_1` is the count of neighbors currently labelled 1. The sigmoid
is evaluated in two branches to remain stable for `|δ| > 700`, which
arises with large β.

### 2.4 Data

Two files from Frag1 (EduceLab-Scrolls) are used: the CT slice
`ctlayer_32.tif` (uint16, 8181 × 6330) and the ink mask `inklabels.png`
(palette PNG, values `{0, 1}`). The 300 × 300 ROI is located at
`(top = 1600, left = 2200)`, inside the papyrus body and with a 30 %
global ink fraction.

### 2.5 Metrics

A hand-written confusion matrix and binary metrics (accuracy, precision,
recall, F1) are computed on three views: full ROI, training, and test.

## 3. Results

Table 1 summarises the test-set metrics across the three scenarios.

**Table 1.** Test-set metrics (bottom half of the ROI).

| Scenario        | β    | Accuracy | Precision | Recall | F1     |
|-----------------|------|----------|-----------|--------|--------|
| `base`          | 0.0  | 0.522    | 0.160     | 0.455  | 0.236  |
| `normal`        | 1.5  | 0.648    | 0.160     | 0.276  | 0.203  |
| `oversmoothed`  | 5.0  | 0.838    | 0.000     | 0.000  | 0.000  |

The `oversmoothed` accuracy is misleadingly high because it reflects the
majority class proportion on the test half (84 % no-ink); the zero F1
confirms modal collapse: the chain locks into the all-zero configuration.
The MRF energy trace decreases monotonically through burn-in and
stabilizes during sampling for β ∈ {0, 1.5}, while for β = 5.0 it is
fixed at the value corresponding to the degenerate configuration
(≈ −9.3 × 10⁵).

β = 1.5 produces the best regularization on the training set
(F1 = 0.689) and sharpens the unary classifier in dense-ink regions of
the ROI. The test-set degradation is attributable to the spatial
heterogeneity of the fragment (the bottom half contains only 16 % ink
versus 44 % in the top half).

Total runtime per scenario is approximately **8 s** on a modern laptop
CPU. The checkerboard schedule is essential: a pure-Python sequential
implementation would take several minutes per scenario.

## 4. Discussion

The three observed regimes correspond, reproducibly, to the three
theoretical behaviours of the Potts model: independence at β = 0,
regularization at moderate β, and modal collapse at high β. The all-zero
initialization of the sampler contributes to the collapse observed at
β = 5.0; a warm start from the unary MAP would reduce this effect.

The unary signal is structurally weak (`μ_ink = 0.326`,
`σ_ink = 0.154`; `μ_noink = 0.409`, `σ_noink = 0.157`). The means lie
just over half a standard deviation apart, which explains the moderate
F1 values even in the best-regularized regime. This is an artefact of
the underlying physical problem — carbon ink on carbonized papyrus —
rather than a limitation of the implementation.

The modular architecture supports immediate extensions: an explicit class
prior, a 3D generalization across CT layers, or replacing the sampler
with ICM or Belief Propagation. Any of these extensions fits without
rewriting the model.

## 5. Conclusions

VoxelScribe shows that the AIMA Chapter 18 machinery — RPMs, grounding,
Gibbs sampling over MRFs — is **sufficient and pedagogically appropriate**
to address the real ink-detection problem of the Vesuvius Challenge at
the scale of a coursework project. The end-to-end from-scratch
implementation, without probabilistic-modelling libraries, makes every
mathematical and algorithmic decision explicit. Results on Frag1
reproduce the three expected β-regimes with execution times compatible
with an iterative experimentation cycle.

## Code and data availability

The code is available in the project directory. Data are publicly
downloadable from `dl.ash2txt.org`. The full technical specification is in
`docs/Implementation_Guide.md`; the post-implementation summary with all
numerical findings is in `docs/implementation_summary.md`.

---

# Referencias / References

1. Russell, S. & Norvig, P. (2021). *Artificial Intelligence: A Modern
   Approach* (4th ed.). Pearson. Chapter 18 — *Probabilistic Programming*.
2. Geman, S. & Geman, D. (1984). *Stochastic Relaxation, Gibbs
   Distributions, and the Bayesian Restoration of Images*. **IEEE TPAMI**,
   6(6), 721–741.
3. Vesuvius Challenge — `scrollprize.org`.
4. EduceLab-Scrolls dataset — Frag1, package `54keV_exposed_surface`.
   `dl.ash2txt.org/fragments/Frag1.volpkg/`.
5. Propuesta de Proyecto — Modelos Estocásticos — VoxelScribe.
   `docs/Propuesta de Proyecto — Modelos Estocásticos — VoxelScribe.pdf`.
6. Documentación interna / Internal documentation:
   `docs/Implementation_Guide.md`,
   `docs/implementation_summary.md`,
   `docs/pipeline.html`,
   `docs/entrega/marco_teorico.md`,
   `docs/entrega/justificacion_del_problema.md`,
   `docs/entrega/manual_de_usuario_y_tecnico.md`.
