# Marco Teórico

**Proyecto:** VoxelScribe — Detección probabilística de tinta en papiros del Vesuvius Challenge mediante Relational Probability Models y Gibbs Sampling.
**Curso:** Modelos Estocásticos — Universidad Nacional de Colombia.
**Referencia principal:** Russell, S. & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4ª ed.). Pearson. **Capítulo 18 — Probabilistic Programming.**

---

## 1. Introducción

Este documento presenta el aparato teórico que sostiene la implementación
del proyecto. Recorre, en orden conceptual: el formalismo de los Relational
Probability Models (RPMs), el procedimiento de *grounding* a una red
concreta, la estructura del Markov Random Field (MRF) resultante, la
definición de los potenciales unario y pairwise, la función de energía
asociada, y finalmente el algoritmo de Gibbs Sampling con el que se realiza
inferencia aproximada del posterior por píxel. Cada sección incluye la
referencia exacta al Capítulo 18 de AIMA, 4ª edición, en la que se basa.

---

## 2. Relational Probability Models (§18.1)

Un **Relational Probability Model (RPM)** describe una distribución de
probabilidad sobre un dominio compuesto por objetos relacionados mediante
funciones. A diferencia de una red bayesiana clásica, donde las variables
aleatorias se enumeran una a una, un RPM se define a nivel de *plantillas*:
se declaran tipos de objetos y funciones cuyo argumento es un objeto del
tipo correspondiente. La instancia particular del modelo —el conjunto
explícito de variables aleatorias— se obtiene en una fase posterior llamada
*grounding* (§18.1.3).

### 2.1 Tipo y funciones del problema

Para el problema de detección de tinta:

- **Tipo:** `Pixel`, indexado por la coordenada `(i, j)` dentro del recorte
  300 × 300 de la imagen CT.
- **Función observada:** `Intensity(i, j)` toma valores en `[0, 1]` (la
  intensidad CT, normalizada a punto flotante). Es la única evidencia
  visible del modelo.
- **Función oculta:** `HasInk(i, j)` toma valores en `{0, 1}` y representa
  la presencia o ausencia de tinta en el píxel. Es la cantidad a inferir.

Estas dos funciones, junto con el tipo `Pixel`, constituyen la *plantilla*
del RPM. No se ha instanciado todavía ninguna variable aleatoria: el modelo
es aún simbólico.

### 2.2 Distribuciones condicionales declaradas en la plantilla

La plantilla declara, también de forma genérica, dos familias de
dependencias condicionales:

1. **Dependencia local con la observación.** Para cada píxel,
   `Intensity(i, j)` depende de `HasInk(i, j)` a través de un programa
   generativo gaussiano (§18.4, ver Sección 5 de este documento).
2. **Dependencia con los vecinos.** `HasInk(i, j)` está acoplada con
   `HasInk(i', j')` para cada vecino directo `(i', j')` a través del
   potencial pairwise de Potts (Sección 6).

Estas dependencias se reescribirán como variables aleatorias concretas tras
el grounding.

---

## 3. Grounding del modelo (§18.1.3)

El **grounding** es el procedimiento por el cual el RPM simbólico se
*despliega* sobre el conjunto concreto de objetos del problema. AIMA usa el
término *unrolling* para referirse a este paso.

En nuestro caso, sea el ROI un recorte de `H × W` píxeles
(`H = W = 300` por configuración por defecto). El grounding produce:

- **`H · W = 90 000`** variables aleatorias `HasInk(i, j)`, una por píxel.
- **`H · W = 90 000`** variables observadas `Intensity(i, j)`.
- Un grafo no dirigido de dependencias en el que cada variable oculta está
  conectada a:
  - su correspondiente intensidad observada (potencial unario), y
  - las variables ocultas de sus cuatro vecinos inmediatos (potencial
    pairwise).

El grounding convierte una plantilla finita y compacta en una red de noventa
mil variables; sin el formalismo RPM sería inviable escribir esa red a mano.

---

## 4. Markov Random Field (MRF) inducido

El grafo no dirigido producido por el grounding es un **Markov Random
Field** sobre la grilla de píxeles, con conexiones de **4-vecindad** entre
variables ocultas.

### 4.1 Markov Blanket por píxel

La propiedad de Markov local del MRF afirma que cada variable es
condicionalmente independiente del resto del modelo dado su Markov blanket.
Para `HasInk(i, j)`, el Markov blanket consta de:

- `Intensity(i, j)` (observada), y
- las cuatro variables `HasInk(i ± 1, j ± 1)` que pertenezcan a la grilla.

Los píxeles del borde tienen, por construcción, dos o tres vecinos en lugar
de cuatro. La implementación respeta esto sin agregar relleno artificial.

### 4.2 Distribución conjunta del MRF

La distribución conjunta sobre el estado completo del MRF, denotado
`s = (s_{ij})`, se factoriza como producto de potenciales:

```
P(s | I) ∝ ∏_{(i,j)} P(I_{ij} | s_{ij})  ·  ∏_{⟨(i,j),(k,l)⟩} ψ_β(s_{ij}, s_{kl})
```

donde `P(I_{ij} | s_{ij})` es el potencial unario, `ψ_β` el potencial
pairwise (modelo de Potts), e `⟨·,·⟩` itera una vez sobre cada par de
píxeles adyacentes.

---

## 5. Potencial unario: programa generativo gaussiano (§18.4)

El potencial unario codifica la verosimilitud condicional de observar la
intensidad CT dado el valor binario de la tinta. Se asume que, condicionado
al valor de la etiqueta, la intensidad sigue una distribución gaussiana
cuyos parámetros dependen únicamente de la clase:

- `P(Intensity(i, j) | HasInk(i, j) = 1) ~ N(μ_ink, σ²_ink)`
- `P(Intensity(i, j) | HasInk(i, j) = 0) ~ N(μ_noink, σ²_noink)`

Este modelo encaja exactamente en la definición de **programa generativo**
de AIMA §18.4: cada observación se interpreta como muestra de un mecanismo
estocástico que depende de la variable latente.

### 5.1 Estimación de parámetros por máxima verosimilitud

Los parámetros se estiman a partir del conjunto de entrenamiento (filas
superiores del ROI) usando la máscara de tinta como supervisión:

- `μ̂_k` = media muestral de las intensidades de la clase `k`.
- `σ̂_k` = desviación estándar muestral con **corrección de Bessel**
  (denominador `n − 1`).

Para los datos de la ejecución de referencia (recorte 300 × 300 en
`top = 1600, left = 2200`):

| Clase    | μ̂        | σ̂        |
|----------|-----------|-----------|
| Tinta    | 0.3263    | 0.1535    |
| No tinta | 0.4088    | 0.1568    |

Las medias se separan en aproximadamente media desviación estándar; la
señal unaria, por tanto, es **débil pero existente**, y es el potencial
pairwise el que aporta la regularización espacial.

### 5.2 Evaluación numéricamente estable

Para evitar el desbordamiento numérico que se produce al evaluar la PDF
gaussiana en escala lineal, se trabaja en escala logarítmica:

```
log P(x | k) = −½ · log(2π σ²_k) − (x − μ_k)² / (2 σ²_k)
```

Esta fórmula está implementada explícitamente en
`src/distributions.py::gaussian_log_pdf`. Toda la cadena de Gibbs opera en
log-espacio; el único punto donde se vuelve al espacio lineal es en el
cálculo de la sigmoide, una sola vez por cada actualización de píxel.

---

## 6. Potencial pairwise: modelo de Potts

El potencial pairwise codifica la coherencia espacial de los trazos de
tinta. Se modela como una versión binaria del modelo de **Potts**:

```
ψ_β(s_{ij}, s_{kl}) = exp( β · 1[s_{ij} = s_{kl}] )
```

donde `1[·]` es la función indicadora. Equivalentemente, en log-espacio,
cada par adyacente contribuye `β` si los vecinos concuerdan y `0` si
difieren.

### 6.1 Interpretación del parámetro β

- `β = 0`: pares independientes. El modelo se reduce a un clasificador
  bayesiano ingenuo por píxel.
- `β > 0`: presiona la cadena hacia configuraciones espacialmente
  coherentes.
- `β → ∞`: colapso modal — la cadena se atrae a configuraciones
  constantes (todo tinta o todo no tinta), eliminando la señal observada.

Los tres escenarios experimentales (β = 0, 1.5, 5.0) recorren este eje y
documentan empíricamente los tres regímenes (ver `docs/implementation_summary.md`
para los resultados numéricos).

---

## 7. Energía del MRF

La **energía** del estado actual es el negativo del log-posterior no
normalizado:

```
E(s) = − Σ_{(i,j)} log P(I_{ij} | s_{ij})
       − β · Σ_{⟨(i,j),(k,l)⟩} 1[s_{ij} = s_{kl}]
```

La segunda suma recorre cada par adyacente **una sola vez**: en la
implementación se suma solo sobre vecinos a la derecha y abajo (`state[:,:-1]
== state[:, 1:]` y análogo vertical), lo que evita el doble conteo.

La energía se monitorea cada 10 sweeps como diagnóstico de convergencia.
Durante el burn-in debería decrecer; durante el muestreo debería estabilizarse
alrededor de un valor (con variabilidad menor) si la cadena ha alcanzado
su distribución estacionaria.

---

## 8. Inferencia: Gibbs Sampling (§18.1.3)

El cálculo exacto de la marginal posterior por píxel es intratable: la
constante de normalización requiere sumar sobre `2^90000` estados. Se
recurre a **Gibbs Sampling**, una cadena de Markov cuya distribución
estacionaria coincide con el posterior objetivo.

### 8.1 Distribución condicional por píxel

Para un píxel `(i, j)` con Markov blanket dado, la distribución
condicional se calcula directamente a partir de los potenciales:

```
log p_k(i, j)  ∝  log P(I_{ij} | s_{ij} = k)  +  β · n_k(i, j)
```

donde `n_k(i, j)` es el número de vecinos de `(i, j)` cuyo valor actual
es `k`. Para el caso binario, la probabilidad condicional se obtiene
aplicando la **sigmoide** a la diferencia de log-probabilidades:

```
δ = log p_1 − log p_0
  = [log P(I | 1) − log P(I | 0)]  +  β · (n_1 − n_0)
  = Δ_unary  +  β · (2 · n_1 − n_total)

P(s_{ij} = 1 | blanket) = σ(δ) = 1 / (1 + exp(−δ))
```

Esta forma compacta es la base del sampler. La sigmoide está implementada
de forma estable en dos ramas para tolerar `|δ|` grandes (ver
`src/gibbs.py::_stable_sigmoid`).

### 8.2 Esquema de actualización en tablero de ajedrez

La especificación de AIMA describe un *sweep* clásico: actualizar todos los
píxeles uno a uno en orden aleatorio. Para una grilla 300 × 300 esto
implica 90 000 actualizaciones por sweep en bucle Python — prohibitivo a la
escala del proyecto.

Se utiliza, en su lugar, el **esquema de tablero de ajedrez** (checkerboard):

- Los píxeles con `(i + j)` par y los píxeles con `(i + j)` impar forman
  dos conjuntos. Dentro de cada conjunto, las variables son
  *condicionalmente independientes* dado el otro conjunto, porque el
  Markov blanket de un píxel de color A está totalmente compuesto por
  píxeles de color B.
- Por consiguiente, todos los píxeles del color A pueden actualizarse en
  paralelo, y luego todos los del color B, sin alterar la distribución
  estacionaria de la cadena.

Esta partición permite vectorizar el sweep completo con operaciones de
NumPy. El costo por sweep pasa de varios minutos a ~10 ms; cada escenario
completo (200 burn-in + 500 muestreo) tarda alrededor de 8 segundos en una
CPU de portátil.

### 8.3 Burn-in y fase de muestreo

La cadena se inicializa con `state = 0` (todo no-tinta), tal como pide la
especificación. La ejecución se divide en dos fases:

1. **Burn-in**: `N_burn = 200` sweeps. Las muestras se descartan; el
   propósito es permitir que la cadena se aleje de la condición inicial
   y se aproxime a la distribución estacionaria.
2. **Muestreo**: `N_sample = 500` sweeps. En cada sweep se incrementa
   `posterior_sum += state` y se acumulan las muestras.

### 8.4 Estimación del marginal

El marginal posterior por píxel se estima como la media muestral de los
estados visitados durante la fase de muestreo:

```
P̂(HasInk(i, j) = 1 | I)  =  posterior_sum(i, j)  /  N_sample
```

Aunque las muestras consecutivas están correlacionadas (no son
independientes), el estimador permanece insesgado; el costo es una mayor
varianza respecto a un muestreo independiente.

### 8.5 Decisión y máscara final

La máscara binaria final se obtiene por umbralización del posterior:

```
ĤasInk_final(i, j)  =  1   si   P̂(HasInk(i, j) = 1 | I) > 0.5
                       0   en caso contrario
```

Este umbral de 0.5 corresponde a la regla de decisión MAP bajo costo 0/1
simétrico.

---

## 9. Cobertura del Capítulo 18

La siguiente tabla resume el mapeo entre las secciones del capítulo y los
artefactos concretos del proyecto. Es el mismo mapeo de la propuesta de
proyecto (§7), actualizado con las referencias a la implementación
realizada.

| Sección AIMA                                         | Aparición en VoxelScribe                                                |
|------------------------------------------------------|--------------------------------------------------------------------------|
| §18.1 — Relational Probability Models                | Tipo `Pixel`, funciones `Intensity` y `HasInk`                          |
| §18.1.3 — Grounding                                  | Instanciación de 90 000 variables sobre el ROI 300 × 300                |
| §18.1.3 — Gibbs Sampling                             | `src/gibbs.py::gibbs_sweep` (tablero) y `run_gibbs` (burn-in + muestreo) |
| §18.4 — Programas generativos                         | Gaussianas por clase en `src/learning.py` + `src/distributions.py`     |
| §18.4.3 — Inferencia en programas generativos        | Acumulación de marginales en la fase de muestreo                        |

Cobertura total estimada: **~80 %** del capítulo, por encima del 50 %
requerido por el curso.

---

## 10. Referencias

1. Russell, S. & Norvig, P. (2021). *Artificial Intelligence: A Modern
   Approach* (4ª ed.). Pearson. Capítulo 18.
2. EduceLab-Scrolls dataset — Vesuvius Challenge.
3. Geman, S. & Geman, D. (1984). *Stochastic Relaxation, Gibbs
   Distributions, and the Bayesian Restoration of Images*. IEEE TPAMI.
4. Documentación interna del proyecto:
   - `docs/Implementation_Guide.md` — especificación técnica.
   - `docs/implementation_summary.md` — hallazgos y métricas
     post-implementación.
   - `docs/pipeline.html` — diagrama de flujo del sistema.
