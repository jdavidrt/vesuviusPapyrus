# Justificación del Problema

**Proyecto:** VoxelScribe — Detección probabilística de tinta en papiros del Vesuvius Challenge.
**Curso:** Modelos Estocásticos — Universidad Nacional de Colombia.
**Capítulo objetivo:** Capítulo 18 — *Probabilistic Programming*. AIMA, 4ª ed. (Russell & Norvig).

---

## 1. Contexto histórico

En el año **79 d. C.**, la erupción del Vesubio sepultó la ciudad de
Herculano y, con ella, la biblioteca de la **Villa de los Papiros**. Los
rollos no fueron destruidos: el calor extremo los **carbonizó**, dejándolos
físicamente intactos pero imposibles de desenrollar sin destruirlos. Por
más de dos milenios, esos textos —probablemente la biblioteca privada
filosófica de Filodemo de Gadara, según los fragmentos abiertos en el
siglo XVIII— han sido inaccesibles.

Desde 2023, el **Vesuvius Challenge** (scrollprize.org) financia la lectura
virtual de estos rollos mediante **tomografía computarizada (CT)** de alta
resolución. La iniciativa ha producido datasets públicos —incluido
EduceLab-Scrolls, del que proviene este proyecto— y un concurso científico
abierto cuyo objetivo es construir algoritmos capaces de detectar la tinta
en las imágenes CT sin desenrollar el material físicamente.

VoxelScribe se inscribe en ese esfuerzo: aplica los conceptos del Capítulo
18 de AIMA al problema concreto de **detección de tinta** sobre un
fragmento público del dataset, el **Frag1**.

---

## 2. El reto técnico de fondo

La dificultad central no es la resolución del escáner sino una propiedad
física de los materiales involucrados.

- La **tinta** utilizada en los papiros herculanenses es de base de
  **carbono**.
- El **papiro carbonizado**, por su parte, es también esencialmente
  carbono.

En consecuencia, la **densidad** registrada por el CT —la única señal
disponible— es **prácticamente la misma** para ambos materiales. La
imagen, aunque resuelta en alta resolución, presenta contraste muy bajo
entre lo que es tinta y lo que es papiro.

Métodos clásicos como la **umbralización** o la **clasificación por
intensidad** funcionan en problemas donde la observación es discriminante;
aquí no lo es. Cualquier decisión sensata debe **combinar** la débil señal
unaria con información **espacial**: la tinta forma trazos continuos y
estructurados, por lo que un píxel marcado como tinta es probable que
tenga vecinos también marcados como tinta. Esta propiedad —coherencia
local— es exactamente la que un Markov Random Field captura con su
potencial pairwise.

El problema, en resumen, **demanda un modelo estocástico**: ni la
intensidad ni la geometría bastan por separado, pero su producto sí.

---

## 3. Pertinencia para el curso de Modelos Estocásticos

El proyecto se planteó como una aplicación directa del **Capítulo 18 —
Probabilistic Programming** de AIMA, 4ª ed. La correspondencia entre los
conceptos del libro y los componentes del proyecto es uno-a-uno:

| Concepto AIMA (§18)                                       | Realización en VoxelScribe                                                                |
|-----------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Tipos y funciones de un **Relational Probability Model**  | Tipo `Pixel`, función observada `Intensity(i, j)`, función oculta `HasInk(i, j)`           |
| **Grounding** del modelo                                  | Instanciación de 90 000 variables aleatorias sobre el ROI de 300 × 300 píxeles            |
| **Markov blanket** y dependencias locales                 | Cada `HasInk(i, j)` depende de su intensidad y sus cuatro vecinos                          |
| **Programa generativo** para la observación               | `P(Intensity \| HasInk = k) ~ N(μ_k, σ²_k)` con parámetros aprendidos por máxima verosimilitud |
| **Gibbs Sampling** como inferencia aproximada             | Sampler completo con burn-in + muestreo, estimación del marginal por promedio              |
| **Convergencia** y diagnósticos                            | Traza de la energía del MRF graficada por sweep                                            |

La cobertura del capítulo se estima en **~80 %**, holgadamente por encima
del 50 % requerido. Crucialmente, el proyecto no usa frameworks
probabilísticos (PyMC, Stan, pgmpy, scipy.stats): todo —Gibbs Sampling,
PDF gaussiana en log-espacio, métricas— se escribe **desde cero con NumPy**.
El estudiante implementa cada pieza del aparato teórico, no lo invoca.

---

## 4. Por qué un enfoque probabilístico (y no otro)

El problema admite, en principio, varias familias de soluciones. Se
discuten las descartadas y la razón:

- **Umbralización directa sobre intensidad CT.** Imposible por la cuasi-
  igualdad de densidades de tinta y papiro carbonizado.
- **Morfología matemática y filtros clásicos** (erosión, dilatación,
  detección de bordes). Aplican operadores deterministas. No cuantifican
  incertidumbre y no se justifican formalmente desde la teoría del curso.
- **Redes neuronales convolucionales** (CNNs, U-Net). Son el estado del
  arte real en el Vesuvius Challenge, pero requieren grandes datasets de
  entrenamiento etiquetados y son una caja negra respecto al modelo
  probabilístico. No corresponden al material del curso, no demuestran
  comprensión del Capítulo 18 y no permiten razonar sobre la
  estructura del posterior.
- **Modelos gráficos probabilísticos vía libreros como pgmpy o PyMC.**
  Excluidos explícitamente por la restricción del proyecto: el objetivo
  pedagógico es **implementar**, no invocar.

El enfoque **RPM + Gibbs Sampling** ofrece varias propiedades deseables
desde la perspectiva del curso:

1. **Cuantificación nativa de la incertidumbre.** El marginal posterior
   por píxel es un valor en `[0, 1]`, no una etiqueta dura. Permite
   distinguir píxeles de alta confianza de píxeles ambiguos.
2. **Separación limpia entre modelo e inferencia.** El modelo (RPM) y el
   algoritmo (Gibbs) son módulos independientes; cambiar `β` no requiere
   tocar el sampler, y cambiar el sampler no requiere tocar el modelo.
3. **Lectura directa del libro.** Cada pieza del código tiene su párrafo
   correspondiente en AIMA, lo que facilita la defensa académica del
   proyecto.
4. **Razonamiento sobre estructura.** El sweep en tablero de ajedrez se
   justifica desde la propiedad de Markov del MRF, no como un truco
   ad-hoc.

---

## 5. Por qué el Fragmento 1 (Frag1)

Dentro del dataset EduceLab-Scrolls, se elige el **Frag1** del paquete
`54keV_exposed_surface` por tres razones operativas:

1. **Disponibilidad pública sin registro.** Las dos URLs requeridas se
   descargan directamente desde `dl.ash2txt.org`.
2. **Ground truth real.** Los fragmentos están físicamente abiertos y
   fueron fotografiados con luz infrarroja, lo que permitió a
   investigadores anotar manualmente una **máscara de tinta**
   (`inklabels.png`). Esa máscara permite evaluar el algoritmo contra una
   referencia verdadera, no contra inferencias auto-generadas.
3. **Tamaño manejable.** Aunque la imagen completa es de
   8181 × 6330 píxeles, un recorte de 300 × 300 píxeles dentro de la zona
   con tinta es suficiente para que el modelo opere a escala didáctica
   (~90 000 variables) sin requerir cómputo de servidor.

Los dos archivos efectivamente usados en el proyecto son:

- `data/raw/ctlayer_32.tif` — imagen CT de 16 bits, una capa.
- `data/raw/inklabels.png` — máscara binaria con valores `{0, 1}`.

---

## 6. Alcance del proyecto

VoxelScribe es un **esqueleto funcional** de extremo a extremo, no un
sistema de producción.

### Lo que el proyecto sí hace

- Aprende los parámetros gaussianos por clase desde los datos.
- Implementa Gibbs Sampling completo sobre un MRF 2D con potenciales
  unario gaussiano y pairwise de Potts.
- Mide la convergencia con la traza de energía.
- Calcula matriz de confusión y métricas binarias (accuracy, precision,
  recall, F1) sobre tres vistas: ROI completo, entrenamiento y prueba.
- Documenta y reproduce **tres escenarios** (`β = 0`, `1.5`, `5.0`)
  generando todos los artefactos requeridos (máscaras, posterior, energía,
  métricas, parámetros, plot comparativo).

### Lo que el proyecto deliberadamente no hace

- **No realiza segmentación 3D ni registro de capas CT**. El proyecto
  trabaja sobre una única capa 2D.
- **No realiza aprendizaje supervisado a la escala del Vesuvius
  Challenge**. No compite con CNNs entrenadas en miles de horas-GPU.
- **No realiza tunneling de hiperparámetros**: `β` se varía sobre tres
  valores prefijados que demuestran tres regímenes cualitativos
  distintos.
- **No usa frameworks probabilísticos**. Esto es una restricción
  pedagógica, no una limitación técnica.

Este alcance se considera **suficiente y adecuado** para los objetivos del
curso: ilustrar conceptos del Capítulo 18 con una aplicación real y
mostrar comprensión profunda del aparato matemático y algorítmico.

---

## 7. Aplicabilidad y valor académico

El proyecto tiene valor en tres niveles:

- **Pedagógico.** Cada concepto del Capítulo 18 aparece, identificado y
  trazable, en el código y en el marco teórico. El estudiante demuestra
  comprensión activa, no recitación.
- **Metodológico.** El esqueleto resultante puede ser extendido sin
  reescribir el modelo: añadir una prior de clase, cambiar la
  distribución unaria, pasar a una vecindad 3D, sustituir el sampler por
  ICM o Belief Propagation. La arquitectura limpia es deliberada.
- **Conexión con un problema científico abierto.** El estudiante trabaja,
  aunque sea sobre un fragmento didáctico, sobre el mismo dataset que el
  estado del arte mundial intenta resolver. La aplicación no es
  artificial.

---

## 8. Resumen

El problema cumple, simultáneamente, varios criterios deseables:

- Tiene **contexto histórico y científico** verificable y motivador.
- Posee una **dificultad técnica real** que excluye soluciones triviales.
- Se mapea, **uno a uno**, con los conceptos del Capítulo 18.
- Cuenta con **datos públicos** y un **ground truth** verificable.
- Permite un **alcance acotado** que se ajusta al tiempo de un proyecto
  de curso (~6 días) sin sacrificar coherencia matemática.

Por estas razones, VoxelScribe es la aplicación apropiada del
*Probabilistic Programming* al problema de detección de tinta en papiros
carbonizados del Vesuvius Challenge.

---

## 9. Referencias

1. Russell, S. & Norvig, P. (2021). *Artificial Intelligence: A Modern
   Approach* (4ª ed.). Pearson. Capítulo 18.
2. Vesuvius Challenge — sitio oficial del concurso. `scrollprize.org`.
3. EduceLab-Scrolls dataset — Frag1, paquete `54keV_exposed_surface`.
4. Propuesta de Proyecto VoxelScribe (documento previo a la
   implementación). `docs/Propuesta de Proyecto — Modelos Estocásticos —
   VoxelScribe.pdf`.
