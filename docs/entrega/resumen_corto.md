# VoxelScribe: Detección Probabilística de Tinta en Papiros Carbonizados mediante Relational Probability Models y Gibbs Sampling

**Curso:** Modelos Estocásticos — Universidad Nacional de Colombia
**Capítulo de referencia:** Russell & Norvig, *AIMA* (4ª ed.), Capítulo 18 — *Probabilistic Programming*.

---

## Resumen

Este trabajo presenta VoxelScribe, un sistema de inferencia probabilística para la detección de tinta en imágenes de tomografía computarizada (CT) de fragmentos de papiro carbonizados procedentes del Vesuvius Challenge, problema en el que la cuasi-identidad de densidad CT entre la tinta de base carbónica y el sustrato carbonizado vuelve insuficiente cualquier clasificador basado únicamente en intensidad. El sistema instancia un *Relational Probability Model* (RPM) sobre una grilla bidimensional de 300 × 300 píxeles —90 000 variables ocultas tras el *grounding*— y estima el marginal posterior por píxel mediante Gibbs Sampling sobre un Markov Random Field con potencial unario gaussiano por clase y potencial pairwise de Potts; toda la matemática —PDF gaussiana en log-espacio, sampler, energía del MRF y métricas— se implementa íntegramente desde cero en NumPy, sin librerías de muestreo, modelado gráfico o métricas, y un esquema de actualización en *tablero de ajedrez* reduce el tiempo por escenario a aproximadamente 8 s en CPU portátil. Sobre el fragmento público Frag1 del dataset EduceLab-Scrolls se evalúan tres regímenes del parámetro de acoplamiento espacial β (0.0, 1.5 y 5.0), reproduciendo los tres comportamientos teóricos esperados —predicción ruidosa sin coherencia, predicción regularizada bajo acoplamiento moderado y colapso modal bajo acoplamiento excesivo— con F1 de 0.236, 0.203 y 0.000 respectivamente sobre el conjunto de prueba, validando empíricamente el aparato del Capítulo 18 de AIMA con una cobertura estimada del 80 %.

**Palabras clave:** Relational Probability Model · Gibbs Sampling · Markov Random Field · modelo de Potts · Vesuvius Challenge · inferencia bayesiana aproximada.

---

## Abstract

This work presents VoxelScribe, a probabilistic inference system for ink detection on computed-tomography (CT) images of carbonized papyrus fragments from the Vesuvius Challenge, a problem in which the near-identical CT density of carbon-based ink and the carbonized substrate renders any intensity-only classifier insufficient. The system instantiates a Relational Probability Model (RPM) over a two-dimensional 300 × 300 pixel grid — 90,000 hidden variables after grounding — and estimates the per-pixel posterior marginal via Gibbs sampling on a Markov Random Field with a per-class Gaussian unary potential and a Potts pairwise potential; all mathematics — log-space Gaussian PDF, sampler, MRF energy and metrics — is implemented entirely from scratch in NumPy, without sampling, graphical-modelling or metrics libraries, and a checkerboard update schedule reduces the per-scenario runtime to approximately 8 s on a laptop CPU. On the public Frag1 fragment of the EduceLab-Scrolls dataset three regimes of the spatial-coupling parameter β (0.0, 1.5 and 5.0) are evaluated, reproducing the three expected theoretical behaviours — noisy prediction without spatial coherence, regularized prediction under moderate coupling and modal collapse under excessive coupling — with test-set F1 scores of 0.236, 0.203 and 0.000 respectively, empirically validating the AIMA Chapter 18 machinery with an estimated 80 % coverage.

**Keywords:** Relational Probability Model · Gibbs sampling · Markov Random Field · Potts model · Vesuvius Challenge · approximate Bayesian inference.
