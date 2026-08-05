# VoxelScribe

> Detección probabilística de tinta en papiros del Vesuvius Challenge mediante
> Relational Probability Models y Gibbs Sampling sobre un Markov Random Field 2D.

## Contexto

Proyecto final de la asignatura **Modelos Estocásticos**, Universidad Nacional
de Colombia.

El trabajo aplica los conceptos del Capítulo 18 (*Probabilistic Programming*) de
*Artificial Intelligence: A Modern Approach* (Russell & Norvig, 4ª ed.) al
problema de detectar tinta en fragmentos de papiro carbonizados por la erupción
del Vesubio en el año 79 d.C., usando imágenes de tomografía computarizada
publicadas por el Vesuvius Challenge.

## Modelo

Cada píxel de la imagen CT es una variable aleatoria binaria `HasInk(i, j)`
dentro de un Relational Probability Model instanciado (*grounding*). La
probabilidad marginal posterior se estima por Gibbs Sampling sobre un MRF con:

- **Potencial unario:** gaussiana sobre la intensidad CT, con parámetros
  `(mu, sigma)` por clase estimados por máxima verosimilitud sobre el
  conjunto de entrenamiento.
- **Potencial pairwise:** modelo de Potts sobre 4-vecindad, controlado por el
  hiperparámetro `beta`.

Se evalúan tres escenarios:

| Escenario | `beta` | Comportamiento esperado |
|-----------|--------|-------------------------|
| `base` | 0.0 | Predicción puramente unaria, ruido tipo "sal y pimienta" |
| `normal` | 1.5 | Coherencia espacial moderada, mejores fronteras |
| `oversmoothed` | 5.0 | Colapso modal, la cadena queda atrapada en una sola clase |

## Pipeline

El diagrama completo del pipeline, módulo por módulo, está en
[pipeline.html](pipeline.html). Ábrelo en un navegador para ver el flujo de
datos desde la carga del TIF hasta la generación de `comparison.png`.

Resumen del flujo:

1. Cargar la slice CT (`.tif`) y la máscara de tinta (`.png`).
2. Recortar el ROI configurado en [src/config.py](src/config.py).
3. Partir el ROI horizontalmente: la mitad superior entrena, la mitad inferior
   evalúa.
4. Estimar `(mu, sigma)` por clase sobre los píxeles de entrenamiento.
5. Pre-calcular log-unarios sobre todo el ROI.
6. Ejecutar Gibbs Sampling (burn-in + muestreo) y acumular la posterior.
7. Calcular métricas sobre ROI completo, train y test.
8. Persistir artefactos y reconstruir el grid comparativo.

## Pila técnica

| Componente | Versión | Uso |
|------------|---------|-----|
| Python | 3.10+ | Lenguaje base |
| NumPy | >= 2.0 | Estructuras numéricas y álgebra |
| Pillow | >= 10.0 | Lectura de TIF y PNG |
| Matplotlib | >= 3.8 | Gráficos |

Toda la matemática del modelo (PDF gaussiana en log-espacio, energía del MRF,
sweeps de Gibbs, matriz de confusión y métricas) se implementa desde cero. No
se usa `scipy`, `scikit-learn`, `scikit-image`, `pymc`, `stan`, `pgmpy` ni
ningún otro framework probabilístico.

## Instalación

Requiere Python 3.10 o superior. Desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Bash (Git Bash / WSL):

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Después de instalar las dependencias hay que colocar los dos archivos de datos
del fragmento público Frag1 del Vesuvius Challenge en:

- `data/raw/ctlayer_32.tif`
- `data/raw/inklabels.png`

(Si los archivos viven con otros nombres, ajustar `CT_FILE` e `INK_FILE` en
[src/config.py](src/config.py).)

## Ejecución

Con el entorno virtual activo, lanzar los tres escenarios:

```powershell
python run_scenario.py --beta 0.0 --tag base
python run_scenario.py --beta 1.5 --tag normal
python run_scenario.py --beta 5.0 --tag oversmoothed
```

Cada corrida tarda alrededor de 8 segundos en un portátil estándar y escribe
sus artefactos en `results/scenario_<tag>/`.

Flags opcionales (los valores por defecto vienen de [src/config.py](src/config.py)):

```
--burn N               sweeps de burn-in        (default 200)
--samples N            sweeps de muestreo       (default 500)
--seed N               semilla del RNG          (default 42)
--energy-log-every N   registrar energía cada N (default 10)
```

Tras cada corrida, `results/comparison.png` y `results/comparison.md` se
reconstruyen automáticamente a partir de todas las carpetas `scenario_*`
presentes en `results/`.

## Salidas

Por escenario, en `results/scenario_<tag>/`:

| Archivo | Contenido |
|---------|-----------|
| `final_mask.png` | Máscara binaria predicha (255 = tinta, 0 = no tinta) |
| `posterior.png` | Marginales `P(HasInk = 1)` por píxel en escala de grises |
| `energy.csv` | Tres columnas: `phase, sweep, energy` |
| `energy.png` | Curva de energía del MRF vs. sweep, con línea en fin de burn-in |
| `metrics.json` | Matriz de confusión y `accuracy / precision / recall / F1` para `full_roi`, `train` y `test` |
| `params.json` | Hiperparámetros, semilla y `(mu, sigma)` aprendidos |

A nivel de `results/`:

| Archivo | Contenido |
|---------|-----------|
| `comparison.png` | Grid lado a lado: CT, ground truth, y predicción por `beta` |
| `comparison.md` | Tabla con métricas de test de todos los escenarios |

## Estructura del proyecto

```
.
├── data/raw/                # TIF + PNG de entrada
├── results/                 # Outputs por escenario (runtime)
├── src/
│   ├── config.py            # Hiperparámetros, paths, semilla
│   ├── data_loader.py       # Carga, crop, split train/test
│   ├── distributions.py     # gaussian_log_pdf desde cero
│   ├── learning.py          # Estimación de mu/sigma por clase
│   ├── gibbs.py             # Sampler (burn-in + muestreo, checkerboard)
│   ├── energy.py            # Energía del MRF
│   ├── metrics.py           # Matriz de confusión y métricas
│   ├── io_utils.py          # Serialización de artefactos
│   └── visualize.py         # Plots de energía y comparativos
├── run_scenario.py          # CLI principal
├── pipeline.html            # Diagrama del pipeline
├── requirements.txt
└── README.md
```
## Autores

**Implementación:** Juan David Ramírez Torres — jdramirezt@unal.edu.co

**Análisis, informe y validación:**
- John Alejandro Pastor Sandoval
- Gabriel Felipe González Bohorquez
- Diego Felipe Cabrejo Suarez
- Mateo Andrés Vivas Acosta

## Referencias

- Russell, S. & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach*
  (4th ed.). Pearson. Capítulo 18.
- Vesuvius Challenge — sitio oficial del concurso.
- EduceLab-Scrolls dataset.

**Profesor:** Jorge Eduardo Ortiz Triviño — jeortizt@unal.edu.co

Universidad Nacional de Colombia — Modelos Estocásticos.
