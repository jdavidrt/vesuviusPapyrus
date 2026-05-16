# VoxelScribe

> Detección probabilística de tinta en papiros del Vesuvius Challenge mediante Relational Probability Models y Gibbs Sampling.

## Contexto

Proyecto final de la asignatura **Modelos Estocásticos**, Universidad Nacional de Colombia.

Aplica los conceptos del Capítulo 18 (*Probabilistic Programming*) de *Artificial Intelligence: A Modern Approach* (Russell & Norvig, 4th ed.) al problema de detectar tinta en fragmentos de papiro carbonizados por la erupción del Vesubio en el 79 d.C., usando imágenes de tomografía computarizada del Vesuvius Challenge.

## Modelo

Cada píxel de la imagen CT es una variable aleatoria binaria `HasInk(i, j)` en un RPM instanciado (grounding). La probabilidad marginal posterior se estima por Gibbs Sampling sobre un Markov Random Field con:

- **Potencial unario**: gaussiana sobre la intensidad CT, parámetros aprendidos por máxima verosimilitud sobre el conjunto de entrenamiento.
- **Potencial pairwise**: modelo de Potts sobre 4-vecindad, controlado por el hiperparámetro beta.

Se evalúan tres escenarios: beta = 0 (sin coherencia espacial), beta = 1.5 (coherencia moderada), beta = 5.0 (sobre-suavizado).

## Pila tecnológica

| Componente | Versión | Uso |
|------------|---------|-----|
| Python | 3.10+ | Lenguaje base |
| NumPy | >= 2.0 | Estructuras numéricas |
| Pillow | >= 10.0 | Lectura de TIF y PNG |
| Matplotlib | >= 3.8 | Gráficos |

Toda la matemática del modelo se implementa desde cero. No se usa `scipy`, `scikit-learn`, `pymc` ni equivalentes.

## Instalación

Solo en Windows. Dos caminos:

**Camino corto** (recomendado): ejecutar el script `setup_voxelscribe.ps1`:

```powershell
powershell -ExecutionPolicy Bypass -File "$HOME\Downloads\setup_voxelscribe.ps1"
```

**Camino manual**: seguir paso a paso `Setup_Windows.md`.

Después de la instalación, descargar manualmente los datos del fragmento público Frag1 desde el portal oficial del Vesuvius Challenge y colocarlos como `data\raw\ct_layer32.tif` y `data\raw\inklabels.png`.

## Ejecución

Con el entorno virtual activo:

```powershell
.\.venv\Scripts\Activate.ps1
python run_scenario.py --beta 1.5 --tag normal
```

Para correr los tres escenarios completos:

```powershell
python run_scenario.py --beta 0.0 --tag base
python run_scenario.py --beta 1.5 --tag normal
python run_scenario.py --beta 5.0 --tag oversmoothed
```

Cada corrida produce en `results/scenario_<tag>/`:

| Archivo | Contenido |
|---------|-----------|
| `final_mask.png` | Máscara binaria predicha |
| `posterior.png` | Probabilidades marginales en escala de grises |
| `energy.csv` | Energía del MRF por sweep |
| `energy.png` | Plot de convergencia del sampler |
| `metrics.json` | Accuracy, precision, recall, F1 |
| `params.json` | Hiperparámetros y semilla usados |

## Estructura del proyecto

```
voxelscribe/
+-- data/raw/                # TIF + PNG de entrada
+-- results/                 # outputs por escenario (runtime)
+-- src/
|   +-- config.py            # hiperparametros, paths, semilla
|   +-- data_loader.py       # carga, crop, split train/test
|   +-- distributions.py     # gaussian_log_pdf desde cero
|   +-- learning.py          # estimacion de mu, sigma por clase
|   +-- gibbs.py             # sampler con burn-in y muestreo
|   +-- energy.py            # diagnostico de convergencia
|   +-- metrics.py           # accuracy, precision, recall, F1
|   +-- io_utils.py          # serializacion de artefactos
|   +-- visualize.py         # plots de energia y mascaras
+-- run_scenario.py          # CLI principal
+-- requirements.txt
+-- Setup_Windows.md
+-- docs/
|   +-- Implementation_Guide.md
|   +-- implementation_summary.md
+-- CLAUDE.md
+-- README.md
```

## Documentación

| Documento | Propósito |
|-----------|-----------|
| `Setup_Windows.md` | Guía paso a paso de instalación |
| `docs/Implementation_Guide.md` | Especificación técnica completa |
| `docs/implementation_summary.md` | Notas posteriores a la implementación, hallazgos, guía de ejecución |
| `CLAUDE.md` | Contexto para asistentes de IA |

## Referencias

- Russell, S. & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. Capítulo 18.
- Vesuvius Challenge (sitio oficial del concurso).
- EduceLab-Scrolls dataset.

## Autores

Por completar al ensamblar el equipo final.
