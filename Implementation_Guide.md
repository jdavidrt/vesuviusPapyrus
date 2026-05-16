# VoxelScribe — Guía de Implementación para Claude Code

Este documento es la especificación operativa para construir el esqueleto funcional de VoxelScribe en una sola sesión autónoma. El objetivo es un pipeline completo que corra los tres escenarios de prueba de extremo a extremo, no una versión pulida. Optimización, tests formales y refactor quedan para etapas posteriores con el equipo.

---

## TL;DR para Claude Code

1. Asume que el entorno ya está configurado (Python 3.10+, venv activo, dependencias instaladas) y los datos están en `data/raw/ct_layer32.tif` y `data/raw/inklabels.png`.
2. Construye el árbol de módulos en `src/` y `run_scenario.py` en la raíz.
3. Implementa cada módulo siguiendo las firmas de función especificadas más abajo.
4. Al final, ejecuta los tres escenarios (β = 0.0, 1.5, 5.0) y genera el plot comparativo.
5. Imprime las métricas en stdout y guarda artefactos en `results/scenario_<tag>/`.
6. Mantén el código simple y legible. Skeleton funcional, no optimizado.

---

## 1. Contexto del proyecto

VoxelScribe detecta tinta en fragmentos de papiro carbonizado del Vesuvius Challenge. Cada píxel de la imagen CT es una variable aleatoria binaria `HasInk(i, j)` cuya probabilidad marginal posterior se estima por Gibbs Sampling sobre un Markov Random Field 2D.

El modelo es un Relational Probability Model (RPM) instanciado: para una imagen de 300×300, el grounding produce 90 000 variables aleatorias conectadas en una grilla de dependencias locales. Cada píxel depende de:

- Su intensidad CT observada (potencial unario gaussiano).
- Sus cuatro vecinos inmediatos (potencial pairwise de Potts).

Referencia teórica: Cap. 18 (Probabilistic Programming) de Russell & Norvig, AIMA 4th ed.

---

## 2. Restricciones técnicas (no negociables)

**Permitido**:

- Python 3.10+
- Librería estándar de Python
- `numpy` (única librería de estructuras de datos)
- `pillow` (solo para leer TIF/PNG)
- `matplotlib` (solo para gráficos)

**Prohibido**:

- `scipy` y cualquiera de sus submódulos (`scipy.stats`, `scipy.special`, etc.)
- `scikit-learn`, `scikit-image`
- `pymc`, `stan`, `pgmpy`, `pyro`, `numpyro`
- Cualquier librería que implemente MCMC, gaussianas, redes bayesianas, MRFs, o métricas de clasificación.

Toda la matemática se implementa desde cero usando NumPy. Si una función parece "demasiado fácil de encontrar en scipy", esa es exactamente la que hay que escribir a mano.

---

## 3. Estructura del proyecto

```
voxelscribe/
├── data/raw/
│   ├── ct_layer32.tif        (ya existe)
│   └── inklabels.png         (ya existe)
├── results/                  (se crea en runtime)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── distributions.py
│   ├── learning.py
│   ├── gibbs.py
│   ├── energy.py
│   ├── metrics.py
│   ├── io_utils.py
│   └── visualize.py
├── run_scenario.py
├── requirements.txt          (ya existe)
└── .gitignore                (ya existe)
```

---

## 4. Configuración base

`src/config.py` define todas las constantes operativas:

```python
# Paths
DATA_DIR = "data/raw"
CT_FILE = "ct_layer32.tif"
INK_FILE = "inklabels.png"
RESULTS_DIR = "results"

# Region of interest (crop)
# Adjust if the default crop does not contain visible ink
CROP_TOP = 0
CROP_LEFT = 0
CROP_H = 300
CROP_W = 300

# Reproducibility
SEED = 42

# Default sampling hyperparameters
N_BURN = 200
N_SAMPLE = 500
ENERGY_LOG_EVERY = 10
```

---

## 5. Referencia matemática

### 5.1 Verosimilitud gaussiana por píxel

Para cada clase k en {0, 1}:

$$\log P(I_{ij} \mid \text{HasInk}=k) = -\frac{1}{2}\log(2\pi\sigma_k^2) - \frac{(I_{ij} - \mu_k)^2}{2\sigma_k^2}$$

Los parámetros (μ_k, σ_k) se estiman por máxima verosimilitud sobre el conjunto de entrenamiento: media muestral y desviación estándar con corrección de Bessel (denominador n − 1).

### 5.2 Distribución condicional de Gibbs

Para un píxel (i, j) con cuatro vecinos (arriba, abajo, izquierda, derecha):

$$\log p_k \propto \log P(I_{ij} \mid \text{HasInk}=k) + \beta \cdot n_k$$

donde n_k es el número de vecinos del píxel que actualmente tienen valor k. La probabilidad final se obtiene aplicando la sigmoide a la diferencia log p_1 − log p_0:

$$P(\text{HasInk}_{ij} = 1 \mid \text{blanket}) = \frac{1}{1 + \exp(-(\log p_1 - \log p_0))}$$

### 5.3 Energía del MRF

$$E(s) = -\sum_{(i,j)} \log P(I_{ij} \mid s_{ij}) - \beta \sum_{\langle (i,j),(k,l)\rangle} \mathbb{1}[s_{ij} = s_{kl}]$$

La suma del segundo término recorre cada par de píxeles adyacentes (4-vecindad) una sola vez. La energía debe decrecer durante el burn-in y estabilizarse en muestreo.

### 5.4 Métricas

A partir de la matriz de confusión (TP, FP, FN, TN):

- accuracy = (TP + TN) / total
- precision = TP / (TP + FP)
- recall = TP / (TP + FN)
- F1 = 2 · precision · recall / (precision + recall)

Si el denominador es cero, retornar 0.0.

---

## 6. Especificación módulo por módulo

Las firmas son la guía mínima. Implementa con docstrings cortas en inglés.

### 6.1 `src/data_loader.py`

```python
def load_ct_image(path: str) -> np.ndarray:
    """Load a 16-bit TIF as float64 normalized to [0, 1]."""

def load_ink_mask(path: str) -> np.ndarray:
    """Load a binary PNG as uint8 array with values in {0, 1}."""

def crop_region(arr: np.ndarray, top: int, left: int, h: int, w: int) -> np.ndarray:
    """Return a (h, w) crop starting at (top, left)."""

def spatial_split(crop_h: int) -> tuple[slice, slice]:
    """Return (train_rows, test_rows) as slice objects splitting horizontally."""
```

### 6.2 `src/distributions.py`

```python
def gaussian_log_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Numerically stable log of Gaussian PDF, evaluated elementwise."""
```

### 6.3 `src/learning.py`

```python
def estimate_class_params(intensities: np.ndarray, labels: np.ndarray) -> dict:
    """Returns {'mu_ink', 'sigma_ink', 'mu_noink', 'sigma_noink'} from train data.
    Uses sample mean and sample std with Bessel correction."""

def precompute_log_unaries(image: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (log_unary_noink, log_unary_ink), each shape (H, W)."""
```

### 6.4 `src/gibbs.py`

```python
def count_neighbors_equal(state: np.ndarray, value: int) -> np.ndarray:
    """For each pixel, count 4-neighbors equal to `value`. Returns (H, W) int array.
    Boundary pixels naturally have fewer than 4 neighbors; do not pad."""

def gibbs_sweep(state: np.ndarray,
                log_unary_noink: np.ndarray,
                log_unary_ink: np.ndarray,
                beta: float,
                rng: np.random.Generator) -> None:
    """One full sweep updating every pixel in random order. Mutates state in place."""

def run_gibbs(image: np.ndarray,
              params: dict,
              beta: float,
              n_burn: int,
              n_sample: int,
              seed: int,
              energy_log_every: int = 10) -> dict:
    """Run the full sampler. Returns:
    {
      'posterior': (H, W) float array of marginal P(HasInk=1),
      'final_mask': (H, W) uint8 array (posterior > 0.5 cast to int),
      'energy_history': list of (sweep_index, energy),
    }
    Initial state is all zeros. Records energy every `energy_log_every` sweeps,
    both during burn-in and sampling phases (with phase tagged in the index)."""
```

### 6.5 `src/energy.py`

```python
def mrf_energy(state: np.ndarray,
               log_unary_noink: np.ndarray,
               log_unary_ink: np.ndarray,
               beta: float) -> float:
    """Total MRF energy for the current state."""
```

### 6.6 `src/metrics.py`

```python
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Returns (TP, FP, FN, TN). Both arrays must be binary."""

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Returns {'accuracy', 'precision', 'recall', 'f1'}."""
```

### 6.7 `src/io_utils.py`

```python
def ensure_dir(path: str) -> None: ...
def save_mask_png(mask: np.ndarray, path: str) -> None: ...
def save_posterior_png(posterior: np.ndarray, path: str) -> None: ...
def save_energy_csv(history: list, path: str) -> None: ...
def save_json(data: dict, path: str) -> None: ...
```

### 6.8 `src/visualize.py`

```python
def plot_energy(history: list, path: str) -> None:
    """Line plot of energy vs sweep index, with vertical line at end of burn-in."""

def plot_comparison(ct: np.ndarray,
                    ground_truth: np.ndarray,
                    predictions: dict,
                    path: str) -> None:
    """Side-by-side panels: CT, ground truth, predictions keyed by beta value."""
```

### 6.9 `run_scenario.py`

CLI con `argparse`:

```
python run_scenario.py --beta 1.5 --burn 200 --samples 500 --seed 42 --tag normal
```

Flujo:

1. Parsear argumentos.
2. Cargar imagen CT y máscara desde `data/raw/`.
3. Recortar al ROI definido en `config.py`.
4. Aplicar split espacial: filas superiores son train, filas inferiores son test.
5. Aprender parámetros gaussianos sobre los píxeles de train.
6. Pre-calcular log-unarios sobre todo el ROI.
7. Correr Gibbs con los hiperparámetros recibidos.
8. Calcular métricas sobre tres conjuntos: ROI completo, train, test.
9. Guardar en `results/scenario_<tag>/`:
   - `final_mask.png`
   - `posterior.png`
   - `energy.csv`
   - `energy.png`
   - `metrics.json` (las tres métricas, una por subconjunto)
   - `params.json` (todos los argumentos y la semilla)
10. Imprimir resumen de métricas en stdout.

---

## 7. Orden de ejecución y aceptación

| Sprint | Módulo | Criterio mínimo de aceptación |
|--------|--------|-------------------------------|
| 1 | `data_loader.py` | Carga sin errores; shapes coinciden; primer crop guardable como PNG visible |
| 2 | `distributions.py` | `gaussian_log_pdf` produce valores finitos para inputs razonables |
| 3 | `learning.py` | `mu_ink` y `mu_noink` son distinguibles en datos reales (no idénticos) |
| 4 | `gibbs.py` (sweep) | Con `beta=0` y semilla fija, dos sweeps independientes producen el mismo resultado |
| 5 | `gibbs.py` (run) + `energy.py` | La energía decrece visiblemente en los primeros 50 sweeps |
| 6 | `metrics.py` | Sobre un par de arrays sintéticos 4×4 hechos a mano, las métricas coinciden con cálculo manual |
| 7 | `run_scenario.py` | Una sola corrida con `--beta 1.5` produce todos los artefactos sin excepciones |
| 8 | Los tres escenarios | β = 0.0, 1.5, 5.0 corren end-to-end y se genera `comparison.png` |

---

## 8. Aceptación final

Al terminar, ejecutar:

```powershell
python run_scenario.py --beta 0.0 --tag base
python run_scenario.py --beta 1.5 --tag normal
python run_scenario.py --beta 5.0 --tag oversmoothed
```

Y verificar que existen:

```
results/
├── scenario_base/
│   ├── final_mask.png
│   ├── posterior.png
│   ├── energy.csv
│   ├── energy.png
│   ├── metrics.json
│   └── params.json
├── scenario_normal/
│   └── (mismos archivos)
├── scenario_oversmoothed/
│   └── (mismos archivos)
├── comparison.png
└── comparison.md
```

`comparison.md` es una tabla con accuracy, precision, recall, F1 de los tres escenarios (métricas sobre test). Generar al final automáticamente o como parte de un script auxiliar `compare_scenarios.py`.

---

## 9. Notas operativas

- Usa `np.random.default_rng(SEED)` para reproducibilidad. Pasa el `rng` como parámetro a las funciones que lo necesiten, no crees uno nuevo internamente.
- Trabaja en log-espacio en todo el sampler. Aplica `np.exp` solo al normalizar. Para el caso binario, usa sigmoide sobre la diferencia de log-probabilidades.
- En `count_neighbors_equal`, los píxeles del borde tienen menos vecinos por construcción. No agregues padding ficticio: usa slicing con cuidado de las fronteras.
- El sweep secuencial (loop Python sobre píxeles en orden aleatorio) es la implementación de referencia. Si una corrida completa supera los 5 minutos, considera vectorización checkerboard: los píxeles donde `(i+j) % 2 == 0` son condicionalmente independientes entre sí dado el resto, así que pueden actualizarse en paralelo con NumPy.
- Usa `raise ValueError` o `raise RuntimeError` para errores. No uses `assert` para control de flujo.
- Comentarios en inglés. Código en inglés. Sin emojis en el código.
- No imprimas progreso excesivo. Un `print` cada 50 sweeps con la energía actual es suficiente.

---

## 10. Salida esperada al final

- Tres subcarpetas en `results/`, una por escenario, cada una con seis archivos.
- Un `comparison.png` con un grid mostrando CT, ground truth, y las tres predicciones.
- Un `comparison.md` con la tabla de métricas.
- En stdout, durante cada corrida, un resumen final con las métricas del escenario.

Si todo lo anterior se produce sin excepciones y la curva de energía decrece, el esqueleto está completo y entregable a la siguiente etapa.
