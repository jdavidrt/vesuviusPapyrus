# Manual de Usuario y Manual Técnico

**Proyecto:** VoxelScribe — Detección probabilística de tinta en papiros del Vesuvius Challenge.
**Curso:** Modelos Estocásticos — Universidad Nacional de Colombia.

Este documento combina, en dos partes claramente separadas:

- **Parte I — Manual de Usuario**: cómo instalar, ejecutar y leer los
  resultados del sistema. Dirigido a quien recibe el código y quiere
  reproducir los experimentos.
- **Parte II — Manual Técnico**: arquitectura interna, módulos,
  estructuras de datos, algoritmos y decisiones de diseño. Dirigido a
  quien mantiene o extiende el código.

---

# Parte I — Manual de Usuario

## 1. Requisitos del sistema

| Componente              | Versión recomendada                       |
|-------------------------|-------------------------------------------|
| Sistema operativo       | Windows 10 / 11 (plataforma de referencia) |
| Python                  | 3.10 o superior                            |
| NumPy                   | ≥ 2.0 (instalado en el venv)               |
| Pillow                  | ≥ 10.0                                     |
| Matplotlib              | ≥ 3.8                                      |
| Espacio en disco        | ~150 MB (incluye datos crudos y resultados) |
| RAM                     | 4 GB son suficientes                      |

No se requiere GPU. Cada escenario corre en ~8 segundos sobre una CPU de
portátil.

## 2. Instalación

La preparación del entorno está descrita en detalle en `Setup_Windows.md`;
en resumen:

```powershell
git clone <repositorio> voxelscribe
cd voxelscribe
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Verificación rápida de que el entorno está sano:

```powershell
python -c "import numpy, PIL, matplotlib; print(numpy.__version__, PIL.__version__, matplotlib.__version__)"
```

## 3. Datos de entrada

El sistema espera dos archivos en `data/raw/`:

- `data/raw/ctlayer_32.tif` — imagen CT de 16 bits, capa 32 del paquete
  `54keV_exposed_surface` del Frag1.
- `data/raw/inklabels.png` — máscara binaria de tinta con valores
  `{0, 1}` (PNG con paleta).

Ambos archivos provienen del Vesuvius Challenge (dataset EduceLab-Scrolls)
y son de descarga pública sin registro. Las URLs exactas están en la
propuesta de proyecto (`docs/Propuesta de Proyecto — Modelos Estocásticos
— VoxelScribe.pdf`, sección 3).

## 4. Estructura de archivos

```
voxelscribe/
├── data/raw/                          (entrada — debe existir antes de correr)
│   ├── ctlayer_32.tif
│   └── inklabels.png
├── docs/                              (especificaciones y entregables)
│   ├── Implementation_Guide.md
│   ├── implementation_summary.md
│   ├── pipeline.html
│   ├── Propuesta de Proyecto ... .pdf
│   └── entrega/
│       ├── marco_teorico.md
│       ├── justificacion_del_problema.md
│       └── manual_de_usuario_y_tecnico.md   ← este archivo
├── results/                           (salida — se crea en runtime)
├── src/                               (código fuente del paquete)
├── run_scenario.py                    (punto de entrada CLI)
├── requirements.txt
└── .venv/                             (entorno virtual local)
```

## 5. Ejecutar un escenario

El programa se invoca por línea de comandos sobre `run_scenario.py`. La
firma básica es:

```powershell
python run_scenario.py --beta <valor> --tag <nombre>
```

Argumentos:

| Argumento              | Tipo  | Por defecto | Significado                                    |
|------------------------|-------|-------------|------------------------------------------------|
| `--beta`               | float | obligatorio | Acoplamiento de Potts (coherencia espacial)    |
| `--tag`                | str   | obligatorio | Sufijo de la subcarpeta de resultados          |
| `--burn`               | int   | 200         | Número de sweeps de burn-in (descartado)       |
| `--samples`            | int   | 500         | Número de sweeps de muestreo                   |
| `--seed`               | int   | 42          | Semilla del generador aleatorio                |
| `--energy-log-every`   | int   | 10          | Frecuencia de registro de la energía           |

Una corrida típica:

```powershell
python run_scenario.py --beta 1.5 --tag normal
```

genera todos los artefactos en `results/scenario_normal/` y reconstruye
los archivos comparativos a nivel del directorio `results/`.

## 6. Los tres escenarios de referencia

La propuesta y la especificación piden tres escenarios. Se ejecutan así:

```powershell
python run_scenario.py --beta 0.0 --tag base
python run_scenario.py --beta 1.5 --tag normal
python run_scenario.py --beta 5.0 --tag oversmoothed
```

Tiempos típicos:

| Escenario        | β    | Duración aprox. |
|------------------|------|------------------|
| `base`           | 0.0  | ~8 s             |
| `normal`         | 1.5  | ~8 s             |
| `oversmoothed`   | 5.0  | ~9 s             |

## 7. Lectura de los resultados

Cada escenario produce, en `results/scenario_<tag>/`, exactamente seis
archivos:

| Archivo               | Tipo  | Qué contiene                                                                          |
|-----------------------|-------|---------------------------------------------------------------------------------------|
| `final_mask.png`      | PNG   | Máscara binaria predicha. `255` = tinta, `0` = no tinta.                              |
| `posterior.png`       | PNG   | Mapa de probabilidad marginal `P(HasInk = 1)` en escala de grises.                    |
| `energy.png`          | PNG   | Traza de la energía del MRF vs. sweep, con línea vertical al final del burn-in.        |
| `energy.csv`          | CSV   | Energía por sweep: columnas `phase, sweep, energy`. Apta para reanalizar.              |
| `metrics.json`        | JSON  | Métricas de clasificación sobre `full_roi`, `train` y `test`.                          |
| `params.json`         | JSON  | Hiperparámetros, semilla, parámetros gaussianos aprendidos.                            |

A nivel global, en `results/`:

| Archivo            | Qué contiene                                                                       |
|--------------------|------------------------------------------------------------------------------------|
| `comparison.png`   | Panel comparativo: CT, etiquetas reales, predicciones por valor de `β`.            |
| `comparison.md`    | Tabla en Markdown con métricas de prueba para cada escenario presente.            |

Estos dos archivos comparativos **se regeneran** al final de cada
ejecución a partir de las subcarpetas `scenario_*` presentes. Si se borra
un escenario, la siguiente corrida lo retira del panel.

## 8. Cómo leer cada artefacto

- **`final_mask.png`.** Se compara visualmente contra `inklabels.png`. El
  `base` (β = 0) muestra ruido tipo sal-y-pimienta; el `normal` (β = 1.5)
  presenta regiones más coherentes; el `oversmoothed` (β = 5.0) debería
  ser uniformemente negro (colapso modal).
- **`posterior.png`.** Píxeles intermedios (gris medio) corresponden a
  regiones de incertidumbre. Una predicción "limpia" exhibe distribución
  bimodal: claras o oscuras, pocas grises.
- **`energy.png`.** Debe **decrecer** durante el burn-in y **estabilizarse**
  durante el muestreo. Una traza plana desde el primer sweep indica que
  la cadena se ha quedado atrapada (caso del `oversmoothed`).
- **`metrics.json`.** Tres bloques: `full_roi`, `train`, `test`. La
  comparación honesta entre escenarios se hace sobre `test`, no sobre
  `full_roi`.

## 9. Inspección rápida desde PowerShell

```powershell
# Leer las métricas con pretty-print
Get-Content results\scenario_normal\metrics.json | ConvertFrom-Json | Format-List

# Ver los parámetros usados (incluye semilla y mu/sigma aprendidos)
Get-Content results\scenario_normal\params.json | ConvertFrom-Json | Format-List

# Abrir los plots desde la terminal
Invoke-Item results\comparison.png
Invoke-Item results\scenario_normal\energy.png
Invoke-Item results\scenario_normal\posterior.png
```

## 10. Resultados típicos (sobre el conjunto de prueba)

Para la configuración por defecto (recorte 300 × 300 en
`top = 1600, left = 2200`, semilla 42, 200 burn-in + 500 muestras):

| Tag             | β   | Accuracy | Precision | Recall | F1     |
|-----------------|-----|----------|-----------|--------|--------|
| `base`          | 0.0 | 0.522    | 0.160     | 0.455  | 0.236  |
| `normal`        | 1.5 | 0.648    | 0.160     | 0.276  | 0.203  |
| `oversmoothed`  | 5.0 | 0.838    | 0.000     | 0.000  | 0.000  |

La accuracy alta del escenario `oversmoothed` es **engañosa**: la
predicción es todo cero y la accuracy refleja simplemente la proporción
mayoritaria del conjunto de prueba. El F1 (que sí refleja el desempeño
real) está en cero y confirma el colapso modal.

## 11. Limpiar resultados

Para repetir los experimentos desde cero:

```powershell
Remove-Item -Recurse -Force results
```

Si solo se desea borrar un escenario:

```powershell
Remove-Item -Recurse -Force results\scenario_smoke
```

La reproducibilidad está garantizada por la semilla: corridas con el mismo
`--seed`, mismo crop y misma configuración producen artefactos
idénticos byte a byte.

## 12. Resolución de problemas

| Síntoma                                                       | Diagnóstico / Acción                                                      |
|----------------------------------------------------------------|---------------------------------------------------------------------------|
| `FileNotFoundError: data/raw/ctlayer_32.tif`                  | Faltan los datos crudos. Descargar Frag1 desde `dl.ash2txt.org`.          |
| `Training set contains no ink pixels (label = 1).`            | El recorte no toca tinta. Ajustar `CROP_TOP` / `CROP_LEFT` en `config.py`. |
| Métricas de F1 todas en cero para `β` moderada                | La cadena no salió de la condición inicial. Aumentar `--burn`.            |
| `ImportError: matplotlib` (o similar)                          | El venv no está activo o `pip install -r requirements.txt` no se corrió.  |
| Resultado distinto cada ejecución                              | La semilla cambió; confirmar `--seed 42`.                                 |
| Permisos al borrar `results/`                                  | Cerrar visores de imagen antes de ejecutar `Remove-Item`.                  |

---

# Parte II — Manual Técnico

## 1. Visión general de la arquitectura

El proyecto está organizado como un **paquete Python plano** dentro de
`src/`, con un script CLI en la raíz que actúa como orquestador. No hay
clases ni jerarquías de objetos: cada módulo expone funciones puras o
funciones que mutan estados explícitos. La separación responde a la
filosofía de **esqueleto funcional** declarada en
`docs/Implementation_Guide.md` (sección 0): la prioridad es claridad y
correctitud, no extensibilidad especulativa.

```
src/
├── config.py           constantes y rutas
├── data_loader.py      lectura, recorte, split espacial
├── distributions.py    PDF gaussiana en log-espacio
├── learning.py         estimación de μ, σ por clase y pre-cálculo de log-unarios
├── energy.py           energía total del MRF (diagnóstico)
├── gibbs.py            muestreador Gibbs en tablero de ajedrez
├── metrics.py          matriz de confusión y métricas binarias
├── io_utils.py         persistencia de imágenes, CSV y JSON
└── visualize.py        gráficos (traza de energía y comparativa de escenarios)
```

Y, en la raíz:

- `run_scenario.py` — punto de entrada CLI; orquesta una corrida completa.

## 2. Diagrama de flujo de datos

```
[ data/raw/*.tif, *.png ]
            │  Pillow
            ▼
   data_loader.py
     load / crop / split
            │  NumPy
            ▼
   learning.py
     μ_ink, σ_ink, μ_noink, σ_noink   ─── distributions.py
            │
            ▼
   gibbs.py — run_gibbs
     ┌─ burn-in 200 sweeps ─┐
     │  gibbs_sweep         │  ← energy.py
     ├─ muestreo 500 sweeps ┤
     │  posterior_sum +=    │
     └──────────────────────┘
            │
            ▼
   metrics.py
     accuracy / precision / recall / F1
            │
            ▼
   io_utils.py + visualize.py
     PNG, CSV, JSON, plot de energía, comparativa
```

Un diagrama visual completo está en `docs/pipeline.html`.

## 3. Descripción detallada por módulo

### 3.1 `src/config.py`

Constantes operativas:

- `DATA_DIR = "data/raw"` — directorio de entrada.
- `CT_FILE = "ctlayer_32.tif"`, `INK_FILE = "inklabels.png"`.
- `RESULTS_DIR = "results"` — directorio de salida.
- `CROP_TOP = 1600`, `CROP_LEFT = 2200`, `CROP_H = CROP_W = 300` —
  región de interés. Fue elegida porque el ROI por defecto del spec
  (`(0, 0, 300, 300)`) cae fuera del cuerpo del papiro y produce un
  recorte de promedio 0.0 sin tinta.
- `SEED = 42` — semilla maestra para reproducibilidad.
- `N_BURN = 200`, `N_SAMPLE = 500`, `ENERGY_LOG_EVERY = 10`.

### 3.2 `src/data_loader.py`

Cuatro funciones puras:

- `load_ct_image(path)`: usa Pillow para leer el TIF. Si la profundidad
  es uint16, divide entre `65535.0`; si es uint8, entre `255.0`. Devuelve
  `np.ndarray` float64 en `[0, 1]`.
- `load_ink_mask(path)`: lee el PNG, descarta canales adicionales y
  binariza con `arr > 0`. **Importante:** el PNG real de Frag1 es una
  imagen con paleta y valores `{0, 1}`, no `{0, 255}`; la implementación
  inicial usaba umbral 127 y eliminaba toda la tinta. Esto está
  documentado en `docs/implementation_summary.md`.
- `crop_region(arr, top, left, h, w)`: recorta y copia. Lanza
  `ValueError` si los índices están fuera del array.
- `spatial_split(crop_h)`: devuelve dos `slice` para las filas superiores
  (entrenamiento) e inferiores (prueba). Para alturas impares, la fila
  central queda en entrenamiento.

### 3.3 `src/distributions.py`

Una sola función:

- `gaussian_log_pdf(x, mu, sigma)`: log de la PDF normal evaluada
  elementwise. Constante precalculada `_LOG_2PI`. Para evitar evaluación
  con `sigma <= 0`, se lanza `ValueError`.

### 3.4 `src/learning.py`

Dos funciones:

- `estimate_class_params(intensities, labels)`: separa por clase, calcula
  media y desviación estándar muestrales con **corrección de Bessel**
  (denominador `n − 1`). Devuelve un diccionario
  `{mu_ink, sigma_ink, mu_noink, sigma_noink}`.
- `precompute_log_unaries(image, params)`: devuelve dos arrays
  `(H, W)` con `log P(I | k)` para `k ∈ {0, 1}`. Se llama una sola vez
  por corrida, amortizando el cálculo a lo largo de los 700 sweeps.

Una constante interna `_MIN_SIGMA = 1e-6` evita división por cero en el
caso degenerado de una clase con un solo elemento.

### 3.5 `src/energy.py`

Una sola función:

- `mrf_energy(state, log_unary_noink, log_unary_ink, beta)`: energía
  total. Suma el término unario con `np.where(state == 1, log_ink, log_noink)`,
  y suma el término pairwise contando concordancias **solo** hacia la
  derecha y hacia abajo (para no doble-contar pares). Devuelve un `float`.

### 3.6 `src/gibbs.py`

Núcleo algorítmico del proyecto. Tres funciones públicas y dos auxiliares:

- `count_neighbors_equal(state, value)`: cuenta, por píxel, cuántos de
  sus cuatro vecinos son iguales a `value`. Los píxeles del borde tienen
  naturalmente 2 ó 3 vecinos; no se aplica padding.
- `_neighbour_totals(shape)` *(auxiliar)*: array constante con el total
  de vecinos válidos por píxel.
- `_stable_sigmoid(x)` *(auxiliar)*: sigmoide en dos ramas para evitar
  `overflow encountered in exp` cuando `|x|` es grande. Para
  `x ≥ 0` evalúa `1 / (1 + exp(−x))`; para `x < 0` evalúa
  `exp(x) / (1 + exp(x))`.
- `gibbs_sweep(state, log_unary_noink, log_unary_ink, beta, rng)`: un
  sweep completo en esquema de **tablero de ajedrez**. Calcula el delta
  para *todos* los píxeles a la vez, pero solo aplica la actualización a
  los píxeles del color activo (par o impar de `(i + j)`), luego repite
  para el otro color. Cada mitad usa una matriz fresca de uniformes
  `rng.random((H, W))`.
- `run_gibbs(image, params, beta, n_burn, n_sample, seed, energy_log_every=10)`:
  inicializa `state = zeros`, ejecuta el burn-in y luego la fase de
  muestreo. Registra la energía cada `energy_log_every` sweeps en una
  lista de tuplas `(phase, sweep, energy)`.

### 3.7 `src/metrics.py`

- `confusion_matrix(y_true, y_pred)`: devuelve `(TP, FP, FN, TN)`. Valida
  que ambos arrays sean binarios.
- `compute_metrics(y_true, y_pred)`: devuelve un diccionario con
  `accuracy, precision, recall, f1` y los conteos `tp, fp, fn, tn`. Cada
  división protegida con `_safe_div` para devolver `0.0` en el caso
  degenerado de denominador cero.

### 3.8 `src/io_utils.py`

- `ensure_dir(path)`: `os.makedirs(path, exist_ok=True)`.
- `save_mask_png(mask, path)`: convierte a `uint8 * 255` y guarda con
  Pillow.
- `save_posterior_png(posterior, path)`: convierte float `[0, 1]` a
  `uint8` y guarda.
- `save_energy_csv(history, path)`: tres columnas `phase, sweep, energy`.
- `save_json(data, path)`: serializa diccionarios, convirtiendo escalares
  y arrays NumPy a tipos Python nativos.

### 3.9 `src/visualize.py`

Uso del backend `Agg` de Matplotlib (sin necesidad de GUI):

- `plot_energy(history, path)`: traza con línea vertical roja en el
  final del burn-in. Los sweeps de la fase de muestreo se desplazan en el
  eje x para mostrarse en continuidad con los de burn-in.
- `plot_comparison(ct, ground_truth, predictions, path)`: panel
  horizontal con CT, ground truth y una predicción por valor de `β`,
  ordenadas crecientemente.

### 3.10 `run_scenario.py`

Orquestador. Pasos:

1. Parsear argumentos (`argparse`).
2. Cargar CT y máscara con `data_loader`.
3. Recortar al ROI configurado.
4. Aplicar split espacial (filas superiores = entrenamiento).
5. Estimar parámetros gaussianos sobre el conjunto de entrenamiento.
6. Llamar a `run_gibbs(...)` que internamente pre-calcula los log-unarios.
7. Calcular métricas sobre `full_roi`, `train` y `test`.
8. Persistir todos los artefactos.
9. Reconstruir `results/comparison.png` y `results/comparison.md` desde
   los `scenario_*` existentes.
10. Imprimir resumen de métricas en `stdout`.

## 4. Estructuras de datos clave

| Estructura          | Forma                  | Tipo       | Significado                                       |
|---------------------|------------------------|------------|---------------------------------------------------|
| `ct`                | `(300, 300)`           | `float64`  | Intensidad CT normalizada a `[0, 1]`.            |
| `ink`               | `(300, 300)`           | `uint8`    | Máscara verdadera binaria `{0, 1}`.              |
| `state`             | `(300, 300)`           | `uint8`    | Estado actual del MRF durante el muestreo.       |
| `log_unary_*`       | `(300, 300)`           | `float64`  | Log-verosimilitud condicional pre-calculada.     |
| `posterior`         | `(300, 300)`           | `float64`  | Marginal estimada `P(HasInk = 1)` por píxel.     |
| `final_mask`        | `(300, 300)`           | `uint8`    | Decisión MAP a partir del posterior.             |
| `energy_history`    | `list[(str, int, float)]` | —       | Tuplas `(fase, sweep, energía)`.                  |

## 5. Algoritmos clave

### 5.1 Sweep de Gibbs en tablero de ajedrez

Para una grilla 2D con potenciales de 4-vecindad, los píxeles con
`(i + j)` par y los píxeles con `(i + j)` impar son condicionalmente
independientes entre sí dados los del otro color. La actualización procede
por capas:

```python
for active_mask in (mask_even, mask_odd):
    n_ink = count_neighbors_equal(state, 1)
    delta = delta_log_unary + beta * (2 * n_ink - n_total)
    p1 = stable_sigmoid(delta)
    u = rng.random((H, W))
    new_vals = (u < p1).astype(uint8)
    state[active_mask] = new_vals[active_mask]
```

La derivación del `delta` se justifica en
`docs/entrega/marco_teorico.md` §8.1.

### 5.2 Conteo de vecinos sin padding

```python
counts = np.zeros_like(match)
counts[1:,  :] += match[:-1, :]     # vecino arriba
counts[:-1, :] += match[1:,  :]     # vecino abajo
counts[:,  1:] += match[:,  :-1]    # vecino izquierda
counts[:,  :-1] += match[:,  1:]    # vecino derecha
```

Los píxeles del borde reciben naturalmente menos contribuciones, lo que
implementa la condición de borde correctamente sin necesidad de relleno.

### 5.3 Sigmoide numéricamente estable

```python
out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
exp_x    = np.exp(x[neg])
out[neg] = exp_x / (1.0 + exp_x)
```

Evita la advertencia *overflow encountered in exp* que se produce con
β = 5.0 cuando el delta excede `±700`.

## 6. Decisiones de diseño y trazabilidad

| Decisión                                                          | Razón                                                                                                 |
|-------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| Sweep en **tablero de ajedrez** (no permutación aleatoria)        | 3 órdenes de magnitud más rápido; preserva la distribución estacionaria; aceptado por el spec.        |
| `state` como `uint8`, no `bool`                                   | Las operaciones de NumPy con `bool` son más lentas y menos legibles; `uint8` permite sumar vecinos directamente. |
| `energy_history` como lista de 3-tuplas (no 2-tuplas)             | La especificación pedía "fase etiquetada en el índice"; explicitarla como tercer campo es más legible. |
| Recorte por defecto `(1600, 2200)`                                | El recorte original `(0, 0)` cae fuera del papiro. Búsqueda heurística encontró región balanceada.    |
| Cadena inicial en ceros                                            | Especificado por el guía; deja un sesgo controlado y reproducible.                                     |
| Estimación de σ con corrección de Bessel                          | Estimador insesgado.                                                                                  |
| Sigmoide en dos ramas                                              | Estable para `|δ| > 700` que aparece naturalmente con β = 5.                                          |
| `Matplotlib` con backend `Agg`                                     | No depende de GUI; corre en headless.                                                                  |
| Reconstrucción de `comparison.*` por glob                          | Cualquier orden de ejecución produce comparativas válidas y consistentes.                              |

## 7. Restricciones técnicas (recordatorio)

- **Permitido:** Python ≥ 3.10, librería estándar, `numpy`, `pillow`,
  `matplotlib`.
- **Prohibido:** `scipy` (todos sus submódulos), `scikit-learn`,
  `scikit-image`, `pymc`, `stan`, `pgmpy`, `pyro`, `numpyro`, cualquier
  librería que implemente MCMC, gaussianas, redes bayesianas, MRFs, o
  métricas de clasificación.

Toda la matemática se implementa desde cero.

## 8. Rendimiento observado

Sobre una CPU portátil (Intel i7 reciente, sin GPU):

| Operación                                | Tiempo aprox.     |
|------------------------------------------|-------------------|
| Carga y normalización de la imagen CT     | ~1 s              |
| Carga de la máscara de tinta              | < 0.1 s           |
| Pre-cálculo de log-unarios `(300, 300)`   | < 5 ms            |
| Un sweep en tablero de ajedrez            | ~10 ms            |
| 200 burn-in + 500 muestreo                | ~7 s              |
| Cálculo de las tres métricas              | < 50 ms           |
| Plot de energía + comparativa             | ~0.5 s            |
| **Total por escenario**                  | **~8 s**           |

## 9. Extensiones futuras (sugerencias para el siguiente equipo)

- **Suite de tests con pytest.** Cada criterio de aceptación de la
  sección 7 de `Implementation_Guide.md` puede convertirse en un test
  determinista.
- **Warm-start desde el MAP unario.** Inicializar la cadena con
  `state = (posterior_unaria > 0.5)` reduciría drásticamente el burn-in
  necesario y evitaría parte del colapso modal observado en β = 5.
- **Prior de clase explícito.** El sampler actual asume implícitamente
  un prior uniforme; en el dataset la clase noink representa ~70 %. Añadir
  `log π_k` al término unario produciría predicciones más conservadoras.
- **Generalización a 3D.** Pasar a una vecindad de 6 vecinos sobre el
  volumen CT completo aprovecharía la información de continuidad de los
  trazos a través de capas.
- **ICM y Belief Propagation como alternativas al sampler.** Útiles para
  comparar tiempo de convergencia y calidad de la aproximación.

---

## 10. Glosario

| Término                | Significado                                                                                  |
|------------------------|----------------------------------------------------------------------------------------------|
| **RPM**                | *Relational Probability Model*. Modelo probabilístico definido a nivel de plantillas.        |
| **Grounding**          | Instanciación del RPM sobre los objetos concretos del problema.                              |
| **MRF**                | *Markov Random Field*. Grafo no dirigido cuyas distribuciones cumplen una propiedad de Markov local. |
| **Markov blanket**     | Conjunto de variables que vuelven a la variable de interés condicionalmente independiente del resto. |
| **Sweep**              | Una pasada completa del sampler que actualiza cada píxel exactamente una vez.                 |
| **Burn-in**            | Sweeps iniciales descartados para permitir que la cadena alcance la distribución estacionaria. |
| **Posterior marginal** | Probabilidad de cada variable individual integrada sobre el resto.                            |
| **β**                  | Hiperparámetro del potencial de Potts; controla la fuerza de la coherencia espacial.         |

---

## 11. Referencias internas

- `docs/Implementation_Guide.md` — especificación técnica completa.
- `docs/implementation_summary.md` — informe post-implementación.
- `docs/pipeline.html` — diagrama visual del flujo.
- `docs/entrega/marco_teorico.md` — marco teórico del proyecto.
- `docs/entrega/justificacion_del_problema.md` — justificación del problema.
