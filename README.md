<div align="center">

# 🔭 Dimensionality Reduction — Interactive Guide

### *Learn PCA, t-SNE and UMAP from scratch — no math degree required*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![UMAP](https://img.shields.io/badge/UMAP--learn-0.5%2B-8B5CF6)](https://umap-learn.readthedocs.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E.svg)](LICENSE)

</div>

---

## 🌍 Introduction *(English)*

> **What if you could see your data the way it truly is — not as a spreadsheet of numbers, but as a living map of relationships?**

This project is an **interactive educational web app** built with Python and Streamlit that teaches one of the most powerful and misunderstood concepts in Machine Learning: **Dimensionality Reduction**.

Modern datasets are huge. A single grayscale image of 28×28 pixels already has **784 dimensions**. A genomics dataset can have **tens of thousands**. Our brains can only perceive 3 dimensions. So how do data scientists *see* what's going on?

They use dimensionality reduction — algorithms that **compress** high-dimensional data into 2D or 3D maps, preserving the most important structure. Think of it as creating a perfect map of a city: the map is not the city, but it captures what matters most.

This app covers **three landmark algorithms**:

| Algorithm | Year | Key Idea |
|-----------|------|----------|
| **PCA** | 1901 | Find the axes of maximum variance (linear) |
| **t-SNE** | 2008 | Preserve local neighborhoods (non-linear) |
| **UMAP** | 2018 | Preserve topological structure (non-linear, fast) |

Each algorithm is explained with **everyday analogies**, **interactive demos**, **real datasets**, and a **quiz** to test your understanding — no equations required.

---

## 🇪🇸 ¿Qué es la Reducción de Dimensionalidad?

### El problema: demasiados datos para el cerebro humano

Imagina que eres médico y tienes los análisis de sangre de 1.000 pacientes. Cada análisis incluye **50 valores distintos**: glucosa, colesterol, hemoglobina, presión, etc. Tu dataset tiene **50 dimensiones**.

¿Cómo visualizas eso? No puedes dibujar una gráfica de 50 ejes. Necesitas una forma de **comprimir esa información** a 2 ó 3 dimensiones sin perder lo importante.

Eso es exactamente lo que hace la **reducción de dimensionalidad**.

```
Dataset original:          Después de reducir:
[50 columnas por paciente] → [2 columnas: Dim1, Dim2]
[1.000 filas]              → [1.000 filas]

¡Ahora puedes hacer un scatter plot y VER los grupos!
```

### 🎒 La analogía de la maleta

> Cuando vas de viaje, no metes en la maleta **todo lo que tienes en casa**.
> Seleccionas sólo lo esencial: la ropa para el clima, los documentos importantes, el cargador.
>
> La reducción de dimensionalidad hace lo mismo con los datos:
> **se queda con la información más importante y descarta el ruido.**

### 📷 La analogía de la fotografía

> Una estatua existe en **3 dimensiones**, pero una fotografía sólo tiene **2**.
> Un buen fotógrafo elige el ángulo que mejor captura la forma de la estatua.
> PCA hace exactamente eso: busca el "ángulo" que muestra la mayor variación en los datos.

### 🗺️ La analogía del mapa

> Un mapa del mundo es una **reducción de 3D a 2D** del planeta.
> Algo se distorsiona (Groenlandia parece enorme), pero la estructura general se conserva.
> t-SNE y UMAP crean mapas así para tus datos, minimizando la distorsión.

---

## 🧰 Los tres algoritmos explicados

### 🧩 PCA — Principal Component Analysis

**En una frase:** *Rota el espacio de datos hasta encontrar la perspectiva que muestra la mayor variación posible.*

**La analogía completa:**
Imagina una nube de puntos en 3D (como un enjambre de abejas en el aire). PCA busca el **plano 2D** donde proyectar esas abejas de forma que la "sombra" resultante sea lo más grande y dispersa posible. Cuanto más grande la sombra, más información conserva.

**¿Cómo funciona paso a paso?**

```
1. CENTRAR    → Resta la media: mueve el origen al centro de los datos
               [1, 5, 3, 8] → [-3, 1, -1, 4]  (media = 4)

2. COVARIANZA → ¿Qué variables se mueven juntas?
               Si cuando "altura" sube también sube "peso" → covarianza alta

3. EIGENVECTORES → Las direcciones principales de la nube de puntos
               PC1 = dirección de máxima varianza
               PC2 = dirección de segunda mayor varianza (⊥ a PC1)
               PC3 = ... y así sucesivamente

4. PROYECTAR  → Comprime los datos a las primeras N componentes
               1000 dimensiones → 2 componentes (conservando 90% de info)
```

**¿Cuándo usarlo?**
- ✅ Antes de entrenar un modelo ML para acelerar el entrenamiento
- ✅ Para eliminar variables redundantes o correlacionadas
- ✅ Cuando necesitas que el resultado sea reproducible (no aleatorio)
- ❌ Cuando las relaciones entre variables son curvilíneas o muy complejas

---

### 🌌 t-SNE — t-distributed Stochastic Neighbor Embedding

**En una frase:** *Organiza los datos en 2D asegurándose de que los puntos similares queden cerca y los distintos queden lejos.*

**La analogía completa:**
Imagina que organizas una fiesta con 200 invitados que no se conocen entre sí, y tu misión es colocarlos en mesas. Mides **cuánto se parecen** entre sí (mismo trabajo, misma ciudad, mismos gustos) y los sientas juntos. Al final, los grupos naturales emergen solos.

t-SNE hace eso con tus datos: mide similitudes y redistribuye los puntos en 2D para que los grupos queden visibles.

**¿Cómo funciona?**

```
PASO 1 — En alta dimensión:
  Para cada punto, calcula la probabilidad de que otro punto sea su "vecino"
  Usa una campana de Gauss: vecinos cercanos → alta probabilidad
                             puntos lejanos  → probabilidad ≈ 0

PASO 2 — En 2D (inicio aleatorio):
  Coloca todos los puntos al azar en un plano 2D

PASO 3 — Optimización iterativa:
  Mueve los puntos para que las probabilidades de vecindad en 2D
  coincidan con las de alta dimensión
  Usa distribución t de Student (colas gruesas → clusters más separados)
  Repite miles de veces hasta converger
```

**El parámetro más importante: Perplejidad**

```
Perplejidad baja (5-10):   → Cada punto sólo mira sus vecinos MÁS cercanos
                              Resultado: muchos clusters pequeños y aislados

Perplejidad media (30):    → Balance entre estructura local y global ← RECOMENDADO

Perplejidad alta (80-100): → Cada punto tiene una visión más amplia
                              Resultado: clusters más grandes y conectados
```

> ⚠️ **Trampa frecuente:** Las distancias *entre* clusters en t-SNE **no son interpretables**. Que dos grupos estén cerca o lejos en el mapa 2D no significa que sean parecidos entre sí. Sólo la distancia *dentro* de un cluster tiene significado.

**¿Cuándo usarlo?**
- ✅ Exploración visual de datos: ¿hay grupos naturales?
- ✅ Verificar que un modelo de clustering tiene sentido
- ❌ Reducción previa a entrenamiento de modelos (no puede transformar datos nuevos)
- ❌ Datasets de más de ~50.000 filas (muy lento)

---

### 🚀 UMAP — Uniform Manifold Approximation and Projection

**En una frase:** *Construye un mapa topológico de los datos y lo "desenrolla" en 2D, preservando tanto los grupos locales como la estructura global.*

**La analogía completa:**
Imagina que tienes una hoja de papel arrugada en una bola. La hoja existe en 3D, pero su "naturaleza real" es 2D. UMAP es el proceso de **desenrollar la bola** para volver a tener la hoja plana, intentando que las distancias originales entre puntos de la hoja se conserven al máximo.

En términos más técnicos: asume que los datos viven en una superficie curva de baja dimensión (un **manifold**) dentro del espacio de alta dimensión, y usa grafos y geometría para "aplanar" esa superficie.

**Comparativa UMAP vs t-SNE:**

```
                    t-SNE           UMAP
─────────────────────────────────────────
Velocidad           🐢 Lento         ⚡ Rápido
Estructura global   ⚠️  Se pierde    ✅ Se preserva
Estructura local    ✅ Excelente     ✅ Muy buena
Nuevos datos        ❌ Imposible     ✅ .transform()
Escalabilidad       ❌ ~50k filas    ✅ Millones de filas
Reproducibilidad    ⚠️  Variable     ✅ Con random_state
```

**Los dos parámetros clave:**

```
n_neighbors (vecinos del grafo):
  Bajo (2-5)   → Ve sólo estructura muy local → clusters muy fragmentados
  Medio (15)   → Balance recomendado
  Alto (50-100)→ Ve estructura global → clusters más grandes y conectados

min_dist (distancia mínima en el mapa):
  0.0  → Puntos muy apretados, clusters ultra-compactos
  0.1  → Valor recomendado para la mayoría de casos
  0.9  → Puntos más distribuidos, menos compactos
```

**¿Cuándo usarlo?**
- ✅ Cuando t-SNE es demasiado lento (datasets grandes)
- ✅ Cuando necesitas un pipeline ML que procese datos nuevos
- ✅ Para preservar tanto estructura local como global
- ✅ Bioinformática, NLP, visión por computador

---

## 🗺️ Mapa de la aplicación

```
🔭 Reducción de Dimensionalidad
│
├── 🏠  Inicio
│       ├── Analogías cotidianas para entender el concepto
│       ├── ¿Por qué importa? (visualización, velocidad, descubrimiento)
│       └── Presentación de los 3 algoritmos con sus pros/contras
│
├── 🧩  PCA
│       ├── Tab 1 — Cómo funciona: vectores principales animados
│       ├── Tab 2 — Demo interactiva: 3 datasets, 2D/3D, loadings
│       ├── Tab 3 — Scree plot: ¿cuántas componentes necesito?
│       └── Tab 4 — Quiz: 3 preguntas con feedback inmediato
│
├── 🌌  t-SNE
│       ├── Tab 1 — Cómo funciona: perplejidad explicada visualmente
│       ├── Tab 2 — Demo interactiva: ajusta perplejidad e iteraciones
│       └── Tab 3 — Quiz: 3 preguntas con feedback inmediato
│
├── 🚀  UMAP
│       ├── Tab 1 — Cómo funciona: manifolds y comparativa con t-SNE
│       ├── Tab 2 — Demo interactiva: ajusta n_neighbors y min_dist
│       └── Tab 3 — Quiz: 3 preguntas con feedback inmediato
│
├── ⚔️  Comparar
│       ├── Los 3 algoritmos aplicados al mismo dataset, lado a lado
│       ├── Barra de progreso en tiempo real
│       └── Guía visual: ¿cuándo usar cada método?
│
└── 🎮  Playground
        ├── Control total de parámetros desde el sidebar
        ├── Visualización 2D y 3D interactiva
        ├── Tabla con los datos reducidos y estadísticas
        └── Glosario expandible de 10 términos clave
```

---

## 📊 Datasets incluidos

### 🌸 Iris — El "Hola Mundo" del Machine Learning
- **150 muestras** de flores, **4 dimensiones** (largo/ancho de sépalo y pétalo)
- **3 clases:** Setosa, Versicolor, Virginica
- Por qué es perfecto para aprender: las clases son casi linealmente separables → PCA funciona muy bien

### 🍷 Vino — Química en acción
- **178 muestras** de vinos italianos, **13 dimensiones** (alcohol, acidez, fenoles, color...)
- **3 clases:** 3 productores distintos
- Por qué es interesante: muchas variables correlacionadas → PCA reduce drásticamente sin perder info

### ✏️ Dígitos — Visión por computador simplificada
- **1.797 imágenes** de dígitos escritos a mano (0-9), **64 dimensiones** (píxeles 8×8)
- **10 clases:** un dígito por clase
- Por qué es impresionante: t-SNE y UMAP separan los 10 dígitos perfectamente en 2D desde 64D

---

## 📂 Estructura del proyecto

```
reduccion-dimensionalidad/
│
├── 📄 Inicio.py               ← Punto de entrada de la app (página principal)
├── 📄 requirements.txt        ← Todas las dependencias con versiones
├── 📄 README.md               ← Esta documentación
├── 📄 .gitignore
│
├── 📁 pages/                  ← Streamlit carga estas páginas automáticamente
│   ├── 1_🧩_PCA.py            ← 4 tabs: teoría · demo · scree plot · quiz
│   ├── 2_🌌_t-SNE.py          ← 3 tabs: teoría · demo interactiva · quiz
│   ├── 3_🚀_UMAP.py           ← 3 tabs: teoría · demo interactiva · quiz
│   ├── 4_⚔️_Comparar.py       ← Los 3 métodos frente a frente
│   └── 5_🎮_Playground.py     ← Laboratorio libre + glosario
│
├── 📁 utils/
│   └── helpers.py             ← Carga de datos, PCA, t-SNE, UMAP, gráficos
│
└── 📁 .streamlit/
    └── config.toml            ← Tema oscuro con colores personalizados
```

---

## 🚀 Instalación y ejecución

### Requisitos previos
- **Python 3.10 o superior** → [Descargar](https://python.org/downloads)
- **pip** (viene incluido con Python)
- **Git** → [Descargar](https://git-scm.com)

### Instalación paso a paso

```bash
# ── 1. Clona el repositorio ────────────────────────────────────────────────────
git clone https://github.com/TU_USUARIO/reduccion-dimensionalidad.git
cd reduccion-dimensionalidad

# ── 2. Crea un entorno virtual (muy recomendado) ───────────────────────────────
python -m venv venv

# Activa el entorno:
source venv/bin/activate          # 🐧 Linux / 🍎 macOS
venv\Scripts\activate             # 🪟 Windows (cmd / PowerShell)

# ── 3. Instala las dependencias ────────────────────────────────────────────────
pip install -r requirements.txt

# ── 4. Lanza la aplicación ─────────────────────────────────────────────────────
streamlit run Inicio.py
```

> 🎉 La app se abrirá automáticamente en **`http://localhost:8501`**

### ¿Problemas con la instalación?

```bash
# Si falla umap-learn en Windows, prueba primero:
pip install numpy --upgrade
pip install umap-learn

# Si tienes Python 3.13+ y hay conflictos:
pip install --pre umap-learn
```

---

## 🛠️ Tecnologías utilizadas

| Librería | Versión | Para qué se usa |
|----------|---------|-----------------|
| [Streamlit](https://streamlit.io) | ≥ 1.32 | Framework de la app web (UI, páginas, widgets) |
| [scikit-learn](https://scikit-learn.org) | ≥ 1.5 | PCA, t-SNE, datasets (Iris, Vino, Dígitos) |
| [UMAP-learn](https://umap-learn.readthedocs.io) | ≥ 0.5.6 | Algoritmo UMAP |
| [Plotly](https://plotly.com/python/) | ≥ 5.20 | Gráficos 2D/3D interactivos |
| [Pandas](https://pandas.pydata.org) | ≥ 2.2 | Manipulación y presentación de datos |
| [NumPy](https://numpy.org) | ≥ 1.26 | Operaciones matemáticas vectorizadas |
| [Pillow](https://pillow.readthedocs.io) | ≥ 10.0 | Procesamiento de imágenes |

---

## 🧠 Conceptos clave (glosario rápido)

| Término | Definición simple |
|---------|-------------------|
| **Dimensión** | Una variable o característica del dataset. Una imagen 28×28 tiene 784 dimensiones. |
| **Varianza** | Cuánto varían los datos. Alta varianza = mucha información. Baja varianza = ruido. |
| **Componente Principal (PC)** | Dirección en el espacio de máxima varianza. PC1 tiene más info que PC2, PC2 más que PC3... |
| **Eigenvector / Autovector** | El vector que define la dirección de una componente principal. |
| **Manifold** | Superficie de baja dimensión "enrollada" dentro de un espacio de alta dimensión. |
| **Perplejidad (t-SNE)** | Cuántos vecinos considera cada punto. Controla el balance local/global. |
| **n_neighbors (UMAP)** | Número de vecinos del grafo. Bajo = local, alto = global. |
| **Embedding** | La representación reducida de los datos en baja dimensión. |
| **Scree plot** | Gráfico de varianza explicada por componente de PCA. Ayuda a decidir cuántas PC usar. |
| **Cluster** | Grupo natural de puntos similares en el espacio de características. |

---

## 📖 Para profundizar

### 📺 Vídeos recomendados (sin matemáticas)
- 🎥 [StatQuest: PCA Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ) — El mejor vídeo de PCA en YouTube
- 🎥 [StatQuest: t-SNE Clearly Explained](https://www.youtube.com/watch?v=NEaUSP4YerM)
- 🎥 [UMAP Uniform Manifold Approximation (PyData)](https://www.youtube.com/watch?v=nq6iPZVUxZU)

### 📄 Papers originales
- 📘 [PCA — Pearson (1901)](https://royalsocietypublishing.org/doi/10.1098/rspl.1901.0035)
- 📘 [t-SNE — van der Maaten & Hinton (2008)](https://www.jmlr.org/papers/v9/vandermaaten08a.html)
- 📘 [UMAP — McInnes, Healy & Melville (2018)](https://arxiv.org/abs/1802.03426)

### 📚 Documentación técnica
- 📗 [scikit-learn: Decomposition (PCA)](https://scikit-learn.org/stable/modules/decomposition.html)
- 📗 [scikit-learn: Manifold learning (t-SNE)](https://scikit-learn.org/stable/modules/manifold.html)
- 📗 [UMAP Documentation](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html)

---

<div align="center">

**¿Te ha resultado útil? ¡Dale una ⭐ al repositorio!**

Hecho con ❤️ para hacer el Machine Learning accesible a todos

*"The goal is to turn data into information, and information into insight."* — Carly Fiorina

</div>
