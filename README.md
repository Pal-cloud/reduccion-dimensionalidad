# 🔭 Reducción de Dimensionalidad — Guía Interactiva

> Una aplicación web educativa e interactiva para aprender **PCA, t-SNE y UMAP** desde cero, sin necesidad de conocimientos matemáticos previos.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org)
[![UMAP](https://img.shields.io/badge/UMAP--learn-0.5%2B-purple)](https://umap-learn.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📸 Vista previa

| Inicio | PCA | Comparativa |
|--------|-----|-------------|
| Explicación con analogías cotidianas | Demo interactiva con scree plot | Los 3 algoritmos lado a lado |

---

## 🤔 ¿Qué es la reducción de dimensionalidad?

Imagina que quieres describir una persona con 1000 medidas distintas: altura, peso, color de ojos, tono de voz...
En Machine Learning, los datasets pueden tener **cientos o miles de variables** (dimensiones).

La **reducción de dimensionalidad** es la técnica que nos permite:
- 👁️ **Visualizar** datos de alta dimensión en 2D/3D
- ⚡ **Acelerar** el entrenamiento de modelos ML
- 🔍 **Descubrir** patrones y grupos ocultos en los datos

---

## 🧰 Algoritmos cubiertos

### 🧩 PCA — Principal Component Analysis
- **Tipo:** Lineal
- **Analogía:** Encontrar el ángulo perfecto para fotografiar un objeto 3D
- **Ideal para:** Reducción previa al entrenamiento de modelos, visualización rápida

### 🌌 t-SNE — t-distributed Stochastic Neighbor Embedding
- **Tipo:** No lineal
- **Analogía:** Organizar una fiesta colocando juntas a las personas que se conocen
- **Ideal para:** Visualización de clusters, exploración de datos

### 🚀 UMAP — Uniform Manifold Approximation and Projection
- **Tipo:** No lineal
- **Analogía:** Construir un mapa del metro preservando las conexiones entre estaciones
- **Ideal para:** Visualización + pipelines ML (puede transformar nuevos datos)

---

## 📂 Estructura del proyecto

```
reduccion-dimensionalidad/
│
├── Inicio.py                  # Página principal y presentación
├── requirements.txt           # Dependencias del proyecto
│
├── pages/
│   ├── 1_🧩_PCA.py            # Explicación + demo + quiz de PCA
│   ├── 2_🌌_t-SNE.py          # Explicación + demo + quiz de t-SNE
│   ├── 3_🚀_UMAP.py           # Explicación + demo + quiz de UMAP
│   ├── 4_⚔️_Comparar.py       # Comparativa visual de los 3 métodos
│   └── 5_🎮_Playground.py     # Laboratorio libre con glosario
│
├── utils/
│   └── helpers.py             # Funciones de carga de datos y algoritmos
│
└── .streamlit/
    └── config.toml            # Tema oscuro personalizado
```

---

## 🚀 Instalación y ejecución

### Requisitos
- Python 3.10 o superior
- pip

### Pasos

```bash
# 1. Clona el repositorio
git clone https://github.com/TU_USUARIO/reduccion-dimensionalidad.git
cd reduccion-dimensionalidad

# 2. (Recomendado) Crea un entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instala las dependencias
pip install -r requirements.txt

# 4. Lanza la aplicación
streamlit run Inicio.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501` 🎉

---

## 🗺️ Contenido de cada página

| Página | Contenido |
|--------|-----------|
| **🏠 Inicio** | Analogías cotidianas, mapa conceptual, presentación de los 3 algoritmos |
| **🧩 PCA** | Cómo funciona paso a paso, scree plot interactivo, loadings, quiz |
| **🌌 t-SNE** | Concepto de perplejidad, efecto de los parámetros, quiz |
| **🚀 UMAP** | Comparativa con t-SNE, manifolds, efecto de n_neighbors y min_dist, quiz |
| **⚔️ Comparar** | Los 3 métodos aplicados al mismo dataset lado a lado + guía de cuándo usar cada uno |
| **🎮 Playground** | Control total de parámetros, estadísticas del resultado, glosario |

---

## 📊 Datasets incluidos

| Dataset | Muestras | Dimensiones | Clases | Descripción |
|---------|----------|-------------|--------|-------------|
| **Iris** 🌸 | 150 | 4 | 3 | Medidas de flores de 3 especies |
| **Vino** 🍷 | 178 | 13 | 3 | Análisis químicos de vinos italianos |
| **Dígitos** ✏️ | 1797 | 64 | 10 | Imágenes 8×8 de números escritos a mano |

---

## 🛠️ Tecnologías utilizadas

| Librería | Uso |
|----------|-----|
| [Streamlit](https://streamlit.io) | Framework de la aplicación web |
| [scikit-learn](https://scikit-learn.org) | PCA, t-SNE y datasets |
| [UMAP-learn](https://umap-learn.readthedocs.io) | Algoritmo UMAP |
| [Plotly](https://plotly.com/python/) | Gráficos interactivos |
| [Pandas](https://pandas.pydata.org) | Manipulación de datos |
| [NumPy](https://numpy.org) | Operaciones numéricas |

---

## 📖 Para aprender más

- 📄 [Paper original de PCA (Pearson, 1901)](https://royalsocietypublishing.org/doi/10.1098/rspl.1901.0035)
- 📄 [Paper original de t-SNE (van der Maaten, 2008)](https://www.jmlr.org/papers/v9/vandermaaten08a.html)
- 📄 [Paper original de UMAP (McInnes, 2018)](https://arxiv.org/abs/1802.03426)
- 🎥 [StatQuest: PCA explicado visualmente](https://www.youtube.com/watch?v=FgakZw6K1QQ)
- 📚 [Documentación de scikit-learn: Decomposition](https://scikit-learn.org/stable/modules/decomposition.html)

---

<p align="center">Hecho con ❤️ para hacer el Machine Learning accesible a todos</p>
