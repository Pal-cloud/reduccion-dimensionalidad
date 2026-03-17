"""Playground — experimenta libremente con todos los parámetros."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import load_dataset, apply_pca, apply_tsne, apply_umap, scatter_2d

st.set_page_config(page_title="Playground", page_icon="🎮", layout="wide")

st.markdown("# 🎮 Playground — Tu laboratorio personal")
st.markdown(
    "Aquí tienes **control total**. Elige dataset, algoritmo y parámetros, "
    "y observa cómo cambia el resultado en tiempo real."
)

# ── Sidebar de control ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    dataset_name = st.selectbox("Dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="pg_ds")
    algorithm = st.selectbox("Algoritmo", ["PCA 🧩", "t-SNE 🌌", "UMAP 🚀"], key="pg_algo")
    st.divider()

    if algorithm == "PCA 🧩":
        n_components = st.radio("Dimensiones de salida", [2, 3], key="pg_pca_nc", horizontal=True)
    elif algorithm == "t-SNE 🌌":
        perp = st.slider("Perplejidad", 5, 100, 30, key="pg_perp")
        n_iter = st.select_slider("Iteraciones", [250, 500, 750, 1000, 2000], value=1000, key="pg_iter")
        n_components = 2
    else:  # UMAP
        nn = st.slider("n_neighbors", 2, 100, 15, key="pg_nn")
        md = st.slider("min_dist", 0.0, 0.99, 0.1, 0.05, key="pg_md")
        n_components = 2

    show_raw = st.checkbox("Mostrar datos originales", value=False, key="pg_raw")
    run_btn = st.button("▶️ Aplicar", type="primary", key="pg_run")

# ── Carga de datos ─────────────────────────────────────────────────────────────
X, y, df_orig, desc = load_dataset(dataset_name)
st.info(desc)

col_plot, col_info = st.columns([2, 1])

# ── Ejecución ──────────────────────────────────────────────────────────────────
cache_key = (
    dataset_name, algorithm,
    locals().get("n_components", 2),
    locals().get("perp", None),
    locals().get("n_iter", None),
    locals().get("nn", None),
    locals().get("md", None),
)

if run_btn or "pg_result" not in st.session_state or st.session_state.get("pg_cfg") != cache_key:
    with st.spinner("Calculando…"):
        if algorithm == "PCA 🧩":
            nc = n_components
            X_r, pca_model = apply_pca(X, n_components=nc)
            y_r = y
        elif algorithm == "t-SNE 🌌":
            if dataset_name == "Dígitos ✏️" and X.shape[0] > 500:
                idx = np.random.RandomState(42).choice(X.shape[0], 500, replace=False)
                X_r = apply_tsne(X[idx], perplexity=perp, n_iter=n_iter)
                y_r = y[idx]
            else:
                X_r = apply_tsne(X, perplexity=perp, n_iter=n_iter)
                y_r = y
            nc = 2
        else:
            X_r = apply_umap(X, n_neighbors=nn, min_dist=md)
            y_r = y
            nc = 2
        st.session_state["pg_result"] = (X_r, y_r, nc)
        st.session_state["pg_cfg"] = cache_key
        if algorithm == "PCA 🧩":
            st.session_state["pg_pca_model"] = pca_model

if "pg_result" in st.session_state:
    X_r, y_r, nc = st.session_state["pg_result"]

    with col_plot:
        if nc == 3:
            from utils.helpers import scatter_3d
            fig = scatter_3d(X_r, y_r, title=f"{algorithm} — {dataset_name}")
        else:
            fig = scatter_2d(X_r, y_r, title=f"{algorithm} — {dataset_name}")
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown("### 📊 Estadísticas del resultado")
        df_res = pd.DataFrame(X_r, columns=[f"Dim {i+1}" for i in range(nc)])
        df_res["Clase"] = y_r
        st.dataframe(df_res.head(15), use_container_width=True, hide_index=True)

        if algorithm == "PCA 🧩" and "pg_pca_model" in st.session_state:
            pca_model = st.session_state["pg_pca_model"]
            var = pca_model.explained_variance_ratio_ * 100
            st.markdown("**Varianza explicada:**")
            for i, v in enumerate(var):
                st.progress(int(v), text=f"PC{i+1}: {v:.1f}%")

# ── Vista de datos crudos ──────────────────────────────────────────────────────
if show_raw:
    st.divider()
    st.markdown("### 🗃️ Datos originales (primeras 10 filas)")
    st.dataframe(df_orig.head(10), use_container_width=True)

# ── Glosario ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("## 📚 Glosario rápido")
terms = {
    "Dimensión": "Una variable o característica del dataset. Una imagen 28×28 tiene 784 dimensiones.",
    "Varianza": "Cuánto varían los datos. Alta varianza = mucha información.",
    "Componente principal (PC)": "Dirección en el espacio de máxima varianza. PC1 tiene más información que PC2.",
    "Manifold": "Superficie de baja dimensión incrustada en un espacio de alta dimensión.",
    "Perplejidad (t-SNE)": "Parámetro que controla cuántos vecinos considera cada punto.",
    "n_neighbors (UMAP)": "Número de vecinos del grafo de vecindad. Controla el equilibrio local/global.",
    "min_dist (UMAP)": "Distancia mínima entre puntos en el mapa 2D. Controla la compactación de clusters.",
    "Cluster": "Grupo natural de puntos similares en el espacio de características.",
    "Embedding": "La representación reducida de los datos en baja dimensión.",
    "Scree plot": "Gráfico de varianza explicada por cada componente de PCA.",
}

cols = st.columns(2)
items = list(terms.items())
half = len(items) // 2
for i, (term, defn) in enumerate(items):
    with cols[0 if i < half else 1]:
        with st.expander(f"**{term}**"):
            st.write(defn)
