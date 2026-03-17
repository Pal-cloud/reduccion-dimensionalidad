"""Página de comparación — PCA vs t-SNE vs UMAP lado a lado."""
import numpy as np
import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import load_dataset, apply_pca, apply_tsne, apply_umap, scatter_2d

st.set_page_config(page_title="Comparar", page_icon="⚔️", layout="wide")

st.markdown("# ⚔️ Comparativa: PCA vs t-SNE vs UMAP")
st.markdown(
    "Aplica los tres algoritmos **sobre el mismo dataset** y compara los resultados "
    "visualmente. ¿Cuál revela mejor la estructura de los datos?"
)

# ── Configuración ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 3])
with col_left:
    dataset_name = st.selectbox(
        "Dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="cmp_ds"
    )
    run_btn = st.button("▶️ Comparar los 3 métodos", type="primary", key="cmp_run")
    st.divider()
    st.markdown("### Parámetros avanzados")
    with st.expander("t-SNE"):
        perp = st.slider("Perplejidad", 5, 80, 30, key="cmp_perp")
        n_iter = st.select_slider("Iteraciones", [250, 500, 1000, 2000], value=1000, key="cmp_iter")
    with st.expander("UMAP"):
        nn = st.slider("n_neighbors", 2, 80, 15, key="cmp_nn")
        md = st.slider("min_dist", 0.0, 0.99, 0.1, 0.05, key="cmp_md")

X, y, _, desc = load_dataset(dataset_name)
st.info(desc)

cfg_key = (dataset_name, perp, n_iter, nn, md)

if run_btn or "cmp_results" not in st.session_state or st.session_state.get("cmp_cfg") != cfg_key:
    progress = st.progress(0, text="Calculando PCA…")
    X_pca, _ = apply_pca(X, n_components=2)
    progress.progress(33, text="Calculando t-SNE… ⏳")
    if dataset_name == "Dígitos ✏️" and X.shape[0] > 500:
        idx = np.random.RandomState(42).choice(X.shape[0], 500, replace=False)
        X_s, y_s = X[idx], y[idx]
        X_pca_s = X_pca[idx]
    else:
        X_s, y_s = X, y
        X_pca_s = X_pca
    X_tsne = apply_tsne(X_s, perplexity=perp, n_iter=n_iter)
    progress.progress(66, text="Calculando UMAP… 🗺️")
    X_umap = apply_umap(X_s, n_neighbors=nn, min_dist=md)
    progress.progress(100, text="¡Listo!")
    progress.empty()
    st.session_state["cmp_results"] = (X_pca_s, X_tsne, X_umap, y_s)
    st.session_state["cmp_cfg"] = cfg_key

if "cmp_results" in st.session_state:
    X_pca_r, X_tsne_r, X_umap_r, y_r = st.session_state["cmp_results"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🧩 PCA")
        st.plotly_chart(
            scatter_2d(X_pca_r, y_r, "PCA"),
            use_container_width=True,
        )
        st.markdown(
            "**🔑 Característica:** Lineal, muy rápido. "
            "Buena separación si las clases son linealmente separables."
        )
    with c2:
        st.markdown("### 🌌 t-SNE")
        st.plotly_chart(
            scatter_2d(X_tsne_r, y_r, f"t-SNE (perp={perp})"),
            use_container_width=True,
        )
        st.markdown(
            "**🔑 Característica:** No lineal, grupos muy compactos. "
            "Lento en datasets grandes."
        )
    with c3:
        st.markdown("### 🚀 UMAP")
        st.plotly_chart(
            scatter_2d(X_umap_r, y_r, f"UMAP (nn={nn}, md={md})"),
            use_container_width=True,
        )
        st.markdown(
            "**🔑 Característica:** No lineal, rápido, escalable. "
            "Preserva estructura local y global."
        )

    st.divider()
    st.markdown("## 📋 ¿Cuándo usar cada uno?")
    import pandas as pd
    df_guide = pd.DataFrame({
        "Situación": [
            "Quiero visualizar rápidamente mis datos",
            "Necesito reducir dimensiones para un modelo ML",
            "Quiero encontrar grupos ocultos",
            "Tengo un dataset enorme (> 100k filas)",
            "Necesito transformar nuevos datos",
            "Mis datos tienen estructura no lineal",
        ],
        "PCA": ["✅", "✅✅", "⚠️ Solo lineal", "✅✅", "✅", "❌"],
        "t-SNE": ["✅✅", "❌", "✅✅✅", "❌ Lento", "❌", "✅✅✅"],
        "UMAP": ["✅✅", "✅✅", "✅✅✅", "✅✅", "✅", "✅✅✅"],
    })
    st.dataframe(df_guide, use_container_width=True, hide_index=True)
