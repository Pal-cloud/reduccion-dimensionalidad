"""Página UMAP — explicación interactiva."""
import numpy as np
import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import load_dataset, apply_umap, scatter_2d

st.set_page_config(page_title="UMAP", page_icon="🚀", layout="wide")

st.markdown("# 🚀 UMAP — El mapa topológico de tus datos")
st.markdown(
    "> *Imagina que construyes un mapa del metro: conservas qué estaciones están "
    "conectadas y cuán lejos están, pero doblas las líneas para que quepan en un papel.*"
)

tab1, tab2, tab3 = st.tabs(["📖 ¿Cómo funciona?", "🎯 Demo interactiva", "🧠 Quiz"])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## UMAP vs t-SNE — hermanos, pero distintos")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            ### ¿Qué es un manifold?
            Un **manifold** (variedad) es una superficie de menor dimensión
            "enrollada" dentro de un espacio de alta dimensión.

            Por ejemplo: la superficie de una pelota es un manifold 2D viviendo
            en un espacio 3D.

            UMAP asume que tus datos **viven en un manifold** y trata de
            "desenrollarlo" para verlo en 2D.

            ### Los pasos simplificados:
            1. **Construye un grafo de vecindad**: conecta cada punto con sus
               `n_neighbors` vecinos más cercanos.
            2. **Asigna pesos** a las conexiones según la distancia.
            3. **Optimiza un layout 2D** que preserve esas conexiones.
            """
        )
        st.success(
            "✅ **Ventaja clave frente a t-SNE:** UMAP **sí puede transformar "
            "nuevos datos** (tiene `transform()`), es mucho más rápido y "
            "preserva mejor la estructura global."
        )

    with col2:
        st.markdown("### Comparativa rápida")
        import pandas as pd
        df_comp = pd.DataFrame({
            "Característica": [
                "Velocidad", "Estructura global", "Estructura local",
                "Transformar nuevos datos", "Reproducibilidad", "Escalabilidad"
            ],
            "PCA": ["⚡⚡⚡", "✅✅✅", "❌", "✅", "✅✅✅", "✅✅✅"],
            "t-SNE": ["⚡", "⚠️", "✅✅✅", "❌", "⚠️", "⚡"],
            "UMAP": ["⚡⚡", "✅✅", "✅✅", "✅", "✅✅", "✅✅"],
        })
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.markdown("### Parámetros clave")
        st.markdown(
            """
            <style>
            .param-card { background: #1E1E2E; border-radius: 8px; padding: 1rem;
                          border-left: 3px solid #FF6B6B; margin-bottom: .8rem; }
            </style>
            <div class="param-card">
                <b style="color:#FF6B6B">n_neighbors</b><br>
                Número de vecinos del grafo. Bajo → estructura local.
                Alto → estructura global.<br>
                <small>💡 Recomendado: 5-50. Típico: 15.</small>
            </div>
            <div class="param-card">
                <b style="color:#FF6B6B">min_dist</b><br>
                Distancia mínima entre puntos en el mapa 2D.
                Bajo → puntos muy juntos (clusters compactos).
                Alto → distribución uniforme.<br>
                <small>💡 Recomendado: 0.0 – 0.9. Típico: 0.1.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🎯 Experimenta con UMAP")
    st.markdown(
        "⏱️ *UMAP es más rápido que t-SNE pero puede tardar unos segundos en datasets grandes.*"
    )

    col_cfg, col_plot = st.columns([1, 2])
    with col_cfg:
        dataset_name = st.selectbox(
            "Dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="umap_ds"
        )
        n_neighbors = st.slider(
            "n_neighbors (vecinos)", min_value=2, max_value=100, value=15, step=1, key="umap_nn"
        )
        min_dist = st.slider(
            "min_dist", min_value=0.0, max_value=0.99, value=0.1, step=0.05, key="umap_md"
        )
        run_btn = st.button("▶️ Ejecutar UMAP", type="primary", key="umap_run")

    X, y, _, desc = load_dataset(dataset_name)
    st.info(desc)

    cache_key = (dataset_name, n_neighbors, min_dist)
    if run_btn or "umap_result" not in st.session_state or st.session_state.get("umap_cfg") != cache_key:
        with st.spinner("Calculando UMAP… 🗺️"):
            X_umap = apply_umap(X, n_neighbors=n_neighbors, min_dist=min_dist)
            st.session_state["umap_result"] = (X_umap, y)
            st.session_state["umap_cfg"] = cache_key

    X_umap, y_stored = st.session_state["umap_result"]
    with col_plot:
        fig = scatter_2d(
            X_umap, y_stored,
            title=f"UMAP — {dataset_name} (n_neighbors={n_neighbors}, min_dist={min_dist})",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💬 ¿Qué observas?")
    st.markdown(
        """
        - Con **min_dist = 0.0**: los clusters son muy compactos y densos.
        - Con **min_dist = 0.9**: los puntos se distribuyen más uniformemente.
        - Con **n_neighbors bajo (2-5)**: la estructura local domina, los grupos se fragmentan.
        - Con **n_neighbors alto (50-100)**: visión más global, grupos más grandes.
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🧠 Comprueba lo que has aprendido")

    questions = [
        {
            "q": "¿Cuál es la principal ventaja de UMAP sobre t-SNE para pipelines de ML?",
            "opts": [
                "UMAP produce visualizaciones más bonitas",
                "UMAP puede transformar nuevos datos no vistos durante el entrenamiento",
                "UMAP siempre es más preciso que t-SNE",
                "UMAP no requiere ningún parámetro",
            ],
            "ans": 1,
            "exp": "UMAP aprende una función de transformación, por lo que **puede proyectar "
                   "nuevos datos** con `.transform()`. Esto lo hace útil en pipelines de ML reales.",
        },
        {
            "q": "¿Qué efecto tiene aumentar `n_neighbors` en UMAP?",
            "opts": [
                "El algoritmo ve más estructura local y los clusters se fragmentan",
                "El algoritmo ve más estructura global y los clusters son más coherentes",
                "Los puntos se colocan más juntos",
                "El algoritmo ignora las distancias",
            ],
            "ans": 1,
            "exp": "Con más vecinos, cada punto tiene una visión más **global** del espacio, "
                   "lo que preserva mejor la estructura de alto nivel.",
        },
        {
            "q": "¿Qué concepto matemático usa UMAP para modelar la estructura de los datos?",
            "opts": [
                "Matrices de covarianza",
                "Árboles de decisión",
                "Manifolds (variedades topológicas)",
                "Redes neuronales",
            ],
            "ans": 2,
            "exp": "UMAP asume que los datos viven en un **manifold** de menor dimensión "
                   "y usa geometría Riemanniana y teoría de categorías para construir el mapa.",
        },
    ]

    for i, item in enumerate(questions):
        st.markdown(f"**Pregunta {i+1}:** {item['q']}")
        choice = st.radio("", item["opts"], key=f"umap_q{i}", index=None)
        if choice is not None:
            if item["opts"].index(choice) == item["ans"]:
                st.success(f"✅ ¡Correcto! {item['exp']}")
            else:
                st.error(f"❌ No exactamente. {item['exp']}")
        st.markdown("---")
