"""Página t-SNE — explicación interactiva."""
import numpy as np
import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import load_dataset, apply_tsne, scatter_2d

st.set_page_config(page_title="t-SNE", page_icon="🌌", layout="wide")

st.markdown("# 🌌 t-SNE — Visualización de grupos ocultos")
st.markdown(
    "> *Como organizar una fiesta: coloca juntas a las personas que se conocen "
    "y separa a las que no tienen relación entre sí.*"
)

tab1, tab2, tab3 = st.tabs(["📖 ¿Cómo funciona?", "🎯 Demo interactiva", "🧠 Quiz"])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## La idea detrás de t-SNE")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            ### El problema que resuelve
            PCA es lineal: sólo dobla y estira el espacio.
            Pero los datos reales tienen formas complejas, como una **esfera,
            una espiral o un toro**.

            t-SNE es **no lineal**: puede doblar, torcer y enrollar el espacio
            para revelar la estructura real de los datos.

            ### ¿Cómo lo hace?
            1. **Mide similitudes en alta dimensión:** ¿qué puntos están cerca?
               Usa una distribución normal (Gaussiana) para calcular probabilidades
               de vecindad.
            2. **Crea un mapa en 2D:** coloca los puntos al azar en 2D.
            3. **Ajusta iterativamente:** mueve los puntos para que las distancias
               en 2D reflejen las similitudes en alta dimensión.
               Usa una distribución **t de Student** (de ahí la "t") que tiene
               colas más gruesas → los grupos quedan más separados.
            """
        )
        st.info(
            "🎯 **t-SNE es excelente para visualización**, pero no para comprimir "
            "datos antes de entrenar modelos. Para eso usa PCA o UMAP."
        )

    with col2:
        st.markdown("### Parámetros clave que debes entender")
        st.markdown(
            """
            <style>
            .param-card { background: #1E1E2E; border-radius: 8px; padding: 1rem;
                          border-left: 3px solid #48CAE4; margin-bottom: .8rem; }
            </style>
            <div class="param-card">
                <b style="color:#48CAE4">Perplejidad (perplexity)</b><br>
                Cuántos vecinos "considera" cada punto. Valores bajos → grupos
                pequeños muy compactos. Valores altos → visión más global.<br>
                <small>💡 Recomendado: entre 5 y 50. Típico: 30.</small>
            </div>
            <div class="param-card">
                <b style="color:#48CAE4">Iteraciones (n_iter)</b><br>
                Cuántos pasos de optimización hace el algoritmo.<br>
                <small>💡 Mínimo 250. Recomendado: 1000+.</small>
            </div>
            <div class="param-card">
                <b style="color:#48CAE4">Tasa de aprendizaje (learning rate)</b><br>
                Qué tan rápido se mueven los puntos en cada iteración.<br>
                <small>💡 scikit-learn lo ajusta automáticamente.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(
            "⚠️ **Atención:** t-SNE es **estocástico** — cada ejecución puede "
            "dar resultados distintos. Las distancias entre clusters NO son "
            "directamente interpretables."
        )


# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🎯 Experimenta con t-SNE")
    st.markdown(
        "⏱️ *t-SNE puede tardar unos segundos. Es normal: es un algoritmo iterativo.*"
    )

    col_cfg, col_plot = st.columns([1, 2])
    with col_cfg:
        dataset_name = st.selectbox(
            "Dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="tsne_ds"
        )
        perplexity = st.slider(
            "Perplejidad", min_value=5, max_value=100, value=30, step=5, key="tsne_perp"
        )
        n_iter = st.select_slider(
            "Iteraciones", options=[250, 500, 750, 1000, 2000], value=1000, key="tsne_iter"
        )
        run_btn = st.button("▶️ Ejecutar t-SNE", type="primary", key="tsne_run")

    X, y, _, desc = load_dataset(dataset_name)
    st.info(desc)

    if run_btn or "tsne_result" not in st.session_state or st.session_state.get("tsne_cfg") != (dataset_name, perplexity, n_iter):
        with st.spinner("Calculando t-SNE… ⏳"):
            # Para dígitos usa submuestra para no tardar demasiado
            if dataset_name == "Dígitos ✏️" and X.shape[0] > 500:
                idx = np.random.RandomState(42).choice(X.shape[0], 500, replace=False)
                X_use, y_use = X[idx], y[idx]
            else:
                X_use, y_use = X, y
            X_tsne = apply_tsne(X_use, perplexity=perplexity, n_iter=n_iter)
            st.session_state["tsne_result"] = (X_tsne, y_use)
            st.session_state["tsne_cfg"] = (dataset_name, perplexity, n_iter)

    X_tsne, y_use = st.session_state["tsne_result"]
    with col_plot:
        fig = scatter_2d(X_tsne, y_use, title=f"t-SNE — {dataset_name} (perp={perplexity})")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💬 ¿Qué observas?")
    st.markdown(
        """
        - ¿Se forman **grupos (clusters) claramente separados**?
        - ¿Qué pasa si bajas la **perplejidad a 5**? (grupos muy pequeños y aislados)
        - ¿Qué pasa si la subes a **80**? (todo más mezclado, visión global)
        - ¿Se parecen los resultados con distintas semillas/iteraciones?
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🧠 Comprueba lo que has aprendido")

    questions = [
        {
            "q": "¿Qué tipo de relaciones captura t-SNE que PCA no puede?",
            "opts": [
                "Relaciones lineales entre variables",
                "Relaciones no lineales y estructuras curvas",
                "Correlaciones de Pearson",
                "Diferencias de escala entre variables",
            ],
            "ans": 1,
            "exp": "t-SNE es **no lineal**, por lo que puede revelar estructuras "
                   "complejas como espirales, esferas o manifolds curvos.",
        },
        {
            "q": "¿Qué significa una perplejidad muy alta en t-SNE?",
            "opts": [
                "El algoritmo considerará más vecinos y verá la estructura global",
                "Los grupos quedarán más pequeños y compactos",
                "El algoritmo irá más rápido",
                "Se perderá toda la información de los datos",
            ],
            "ans": 0,
            "exp": "Una perplejidad alta hace que cada punto considere más vecinos, "
                   "dando una visión más **global** de los datos. Perplejidad baja "
                   "→ énfasis en estructura local.",
        },
        {
            "q": "¿Por qué no se puede usar t-SNE para comprimir datos antes de entrenar un modelo?",
            "opts": [
                "Porque t-SNE sólo funciona con datasets pequeños",
                "Porque t-SNE no tiene función de transformación para nuevos datos",
                "Porque t-SNE siempre distorsiona las distancias",
                "Porque t-SNE es demasiado lento",
            ],
            "ans": 1,
            "exp": "t-SNE **no puede transformar nuevos datos** que no ha visto durante el entrenamiento. "
                   "Para usar reducción en pipelines de ML, usa PCA o UMAP.",
        },
    ]

    for i, item in enumerate(questions):
        st.markdown(f"**Pregunta {i+1}:** {item['q']}")
        choice = st.radio("", item["opts"], key=f"tsne_q{i}", index=None)
        if choice is not None:
            if item["opts"].index(choice) == item["ans"]:
                st.success(f"✅ ¡Correcto! {item['exp']}")
            else:
                st.error(f"❌ No exactamente. {item['exp']}")
        st.markdown("---")
