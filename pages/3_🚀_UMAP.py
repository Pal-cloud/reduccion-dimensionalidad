"""Página UMAP — explicación interactiva."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import load_dataset, apply_umap, scatter_2d

st.set_page_config(page_title="UMAP", page_icon="🚀", layout="wide")

st.markdown("""
<style>
.section-label { font-size:.75rem; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:#FF6B6B; margin-bottom:.2rem; }
.concept-box { background:#1E1E2E; border-radius:10px; padding:1rem 1.2rem;
    border-left:4px solid; margin-bottom:.8rem; }
.step-block { display:flex; gap:1rem; align-items:flex-start; margin-bottom:1rem; }
.step-num { background:#FF6B6B; color:#fff; font-weight:800; border-radius:50%;
    width:2rem; height:2rem; min-width:2rem;
    display:flex; align-items:center; justify-content:center; font-size:.9rem; }
.step-body { color:#D1D5DB; font-size:.93rem; line-height:1.55; padding-top:.1rem; }
.callout-green  { background:#052e16; border:1px solid #16a34a; border-radius:8px;
    padding:.9rem 1.1rem; color:#86efac; font-size:.93rem; margin-bottom:.6rem; }
.callout-yellow { background:#1c1700; border:1px solid #ca8a04; border-radius:8px;
    padding:.9rem 1.1rem; color:#fde047; font-size:.93rem; margin-bottom:.6rem; }
.callout-blue   { background:#0c1a2e; border:1px solid #2563eb; border-radius:8px;
    padding:.9rem 1.1rem; color:#93c5fd; font-size:.93rem; margin-bottom:.6rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🚀 UMAP — Topological Map of Your Data")
st.markdown(
    "> *Like building a subway map: you keep which stations are connected and how far apart "
    "they are, but you bend and fold the lines to fit everything on a single page.*"
)

tab1, tab2, tab3 = st.tabs(["📖 How it works", "🎯 Interactive demo", "🧠 Quiz"])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">Conceptual foundation</p>', unsafe_allow_html=True)
    st.markdown("## UMAP — the smarter, faster sibling of t-SNE")

    col1, col2 = st.columns([1.1, 1], gap="large")
    with col1:
        st.markdown("""
### What is a manifold?

A **manifold** is a lower-dimensional surface *hidden inside* a high-dimensional space.

**Real-world examples:**
- 🌍 The Earth's surface is a 2D manifold living in 3D space
- 🧻 A crumpled sheet of paper is a 2D manifold in 3D
- 🍩 The surface of a donut (torus) is a 2D manifold in 3D

UMAP's key assumption: **your data lives on (or near) a low-dimensional manifold**.
Its job is to find that manifold and "unroll" it into 2D.
""")

        st.markdown("### The 3-step algorithm")
        steps = [
            ("Build a neighbourhood graph",
             "For each data point, find its <code>n_neighbors</code> nearest neighbours "
             "and connect them with edges. Assign weights to edges based on distance "
             "(closer = stronger connection). This creates a <strong>weighted graph</strong> "
             "that captures the local topology of the data."),
            ("Compute fuzzy simplicial sets",
             "Convert the graph into a fuzzy topological representation. "
             "This uses Riemannian geometry to account for the fact that distances "
             "may be measured differently in different regions of the manifold. "
             "<em>(Don't worry about the maths — the key idea is: build a reliable map of who's near who.)</em>"),
            ("Optimise the 2D layout",
             "Initialise points in 2D (often using PCA for stability). "
             "Then iteratively move points so that the 2D neighbourhood graph "
             "matches the high-dimensional graph as closely as possible. "
             "Uses <code>min_dist</code> to control how tightly points cluster together."),
        ]
        for i, (title, body) in enumerate(steps, 1):
            st.markdown(
                f'<div class="step-block">'
                f'<div class="step-num">{i}</div>'
                f'<div class="step-body"><strong style="color:#E5E7EB">{title}</strong>'
                f'<br>{body}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="callout-green">✅ <strong>UMAP\'s key advantage over t-SNE:</strong> '
            'It learns a proper mapping function, so it <strong>can transform new data</strong> '
            'with <code>.transform()</code>. This makes it suitable for real ML pipelines. '
            'It\'s also typically 5–10× faster than t-SNE.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="callout-blue">💡 <strong>Unlike t-SNE</strong>, UMAP also preserves '
            '<strong>global structure</strong> — the relative positions between clusters '
            'are more meaningful. Clusters that are far apart in UMAP tend to be genuinely '
            'more different from each other.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="callout-yellow">⚠️ <strong>Caveat:</strong> Like t-SNE, UMAP still '
            'distorts some distances. Treat the map as a guide, not a precise ruler. '
            'Always verify important findings with quantitative methods.</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("### Full comparison: PCA vs t-SNE vs UMAP")
        df_comp = pd.DataFrame({
            "Feature": [
                "Type", "Speed", "Global structure", "Local structure",
                "New data (.transform)", "Reproducible", "Scalability", "Best use case"
            ],
            "🧩 PCA": ["Linear", "⚡⚡⚡ Very fast", "✅✅✅", "⚠️ Weak",
                       "✅", "✅✅✅", "✅✅✅ Millions", "Preprocessing, fast viz"],
            "🌌 t-SNE": ["Non-linear", "🐢 Slow", "⚠️ Often lost", "✅✅✅",
                         "❌", "⚠️", "❌ ~50k rows", "Cluster exploration"],
            "🚀 UMAP": ["Non-linear", "⚡⚡ Fast", "✅✅", "✅✅",
                        "✅", "✅✅", "✅✅ Millions", "Everything"],
        })
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.markdown("### 🎚️ The two key parameters")
        params = [
            ("#FF6B6B", "n_neighbors — neighbourhood size",
             [("2–5", "Only immediate neighbours. Tiny, fragmented clusters. "
               "High-resolution local structure but noisy global layout."),
              ("15 ← default", "Good balance. Recommended starting point."),
              ("50–100", "Wide neighbourhood. Cohesive global structure, "
               "but local sub-clusters may merge.")]),
            ("#F59E0B", "min_dist — cluster compactness",
             [("0.0", "Points are packed as tightly as possible. "
               "Very compact, dense clusters."),
              ("0.1 ← default", "Slight spread. Good visual separation."),
              ("0.5–0.9", "Points spread uniformly. "
               "Useful when you care more about global layout than cluster tightness.")]),
        ]
        for color, param_name, values in params:
            rows_html = "".join(
                f'<tr><td style="color:{color};padding:.3rem .5rem;font-weight:600">{v}</td>'
                f'<td style="color:#D1D5DB;padding:.3rem .5rem;font-size:.87rem">{d}</td></tr>'
                for v, d in values
            )
            st.markdown(
                f'<div class="concept-box" style="border-left-color:{color}">'
                f'<strong style="color:{color}">{param_name}</strong>'
                f'<table style="width:100%;margin-top:.5rem;border-collapse:collapse">'
                f'{rows_html}</table></div>',
                unsafe_allow_html=True,
            )

        # Visual: n_neighbors effect diagram
        st.markdown("### 📐 n_neighbors: local vs global")
        categories = ["Local detail", "Cluster coherence", "Global layout", "Speed"]
        low_nn  = [5, 2, 1, 4]
        high_nn = [2, 5, 5, 3]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=low_nn + [low_nn[0]], theta=categories + [categories[0]],
                                             fill="toself", name="n_neighbors = 5",
                                             line_color="#FF6B6B"))
        fig_radar.add_trace(go.Scatterpolar(r=high_nn + [high_nn[0]], theta=categories + [categories[0]],
                                             fill="toself", name="n_neighbors = 50",
                                             line_color="#48CAE4"))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            template="plotly_dark", height=260,
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">Hands-on</p>', unsafe_allow_html=True)
    st.markdown("## 🎯 Experiment with UMAP")
    st.markdown("⏱️ *UMAP is faster than t-SNE but may take a few seconds on large datasets.*")

    col_cfg, col_plot = st.columns([1, 2])
    with col_cfg:
        dataset_name = st.selectbox("Dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="umap_ds")
        n_neighbors = st.slider("n_neighbors", min_value=2, max_value=100, value=15, step=1, key="umap_nn")
        min_dist = st.slider("min_dist", min_value=0.0, max_value=0.99, value=0.1, step=0.05, key="umap_md")
        run_btn = st.button("▶️ Run UMAP", type="primary", key="umap_run")
        st.markdown("---")
        st.markdown("""
**💡 Experiments to try:**
- Set `min_dist = 0.0` → ultra-compact clusters
- Set `min_dist = 0.9` → points spread uniformly
- Set `n_neighbors = 3` → fragmented micro-clusters
- Set `n_neighbors = 80` → cohesive global layout
- Compare with the t-SNE result on the same dataset
""")

    X, y, _, desc = load_dataset(dataset_name)
    st.info(desc)

    cache_key = (dataset_name, n_neighbors, min_dist)
    if (run_btn
            or "umap_result" not in st.session_state
            or st.session_state.get("umap_cfg") != cache_key):
        with st.spinner("Running UMAP… 🗺️"):
            X_umap = apply_umap(X, n_neighbors=n_neighbors, min_dist=min_dist)
            st.session_state["umap_result"] = (X_umap, y)
            st.session_state["umap_cfg"] = cache_key

    X_umap, y_stored = st.session_state["umap_result"]
    with col_plot:
        fig = scatter_2d(
            X_umap, y_stored,
            title=f"UMAP — {dataset_name}  (n_neighbors={n_neighbors}, min_dist={min_dist})",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💬 What do you observe?")
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        st.markdown("""
**Questions:**
- Are the clusters well separated?
- Do clusters farther apart seem more different conceptually?
- Does the structure change dramatically with `n_neighbors`?
""")
    with col_o2:
        st.markdown("""
**UMAP vs t-SNE:**
- UMAP tends to preserve more global structure
- UMAP usually runs faster
- Both can separate classes well — compare them in the ⚔️ Compare page
""")

# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-label">Test yourself</p>', unsafe_allow_html=True)
    st.markdown("## 🧠 Quiz — check your understanding")
    st.markdown("---")

    questions = [
        {
            "q": "What is the main advantage of UMAP over t-SNE for ML pipelines?",
            "opts": [
                "UMAP always produces more beautiful visualisations",
                "UMAP can transform new, unseen data points with .transform()",
                "UMAP is always more accurate than t-SNE",
                "UMAP requires no parameters at all",
            ],
            "ans": 1,
            "exp": "UMAP learns a proper mapping function, so it can **project new data** "
                   "with `.transform()`. This is essential for production ML pipelines "
                   "where new samples arrive after training.",
        },
        {
            "q": "What happens when you increase `n_neighbors` in UMAP?",
            "opts": [
                "The algorithm focuses on finer local structure and clusters fragment",
                "The algorithm gains a broader global view and clusters become more coherent",
                "Points are placed closer together on the map",
                "The algorithm ignores all distances",
            ],
            "ans": 1,
            "exp": "More neighbours means each point has a **wider view** of the data space, "
                   "which emphasises global structure. Fewer neighbours → fine local detail.",
        },
        {
            "q": "What does `min_dist = 0.0` do in UMAP?",
            "opts": [
                "Points are spread as uniformly as possible across the 2D map",
                "The algorithm ignores all local structure",
                "Points are packed as tightly as possible, creating very compact clusters",
                "UMAP becomes identical to PCA",
            ],
            "ans": 2,
            "exp": "`min_dist = 0.0` allows points to be placed right on top of each other "
                   "in the 2D map, creating **ultra-compact, dense clusters**. "
                   "Higher values spread points out more.",
        },
        {
            "q": "What mathematical concept does UMAP use to model data structure?",
            "opts": [
                "Covariance matrices (like PCA)",
                "Decision trees",
                "Manifolds — lower-dimensional surfaces embedded in high-dimensional space",
                "Neural networks",
            ],
            "ans": 2,
            "exp": "UMAP assumes data lies on a **manifold** and uses fuzzy topological "
                   "representations and Riemannian geometry to build the neighbourhood graph. "
                   "This is what makes it so powerful for non-linear data.",
        },
    ]

    for i, item in enumerate(questions):
        st.markdown(f"**Question {i+1} of {len(questions)}:** {item['q']}")
        choice = st.radio("", item["opts"], key=f"umap_q{i}", index=None)
        if choice is not None:
            if item["opts"].index(choice) == item["ans"]:
                st.success(f"✅ Correct! {item['exp']}")
            else:
                correct = item["opts"][item["ans"]]
                st.error(f"❌ Not quite. Correct answer: **{correct}**\n\n{item['exp']}")
        st.markdown("---")

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
