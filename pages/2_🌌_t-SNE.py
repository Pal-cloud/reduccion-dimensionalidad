"""Página t-SNE — explicación interactiva."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import load_dataset, apply_tsne, scatter_2d

st.set_page_config(page_title="t-SNE", page_icon="🌌", layout="wide")

st.markdown("""
<style>
.section-label { font-size:.75rem; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:#48CAE4; margin-bottom:.2rem; }
.concept-box { background:#1E1E2E; border-radius:10px; padding:1rem 1.2rem;
    border-left:4px solid; margin-bottom:.8rem; }
.step-block { display:flex; gap:1rem; align-items:flex-start; margin-bottom:1rem; }
.step-num { background:#48CAE4; color:#000; font-weight:800; border-radius:50%;
    width:2rem; height:2rem; min-width:2rem;
    display:flex; align-items:center; justify-content:center; font-size:.9rem; }
.step-body { color:#D1D5DB; font-size:.93rem; line-height:1.55; padding-top:.1rem; }
.callout-green  { background:#052e16; border:1px solid #16a34a; border-radius:8px;
    padding:.9rem 1.1rem; color:#86efac; font-size:.93rem; margin-bottom:.6rem; }
.callout-yellow { background:#1c1700; border:1px solid #ca8a04; border-radius:8px;
    padding:.9rem 1.1rem; color:#fde047; font-size:.93rem; margin-bottom:.6rem; }
.callout-red    { background:#2d0a0a; border:1px solid #dc2626; border-radius:8px;
    padding:.9rem 1.1rem; color:#fca5a5; font-size:.93rem; margin-bottom:.6rem; }
.perp-row { display:flex; gap:.5rem; align-items:center; margin-bottom:.5rem; }
.perp-badge { border-radius:6px; padding:.3rem .7rem; font-size:.82rem;
    font-weight:600; white-space:nowrap; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🌌 t-SNE — Visualising Hidden Clusters")
st.markdown(
    "> *Like seating guests at a wedding: place people who know each other together, "
    "and strangers far apart. The social groups emerge by themselves.*"
)

tab1, tab2, tab3 = st.tabs(["📖 How it works", "🎯 Interactive demo", "🧠 Quiz"])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">Conceptual foundation</p>', unsafe_allow_html=True)
    st.markdown("## The idea behind t-SNE")

    col1, col2 = st.columns([1.1, 1], gap="large")
    with col1:
        st.markdown("""
### Why PCA is not always enough

PCA is **linear** — it can only rotate and stretch the data space.
But many real datasets live on **curved surfaces** hidden inside high-dimensional space:
think of a spiral, a sphere, or a Swiss roll.

When PCA tries to flatten a spiral into 2D, it overlaps all the loops.
t-SNE can **unfold** it correctly.

### The core idea in plain English

Think of each data point as a person at a huge party.
t-SNE measures **how well two people know each other** (their similarity in high-D space),
then rearranges everyone in a room (2D map) so that:
- Good friends → placed close together
- Strangers → pushed far apart

It does this thousands of times, nudging everyone slightly, until the arrangement
is as consistent as possible with the original friendships.
""")

        st.markdown("### The 3-step algorithm")
        steps = [
            ("Measure similarities in high-D",
             "For each point, compute the probability that every other point is its 'neighbour'. "
             "Uses a <strong>Gaussian bell curve</strong>: nearby points get high probability, "
             "distant points get near-zero. The <em>perplexity</em> parameter controls "
             "how wide the bell is — wider bell = more neighbours considered."),
            ("Place points randomly in 2D",
             "All points start at random positions in a 2D plane. "
             "This is the initial noisy map that will be refined."),
            ("Iterative optimisation",
             "At each step, compute similarities in the 2D map using a "
             "<strong>t-Student distribution</strong> (heavier tails than Gaussian). "
             "Compare 2D similarities vs high-D similarities. "
             "Move points to reduce the mismatch. Repeat 250–2000 times. "
             "The t-Student tails push clusters farther apart, creating clean separations."),
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
            '<div class="callout-green">✅ <strong>What t-SNE is great at:</strong> '
            'Revealing tight, well-separated clusters in data with complex non-linear structure. '
            'Often produces the most visually compelling 2D maps.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="callout-red">🚫 <strong>Critical warning:</strong> '
            'Distances <em>between</em> clusters in a t-SNE plot are <strong>NOT meaningful</strong>. '
            'Two clusters being close or far on the map says nothing about how similar they '
            'actually are. Only the <em>tightness</em> within a cluster matters.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="callout-yellow">⚠️ <strong>Limitation:</strong> '
            't-SNE cannot transform new data points. '
            'Every run on a new dataset starts from scratch. '
            'For ML pipelines, use UMAP instead.</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("### 🎚️ Perplexity — the most important parameter")
        st.markdown(
            "Perplexity roughly equals the number of **effective neighbours** "
            "each point considers. It is the single most impactful parameter in t-SNE."
        )

        perp_examples = [
            ("#FF6B6B", "Perplexity 5", "Each point only looks at its 5 nearest neighbours. "
             "Result: many tiny, isolated micro-clusters. "
             "Good for spotting very fine-grained sub-groups."),
            ("#F59E0B", "Perplexity 15", "Moderate local focus. "
             "Clusters are more coherent but might split large groups into sub-groups."),
            ("#48CAE4", "Perplexity 30 ← recommended", "Balance between local structure "
             "(tight clusters) and global awareness. Good starting point for most datasets."),
            ("#6C63FF", "Perplexity 80+", "Each point sees a large neighbourhood. "
             "Clusters become bigger and more spread out. "
             "Useful for understanding the global layout, but local detail is lost."),
        ]
        for color, label, desc in perp_examples:
            st.markdown(
                f'<div class="concept-box" style="border-left-color:{color}">'
                f'<span class="perp-badge" style="background:{color}33;color:{color}">{label}</span>'
                f'<p style="margin:.4rem 0 0;color:#D1D5DB;font-size:.88rem">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("### 📐 Why t-Student instead of Gaussian?")
        # Mini chart comparing Gaussian vs t-Student tails
        x = np.linspace(-5, 5, 300)
        gauss = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
        t_dist = (1 + x**2) ** (-1)  # simplified t(1)
        t_dist = t_dist / t_dist.max() * gauss.max()

        fig_dists = go.Figure()
        fig_dists.add_trace(go.Scatter(x=x, y=gauss, name="Gaussian (high-D)",
                                       line=dict(color="#6C63FF", width=2.5)))
        fig_dists.add_trace(go.Scatter(x=x, y=t_dist, name="t-Student (2D map)",
                                       line=dict(color="#FF6B6B", width=2.5, dash="dot")))
        fig_dists.update_layout(
            template="plotly_dark", height=220,
            margin=dict(l=20, r=10, t=10, b=30),
            legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(size=11)),
            yaxis_title="Density", xaxis_title="Distance",
        )
        st.plotly_chart(fig_dists, use_container_width=True)
        st.caption(
            "The t-Student has **heavier tails** — distant points in 2D are allowed to be "
            "even further apart, which **pushes clusters apart** and creates cleaner maps."
        )

        st.markdown("### ⏱️ Iterations matter")
        st.markdown("""
| Iterations | Effect |
|-----------|--------|
| 250 | Very rough map, clusters barely visible |
| 500 | Clusters start to form |
| 1000 | Good quality, recommended default |
| 2000+ | Minor refinement, diminishing returns |

Run with **at least 1000 iterations** before drawing conclusions.
""")

# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">Hands-on</p>', unsafe_allow_html=True)
    st.markdown("## 🎯 Experiment with t-SNE")
    st.markdown("⏱️ *t-SNE can take a few seconds — it's an iterative optimisation algorithm.*")

    col_cfg, col_plot = st.columns([1, 2])
    with col_cfg:
        dataset_name = st.selectbox("Dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="tsne_ds")
        perplexity = st.slider("Perplexity", min_value=5, max_value=100, value=30, step=5, key="tsne_perp")
        n_iter = st.select_slider("Iterations", options=[250, 500, 750, 1000, 2000], value=1000, key="tsne_iter")
        run_btn = st.button("▶️ Run t-SNE", type="primary", key="tsne_run")
        st.markdown("---")
        st.markdown("""
**💡 Experiments to try:**
- Set perplexity to **5** → notice the tiny isolated clusters
- Set perplexity to **80** → notice how clusters merge
- Use **Digits ✏️** → can t-SNE separate all 10 digits from 64 pixels?
- Run twice with the same settings → results change slightly (stochastic)
""")

    X, y, _, desc = load_dataset(dataset_name)
    st.info(desc)

    if (run_btn
            or "tsne_result" not in st.session_state
            or st.session_state.get("tsne_cfg") != (dataset_name, perplexity, n_iter)):
        with st.spinner("Running t-SNE… ⏳"):
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
        fig = scatter_2d(X_tsne, y_use,
                         title=f"t-SNE — {dataset_name}  (perplexity={perplexity}, iter={n_iter})")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💬 What do you observe?")
    col_obs1, col_obs2 = st.columns(2)
    with col_obs1:
        st.markdown("""
**Questions to ask yourself:**
- Are the classes separated into distinct islands?
- Are there any classes that overlap or mix?
- Do some classes form multiple sub-clusters?
""")
    with col_obs2:
        st.markdown("""
**Remember:**
- Cluster *size* and *distance between clusters* are NOT meaningful in t-SNE
- Only cluster *tightness* and *separation* matter
- Different runs give different layouts (but similar cluster structure)
""")

# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-label">Test yourself</p>', unsafe_allow_html=True)
    st.markdown("## 🧠 Quiz — check your understanding")
    st.markdown("---")

    questions = [
        {
            "q": "What type of structure can t-SNE reveal that PCA cannot?",
            "opts": [
                "Linear correlations between variables",
                "Non-linear relationships and curved manifolds",
                "Pearson correlations",
                "Scale differences between variables",
            ],
            "ans": 1,
            "exp": "t-SNE is **non-linear**, so it can reveal complex structures "
                   "like spirals, spheres, and curved manifolds that PCA would flatten incorrectly.",
        },
        {
            "q": "What does a very HIGH perplexity value do in t-SNE?",
            "opts": [
                "Each point considers more neighbours → global structure is emphasised",
                "Clusters become smaller and more compact",
                "The algorithm runs faster",
                "All information in the data is lost",
            ],
            "ans": 0,
            "exp": "High perplexity makes each point consider a **wider neighbourhood**, "
                   "giving a more global view of the data. Low perplexity = local focus.",
        },
        {
            "q": "Why can't you use t-SNE to preprocess data for a Machine Learning pipeline?",
            "opts": [
                "Because t-SNE only works with small datasets",
                "Because t-SNE cannot transform new, unseen data points",
                "Because t-SNE always distorts distances",
                "Because t-SNE is too slow for any practical use",
            ],
            "ans": 1,
            "exp": "t-SNE **has no transform function** — it cannot project new data "
                   "that wasn't seen during fitting. Use PCA or UMAP for production pipelines.",
        },
        {
            "q": "What is dangerous about interpreting distances BETWEEN clusters in a t-SNE plot?",
            "opts": [
                "Nothing — distances are always reliable in t-SNE",
                "Farther clusters are always less similar",
                "Inter-cluster distances are NOT meaningful — only intra-cluster tightness is",
                "Closer clusters are always more similar",
            ],
            "ans": 2,
            "exp": "This is the most common t-SNE mistake! The distance between two clusters "
                   "on the map tells you nothing about their actual similarity. "
                   "Only the compactness within a cluster has meaning.",
        },
    ]

    for i, item in enumerate(questions):
        st.markdown(f"**Question {i+1} of {len(questions)}:** {item['q']}")
        choice = st.radio("", item["opts"], key=f"tsne_q{i}", index=None)
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
