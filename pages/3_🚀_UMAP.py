"""Página UMAP — explicación interactiva y visual en español."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import load_dataset, apply_tsne, apply_umap, scatter_2d

st.set_page_config(page_title="UMAP", page_icon="🚀", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-label { font-size:.75rem; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:#A78BFA; margin-bottom:.2rem; }
.step-block { display:flex; gap:1rem; align-items:flex-start; margin-bottom:1.1rem; }
.step-num { background:#A78BFA; color:#0f172a; font-weight:800; border-radius:50%;
    width:2.2rem; height:2.2rem; min-width:2.2rem;
    display:flex; align-items:center; justify-content:center; font-size:.95rem; }
.step-body { color:#D1D5DB; font-size:.94rem; line-height:1.6; padding-top:.15rem; }
.callout-green  { background:#052e16; border:1px solid #16a34a; border-radius:8px;
    padding:.9rem 1.1rem; color:#86efac; font-size:.93rem; margin-bottom:.6rem; }
.callout-yellow { background:#1c1700; border:1px solid #ca8a04; border-radius:8px;
    padding:.9rem 1.1rem; color:#fde047; font-size:.93rem; margin-bottom:.6rem; }
.callout-blue   { background:#0c1a2e; border:1px solid #2563eb; border-radius:8px;
    padding:.9rem 1.1rem; color:#93c5fd; font-size:.93rem; margin-bottom:.6rem; }
.callout-purple { background:#1a0f2e; border:1px solid #7c3aed; border-radius:8px;
    padding:.9rem 1.1rem; color:#c4b5fd; font-size:.93rem; margin-bottom:.6rem; }
.big-metric { background:#1E1E2E; border-radius:12px; padding:1.2rem;
    text-align:center; border:1px solid #374151; }
.big-metric .val { font-size:2.5rem; font-weight:800; color:#A78BFA; }
.big-metric .lbl { font-size:.85rem; color:#9CA3AF; margin-top:.2rem; }
.vs-table td { padding: .4rem .8rem; font-size:.9rem; }
.vs-table th { padding: .4rem .8rem; font-size:.9rem; color:#A78BFA; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🚀 UMAP — Uniform Manifold Approximation and Projection")
st.markdown(
    "> *Como desenrollar una hoja de papel arrugada: revela la forma real de los datos "
    "preservando tanto los detalles locales como la estructura global.*"
)

tab1, tab2, tab3 = st.tabs([
    "📖 ¿Cómo funciona?",
    "🎯 Demo interactiva",
    "🧠 Quiz",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TEORÍA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">Fundamento conceptual</p>', unsafe_allow_html=True)
    st.markdown("## ¿Qué hace UMAP exactamente?")

    col_text, col_plot = st.columns([1, 1], gap="large")

    with col_text:
        st.markdown("""
Imagina que tienes una hoja de papel con puntos dibujados, y la arrugaste en una bola.
La bola ahora existe en 3D, pero la "naturaleza real" del papel es 2D.

UMAP es el proceso de **desenrollar esa bola** para volver a tener la hoja plana,
intentando que las distancias entre los puntos del papel se conserven al máximo posible.

En términos más precisos: UMAP asume que los datos viven en una **superficie curva
de baja dimensión** (llamada *manifold*) dentro del espacio de alta dimensión,
y usa matemáticas de grafos y topología para "aplanar" esa superficie.

> **La diferencia clave con t-SNE:**  
> t-SNE sólo preserva bien la estructura **local** (vecindarios).  
> UMAP preserva también la estructura **global** — la relación entre grupos distantes.
""")

        st.markdown("### 🪜 El algoritmo en 3 pasos")
        pasos = [
            ("Construir el grafo de vecindad (alta dimensión)",
             "Para cada punto, encuentra sus <strong>n_neighbors</strong> vecinos más cercanos "
             "y crea una arista ponderada (conexión) entre ellos. "
             "El peso de cada arista refleja qué tan cercanos son los puntos. "
             "El resultado es un <em>grafo difuso</em> que captura la topología de los datos."),
            ("Construir el grafo objetivo (baja dimensión)",
             "Crea un grafo equivalente en el espacio 2D de destino, "
             "con los puntos colocados inicialmente de forma aleatoria. "
             "El objetivo es que este grafo se parezca lo más posible al de alta dimensión."),
            ("Optimizar minimizando la diferencia entre grafos",
             "Mueve los puntos en 2D iterativamente usando descenso de gradiente "
             "para que el grafo de baja dimensión se parezca al de alta dimensión. "
             "UMAP usa una función de coste llamada <em>entropía cruzada difusa</em>. "
             "Esto es mucho más rápido que t-SNE y también preserva la estructura global."),
        ]
        for i, (titulo, cuerpo) in enumerate(pasos, 1):
            st.markdown(
                f'<div class="step-block">'
                f'<div class="step-num">{i}</div>'
                f'<div class="step-body"><strong style="color:#E5E7EB">{titulo}</strong>'
                f'<br>{cuerpo}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="callout-purple">🔑 <strong>Concepto clave — Manifold:</strong> '
            'Un manifold es una superficie que localmente parece plana, aunque globalmente '
            'sea curva. La Tierra es un manifold 2D que vive en 3D. '
            'UMAP asume que tus datos forman un manifold de baja dimensión '
            'dentro del espacio de alta dimensión y lo "desenrolla".'
            '</div>', unsafe_allow_html=True)

    with col_plot:
        st.markdown("### 🌐 Visualizando un manifold en 3D")
        st.markdown(
            "El ejemplo clásico: los datos viven en la superficie de un **rollo suizo** en 3D, "
            "pero su estructura real es 2D. UMAP lo detecta y lo aplana."
        )
        np.random.seed(42)
        n_manifold = 500
        t_m = 1.5 * np.pi * (1 + 2 * np.random.rand(n_manifold))
        h_m = 21 * np.random.rand(n_manifold)
        x_m = t_m * np.cos(t_m)
        y_m = h_m
        z_m = t_m * np.sin(t_m)
        color_m = t_m

        fig_manifold = go.Figure(data=[go.Scatter3d(
            x=x_m, y=y_m, z=z_m,
            mode="markers",
            marker=dict(
                size=4, color=color_m,
                colorscale="Viridis", opacity=0.85,
                colorbar=dict(title="Posición<br>en rollo", len=0.6)
            )
        )])
        fig_manifold.update_layout(
            template="plotly_dark", height=320,
            title="Rollo suizo — manifold 2D en espacio 3D",
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                bgcolor="#0e1117"),
            margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_manifold, use_container_width=True)

        st.markdown(
            '<div class="callout-blue">💡 UMAP detectaría que estos puntos viven en '
            'una superficie 2D y los "desenrollaría" para que la gradación de colores '
            'quede como una banda lineal — preservando las distancias reales entre puntos.'
            '</div>', unsafe_allow_html=True)

    st.divider()

    # ── Los dos parámetros clave ───────────────────────────────────────────
    st.markdown("## 🎛️ Los dos parámetros más importantes")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("### 🔗 `n_neighbors` — vecinos del grafo")
        st.markdown("""
Controla cuántos vecinos considera cada punto al construir el grafo.

| Valor | Efecto |
|-------|--------|
| **2–5** (muy bajo) | Ve sólo estructura muy local → clusters muy fragmentados |
| **15** (recomendado) | Balance entre detalle local y estructura global |
| **50–200** (alto) | Ve estructura global → clusters más grandes y conectados |

> **Analogía:** si sólo miras a tus 2 vecinos más cercanos, no ves el barrio.
> Si miras a 200, ves toda la ciudad pero pierdes el detalle del bloque.
""")
        st.markdown(
            '<div class="callout-green">✅ Para <strong>exploración inicial</strong>: '
            'usa n_neighbors=15. Para <strong>estructuras globales</strong>: '
            'sube a 50–100.</div>', unsafe_allow_html=True)

    with col_p2:
        st.markdown("### 📏 `min_dist` — distancia mínima en el mapa")
        st.markdown("""
Controla qué tan apretados están los puntos dentro de cada cluster en el resultado.

| Valor | Efecto |
|-------|--------|
| **0.0** | Clusters ultra-compactos, puntos muy juntos |
| **0.1** (recomendado) | Clusters compactos pero con estructura interna visible |
| **0.5** | Clusters más distribuidos |
| **0.9** | Puntos muy dispersos, sin estructura clara de clusters |

> **Analogía:** min_dist es como el zoom de la cámara al fotografiar los clusters.
> Bajo = zoom máximo, puntos muy juntos. Alto = gran angular, puntos dispersos.
""")
        st.markdown(
            '<div class="callout-blue">💡 Para <strong>visualización de clusters</strong>: '
            'usa min_dist=0.1. Para <strong>ver estructura continua</strong> (sin clusters): '
            'sube a 0.5–0.9.</div>', unsafe_allow_html=True)

    st.divider()

    # ── UMAP vs t-SNE ──────────────────────────────────────────────────────
    st.markdown("## ⚔️ UMAP vs t-SNE — comparativa completa")

    comparativa = {
        "Característica": [
            "Velocidad", "Estructura local", "Estructura global",
            "Transformar datos nuevos", "Escalabilidad", "Reproducibilidad",
            "Fundamento matemático", "Año de publicación",
        ],
        "t-SNE 🌌": [
            "🐢 Lento (O(n²))", "✅ Excelente", "⚠️ Se pierde parcialmente",
            "❌ Imposible", "❌ ~50k filas máx.", "⚠️ Variable (inicio aleatorio)",
            "Estadístico (probabilidades)", "2008",
        ],
        "UMAP 🚀": [
            "⚡ Rápido (O(n^1.14))", "✅ Muy buena", "✅ Se preserva bien",
            "✅ Sí (.transform())", "✅ Millones de filas", "✅ Con random_state",
            "Topológico (grafos)", "2018",
        ],
    }
    df_comp = pd.DataFrame(comparativa)
    st.dataframe(
        df_comp.set_index("Característica"),
        use_container_width=True,
        height=320)

    st.markdown(
        '<div class="callout-yellow">⚠️ <strong>¿Cuándo elegir t-SNE sobre UMAP?</strong> '
        'Cuando la separación visual de clusters compactos es lo más importante y el dataset '
        'no es muy grande. t-SNE a veces produce clusters más "nítidos" visualmente, '
        'aunque UMAP es generalmente superior en todo lo demás.</div>',
        unsafe_allow_html=True)

    st.divider()

    # ── Cuándo usar ────────────────────────────────────────────────────────
    st.markdown("## ✅❌ ¿Cuándo usar UMAP?")
    col_si, col_no = st.columns(2)
    with col_si:
        st.markdown("### ✅ Úsalo cuando...")
        st.markdown("""
- t-SNE es **demasiado lento** para tu dataset
- Necesitas un pipeline que **transforme datos nuevos**
- Quieres preservar tanto **estructura local como global**
- Trabajas en **bioinformática** (scRNA-seq, proteómica)
- Tu dataset tiene más de **50.000 filas**
- Necesitas **reproducibilidad** (fija `random_state`)
- Quieres usar la reducción como **preprocesamiento para ML**
""")
    with col_no:
        st.markdown("### ❌ Considera otras opciones cuando...")
        st.markdown("""
- Los datos tienen relaciones **puramente lineales** → PCA es más interpretable
- Necesitas que los **loadings** sean interpretables → PCA
- El tiempo de instalación es un problema (umap-learn requiere compiladores)
- La separación visual de clusters pequeños es crítica → t-SNE puede ser mejor
- Los datos tienen **muy pocas dimensiones** (2-5D) → visualiza directamente
""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEMO INTERACTIVA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">Manos a la obra</p>', unsafe_allow_html=True)
    st.markdown("## 🎯 Prueba UMAP con datos reales")

    col_cfg, col_plot = st.columns([1, 2.2])
    with col_cfg:
        dataset_name = st.selectbox("Dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="umap_ds")
        n_neighbors = st.slider("n_neighbors (vecinos)", 2, 100, 15, key="umap_nn")
        min_dist = st.slider("min_dist (separación)", 0.0, 0.99, 0.1, 0.05, key="umap_md")
        st.markdown("---")
        st.markdown("""
**💡 Qué observar:**
- Aumenta n_neighbors: ¿los clusters se conectan?
- Bájalo a 2–5: ¿aparecen sub-clusters?
- Sube min_dist a 0.9: ¿los puntos se dispersan?
- ¿Ves que los clusters mantienen su posición relativa?
  Eso es la **estructura global** que UMAP preserva.
""")
        st.markdown(
            '<div class="callout-purple">🚀 UMAP es mucho más rápido que t-SNE, '
            'especialmente para datasets grandes. Prueba cambiando parámetros — '
            'notarás la diferencia de velocidad.</div>', unsafe_allow_html=True)

    X, y, df_orig, desc = load_dataset(dataset_name)
    st.info(desc)

    with col_plot:
        with st.spinner("🔄 Ejecutando UMAP..."):
            X_umap = apply_umap(X, n_neighbors=n_neighbors, min_dist=min_dist)

        fig_umap = scatter_2d(
            X_umap, y,
            title=f"UMAP — {dataset_name} (n_neighbors={n_neighbors}, min_dist={min_dist})",
            x_label="UMAP 1",
            y_label="UMAP 2")
        st.plotly_chart(fig_umap, use_container_width=True)

        n_clusters = len(np.unique(y))
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Dimensiones originales", X.shape[1])
        col_m2.metric("Dimensiones reducidas", 2)
        col_m3.metric("Clases en el dataset", n_clusters)

    st.divider()

    # ── Comparativa UMAP vs t-SNE en vivo ─────────────────────────────────
    st.markdown("## ⚡ UMAP vs t-SNE — cara a cara en el mismo dataset")
    st.markdown(
        "Compara los resultados de ambos algoritmos con el **mismo dataset**. "
        "Observa las diferencias en la estructura global y la velocidad."
    )

    ds_vs = st.selectbox("Dataset para comparar", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="umap_vs_ds")

    if st.button("🔄 Comparar UMAP vs t-SNE", key="umap_vs_btn"):
        X_vs, y_vs, _, _ = load_dataset(ds_vs)
        col_u, col_t = st.columns(2)

        with col_u:
            with st.spinner("Ejecutando UMAP..."):
                import time
                t0 = time.time()
                Xu = apply_umap(X_vs, n_neighbors=15, min_dist=0.1)
                t_umap = time.time() - t0
            fig_u = scatter_2d(Xu, y_vs, title="UMAP (n_neighbors=15)",
                               x_label="UMAP 1", y_label="UMAP 2")
            fig_u.update_layout(height=330, showlegend=False,
                                margin=dict(l=10, r=10, t=40, b=20))
            st.plotly_chart(fig_u, use_container_width=True)
            st.metric("⏱️ Tiempo UMAP", f"{t_umap:.2f}s")
            st.markdown(
                '<div class="callout-purple">🚀 UMAP preserva la <strong>estructura global</strong>: '
                'la posición relativa de los clusters tiene sentido.</div>',
                unsafe_allow_html=True)

        with col_t:
            with st.spinner("Ejecutando t-SNE..."):
                t0 = time.time()
                Xt = apply_tsne(X_vs, perplexity=30, n_iter=500)
                t_tsne = time.time() - t0
            fig_t = scatter_2d(Xt, y_vs, title="t-SNE (perp=30)",
                               x_label="Dim 1", y_label="Dim 2")
            fig_t.update_layout(height=330, showlegend=False,
                                margin=dict(l=10, r=10, t=40, b=20))
            st.plotly_chart(fig_t, use_container_width=True)
            st.metric("⏱️ Tiempo t-SNE", f"{t_tsne:.2f}s")
            st.markdown(
                '<div class="callout-yellow">⚠️ t-SNE produce clusters más compactos, '
                'pero la posición relativa entre ellos <strong>no es interpretable</strong>.'
                '</div>', unsafe_allow_html=True)

        speedup = t_tsne / max(t_umap, 0.001)
        st.success(f"⚡ UMAP fue **{speedup:.1f}× más rápido** que t-SNE en este dataset.")
    else:
        st.info("👆 Pulsa el botón para lanzar la comparativa en vivo")

    st.divider()

    # ── Métricas de calidad ───────────────────────────────────────────────
    st.markdown("## 📏 Calidad del embedding UMAP")

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown("### 🎯 Preservación de estructura")
        st.markdown(
            "KNN en el espacio reducido. Si UMAP preservó la estructura, "
            "los puntos de la misma clase seguirán cerca."
        )
        try:
            knn = KNeighborsClassifier(n_neighbors=5)
            y_num = np.array([list(np.unique(y)).index(yi) for yi in y])
            scores = cross_val_score(knn, X_umap, y_num, cv=5, scoring="accuracy")
            acc = scores.mean() * 100
            st.markdown(
                f'<div class="big-metric"><div class="val">{acc:.1f}%</div>'
                f'<div class="lbl">Precisión KNN-5 en espacio UMAP (CV-5)</div></div>',
                unsafe_allow_html=True)
            if acc >= 90:
                st.success("✅ Excelente preservación de estructura")
            elif acc >= 70:
                st.info("👍 Buena preservación — prueba ajustando n_neighbors")
            else:
                st.warning("⚠️ Prueba con n_neighbors=15 y min_dist=0.1")
        except Exception as e:
            st.error(f"Error: {e}")

    with col_q2:
        st.markdown("### 📊 Compacidad de clusters")
        try:
            unique_labels = np.unique(y)
            intra_dists = []
            centroids = []
            for lbl in unique_labels:
                pts = X_umap[y == lbl]
                centroid = pts.mean(axis=0)
                centroids.append(centroid)
                intra_dists.append(np.mean(np.linalg.norm(pts - centroid, axis=1)))

            centroids = np.array(centroids)
            inter_dist = 0.0
            count = 0
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    inter_dist += np.linalg.norm(centroids[i] - centroids[j])
                    count += 1
            inter_dist /= max(count, 1)

            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Distancia intra-cluster (media)", f"{np.mean(intra_dists):.2f}")
            col_m2.metric("Distancia inter-cluster (media)", f"{inter_dist:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — QUIZ
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-label">Comprueba lo que sabes</p>', unsafe_allow_html=True)
    st.markdown("## 🧠 Quiz — pon a prueba tu comprensión de UMAP")
    st.markdown("4 preguntas con feedback inmediato y explicación detallada.")
    st.markdown("---")

    preguntas = [
        {
            "q": "¿Qué concepto matemático es fundamental en UMAP pero no en PCA ni t-SNE?",
            "opts": [
                "La descomposición en valores singulares (SVD)",
                "La topología y los grafos difusos (fuzzy graphs)",
                "Las distribuciones gaussianas multivariantes",
                "La regresión lineal por mínimos cuadrados",
            ],
            "ans": 1,
            "exp": "UMAP se basa en **topología algebraica** y representa los datos como "
                   "un **grafo difuso** (fuzzy simplicial complex). Esto le permite capturar "
                   "la forma topológica de los datos y es lo que lo diferencia fundamentalmente "
                   "de PCA (lineal-algebraico) y t-SNE (estadístico-probabilístico).",
        },
        {
            "q": "¿Cuál es la principal ventaja de UMAP sobre t-SNE?",
            "opts": [
                "UMAP siempre produce clusters más compactos visualmente",
                "UMAP es más antiguo y por tanto más probado",
                "UMAP es más rápido y puede transformar datos nuevos con .transform()",
                "UMAP no requiere fijar ningún parámetro",
            ],
            "ans": 2,
            "exp": "UMAP tiene **dos ventajas clave** sobre t-SNE: es mucho más rápido "
                   "(escala mejor con el tamaño del dataset) y puede transformar datos nuevos "
                   "usando `.transform()`. Esto lo hace útil como preprocesamiento en "
                   "pipelines de Machine Learning reales.",
        },
        {
            "q": "Si aumentas mucho n_neighbors en UMAP, ¿qué efecto esperas?",
            "opts": [
                "Los clusters se fragmentan en muchos sub-clusters",
                "El algoritmo se vuelve determinista sin necesidad de random_state",
                "El embedding preserva más estructura global y los clusters se conectan más",
                "Los puntos se distribuyen uniformemente sin estructura visible",
            ],
            "ans": 2,
            "exp": "Un **n_neighbors alto** (50-200) significa que cada punto considera "
                   "muchos vecinos, lo que da a UMAP una visión más 'global' de los datos. "
                   "El resultado son clusters más grandes y conectados que reflejan la "
                   "estructura a gran escala del dataset.",
        },
        {
            "q": "¿Qué es un 'manifold' en el contexto de UMAP?",
            "opts": [
                "Un tipo de gráfico para visualizar eigenvalues",
                "El número máximo de dimensiones que puede manejar UMAP",
                "Una superficie de baja dimensión 'enrollada' dentro de un espacio de alta dimensión",
                "El parámetro que controla la compacidad de los clusters",
            ],
            "ans": 2,
            "exp": "Un **manifold** es una superficie que localmente parece plana pero "
                   "globalmente puede ser curva. La Tierra es un manifold 2D en un espacio 3D. "
                   "UMAP asume que los datos de alta dimensión viven en un manifold de baja "
                   "dimensión y lo 'desenrolla' para visualizarlo en 2D.",
        },
    ]

    score = 0
    answered = 0
    for i, item in enumerate(preguntas):
        st.markdown(f"**Pregunta {i+1} de {len(preguntas)}:** {item['q']}")
        choice = st.radio("", item["opts"], key=f"umap_q{i}", index=None)
        if choice is not None:
            answered += 1
            if item["opts"].index(choice) == item["ans"]:
                score += 1
                st.success(f"✅ ¡Correcto! {item['exp']}")
            else:
                correcta = item["opts"][item["ans"]]
                st.error(f"❌ No del todo. La respuesta correcta es: **{correcta}**\n\n{item['exp']}")
        st.markdown("---")

    if answered == len(preguntas):
        pct = int(score / len(preguntas) * 100)
        if pct == 100:
            st.balloons()
            st.success(f"🏆 ¡Perfecto! {score}/{len(preguntas)} — ¡Dominas UMAP!")
        elif pct >= 50:
            st.info(f"👍 Bien: {score}/{len(preguntas)}. Repasa las explicaciones y vuelve a intentarlo.")
        else:
            st.warning(f"📚 {score}/{len(preguntas)}. Vuelve a la pestaña '¿Cómo funciona?' con atención.")
