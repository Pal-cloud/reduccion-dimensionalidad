"""Página principal — Inicio y presentación del proyecto."""
import streamlit as st

st.set_page_config(
    page_title="Reducción de Dimensionalidad",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS extra ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .hero-title   { font-size: 3rem; font-weight: 800; text-align: center;
                    background: linear-gradient(90deg, #6C63FF, #48CAE4);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-sub     { font-size: 1.2rem; text-align: center; color: #aaa; margin-bottom: 2rem; }
    .card         { background: #1E1E2E; border-radius: 12px; padding: 1.4rem;
                    border-left: 4px solid #6C63FF; margin-bottom: 1rem; }
    .card h3      { margin-top: 0; color: #6C63FF; }
    .algo-badge   { display: inline-block; padding: .25rem .7rem; border-radius: 999px;
                    font-size: .85rem; font-weight: 600; margin: .15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🔭 Reducción de Dimensionalidad</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Aprende a ver la esencia de los datos eliminando lo que sobra</p>',
    unsafe_allow_html=True,
)

st.divider()

# ── Analogía principal ─────────────────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1])
with col1:
    st.markdown("## 🤔 ¿Qué es la dimensionalidad?")
    st.markdown(
        """
        Imagina que quieres describir una **persona**. Podrías usar:

        - Altura, peso, color de ojos, tono de voz, tamaño del pie…

        Cada una de esas medidas es una **dimensión**. En el mundo real los datos
        pueden tener **cientos o miles** de dimensiones (píxeles de una imagen, genes,
        palabras de un texto…).

        El problema es que **nuestro cerebro sólo puede visualizar hasta 3 dimensiones**,
        y muchos algoritmos funcionan mucho peor con demasiadas variables.
        """
    )
    st.info(
        "🎒 **Analogía de la mochila:** Cuando haces la maleta, no metes TODO lo que tienes "
        "en casa. Seleccionas sólo lo esencial. La reducción de dimensionalidad hace lo mismo "
        "con los datos: **se queda con la información más importante y descarta el ruido.**"
    )

with col2:
    st.markdown("## 🗺️ Hoja de ruta")
    st.markdown(
        """
        Navega por las páginas del menú lateral para explorar:

        | Página | Qué aprenderás |
        |--------|---------------|
        | 🧩 **PCA** | Comprimir datos conservando la varianza |
        | 🌌 **t-SNE** | Revelar grupos ocultos en los datos |
        | 🚀 **UMAP** | Alternativa rápida y escalable a t-SNE |
        | ⚔️ **Comparar** | Los 3 métodos frente a frente |
        | 🎮 **Playground** | Experimenta con tus propios parámetros |
        """
    )

st.divider()

# ── ¿Por qué importa? ──────────────────────────────────────────────────────────
st.markdown("## 💡 ¿Por qué es importante?")

c1, c2, c3 = st.columns(3)
cards = [
    ("👁️ Visualización", "#6C63FF",
     "Transforma datos de 100+ dimensiones en gráficos 2D/3D que los humanos "
     "podemos entender de un vistazo."),
    ("⚡ Velocidad", "#48CAE4",
     "Los algoritmos de Machine Learning entrenan **mucho más rápido** con menos "
     "variables, sin perder precisión."),
    ("🔍 Descubrimiento", "#FF6B6B",
     "Ayuda a encontrar **patrones ocultos** y grupos naturales que no éramos "
     "capaces de ver en la tabla de datos."),
]
for col, (title, color, text) in zip([c1, c2, c3], cards):
    with col:
        st.markdown(
            f"""
            <div class="card">
                <h3 style="color:{color}">{title}</h3>
                <p>{text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── Los 3 algoritmos en una línea ─────────────────────────────────────────────
st.markdown("## 🧰 Los 3 algoritmos que veremos")
a1, a2, a3 = st.columns(3)

algo_info = [
    ("🧩 PCA", "Principal Component Analysis",
     "Como encontrar el ángulo perfecto para fotografiar un objeto en 3D: "
     "elige la perspectiva que muestra MÁS información.",
     "Lineal · Rápido · Reproducible", "#6C63FF"),
    ("🌌 t-SNE", "t-distributed Stochastic Neighbor Embedding",
     "Como organizar una fiesta: coloca a las personas que se conocen juntas "
     "y separa a las que no tienen relación.",
     "No lineal · Grupos claros · Lento en datos grandes", "#48CAE4"),
    ("🚀 UMAP", "Uniform Manifold Approximation and Projection",
     "Igual que t-SNE pero usando un mapa topológico del espacio. "
     "Más rápido y preserva mejor la estructura global.",
     "No lineal · Rápido · Escalable", "#FF6B6B"),
]
for col, (name, full, analogy, props, color) in zip([a1, a2, a3], algo_info):
    with col:
        st.markdown(
            f"""
            <div class="card" style="border-left-color:{color}">
                <h3 style="color:{color}">{name}</h3>
                <p><em>{full}</em></p>
                <p>{analogy}</p>
                <small style="color:#888">{props}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    "<p style='text-align:center;color:#555;font-size:.9rem'>"
    "Hecho con ❤️ usando Python · Streamlit · scikit-learn · UMAP · Plotly"
    "</p>",
    unsafe_allow_html=True,
)
