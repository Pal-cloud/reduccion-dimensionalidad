"""Página de inicio — Guía interactiva de reducción de dimensionalidad en español."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

st.set_page_config(
    page_title="Reducción de Dimensionalidad — Guía Interactiva",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-size: 3.2rem; font-weight: 800; text-align: center;
    background: linear-gradient(135deg, #6C63FF 0%, #48CAE4 50%, #FF6B6B 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2; margin-bottom: .4rem;
}
.hero-sub {
    font-size: 1.25rem; text-align: center; color: #9CA3AF; margin-bottom: .5rem;
}
.hero-badge {
    display: flex; justify-content: center; gap: .5rem;
    flex-wrap: wrap; margin-bottom: 2rem;
}
.badge {
    background: #1E1E2E; border: 1px solid #374151;
    border-radius: 999px; padding: .3rem .9rem;
    font-size: .8rem; color: #D1D5DB;
}
.section-label {
    font-size: .75rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: #6C63FF; margin-bottom: .3rem;
}
.nav-card {
    background: #1E1E2E;
    border: 1px solid #2D2D3F;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: .8rem;
    transition: border-color .2s;
    cursor: default;
}
.nav-card:hover { border-color: #6C63FF; }
.nav-card .icon { font-size: 2rem; margin-bottom: .4rem; }
.nav-card h3 { margin: 0 0 .3rem 0; font-size: 1.05rem; font-weight: 700; }
.nav-card p  { margin: 0; color: #9CA3AF; font-size: .9rem; line-height: 1.5; }
.nav-card .tags { margin-top: .6rem; }
.tag {
    display: inline-block; border-radius: 999px;
    padding: .15rem .6rem; font-size: .75rem; font-weight: 600;
    margin-right: .25rem;
}
.concept-box {
    background: #1E1E2E; border-radius: 12px;
    padding: 1.2rem 1.4rem; border-left: 4px solid;
    margin-bottom: .8rem;
}
.step-row { display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1rem; }
.step-num {
    background: #6C63FF; color: white; font-weight: 800;
    border-radius: 50%; width: 2rem; height: 2rem; min-width: 2rem;
    display: flex; align-items: center; justify-content: center;
    font-size: .9rem;
}
.step-text { color: #D1D5DB; font-size: .95rem; line-height: 1.5; padding-top: .15rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HÉROE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="hero-title">🔭 Reducción de Dimensionalidad</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Guía interactiva de PCA, t-SNE y UMAP — '
    'sin fórmulas, sin miedo, con ejemplos reales</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-badge">'
    '<span class="badge">🐍 Python</span>'
    '<span class="badge">📊 scikit-learn</span>'
    '<span class="badge">🚀 UMAP</span>'
    '<span class="badge">📈 Plotly</span>'
    '<span class="badge">🎛️ Streamlit</span>'
    '<span class="badge">🆓 Código abierto</span>'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ¿QUÉ ES? + DEMO EN VIVO
# ══════════════════════════════════════════════════════════════════════════════
col_what, col_live = st.columns([1.1, 1], gap="large")

with col_what:
    st.markdown('<p class="section-label">La idea central</p>', unsafe_allow_html=True)
    st.markdown("## ¿Qué es la reducción de dimensionalidad?")
    st.markdown("""
Los datos modernos son enormes. Una imagen en escala de grises de 28×28 píxeles
tiene **784 dimensiones**. Un dataset genómico puede tener **decenas de miles**.
Nuestros cerebros sólo pueden visualizar 3.

La **reducción de dimensionalidad** comprime datos de alta dimensión en mapas 2D o 3D
conservando la estructura más importante — permitiéndote *ver* patrones que eran
completamente invisibles en una hoja de cálculo.
""")

    # Tres analogías cotidianas
    analogias = [
        ("#6C63FF",
         "🎒 La maleta de viaje",
         "No te llevas toda la casa de viaje. Empacas sólo lo esencial. "
         "La reducción de dimensionalidad hace lo mismo: guarda las características "
         "más informativas y descarta el ruido."),
        ("#48CAE4",
         "📷 La fotografía",
         "Una escultura existe en 3D, pero la foto es 2D. Un buen fotógrafo elige "
         "el ángulo que muestra más detalle. PCA encuentra exactamente ese ángulo "
         "para tus datos."),
        ("#FF6B6B",
         "🗺️ El mapa de la ciudad",
         "Un mapa no es la ciudad — es una compresión 2D de la realidad 3D. "
         "t-SNE y UMAP construyen mapas así para tus datos, minimizando la distorsión."),
    ]
    for color, titulo, texto in analogias:
        st.markdown(
            f'<div class="concept-box" style="border-left-color:{color}">'
            f'<strong style="color:{color}">{titulo}</strong>'
            f'<p style="margin:.4rem 0 0 0;color:#D1D5DB;font-size:.92rem">{texto}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

with col_live:
    st.markdown('<p class="section-label">Vista previa en vivo</p>', unsafe_allow_html=True)
    st.markdown("## Dataset Iris: 4D → 2D con PCA")
    st.markdown(
        "Abajo tienes una proyección real con PCA del dataset de flores Iris. "
        "150 flores descritas por **4 medidas** comprimidas en **2 dimensiones**. "
        "Fíjate cómo las tres especies forman grupos naturales bien separados."
    )

    # ── mini-demo en vivo ──────────────────────────────────────────────────
    iris = load_iris()
    X_scaled = (iris.data - iris.data.mean(axis=0)) / iris.data.std(axis=0)
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X_scaled)
    colores = ["#6C63FF", "#48CAE4", "#FF6B6B"]
    nombres = iris.target_names

    fig_home = go.Figure()
    for i, (nombre, color) in enumerate(zip(nombres, colores)):
        mask = iris.target == i
        fig_home.add_trace(go.Scatter(
            x=Xp[mask, 0], y=Xp[mask, 1],
            mode="markers",
            name=nombre.capitalize(),
            marker=dict(color=color, size=9, opacity=0.85,
                        line=dict(width=0.5, color="white")),
        ))
    v = pca.explained_variance_ratio_
    fig_home.update_layout(
        template="plotly_dark",
        height=340,
        xaxis_title=f"CP1 — {v[0]*100:.1f}% de varianza",
        yaxis_title=f"CP2 — {v[1]*100:.1f}% de varianza",
        legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(size=12)),
        margin=dict(l=30, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_home, use_container_width=True)
    st.caption(
        f"✅ Con sólo 2 números por flor capturamos "
        f"**{(v[0]+v[1])*100:.1f}%** de toda la información contenida en 4 variables."
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ¿POR QUÉ IMPORTA?
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">¿Por qué importa?</p>', unsafe_allow_html=True)
st.markdown("## Tres razones por las que todo científico de datos necesita esto")

r1, r2, r3 = st.columns(3, gap="medium")
razones = [
    ("👁️", "Visualización", "#6C63FF",
     "Convierte una hoja de cálculo con 500 columnas en un diagrama 2D que entiendes "
     "en segundos. Detecta grupos, valores atípicos y tendencias al instante."),
    ("⚡", "Velocidad", "#48CAE4",
     "Los modelos de Machine Learning entrenan **10×–100× más rápido** con datos "
     "comprimidos. Menos dimensiones = menos parámetros = menos sobreajuste."),
    ("🔍", "Descubrimiento", "#FF6B6B",
     "Revela grupos y relaciones ocultas que son completamente invisibles en tablas "
     "de alta dimensión sin procesar. Deja que los datos te sorprendan."),
]
for col, (icono, titulo, color, texto) in zip([r1, r2, r3], razones):
    with col:
        st.markdown(
            f'<div class="nav-card" style="border-left: 4px solid {color};">'
            f'<div class="icon">{icono}</div>'
            f'<h3 style="color:{color}">{titulo}</h3>'
            f'<p>{texto}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TARJETAS DE NAVEGACIÓN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Explorar la app</p>', unsafe_allow_html=True)
st.markdown("## Elige un tema y sumérgete")
st.markdown(
    "Usa la **barra lateral** o haz clic en cualquier tarjeta para saltar a una sección. "
    "Cada página incluye teoría, una demo interactiva y un quiz."
)

paginas = [
    ("🧩", "PCA", "Análisis de Componentes Principales",
     "El clásico. Rota tus datos para encontrar las direcciones de máxima varianza. "
     "Ideal para velocidad, reproducibilidad y preprocesamiento de pipelines de ML.",
     "Lineal", "Muy rápido", "Reproducible", "#6C63FF", "pages/1_🧩_PCA"),
    ("🌌", "t-SNE", "Inserción Estocástica de Vecinos con t de Student",
     "Coloca puntos similares cerca en 2D. Revela clústeres ocultos incluso "
     "en estructuras de datos altamente no lineales.",
     "No lineal", "Mejor para clústeres", "Lento en datos grandes", "#48CAE4", "pages/2_🌌_t-SNE"),
    ("🚀", "UMAP", "Aproximación y Proyección de Variedades Uniformes",
     "Más rápido y escalable que t-SNE. Preserva tanto los clústeres locales "
     "COMO la estructura global. Puede transformar datos nuevos.",
     "No lineal", "Rápido", "Escalable", "#FF6B6B", "pages/3_🚀_UMAP"),
    ("⚔️", "Comparar", "PCA vs t-SNE vs UMAP cara a cara",
     "Aplica los tres algoritmos al mismo dataset simultáneamente "
     "y observa las diferencias con tus propios ojos.",
     "Todos los métodos", "Mismo dataset", "Lado a lado", "#F59E0B", "pages/4_⚔️_Comparar"),
    ("🎮", "Laboratorio", "Tu laboratorio personal",
     "Control total sobre cada parámetro. Explora libremente, "
     "descarga resultados y consulta el glosario integrado.",
     "Cualquier algoritmo", "Cualquier dataset", "Exploración libre", "#22C55E", "pages/5_🎮_Playground"),
]

col_a, col_b = st.columns(2, gap="medium")
for i, (icono, nombre, completo, desc, t1, t2, t3, color, _) in enumerate(paginas):
    tag_html = (
        f'<span class="tag" style="background:{color}22;color:{color}">{t1}</span>'
        f'<span class="tag" style="background:#ffffff11;color:#9CA3AF">{t2}</span>'
        f'<span class="tag" style="background:#ffffff11;color:#9CA3AF">{t3}</span>'
    )
    card_html = (
        f'<div class="nav-card" style="border-top: 3px solid {color};">'
        f'<div class="icon">{icono}</div>'
        f'<h3 style="color:{color}">{nombre} — '
        f'<span style="font-weight:400;color:#9CA3AF;font-size:.9rem">{completo}</span></h3>'
        f'<p>{desc}</p>'
        f'<div class="tags">{tag_html}</div>'
        f'</div>'
    )
    destino = col_a if i % 2 == 0 else col_b
    with destino:
        st.markdown(card_html, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# RUTA DE APRENDIZAJE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Cómo empezar</p>', unsafe_allow_html=True)
st.markdown("## Ruta de aprendizaje sugerida")

col_pasos, col_datos = st.columns([1, 1], gap="large")

with col_pasos:
    pasos = [
        ("Empieza aquí",
         "Lee esta página de inicio para entender el panorama general."),
        ("Aprende PCA",
         "Ve a 🧩 PCA. Lee cómo funciona y prueba la demo con el dataset Iris."),
        ("Aprende t-SNE",
         "Ve a 🌌 t-SNE. Ajusta el deslizador de perplejidad y observa cómo cambian los clústeres."),
        ("Aprende UMAP",
         "Ve a 🚀 UMAP. Compáralo mentalmente con t-SNE — ¿qué es diferente?"),
        ("Compara los tres",
         "Ve a ⚔️ Comparar y ejecuta los tres en el mismo dataset al mismo tiempo."),
        ("Experimenta libremente",
         "Ve a 🎮 Laboratorio. Prueba cada combinación y consulta el glosario."),
    ]
    for num, (titulo, texto) in enumerate(pasos, 1):
        st.markdown(
            f'<div class="step-row">'
            f'<div class="step-num">{num}</div>'
            f'<div class="step-text"><strong style="color:#E5E7EB">{titulo}:</strong> {texto}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

with col_datos:
    st.markdown("### 📊 Datasets disponibles en toda la app")
    datasets = [
        ("🌸", "Iris", "150 flores · 4 dimensiones · 3 especies",
         "El 'Hola Mundo' del ML. Ideal para principiantes — las clases son casi linealmente separables.",
         "#6C63FF"),
        ("🍷", "Vino", "178 vinos · 13 dimensiones · 3 productores",
         "Química de vinos italianos. Muchas variables correlacionadas — aquí brilla PCA.",
         "#48CAE4"),
        ("✏️", "Dígitos", "1.797 imágenes · 64 dimensiones · 10 clases",
         "Dígitos escritos a mano (0–9). t-SNE y UMAP separan los 10 grupos perfectamente desde 64D.",
         "#FF6B6B"),
    ]
    for emoji, nombre, meta, por_que, color in datasets:
        st.markdown(
            f'<div class="concept-box" style="border-left-color:{color};margin-bottom:.7rem">'
            f'<strong style="color:{color}">{emoji} {nombre}</strong> '
            f'<span style="color:#6B7280;font-size:.82rem">{meta}</span>'
            f'<p style="margin:.35rem 0 0 0;color:#9CA3AF;font-size:.88rem">{por_que}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TABLA DE REFERENCIA RÁPIDA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Referencia rápida</p>', unsafe_allow_html=True)
st.markdown("## Tabla comparativa de algoritmos")

df_ref = pd.DataFrame({
    "Característica": [
        "Tipo", "Velocidad", "Preserva estructura global",
        "Preserva estructura local", "Transforma datos nuevos",
        "Mejor para", "Parámetro principal",
    ],
    "🧩 PCA": [
        "Lineal", "⚡⚡⚡ Muy rápido", "✅ Sí", "⚠️ Parcialmente",
        "✅ Sí", "Preprocesamiento, viz rápida", "n_components",
    ],
    "🌌 t-SNE": [
        "No lineal", "🐢 Lento", "⚠️ A menudo perdida", "✅✅ Excelente",
        "❌ No", "Visualización de clústeres", "perplexity",
    ],
    "🚀 UMAP": [
        "No lineal", "⚡⚡ Rápido", "✅ Sí", "✅✅ Muy buena",
        "✅ Sí", "Datasets grandes, pipelines", "n_neighbors, min_dist",
    ],
})
st.dataframe(df_ref, use_container_width=True, hide_index=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ DE REFLEXIÓN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Reflexión inicial</p>', unsafe_allow_html=True)
st.markdown("## ¿Ya tienes una idea base?")

col_q1, col_q2 = st.columns(2, gap="large")

with col_q1:
    respuesta = st.radio(
        "Si tienes un dataset con 500 columnas y quieres visualizarlo rápidamente "
        "sin perder demasiada información, ¿qué técnica usarías primero?",
        [
            "t-SNE — porque revela clústeres muy bien",
            "PCA — porque es rápido y reproducible ✅",
            "UMAP — porque es el más moderno",
            "Ninguna, usaría todas las 500 columnas directamente",
        ],
        index=None,
        key="quiz_home",
    )
    if respuesta is not None:
        if "PCA" in respuesta:
            st.success(
                "🎉 ¡Correcto! PCA es la opción ideal para una primera exploración rápida. "
                "Es determinista, muy veloz y explica cuánta varianza conserva cada componente."
            )
        elif "Ninguna" in respuesta:
            st.error(
                "❌ Con 500 columnas es imposible visualizar nada directamente. "
                "La reducción de dimensionalidad es exactamente la solución a este problema."
            )
        else:
            st.warning(
                "⚠️ No está mal pensar en t-SNE o UMAP, pero para un primer vistazo rápido "
                "PCA es mejor: es más rápido, reproducible y fácil de interpretar. "
                "Guarda t-SNE/UMAP para cuando necesites explorar la estructura de clústeres."
            )

with col_q2:
    st.markdown(
        '<div class="concept-box" style="border-left-color:#6C63FF;">'
        '<strong style="color:#6C63FF">💡 Consejo para empezar</strong>'
        '<p style="margin:.5rem 0 0 0;color:#D1D5DB;font-size:.93rem">'
        'No necesitas saber matemáticas avanzadas para usar esta app. '
        'Cada página explica los conceptos con analogías cotidianas, '
        'visualizaciones interactivas y preguntas de reflexión.<br><br>'
        'Empieza siempre por <strong style="color:#6C63FF">🧩 PCA</strong> — '
        'es el algoritmo más sencillo y el mejor punto de partida para '
        'entender la idea general.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="concept-box" style="border-left-color:#48CAE4;">'
        '<strong style="color:#48CAE4">🧭 ¿Qué encontrarás en cada página?</strong>'
        '<p style="margin:.5rem 0 0 0;color:#D1D5DB;font-size:.93rem">'
        '📖 <strong>Teoría visual</strong> — explicaciones con analogías y diagramas<br>'
        '🎛️ <strong>Demo interactiva</strong> — ajusta parámetros y ve el resultado<br>'
        '📊 <strong>Métricas</strong> — entiende qué tan buena es la proyección<br>'
        '❓ <strong>Quiz</strong> — comprueba lo que aprendiste'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.markdown(
    "<p style='text-align:center;color:#4B5563;font-size:.88rem'>"
    "Construido con ❤️ usando Python · Streamlit · scikit-learn · UMAP-learn · Plotly"
    "</p>",
    unsafe_allow_html=True,
)
