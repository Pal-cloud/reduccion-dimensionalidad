"""Modo presentación — diapositivas step-by-step para explicar reducción de dimensionalidad."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.preprocessing import StandardScaler
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="Modo Presentación",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Slide container ─────────────────────────────────── */
.slide-wrap {
    background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 100%);
    border: 1px solid #2d2d4e;
    border-radius: 20px;
    padding: 2.8rem 3.2rem 2.4rem 3.2rem;
    min-height: 420px;
    position: relative;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
}
.slide-number {
    position: absolute; top: 1.2rem; right: 1.6rem;
    font-size: .78rem; color: #4B5563; font-weight: 600;
    letter-spacing: .06em;
}
.slide-chapter {
    font-size: .72rem; font-weight: 700; letter-spacing: .13em;
    text-transform: uppercase; margin-bottom: .6rem;
}
.slide-title {
    font-size: 2.4rem; font-weight: 800; line-height: 1.15;
    margin-bottom: .9rem;
}
.slide-subtitle {
    font-size: 1.15rem; color: #9CA3AF; line-height: 1.6;
    max-width: 780px; margin-bottom: 1.4rem;
}
.slide-body {
    font-size: 1rem; color: #CBD5E1; line-height: 1.75;
}

/* ── Bullet cards ────────────────────────────────────── */
.bullet-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: .9rem; margin-top: 1.2rem;
}
.bullet-card {
    background: #1e1e35; border: 1px solid #2d2d50;
    border-radius: 14px; padding: 1.1rem 1.3rem;
}
.bullet-card .bc-icon { font-size: 1.8rem; margin-bottom: .5rem; }
.bullet-card .bc-title { font-size: .97rem; font-weight: 700; color: #F9FAFB;
    margin-bottom: .3rem; }
.bullet-card .bc-body { font-size: .88rem; color: #9CA3AF; line-height: 1.55; }

/* ── Comparison row ──────────────────────────────────── */
.cmp-row {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: .9rem; margin-top: 1rem;
}
.cmp-card {
    border-radius: 14px; padding: 1.1rem 1.3rem;
    border-top: 4px solid;
}
.cmp-card .cmp-name { font-size: 1.05rem; font-weight: 800; margin-bottom: .5rem; }
.cmp-card .cmp-row-inner { font-size: .86rem; color: #9CA3AF;
    line-height: 1.6; margin-bottom: .3rem; }
.cmp-card .cmp-row-inner strong { color: #E5E7EB; }

/* ── Highlight quote ─────────────────────────────────── */
.slide-quote {
    border-left: 5px solid; border-radius: 0 10px 10px 0;
    padding: 1rem 1.4rem; margin: 1rem 0;
    font-size: 1.05rem; font-style: italic; color: #E5E7EB;
    background: #111827;
}

/* ── Step list ───────────────────────────────────────── */
.step-list { margin-top: .8rem; }
.step-item {
    display: flex; gap: 1rem; align-items: flex-start;
    margin-bottom: .9rem;
}
.step-dot {
    width: 2rem; height: 2rem; min-width: 2rem;
    border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-weight: 800; font-size: .9rem;
    color: #0f0f1a;
}
.step-dot-text { font-size: .97rem; color: #CBD5E1; line-height: 1.55;
    padding-top: .15rem; }
.step-dot-text strong { color: #F9FAFB; }

/* ── Callout boxes ───────────────────────────────────── */
.callout-green { background:#052e16; border:1px solid #16a34a; border-radius:8px;
    padding:.75rem 1rem; color:#86efac; font-size:.93rem; margin-top:.6rem; }
.callout-yellow { background:#1c1700; border:1px solid #ca8a04; border-radius:8px;
    padding:.75rem 1rem; color:#fde047; font-size:.93rem; margin-top:.6rem; }
.callout-red { background:#1c0505; border:1px solid #dc2626; border-radius:8px;
    padding:.75rem 1rem; color:#fca5a5; font-size:.93rem; margin-top:.6rem; }
.callout-blue { background:#0c1a2e; border:1px solid #2563eb; border-radius:8px;
    padding:.75rem 1rem; color:#93c5fd; font-size:.93rem; margin-top:.6rem; }

/* ── Nav bar ─────────────────────────────────────────── */
.nav-hint { text-align:center; color:#4B5563; font-size:.82rem; margin-top:.5rem; }

/* ── Progress bar ────────────────────────────────────── */
.prog-bar-wrap { background:#1f2937; border-radius:999px; height:5px;
    margin-bottom:.8rem; overflow:hidden; }
.prog-bar-fill { height:5px; border-radius:999px;
    background: linear-gradient(90deg, #6C63FF, #48CAE4, #FF6B6B); }

/* ── Table of contents ───────────────────────────────── */
.toc-item {
    display:flex; align-items:center; gap:.9rem; padding:.7rem .9rem;
    border-radius:10px; margin-bottom:.4rem;
    background:#111827; border:1px solid #1f2937;
    font-size:.93rem; color:#9CA3AF;
}
.toc-item.active { background:#1e1e35; border-color:#6C63FF; color:#F9FAFB; }
.toc-num { font-weight:800; font-size:.85rem; min-width:1.6rem; color:#6C63FF; }
</style>
""", unsafe_allow_html=True)

# ── Datos precalculados ──────────────────────────────────────────────────────
@st.cache_data
def _precompute():
    iris   = load_iris()
    digits = load_digits()
    wine   = load_wine()

    scaler = StandardScaler()

    # Iris PCA
    Xi = scaler.fit_transform(iris.data)
    pca_iris = PCA(n_components=2, random_state=42)
    Xip = pca_iris.fit_transform(Xi)

    # Digits t-SNE (submuestra para velocidad)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(digits.data), 600, replace=False)
    Xd_sub = scaler.fit_transform(digits.data[idx])
    yd_sub = digits.target[idx]
    tsne_dig = TSNE(n_components=2, perplexity=30, max_iter=500,
                    random_state=42, init="pca")
    Xdt = tsne_dig.fit_transform(Xd_sub)

    # Wine PCA 3 components
    Xw = scaler.fit_transform(wine.data)
    pca_wine = PCA(n_components=3, random_state=42)
    Xwp = pca_wine.fit_transform(Xw)
    evr_wine = pca_wine.explained_variance_ratio_

    # Iris full PCA variance
    pca_all = PCA(random_state=42).fit(Xi)
    evr_iris = pca_all.explained_variance_ratio_

    return (iris, digits, wine,
            Xip, pca_iris,
            Xdt, yd_sub,
            Xwp, evr_wine,
            evr_iris)

(iris, digits, wine,
 Xip, pca_iris,
 Xdt, yd_sub,
 Xwp, evr_wine,
 evr_iris) = _precompute()

# ── Definición de diapositivas ───────────────────────────────────────────────
SLIDES_META = [
    # (capítulo, título, emoji color)
    ("Apertura",            "Bienvenida y agenda",                    "#6C63FF"),
    ("El problema",         "El problema de las muchas dimensiones",   "#6C63FF"),
    ("El problema",         "Por qué no podemos visualizar datos 4D+", "#6C63FF"),
    ("La solución",         "¿Qué es la reducción de dimensionalidad?","#48CAE4"),
    ("La solución",         "Tres analogías cotidianas",               "#48CAE4"),
    ("Algoritmo 1 — PCA",   "PCA: la fotografía perfecta",            "#6C63FF"),
    ("Algoritmo 1 — PCA",   "PCA en acción: dataset Iris",            "#6C63FF"),
    ("Algoritmo 1 — PCA",   "¿Cuánta información conserva PCA?",      "#6C63FF"),
    ("Algoritmo 2 — t-SNE", "t-SNE: el organizador de fiestas",       "#48CAE4"),
    ("Algoritmo 2 — t-SNE", "t-SNE en acción: dígitos escritos",      "#48CAE4"),
    ("Algoritmo 2 — t-SNE", "Cuidado: lo que t-SNE NO puede hacer",   "#48CAE4"),
    ("Algoritmo 3 — UMAP",  "UMAP: lo mejor de los dos mundos",       "#FF6B6B"),
    ("Comparativa",         "PCA vs t-SNE vs UMAP: tabla de decisión","#F59E0B"),
    ("Cierre",              "¿Cuándo usar cada uno? Guía rápida",      "#22C55E"),
    ("Cierre",              "Recursos, preguntas y próximos pasos",    "#22C55E"),
]
TOTAL = len(SLIDES_META)

# ── Estado de la diapositiva actual ─────────────────────────────────────────
if "slide_idx" not in st.session_state:
    st.session_state.slide_idx = 0

def _prev():
    st.session_state.slide_idx = max(0, st.session_state.slide_idx - 1)

def _next():
    st.session_state.slide_idx = min(TOTAL - 1, st.session_state.slide_idx + 1)

def _slider_change():
    st.session_state.slide_idx = st.session_state._slider_val - 1

def _cap_go():
    cap_sel = st.session_state.cap_jump
    first = next(i for i, m in enumerate(SLIDES_META) if m[0] == cap_sel)
    st.session_state.slide_idx = first

# ── Header de control ────────────────────────────────────────────────────────
hdr_left, hdr_mid, hdr_right = st.columns([1, 2, 1])

with hdr_left:
    st.markdown("### 🎤 Modo Presentación")
    st.caption("Reducción de Dimensionalidad")

with hdr_mid:
    st.slider(
        "Diapositiva", 1, TOTAL,
        value=st.session_state.slide_idx + 1,
        key="_slider_val",
        label_visibility="collapsed",
        on_change=_slider_change,
    )

with hdr_right:
    capitulos = list(dict.fromkeys(m[0] for m in SLIDES_META))
    st.selectbox("Ir a capítulo", capitulos, key="cap_jump",
                 label_visibility="collapsed")
    st.button("↩ Ir", key="cap_go", on_click=_cap_go)

# ── Barra de progreso ────────────────────────────────────────────────────────
pct = (st.session_state.slide_idx + 1) / TOTAL * 100
st.markdown(
    f'<div class="prog-bar-wrap"><div class="prog-bar-fill" style="width:{pct:.1f}%"></div></div>',
    unsafe_allow_html=True,
)

# ── Botones de navegación (arriba) ───────────────────────────────────────────
nav_prev, nav_info, nav_next = st.columns([1, 3, 1])
with nav_prev:
    st.button("◀ Anterior",
              disabled=(st.session_state.slide_idx == 0),
              use_container_width=True,
              on_click=_prev,
              key="btn_prev")
with nav_info:
    cap, tit, col = SLIDES_META[st.session_state.slide_idx]
    st.markdown(
        f"<p style='text-align:center;color:{col};font-size:.82rem;"
        f"font-weight:700;letter-spacing:.08em;text-transform:uppercase;"
        f"margin:.3rem 0'>{cap} &nbsp;·&nbsp; "
        f"<span style='color:#9CA3AF;font-weight:400'>{tit}</span></p>",
        unsafe_allow_html=True,
    )
with nav_next:
    st.button("Siguiente ▶",
              disabled=(st.session_state.slide_idx == TOTAL - 1),
              use_container_width=True,
              type="primary",
              on_click=_next,
              key="btn_next")

st.markdown("")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        CONTENIDO DE DIAPOSITIVAS                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

s = st.session_state.slide_idx

# ─────────────────────────────────────────────────────────────────────────────
# DIAPOSITIVA 0 — Bienvenida y agenda
# ─────────────────────────────────────────────────────────────────────────────
if s == 0:
    main_col, toc_col = st.columns([1.4, 1], gap="large")

    with main_col:
        st.markdown(
            f'<div class="slide-wrap">'
            f'<div class="slide-number">1 / {TOTAL}</div>'
            f'<div class="slide-chapter" style="color:#6C63FF">🎤 Apertura</div>'
            f'<div class="slide-title" style="background:linear-gradient(135deg,#6C63FF,#48CAE4,#FF6B6B);'
            f'-webkit-background-clip:text;-webkit-text-fill-color:transparent">'
            f'Reducción de Dimensionalidad</div>'
            f'<div class="slide-subtitle">Una guía visual de PCA, t-SNE y UMAP —<br>'
            f'sin fórmulas, con ejemplos reales y demos interactivas.</div>'
            f'<div class="slide-body">'
            f'En los próximos <strong>35 minutos</strong> aprenderás a responder:<br><br>'
            f'❓ ¿Por qué no puedo simplemente graficar mis datos directamente?<br>'
            f'🧩 ¿Cómo funciona PCA? ¿Y t-SNE? ¿Y UMAP?<br>'
            f'🎯 ¿Cuál debo usar para mi caso concreto?<br>'
            f'🚀 ¿Cómo puedo probarlo yo mismo ahora?'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with toc_col:
        st.markdown("#### 🗂️ Agenda de la presentación")
        agenda = [
            ("El problema",         "¿Por qué existen estas técnicas?"),
            ("La solución",         "Idea intuitiva y analogías"),
            ("PCA",                 "El clásico lineal"),
            ("t-SNE",               "Para clústeres no lineales"),
            ("UMAP",                "Más rápido y escalable"),
            ("Comparativa",         "¿Cuándo usar cada uno?"),
            ("Cierre",              "Recursos y próximos pasos"),
        ]
        for i, (cap, desc) in enumerate(agenda):
            active = ""
            st.markdown(
                f'<div class="toc-item{active}">'
                f'<span class="toc-num">{i+1}</span>'
                f'<span><strong style="color:#E5E7EB">{cap}</strong>'
                f'<br><span style="font-size:.82rem">{desc}</span></span>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# DIAPOSITIVA 1 — El problema de las muchas dimensiones
# ─────────────────────────────────────────────────────────────────────────────
elif s == 1:
    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.markdown(
            f'<div class="slide-wrap">'
            f'<div class="slide-number">2 / {TOTAL}</div>'
            f'<div class="slide-chapter" style="color:#6C63FF">🔍 El problema</div>'
            f'<div class="slide-title" style="color:#F9FAFB">El problema de las<br>muchas dimensiones</div>'
            f'<div class="slide-subtitle">Los datos modernos tienen cientos o miles de columnas. '
            f'Ningún humano puede leerlos directamente.</div>'
            f'<div class="slide-body">'
            f'<div class="bullet-grid">'
            f'<div class="bullet-card">'
            f'<div class="bc-icon">🖼️</div>'
            f'<div class="bc-title">Imagen 28×28 px</div>'
            f'<div class="bc-body"><strong style="color:#6C63FF">784 dimensiones</strong> — '
            f'una por píxel. Imposible de graficar directamente.</div>'
            f'</div>'
            f'<div class="bullet-card">'
            f'<div class="bc-icon">🧬</div>'
            f'<div class="bc-title">Genómica</div>'
            f'<div class="bc-body"><strong style="color:#48CAE4">20.000+ dimensiones</strong> — '
            f'un gen por columna. Los análisis directos son inviables.</div>'
            f'</div>'
            f'<div class="bullet-card">'
            f'<div class="bc-icon">🛒</div>'
            f'<div class="bc-title">Recomendaciones</div>'
            f'<div class="bc-body"><strong style="color:#FF6B6B">Millones de productos</strong> — '
            f'cada usuario como un vector de preferencias enorme.</div>'
            f'</div>'
            f'<div class="bullet-card">'
            f'<div class="bc-icon">🩺</div>'
            f'<div class="bc-title">Medicina</div>'
            f'<div class="bc-body"><strong style="color:#F59E0B">Análisis clínicos</strong> — '
            f'cientos de marcadores por paciente. Difícil ver patrones.</div>'
            f'</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### 🗃️ Ejemplo: dataset real con muchas columnas")
        # Simular una "hoja de cálculo con muchas columnas"
        np.random.seed(1)
        fake_cols = [f"var_{i}" for i in range(1, 18)]
        fake_data = np.random.randn(8, 17).round(2)
        df_fake = pd.DataFrame(fake_data, columns=fake_cols)
        df_fake.index = [f"muestra_{i+1}" for i in range(8)]
        st.dataframe(df_fake, use_container_width=True, height=260)
        st.caption("Con 17 columnas ya es difícil leer. Imagina 500 o 20.000.")

        st.markdown(
            '<div class="callout-blue">💡 <strong>El objetivo:</strong> convertir esta tabla '
            'en un gráfico 2D que cualquier persona pueda entender en segundos.</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# DIAPOSITIVA 2 — Por qué no podemos visualizar 4D+
# ─────────────────────────────────────────────────────────────────────────────
elif s == 2:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown(
            f'<div class="slide-wrap">'
            f'<div class="slide-number">3 / {TOTAL}</div>'
            f'<div class="slide-chapter" style="color:#6C63FF">🔍 El problema</div>'
            f'<div class="slide-title" style="color:#F9FAFB">¿Por qué no podemos<br>visualizar 4D+?</div>'
            f'<div class="slide-subtitle">Nuestro cerebro sólo procesa 2 o 3 dimensiones simultáneamente.</div>'
            f'<div class="slide-body">'
            f'<div class="step-list">'
            f'<div class="step-item"><div class="step-dot" style="background:#6C63FF">1D</div>'
            f'<div class="step-dot-text"><strong>Una dimensión:</strong> una línea numérica. '
            f'Fácil — sólo hay "más" o "menos".</div></div>'
            f'<div class="step-item"><div class="step-dot" style="background:#48CAE4">2D</div>'
            f'<div class="step-dot-text"><strong>Dos dimensiones:</strong> un gráfico XY. '
            f'Fácil — lo usamos a diario en mapas y hojas de cálculo.</div></div>'
            f'<div class="step-item"><div class="step-dot" style="background:#FF6B6B">3D</div>'
            f'<div class="step-dot-text"><strong>Tres dimensiones:</strong> el mundo físico. '
            f'Todavía manejable con gráficos rotables en pantalla.</div></div>'
            f'<div class="step-item"><div class="step-dot" style="background:#F59E0B">4D+</div>'
            f'<div class="step-dot-text"><strong>Cuatro o más:</strong> '
            f'<em>imposible de visualizar directamente</em>. '
            f'Aquí es donde la reducción de dimensionalidad se hace imprescindible.</div></div>'
            f'</div>'
            f'<div class="callout-yellow" style="margin-top:1rem">⚠️ <strong>La maldición de la dimensionalidad:</strong> '
            f'a más dimensiones, los datos se vuelven cada vez más dispersos y difíciles de analizar. '
            f'Los algoritmos de ML sufren si no reducimos primero.</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### 📊 De 4D a 2D: el dataset Iris")
        st.caption(
            "Iris tiene 4 dimensiones. Aquí mostramos sólo 2 a la vez — "
            "y ya se ve claramente que hay 3 grupos."
        )
        # scatter matrix simplificado: 2 pares de variables
        fig = go.Figure()
        colores_iris = ["#6C63FF", "#48CAE4", "#FF6B6B"]
        for i, (nombre, color) in enumerate(zip(iris.target_names, colores_iris)):
            mask = iris.target == i
            fig.add_trace(go.Scatter(
                x=iris.data[mask, 2], y=iris.data[mask, 3],
                mode="markers", name=nombre.capitalize(),
                marker=dict(color=color, size=8, opacity=0.8,
                            line=dict(width=0.5, color="white")),
            ))
        fig.update_layout(
            template="plotly_dark", height=300,
            xaxis_title="largo del pétalo (cm) — dim 3",
            yaxis_title="ancho del pétalo (cm) — dim 4",
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=30, r=10, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Con sólo 2 de las 4 variables ya separamos bien los grupos. "
            "PCA elige automáticamente las 2 mejores 'proyecciones'."
        )
        st.markdown(
            '<div class="callout-green">✅ <strong>La reducción de dimensionalidad</strong> '
            'hace este proceso de forma óptima y automática para <em>cualquier</em> número '
            'de dimensiones, no sólo 4.</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# DIAPOSITIVA 3 — Qué es la reducción de dimensionalidad
# ─────────────────────────────────────────────────────────────────────────────
elif s == 3:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown(
            f'<div class="slide-wrap">'
            f'<div class="slide-number">4 / {TOTAL}</div>'
            f'<div class="slide-chapter" style="color:#48CAE4">💡 La solución</div>'
            f'<div class="slide-title" style="color:#48CAE4">¿Qué es la reducción<br>de dimensionalidad?</div>'
            f'<div class="slide-subtitle">Comprimir datos de alta dimensión en 2D o 3D '
            f'conservando la estructura más importante.</div>'
            f'<div class="slide-body">'
            f'<div class="slide-quote" style="border-left-color:#48CAE4;color:#BAE6FD">'
            f'"Tomar una tabla con 500 columnas y convertirla en un dibujo '
            f'de puntos donde lo similar queda junto y lo diferente queda separado."'
            f'</div>'
            f'<div class="step-list">'
            f'<div class="step-item"><div class="step-dot" style="background:#6C63FF">→</div>'
            f'<div class="step-dot-text"><strong>Entrada:</strong> dataset con N dimensiones '
            f'(columnas). Puede ser 4, 64, 500 o 20.000.</div></div>'
            f'<div class="step-item"><div class="step-dot" style="background:#48CAE4">⚙️</div>'
            f'<div class="step-dot-text"><strong>Proceso:</strong> el algoritmo aprende '
            f'qué combinaciones de columnas capturan más información.</div></div>'
            f'<div class="step-item"><div class="step-dot" style="background:#FF6B6B">←</div>'
            f'<div class="step-dot-text"><strong>Salida:</strong> dataset con 2 ó 3 dimensiones. '
            f'Listo para graficar y explorar visualmente.</div></div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### 🎯 El resultado en una imagen")
        # PCA iris ya calculado
        fig = go.Figure()
        colores_iris = ["#6C63FF", "#48CAE4", "#FF6B6B"]
        for i, (nombre, color) in enumerate(zip(iris.target_names, colores_iris)):
            mask = iris.target == i
            fig.add_trace(go.Scatter(
                x=Xip[mask, 0], y=Xip[mask, 1],
                mode="markers", name=nombre.capitalize(),
                marker=dict(color=color, size=9, opacity=0.85,
                            line=dict(width=0.5, color="white")),
            ))
        v = pca_iris.explained_variance_ratio_
        fig.update_layout(
            template="plotly_dark", height=320,
            xaxis_title=f"Dim 1 — {v[0]*100:.0f}% varianza",
            yaxis_title=f"Dim 2 — {v[1]*100:.0f}% varianza",
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=30, r=10, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.success(
            f"✅ 4 dimensiones → 2 dimensiones, conservando "
            f"**{(v[0]+v[1])*100:.0f}%** de la información. "
            "Los tres grupos son claramente visibles."
        )

# ─────────────────────────────────────────────────────────────────────────────
# DIAPOSITIVA 4 — Tres analogías cotidianas
# ─────────────────────────────────────────────────────────────────────────────
elif s == 4:
    st.markdown(
        f'<div class="slide-wrap">'
        f'<div class="slide-number">5 / {TOTAL}</div>'
        f'<div class="slide-chapter" style="color:#48CAE4">💡 La solución</div>'
        f'<div class="slide-title" style="color:#F9FAFB">Tres analogías<br>para entenderlo</div>'
        f'<div class="slide-subtitle">No necesitas matemáticas — sólo intuición cotidiana.</div>'
        f'<div class="bullet-grid">'
        f'<div class="bullet-card" style="border-left:4px solid #6C63FF">'
        f'<div class="bc-icon">🎒</div>'
        f'<div class="bc-title" style="color:#6C63FF">La maleta de viaje</div>'
        f'<div class="bc-body">No te llevas toda la casa — empacas sólo lo esencial. '
        f'La reducción de dimensionalidad descarta el "ruido" de los datos '
        f'y se queda con lo que realmente importa.<br><br>'
        f'<em>Resultado: una maleta pequeña que contiene casi todo lo que necesitas.</em></div>'
        f'</div>'
        f'<div class="bullet-card" style="border-left:4px solid #48CAE4">'
        f'<div class="bc-icon">📷</div>'
        f'<div class="bc-title" style="color:#48CAE4">La fotografía de una escultura</div>'
        f'<div class="bc-body">Una escultura es 3D, la foto es 2D. Un buen fotógrafo '
        f'elige el ángulo que muestra más detalle. PCA hace lo mismo: '
        f'encuentra el mejor ángulo para "fotografiar" tus datos.<br><br>'
        f'<em>CP1 y CP2 son los dos ejes de esa foto óptima.</em></div>'
        f'</div>'
        f'<div class="bullet-card" style="border-left:4px solid #FF6B6B">'
        f'<div class="bc-icon">🗺️</div>'
        f'<div class="bc-title" style="color:#FF6B6B">El mapa del metro</div>'
        f'<div class="bc-body">El metro existe en 3D (curvas, túneles, desniveles), '
        f'pero el mapa es 2D y plano. No es exacto, pero es suficiente para orientarse. '
        f't-SNE y UMAP construyen ese tipo de mapa para tus datos.<br><br>'
        f'<em>Lo que importa: las distancias locales, no la geometría global exacta.</em></div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# DIAPOSITIVA 5 — PCA: la fotografía perfecta
# ─────────────────────────────────────────────────────────────────────────────
elif s == 5:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown(
            f'<div class="slide-wrap">'
            f'<div class="slide-number">6 / {TOTAL}</div>'
            f'<div class="slide-chapter" style="color:#6C63FF">🧩 Algoritmo 1 — PCA</div>'
            f'<div class="slide-title" style="color:#6C63FF">PCA: la fotografía perfecta</div>'
            f'<div class="slide-subtitle">Análisis de Componentes Principales — '
            f'el algoritmo de reducción de dimensionalidad más antiguo y más usado.</div>'
            f'<div class="slide-body">'
            f'<div class="step-list">'
            f'<div class="step-item"><div class="step-dot" style="background:#6C63FF">1</div>'
            f'<div class="step-dot-text"><strong>Centrar los datos:</strong> '
            f'restar la media para que la nube quede centrada en el origen.</div></div>'
            f'<div class="step-item"><div class="step-dot" style="background:#6C63FF">2</div>'
            f'<div class="step-dot-text"><strong>Medir correlaciones:</strong> '
            f'calcular la matriz de covarianza — qué variables van juntas.</div></div>'
            f'<div class="step-item"><div class="step-dot" style="background:#6C63FF">3</div>'
            f'<div class="step-dot-text"><strong>Hallar los ejes principales (eigenvectores):</strong> '
            f'CP1 apunta a la máxima varianza. CP2 es perpendicular a CP1.</div></div>'
            f'<div class="step-item"><div class="step-dot" style="background:#6C63FF">4</div>'
            f'<div class="step-dot-text"><strong>Proyectar:</strong> '
            f'multiplicar los datos por los primeros N componentes → nuevo dataset comprimido.</div></div>'
            f'</div>'
            f'<div class="callout-green" style="margin-top:.8rem">✅ <strong>Ventajas de PCA:</strong> '
            f'muy rápido · determinista · explica cuánta varianza conserva · '
            f'puede transformar datos nuevos.</div>'
            f'<div class="callout-yellow">⚠️ <strong>Limitación:</strong> '
            f'sólo captura relaciones <em>lineales</em>. Si los datos tienen '
            f'estructura curva, usar t-SNE o UMAP.</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### 📐 Las componentes principales en vivo")
        np.random.seed(7)
        cov = [[3, 2.5], [2.5, 3]]
        raw = np.random.multivariate_normal([0, 0], cov, 300)
        pca_demo = PCA(n_components=2).fit(raw)
        v1 = pca_demo.components_[0] * 3.5
        v2 = pca_demo.components_[1] * 1.6

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=raw[:, 0], y=raw[:, 1], mode="markers",
            marker=dict(color="#6C63FF", opacity=0.35, size=7),
            name="Datos originales"))
        fig.add_annotation(ax=0, ay=0, x=v1[0], y=v1[1],
                            axref="x", ayref="y", xref="x", yref="y",
                            arrowhead=3, arrowwidth=3, arrowcolor="#FF6B6B",
                            font=dict(color="#FF6B6B", size=12),
                            text="  CP1 — máx. varianza")
        fig.add_annotation(ax=0, ay=0, x=v2[0], y=v2[1],
                            axref="x", ayref="y", xref="x", yref="y",
                            arrowhead=3, arrowwidth=2, arrowcolor="#48CAE4",
                            font=dict(color="#48CAE4", size=12), text="  CP2")
        fig.update_layout(
            template="plotly_dark", height=300, showlegend=False,
            xaxis_title="Variable X", yaxis_title="Variable Y",
            margin=dict(l=30, r=10, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"CP1 captura el **{pca_demo.explained_variance_ratio_[0]*100:.0f}%** "
            f"de la varianza. CP2 captura el **{pca_demo.explained_variance_ratio_[1]*100:.0f}%** adicional. "
            "Las flechas apuntan a los ejes de mayor dispersión."
        )
        st.markdown("#### 📐 Fórmulas esenciales")
        st.markdown(r"""
| | Fórmula | Significado |
|--|---------|-------------|
| Covarianza | $\mathbf{C}=\frac{1}{n-1}\mathbf{X}^\top\mathbf{X}$ | Mide correlación entre variables |
| Eigenvectores | $\mathbf{C}\mathbf{v}=\lambda\mathbf{v}$ | $\mathbf{v}$ = dirección; $\lambda$ = varianza capturada |
| Proyección | $\mathbf{Z}=\mathbf{X}\mathbf{W}_k$ | Comprime a $k$ dimensiones |
| Var. explicada | $\lambda_i/\sum\lambda_j$ | % de información conservada por $CP_i$ |
""")

# ─────────────────────────────────────────────────────────────────────────────
# DIAPOSITIVA 6 — PCA en acción: dataset Iris
# ─────────────────────────────────────────────────────────────────────────────
elif s == 6:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown(
            f'<div class="slide-wrap">'
            f'<div class="slide-number">7 / {TOTAL}</div>'
            f'<div class="slide-chapter" style="color:#6C63FF">🧩 Algoritmo 1 — PCA</div>'
            f'<div class="slide-title" style="color:#6C63FF">PCA en acción:<br>dataset Iris</div>'
            f'<div class="slide-subtitle">150 flores · 4 dimensiones → 2 dimensiones.</div>'
            f'<div class="slide-body">'
            f'<strong>¿Qué es el dataset Iris?</strong><br>'
            f'Creado en 1936 por Ronald Fisher. Medidas físicas de 150 flores de '
            f'tres especies: <em>Setosa</em>, <em>Versicolor</em> y <em>Virginica</em>.'
            f'<br><br>'
            f'<strong>Las 4 variables originales:</strong><br>'
            f'• Largo del sépalo (cm)<br>'
            f'• Ancho del sépalo (cm)<br>'
            f'• Largo del pétalo (cm)<br>'
            f'• Ancho del pétalo (cm)<br><br>'
            f'<strong>Lo que hace PCA:</strong> combina estas 4 variables en 2 nuevas '
            f'"súper-variables" que capturan la mayor parte de la información.'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### 🌸 Resultado: 4D → 2D con PCA")
        fig = go.Figure()
        colores_iris = ["#6C63FF", "#48CAE4", "#FF6B6B"]
        for i, (nombre, color) in enumerate(zip(iris.target_names, colores_iris)):
            mask = iris.target == i
            fig.add_trace(go.Scatter(
                x=Xip[mask, 0], y=Xip[mask, 1],
                mode="markers", name=nombre.capitalize(),
                marker=dict(color=color, size=10, opacity=0.85,
                            line=dict(width=0.5, color="white")),
            ))
        v = pca_iris.explained_variance_ratio_
        fig.update_layout(
            template="plotly_dark", height=350,
            xaxis_title=f"CP1 — {v[0]*100:.1f}% de varianza",
            yaxis_title=f"CP2 — {v[1]*100:.1f}% de varianza",
            legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(size=13)),
            margin=dict(l=30, r=10, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            f"✅ Con sólo **2 números** por flor conservamos el "
            f"**{(v[0]+v[1])*100:.0f}%** de la información original. "
            "La especie Setosa queda completamente separada."
        )

# ─────────────────────────────────────────────────────────────────────────────
# DIAPOSITIVA 7 — ¿Cuánta información conserva PCA?
# ─────────────────────────────────────────────────────────────────────────────
elif s == 7:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown(
            f'<div class="slide-wrap">'
            f'<div class="slide-number">8 / {TOTAL}</div>'
            f'<div class="slide-chapter" style="color:#6C63FF">🧩 Algoritmo 1 — PCA</div>'
            f'<div class="slide-title" style="color:#6C63FF">¿Cuánta información<br>conserva PCA?</div>'
            f'<div class="slide-subtitle">El "Scree Plot" — el gráfico que responde esta pregunta.</div>'
            f'<div class="slide-body">'
            f'<strong>Varianza explicada acumulada:</strong><br>'
            f'Cada componente principal añade un poco más de información. '
            f'El scree plot muestra cuántas componentes necesitamos para capturar, '
            f'digamos, el 95% de la información total.<br><br>'
            f'<strong>Regla práctica:</strong><br>'
            f'• <span style="color:#86efac">90%+ de varianza</span> → excelente compresión<br>'
            f'• <span style="color:#fde047">70–90%</span> → aceptable para exploración<br>'
            f'• <span style="color:#fca5a5">Menos del 50%</span> → considera t-SNE o UMAP<br><br>'
            f'<strong>Para Iris:</strong> ¡con sólo 2 componentes capturamos ~97%!'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### 📊 Scree Plot — Iris dataset")
        cumvar = np.cumsum(evr_iris) * 100
        n_comp = len(evr_iris)
        fig = go.Figure()
        fig.add_bar(
            x=list(range(1, n_comp + 1)),
            y=evr_iris * 100,
            name="Varianza por componente",
            marker_color="#6C63FF",
        )
        fig.add_scatter(
            x=list(range(1, n_comp + 1)),
            y=cumvar,
            mode="lines+markers",
            name="Varianza acumulada",
            line=dict(color="#FF6B6B", width=2),
            marker=dict(size=7),
        )
        fig.add_hline(y=95, line_dash="dash", line_color="#F59E0B",
                      annotation_text="95% umbral", annotation_position="right")
        fig.update_layout(
            template="plotly_dark", height=320,
            xaxis_title="Número de componente",
            yaxis_title="Varianza explicada (%)",
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=30, r=60, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Con **2 componentes** ya superamos el 95%. "
            f"El codo del gráfico sugiere que 2 es el número óptimo para Iris."
        )

# ─────────────────────────────────────────────────────────────────────────────
# DIAPOSITIVA 8 — t-SNE: el organizador de fiestas
# ─────────────────────────────────────────────────────────────────────────────
elif s == 8:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown(
            f'<div class="slide-wrap">'
            f'<div class="slide-number">9 / {TOTAL}</div>'
            f'<div class="slide-chapter" style="color:#38BDF8">🌌 Algoritmo 2 — t-SNE</div>'
            f'<div class="slide-title" style="color:#38BDF8">t-SNE: el organizador<br>de fiestas</div>'
            f'<div class="slide-subtitle">t-distributed Stochastic Neighbor Embedding — '
            f'especialista en revelar clústeres ocultos.</div>'
            f'<div class="slide-body">'
            f'<div class="slide-quote" style="border-left-color:#38BDF8;color:#BAE6FD">'
            f'"Imagina una fiesta con 500 invitados. Tu misión: sentar juntas a las '
            f'personas que se parecen. Los grupos naturales emergen solos."'
            f'</div>'
            f'<div class="step-list">'
            f'<div class="step-item"><div class="step-dot" style="background:#38BDF8;color:#0f0f1a">1</div>'
            f'<div class="step-dot-text"><strong>Medir similitudes en alta dimensión:</strong> '
            f'para cada punto, calcula la probabilidad de que cada vecino sea realmente cercano (curva gaussiana).</div></div>'
            f'<div class="step-item"><div class="step-dot" style="background:#38BDF8;color:#0f0f1a">2</div>'
            f'<div class="step-dot-text"><strong>Colocar en 2D aleatoriamente:</strong> '
            f'posición inicial aleatoria — sólo el punto de partida.</div></div>'
            f'<div class="step-item"><div class="step-dot" style="background:#38BDF8;color:#0f0f1a">3</div>'
            f'<div class="step-dot-text"><strong>Optimizar iterativamente:</strong> '
            f'mover puntos para que los vecinos cercanos en alta dimensión queden cerca en 2D. '
            f'Se repite cientos o miles de veces.</div></div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### 🔑 El parámetro clave: Perplejidad")
        st.markdown(
            "La **perplejidad** controla cuántos vecinos 've' cada punto. "
            "Cambia el resultado dramáticamente."
        )
        perp_data = {
            "Perplejidad": ["5 (baja)", "30 (media ✅)", "80 (alta)"],
            "Efecto": [
                "Clusters muy pequeños y separados",
                "Balance ideal: clusters bien formados",
                "Clusters grandes y difusos",
            ],
            "Cuándo usar": [
                "Datos muy locales",
                "Uso general (recomendado)",
                "Datasets muy grandes",
            ]
        }
        st.dataframe(pd.DataFrame(perp_data), hide_index=True, use_container_width=True)

        st.markdown(
            '<div class="callout-blue">💡 <strong>Regla de oro:</strong> '
            'la perplejidad debe ser mucho menor que el número de muestras. '
            'Para 1.000 muestras → perplejidad 30-50 es ideal.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="callout-yellow">⚠️ <strong>Importante:</strong> '
            't-SNE es <em>no determinista</em> — ejecutarlo dos veces con '
            'distintas semillas da resultados distintos. '
            'Siempre fijar <code>random_state</code>.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### 📐 Fórmulas esenciales")
        st.markdown(r"""
| | Fórmula | Significado |
|--|---------|-------------|
| Alta dim. | $p_{j\|i}\propto e^{-\|x_i-x_j\|^2/2\sigma_i^2}$ | Similitud Gaussiana; $\sigma_i$ lo fija la perplejidad |
| Baja dim. | $q_{ij}\propto(1+\|y_i-y_j\|^2)^{-1}$ | Distribución **t de Student** — colas gruesas = clusters separados |
| Coste | $\mathcal{L}=\text{KL}(P\|Q)$ | Minimiza diferencia entre las dos distribuciones |
""")

# ─────────────────────────────────────────────────────────────────────────────
# DIAPOSITIVA 9 — t-SNE en acción: dígitos
# ─────────────────────────────────────────────────────────────────────────────
elif s == 9:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown(
            f'<div class="slide-wrap">'
            f'<div class="slide-number">10 / {TOTAL}</div>'
            f'<div class="slide-chapter" style="color:#38BDF8">🌌 Algoritmo 2 — t-SNE</div>'
            f'<div class="slide-title" style="color:#38BDF8">t-SNE en acción:<br>dígitos escritos a mano</div>'
            f'<div class="slide-subtitle">600 imágenes · 64 dimensiones → 2 dimensiones.</div>'
            f'<div class="slide-body">'
            f'<strong>¿Qué son los dígitos?</strong><br>'
            f'Imágenes de dígitos del 0 al 9 escritos a mano, escaneadas en cuadrículas '
            f'de 8×8 píxeles. Cada píxel es una variable → <strong>64 dimensiones</strong>.'
            f'<br><br>'
            f'<strong>El reto:</strong> es completamente imposible visualizar 64 dimensiones directamente.<br><br>'
            f'<strong>t-SNE lo resuelve:</strong> crea un mapa 2D donde los 10 dígitos '
            f'forman <strong>10 islas perfectamente separadas</strong>. '
            f'Esto es exactamente lo que queremos ver: ¿los dígitos similares quedan juntos?'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### ✏️ Resultado: 64D → 2D con t-SNE (600 imágenes)")
        colores_dig = px.colors.qualitative.Vivid
        fig = go.Figure()
        for d in range(10):
            mask = yd_sub == d
            fig.add_trace(go.Scatter(
                x=Xdt[mask, 0], y=Xdt[mask, 1],
                mode="markers", name=str(d),
                marker=dict(color=colores_dig[d % len(colores_dig)],
                            size=8, opacity=0.85,
                            line=dict(width=0.4, color="white")),
            ))
        fig.update_layout(
            template="plotly_dark", height=380,
            xaxis_title="Dimensión 1 (t-SNE)",
            yaxis_title="Dimensión 2 (t-SNE)",
            legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(size=11),
                        title="Dígito"),
            margin=dict(l=30, r=10, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "✅ Cada color es un dígito (0–9). Las 10 'islas' bien separadas "
            "demuestran que t-SNE ha encontrado estructura oculta en los datos."
        )

# ─────────────────────────────────────────────────────────────────────────────
# DIAPOSITIVA 10 — Lo que t-SNE NO puede hacer
# ─────────────────────────────────────────────────────────────────────────────
elif s == 10:
    st.markdown(
        f'<div class="slide-wrap">'
        f'<div class="slide-number">11 / {TOTAL}</div>'
        f'<div class="slide-chapter" style="color:#38BDF8">🌌 Algoritmo 2 — t-SNE</div>'
        f'<div class="slide-title" style="color:#F9FAFB">Cuidado: lo que t-SNE<br>NO puede hacer</div>'
        f'<div class="slide-subtitle">Conocer las limitaciones es tan importante como conocer las fortalezas.</div>'
        f'<div class="bullet-grid">'
        f'<div class="bullet-card" style="border-left:4px solid #dc2626">'
        f'<div class="bc-icon">🚫</div>'
        f'<div class="bc-title" style="color:#fca5a5">Las distancias entre clusters no son interpretables</div>'
        f'<div class="bc-body">Que dos grupos estén cerca o lejos en el mapa 2D '
        f'<strong>no significa nada</strong> sobre si son parecidos entre sí. '
        f'Sólo la distancia <em>dentro</em> de un cluster tiene sentido.</div>'
        f'</div>'
        f'<div class="bullet-card" style="border-left:4px solid #dc2626">'
        f'<div class="bc-icon">🚫</div>'
        f'<div class="bc-title" style="color:#fca5a5">No transforma datos nuevos</div>'
        f'<div class="bc-body">Si tienes datos nuevos que llegaron después, '
        f'<strong>no puedes proyectarlos</strong> en el mapa existente. '
        f'Hay que reejecutar t-SNE desde cero con todos los datos.</div>'
        f'</div>'
        f'<div class="bullet-card" style="border-left:4px solid #ca8a04">'
        f'<div class="bc-icon">⚠️</div>'
        f'<div class="bc-title" style="color:#fde047">Lento en datasets grandes</div>'
        f'<div class="bc-body">Con más de ~50.000 muestras t-SNE se vuelve muy lento. '
        f'La complejidad es O(N²). Para datasets grandes usar UMAP.</div>'
        f'</div>'
        f'<div class="bullet-card" style="border-left:4px solid #ca8a04">'
        f'<div class="bc-icon">⚠️</div>'
        f'<div class="bc-title" style="color:#fde047">No determinista por defecto</div>'
        f'<div class="bc-body">Dos ejecuciones con distintas semillas dan mapas distintos. '
        f'Siempre usar <code>random_state=42</code> para reproducibilidad. '
        f'La estructura de clusters se mantiene, pero su posición cambia.</div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# DIAPOSITIVA 11 — UMAP: lo mejor de los dos mundos
# ─────────────────────────────────────────────────────────────────────────────
elif s == 11:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown(
            f'<div class="slide-wrap">'
            f'<div class="slide-number">12 / {TOTAL}</div>'
            f'<div class="slide-chapter" style="color:#FF6B6B">🚀 Algoritmo 3 — UMAP</div>'
            f'<div class="slide-title" style="color:#FF6B6B">UMAP: lo mejor<br>de los dos mundos</div>'
            f'<div class="slide-subtitle">Uniform Manifold Approximation and Projection — '
            f'más rápido que t-SNE y preserva también la estructura global.</div>'
            f'<div class="slide-body">'
            f'<div class="slide-quote" style="border-left-color:#FF6B6B;color:#FCA5A5">'
            f'"UMAP construye un grafo de vecindario en alta dimensión y '
            f'luego lo \u201cdibuja\u201d en 2D manteniendo tanto los grupos locales '
            f'como la estructura general del dataset."'
            f'</div>'
            f'<strong>Dos parámetros principales:</strong><br><br>'
            f'<div class="step-list">'
            f'<div class="step-item">'
            f'<div class="step-dot" style="background:#FF6B6B">n</div>'
            f'<div class="step-dot-text"><strong>n_neighbors (vecinos):</strong> '
            f'cuántos vecinos considera el grafo. Bajo → estructura local. '
            f'Alto → estructura global. Valor típico: 15.</div></div>'
            f'<div class="step-item">'
            f'<div class="step-dot" style="background:#FF6B6B">d</div>'
            f'<div class="step-dot-text"><strong>min_dist (distancia mínima):</strong> '
            f'qué tan apretados quedan los puntos. '
            f'0 = muy compacto. 0.9 = muy disperso. Valor típico: 0.1.</div></div>'
            f'</div>'
            f'<div class="callout-green" style="margin-top:.8rem">✅ <strong>Ventaja clave:</strong> '
            f'puede transformar datos nuevos (como PCA) y es mucho más rápido que t-SNE '
            f'en datasets de más de 10.000 muestras.</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### 🆚 UMAP vs t-SNE: diferencias clave")
        df_vs = pd.DataFrame({
            "Característica": [
                "Velocidad", "Estructura global", "Estructura local",
                "Datos nuevos", "Reproducible", "Parámetros"
            ],
            "🌌 t-SNE": [
                "🐢 Lento (O(N²))",
                "⚠️ A menudo perdida",
                "✅✅ Excelente",
                "❌ No",
                "⚠️ Con random_state",
                "perplexity, n_iter"
            ],
            "🚀 UMAP": [
                "⚡ Rápido (O(N))",
                "✅ Conservada",
                "✅ Muy buena",
                "✅ Sí",
                "✅ Con random_state",
                "n_neighbors, min_dist"
            ],
        })
        st.dataframe(df_vs, hide_index=True, use_container_width=True)

        st.markdown(
            '<div class="callout-blue" style="margin-top:.8rem">💡 <strong>Cuándo elegir UMAP sobre t-SNE:</strong> '
            'si el dataset tiene más de 10.000 muestras, '
            'si necesitas proyectar datos nuevos, '
            'o si la estructura global del dataset importa.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### 📐 Fórmulas esenciales")
        st.markdown(r"""
| | Fórmula | Significado |
|--|---------|-------------|
| Peso grafo | $w_{ij}=\exp(-(d_{ij}-\rho_i)/\sigma_i)$ | Fuerza de conexión; $\rho_i$ normaliza escala local |
| Simetría | $\bar{w}_{ij}=w_{ij}+w_{ji}-w_{ij}w_{ji}$ | Convierte grafo dirigido en no dirigido |
| Baja dim. | $q_{ij}=(1+a\|y_i-y_j\|^{2b})^{-1}$ | Similitud en embedding; $a,b$ controlados por `min_dist` |
| Coste | $\mathcal{L}=\sum w_{ij}\log\frac{w_{ij}}{q_{ij}}+(1-w_{ij})\log\frac{1-w_{ij}}{1-q_{ij}}$ | Entropía cruzada binaria — más rápida que KL |
""")

# ─────────────────────────────────────────────────────────────────────────────
# DIAPOSITIVA 12 — Tabla de decisión comparativa
# ─────────────────────────────────────────────────────────────────────────────
elif s == 12:
    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.markdown(
            f'<div class="slide-wrap">'
            f'<div class="slide-number">13 / {TOTAL}</div>'
            f'<div class="slide-chapter" style="color:#F59E0B">⚔️ Comparativa</div>'
            f'<div class="slide-title" style="color:#F9FAFB">PCA vs t-SNE vs UMAP:<br>tabla de decisión</div>'
            f'<div class="slide-subtitle">¿Cuál algoritmo es el correcto para tu caso?</div>'
            f'<div class="slide-body">'
            f'<div class="cmp-row">'
            # PCA
            f'<div class="cmp-card" style="background:#1a1a30;border-top-color:#6C63FF">'
            f'<div class="cmp-name" style="color:#6C63FF">🧩 PCA</div>'
            f'<div class="cmp-row-inner"><strong>Tipo:</strong> Lineal</div>'
            f'<div class="cmp-row-inner"><strong>Velocidad:</strong> ⚡⚡⚡</div>'
            f'<div class="cmp-row-inner"><strong>Global:</strong> ✅</div>'
            f'<div class="cmp-row-inner"><strong>Local:</strong> ⚠️</div>'
            f'<div class="cmp-row-inner"><strong>Datos nuevos:</strong> ✅</div>'
            f'<div class="cmp-row-inner"><strong>Mejor para:</strong> primera exploración, preprocesamiento ML</div>'
            f'</div>'
            # t-SNE
            f'<div class="cmp-card" style="background:#1a1a30;border-top-color:#38BDF8">'
            f'<div class="cmp-name" style="color:#38BDF8">🌌 t-SNE</div>'
            f'<div class="cmp-row-inner"><strong>Tipo:</strong> No lineal</div>'
            f'<div class="cmp-row-inner"><strong>Velocidad:</strong> 🐢</div>'
            f'<div class="cmp-row-inner"><strong>Global:</strong> ⚠️</div>'
            f'<div class="cmp-row-inner"><strong>Local:</strong> ✅✅</div>'
            f'<div class="cmp-row-inner"><strong>Datos nuevos:</strong> ❌</div>'
            f'<div class="cmp-row-inner"><strong>Mejor para:</strong> visualización de clusters &lt;50K muestras</div>'
            f'</div>'
            # UMAP
            f'<div class="cmp-card" style="background:#1a1a30;border-top-color:#FF6B6B">'
            f'<div class="cmp-name" style="color:#FF6B6B">🚀 UMAP</div>'
            f'<div class="cmp-row-inner"><strong>Tipo:</strong> No lineal</div>'
            f'<div class="cmp-row-inner"><strong>Velocidad:</strong> ⚡⚡</div>'
            f'<div class="cmp-row-inner"><strong>Global:</strong> ✅</div>'
            f'<div class="cmp-row-inner"><strong>Local:</strong> ✅✅</div>'
            f'<div class="cmp-row-inner"><strong>Datos nuevos:</strong> ✅</div>'
            f'<div class="cmp-row-inner"><strong>Mejor para:</strong> datasets grandes, pipelines, genómica</div>'
            f'</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### 🌸 Los tres algoritmos en el mismo dataset (Iris)")
        scaler_tmp = StandardScaler()
        Xi_tmp = scaler_tmp.fit_transform(iris.data)

        pca_tmp = PCA(n_components=2, random_state=42)
        Xpca = pca_tmp.fit_transform(Xi_tmp)

        try:
            import umap as umap_lib
            reducer = umap_lib.UMAP(n_components=2, n_neighbors=15,
                                    min_dist=0.1, random_state=42)
            Xumap = reducer.fit_transform(Xi_tmp)
            has_umap = True
        except Exception:
            has_umap = False

        tab_pca, tab_tsne = st.tabs(["🧩 PCA", "🌌 t-SNE (cached)"])
        colores_iris = ["#6C63FF", "#48CAE4", "#FF6B6B"]

        with tab_pca:
            fig = go.Figure()
            for i, (nombre, color) in enumerate(zip(iris.target_names, colores_iris)):
                mask = iris.target == i
                fig.add_trace(go.Scatter(
                    x=Xpca[mask, 0], y=Xpca[mask, 1],
                    mode="markers", name=nombre.capitalize(),
                    marker=dict(color=color, size=8, opacity=0.8)))
            fig.update_layout(template="plotly_dark", height=260,
                              xaxis_title="CP1", yaxis_title="CP2",
                              margin=dict(l=20, r=10, t=10, b=30),
                              legend=dict(bgcolor="rgba(0,0,0,0.3)"))
            st.plotly_chart(fig, use_container_width=True)

        with tab_tsne:
            fig2 = go.Figure()
            for i, (nombre, color) in enumerate(zip(iris.target_names, colores_iris)):
                mask = iris.target == i
                fig2.add_trace(go.Scatter(
                    x=Xip[mask, 0], y=Xip[mask, 1],
                    mode="markers", name=nombre.capitalize(),
                    marker=dict(color=color, size=8, opacity=0.8)))
            fig2.update_layout(template="plotly_dark", height=260,
                               xaxis_title="Dim 1", yaxis_title="Dim 2",
                               margin=dict(l=20, r=10, t=10, b=30),
                               legend=dict(bgcolor="rgba(0,0,0,0.3)"))
            st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# DIAPOSITIVA 13 — Guía rápida de uso
# ─────────────────────────────────────────────────────────────────────────────
elif s == 13:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown(
            f'<div class="slide-wrap">'
            f'<div class="slide-number">14 / {TOTAL}</div>'
            f'<div class="slide-chapter" style="color:#22C55E">✅ Cierre</div>'
            f'<div class="slide-title" style="color:#22C55E">¿Cuándo usar cada uno?<br>Guía rápida</div>'
            f'<div class="slide-subtitle">Un árbol de decisión simple para elegir el algoritmo correcto.</div>'
            f'<div class="slide-body">'
            f'<div class="step-list">'
            f'<div class="step-item">'
            f'<div class="step-dot" style="background:#6C63FF">1</div>'
            f'<div class="step-dot-text"><strong>¿Quieres una exploración rápida inicial?</strong><br>'
            f'→ Empieza siempre con <span style="color:#6C63FF"><strong>PCA</strong></span>. '
            f'Es el más rápido, determinista y fácil de interpretar.</div></div>'
            f'<div class="step-item">'
            f'<div class="step-dot" style="background:#38BDF8">2</div>'
            f'<div class="step-dot-text"><strong>¿Tienes menos de 50.000 muestras y quieres ver clusters?</strong><br>'
            f'→ Usa <span style="color:#38BDF8"><strong>t-SNE</strong></span>. '
            f'Da los clusters más compactos y visualmente limpios.</div></div>'
            f'<div class="step-item">'
            f'<div class="step-dot" style="background:#FF6B6B">3</div>'
            f'<div class="step-dot-text"><strong>¿Dataset grande, necesitas escalar o proyectar datos nuevos?</strong><br>'
            f'→ Usa <span style="color:#FF6B6B"><strong>UMAP</strong></span>. '
            f'Más rápido y preserva también la estructura global.</div></div>'
            f'<div class="step-item">'
            f'<div class="step-dot" style="background:#F59E0B">4</div>'
            f'<div class="step-dot-text"><strong>¿Quieres el mejor resultado posible?</strong><br>'
            f'→ Aplica los tres y <span style="color:#F59E0B"><strong>compara visualmente</strong></span>. '
            f'La página ⚔️ Comparar de la app hace esto en un clic.</div></div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### 📋 Tabla de referencia rápida")
        df_quick = pd.DataFrame({
            "Situación": [
                "Primera exploración de cualquier dataset",
                "Dataset con muchas correlaciones entre variables",
                "Quiero ver clusters bien definidos",
                "Dataset con >10.000 muestras",
                "Necesito proyectar datos nuevos",
                "Antes de entrenar un modelo de ML",
                "Datos con estructura no lineal (espirales, círculos...)",
            ],
            "Algoritmo recomendado": [
                "🧩 PCA",
                "🧩 PCA",
                "🌌 t-SNE",
                "🚀 UMAP",
                "🧩 PCA o 🚀 UMAP",
                "🧩 PCA",
                "🌌 t-SNE o 🚀 UMAP",
            ]
        })
        st.dataframe(df_quick, hide_index=True, use_container_width=True)
        st.markdown(
            '<div class="callout-green" style="margin-top:.6rem">✅ <strong>Consejo profesional:</strong> '
            'en la práctica, los mejores análisis usan los tres. '
            'PCA primero para reducir ruido, luego t-SNE o UMAP para la visualización final.</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# DIAPOSITIVA 14 — Cierre, recursos y próximos pasos
# ─────────────────────────────────────────────────────────────────────────────
elif s == 14:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown(
            f'<div class="slide-wrap">'
            f'<div class="slide-number">15 / {TOTAL}</div>'
            f'<div class="slide-chapter" style="color:#22C55E">🎉 Cierre</div>'
            f'<div class="slide-title" style="color:#22C55E">Recursos y<br>próximos pasos</div>'
            f'<div class="slide-subtitle">¿Qué hacer después de esta presentación?</div>'
            f'<div class="slide-body">'
            f'<div class="step-list">'
            f'<div class="step-item">'
            f'<div class="step-dot" style="background:#6C63FF">🧩</div>'
            f'<div class="step-dot-text">Ir a la página <strong>PCA</strong> de la app y '
            f'ejecutar la demo con el dataset Iris. Ajustar el deslizador de ángulo.</div></div>'
            f'<div class="step-item">'
            f'<div class="step-dot" style="background:#38BDF8">🌌</div>'
            f'<div class="step-dot-text">Ir a <strong>t-SNE</strong> y cambiar la perplejidad '
            f'de 5 a 80. Observar cómo cambian los clusters.</div></div>'
            f'<div class="step-item">'
            f'<div class="step-dot" style="background:#FF6B6B">🚀</div>'
            f'<div class="step-dot-text">Ir a <strong>UMAP</strong> y comparar mentalmente '
            f'con t-SNE en el mismo dataset. ¿Qué diferencias ves?</div></div>'
            f'<div class="step-item">'
            f'<div class="step-dot" style="background:#F59E0B">⚔️</div>'
            f'<div class="step-dot-text">Ir a <strong>Comparar</strong> y ejecutar los '
            f'tres algoritmos al mismo tiempo en Dígitos.</div></div>'
            f'<div class="step-item">'
            f'<div class="step-dot" style="background:#22C55E">🎮</div>'
            f'<div class="step-dot-text">Ir al <strong>Laboratorio</strong> y '
            f'explorar libremente. Consultar el glosario integrado.</div></div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### 📚 Recursos para profundizar")
        recursos = [
            ("📄", "Paper original t-SNE",
             "van der Maaten & Hinton (2008) — el artículo que lo cambió todo.",
             "https://jmlr.org/papers/v9/vandermaaten08a.html"),
            ("📄", "Paper original UMAP",
             "McInnes, Healy & Melville (2018) — la base teórica de UMAP.",
             "https://arxiv.org/abs/1802.03426"),
            ("🌐", "Documentación scikit-learn",
             "PCA y t-SNE con ejemplos y parámetros completos.",
             "https://scikit-learn.org/stable/modules/decomposition.html"),
            ("🌐", "Documentación UMAP",
             "Guía oficial con tutoriales y casos de uso.",
             "https://umap-learn.readthedocs.io"),
            ("🎮", "Distill: How to Use t-SNE Effectively",
             "Artículo interactivo imprescindible sobre las trampas de t-SNE.",
             "https://distill.pub/2016/misread-tsne/"),
        ]
        for emoji, titulo, desc, url in recursos:
            st.markdown(
                f'<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;'
                f'padding:.75rem 1rem;margin-bottom:.5rem">'
                f'<strong style="color:#E5E7EB">{emoji} {titulo}</strong><br>'
                f'<span style="color:#9CA3AF;font-size:.88rem">{desc}</span><br>'
                f'<a href="{url}" target="_blank" '
                f'style="color:#6C63FF;font-size:.82rem">{url}</a>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="callout-green" style="margin-top:.6rem;text-align:center">'
            '🎉 <strong>¡Gracias por seguir la presentación!</strong><br>'
            'Usa los botones de arriba para revisitar cualquier diapositiva.'
            '</div>',
            unsafe_allow_html=True,
        )

# ── Botones de navegación (abajo) ────────────────────────────────────────────
st.markdown("")
bot_prev, bot_info, bot_next = st.columns([1, 3, 1])
with bot_prev:
    st.button("◀ Anterior", key="bot_prev",
              disabled=(st.session_state.slide_idx == 0),
              use_container_width=True,
              on_click=_prev)
with bot_info:
    st.markdown(
        f"<p class='nav-hint'>"
        f"Diapositiva <strong>{st.session_state.slide_idx + 1}</strong> de "
        f"<strong>{TOTAL}</strong> · "
        f"Usa los botones, el deslizador o el menú de capítulos para navegar."
        f"</p>",
        unsafe_allow_html=True,
    )
with bot_next:
    st.button("Siguiente ▶", key="bot_next",
              disabled=(st.session_state.slide_idx == TOTAL - 1),
              use_container_width=True, type="primary",
              on_click=_next)
