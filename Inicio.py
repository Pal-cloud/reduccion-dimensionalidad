"""Página de inicio — Guía interactiva de reducción de dimensionalidad en español."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.preprocessing import StandardScaler

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

/* ── Dataset cards ─────────────────────────────────────────────── */
.ds-card {
    background: #1E1E2E;
    border: 1px solid #2D2D3F;
    border-radius: 16px;
    padding: 1.5rem 1.7rem;
    margin-bottom: 1rem;
    border-top: 4px solid;
}
.ds-card h3 { margin: 0 0 .3rem 0; font-size: 1.2rem; font-weight: 800; }
.ds-card .ds-meta {
    display: flex; flex-wrap: wrap; gap: .4rem; margin: .6rem 0 .9rem 0;
}
.ds-pill {
    background: #0f172a; border: 1px solid #334155;
    border-radius: 999px; padding: .2rem .75rem;
    font-size: .78rem; color: #94A3B8; font-weight: 600;
}
.ds-card p { color: #CBD5E1; font-size: .92rem; line-height: 1.65; margin: 0; }
.ds-link {
    display: inline-block; margin-top: .8rem;
    font-size: .82rem; color: #6C63FF;
    text-decoration: none; border-bottom: 1px dashed #6C63FF;
}
.why-box {
    background: #0D1117; border-radius: 10px;
    padding: .85rem 1.1rem; margin-top: .8rem;
    border-left: 3px solid;
}
.why-box strong { font-size: .82rem; text-transform: uppercase;
    letter-spacing: .06em; }
.why-box p { color: #94A3B8; font-size: .88rem; margin: .3rem 0 0 0;
    line-height: 1.55; }
.callout-info {
    background: #0c1a2e; border: 1px solid #2563eb; border-radius: 10px;
    padding: .9rem 1.2rem; color: #93c5fd; font-size: .92rem;
    line-height: 1.6; margin-bottom: .8rem;
}

/* ── Glosario básico ───────────────────────────────────────────── */
.glosario-basico {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: .8rem;
}
.glosario-basico h4 { margin: 0 0 .25rem 0; font-size: 1rem; font-weight: 700; }
.glosario-basico .def { color: #9CA3AF; font-size: .9rem; line-height: 1.6; margin: 0; }
.glosario-basico .ej {
    margin-top: .5rem; background: #0f172a; border-radius: 6px;
    padding: .5rem .8rem; color: #6EE7B7; font-size: .85rem;
    border-left: 3px solid #10B981;
}

/* ── Pasos "qué verás en pantalla" ────────────────────────────── */
.pantalla-paso {
    display: flex; gap: 1rem; align-items: flex-start;
    background: #111827; border-radius: 10px; padding: 1rem 1.2rem;
    margin-bottom: .7rem; border: 1px solid #1f2937;
}
.pantalla-icono { font-size: 1.8rem; min-width: 2.5rem; text-align: center; }
.pantalla-texto h4 { margin: 0 0 .2rem 0; font-size: .97rem; color: #F9FAFB; }
.pantalla-texto p  { margin: 0; font-size: .88rem; color: #9CA3AF; line-height: 1.55; }

/* ── Cómo leer la gráfica de portada ──────────────────────────── */
.read-home {
    background: #0f1b2d; border: 1px solid #1e40af;
    border-left: 5px solid #3b82f6; border-radius: 10px;
    padding: 1rem 1.3rem; margin-top: .6rem;
}
.read-home .rt { color: #60a5fa; font-weight: 700; font-size: .82rem;
    text-transform: uppercase; letter-spacing: .07em; margin-bottom: .5rem; }
.read-home ul { margin: 0; padding-left: 1.2rem; }
.read-home li { color: #bfdbfe; font-size: .9rem; line-height: 1.7; margin-bottom: .15rem; }
.read-home li strong { color: #93c5fd; }
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
# BLOQUE CERO — 4 conceptos esenciales antes de empezar
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🔑 Antes de empezar: 4 conceptos que necesitas entender (empieza aquí)", expanded=True):
    st.markdown(
        "No necesitas saber matemáticas. Sólo necesitas entender estas 4 ideas "
        "para que todo lo demás tenga sentido:"
    )

    g1, g2 = st.columns(2, gap="medium")

    with g1:
        st.markdown(
            '<div class="glosario-basico">'
            '<h4>1️⃣ ¿Qué es una "dimensión"?</h4>'
            '<p class="def">Una dimensión es simplemente <strong>una columna en una tabla de datos</strong>, '
            'es decir, una característica o variable medida.<br><br>'
            'Si tienes datos de personas con altura, peso, edad y salario → tienes <strong>4 dimensiones</strong>.<br>'
            'Una foto de 28×28 píxeles tiene <strong>784 dimensiones</strong> (una por píxel).<br>'
            'Un análisis de sangre con 200 marcadores tiene <strong>200 dimensiones</strong>.</p>'
            '<div class="ej">💡 Ejemplo: el dataset de flores Iris tiene 4 dimensiones: '
            'largo del sépalo, ancho del sépalo, largo del pétalo, ancho del pétalo.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="glosario-basico">'
            '<h4>2️⃣ ¿Por qué es un problema tener muchas dimensiones?</h4>'
            '<p class="def">Nuestros ojos y cerebro sólo pueden procesar gráficas de '
            '<strong>2 ó 3 dimensiones</strong> a la vez.<br><br>'
            'Si tienes un dataset con 64 columnas, es como intentar visualizar '
            'un objeto en un espacio de 64 ejes — es imposible.<br><br>'
            'La solución es <strong>comprimir</strong> esas 64 columnas en 2, '
            'perdiendo la menor información posible.</p>'
            '<div class="ej">💡 Ejemplo: los dígitos escritos a mano tienen 64 dimensiones '
            '(64 píxeles). Nadie puede visualizar 64 ejes a la vez.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with g2:
        st.markdown(
            '<div class="glosario-basico">'
            '<h4>3️⃣ ¿Qué es un gráfico de dispersión (scatter plot)?</h4>'
            '<p class="def">Es el tipo de gráfico que verás en toda esta app. '
            'Cada <strong>punto representa una muestra</strong> (una flor, un vino, una imagen).<br><br>'
            'Su posición en el plano viene dada por las 2 nuevas dimensiones calculadas por el algoritmo.<br><br>'
            'El <strong>color de cada punto es su categoría real</strong> '
            '(la etiqueta que ya existía en los datos, no algo inventado).</p>'
            '<div class="ej">💡 Si ves puntos del mismo color agrupados juntos → '
            'el algoritmo ha detectado que esas muestras son similares. ¡Eso es éxito!</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="glosario-basico">'
            '<h4>4️⃣ ¿Qué significa "varianza" o "información conservada"?</h4>'
            '<p class="def">La <strong>varianza</strong> mide cuánto varían los datos — '
            'si todos los valores son iguales, la varianza es 0.<br><br>'
            'Cuando comprimimos datos, inevitablemente perdemos algo. '
            'El porcentaje de <strong>varianza conservada</strong> nos dice '
            'cuánta información del dataset original hemos mantenido.<br><br>'
            '<strong>95% de varianza conservada</strong> = excelente compresión.<br>'
            '<strong>50% de varianza conservada</strong> = hemos perdido la mitad.</p>'
            '<div class="ej">💡 PCA con Iris conserva ~97% de la varianza con solo 2 dimensiones. '
            'Es como comprimir un archivo y que casi no se note la diferencia.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="background:#052e16;border:1px solid #16a34a;border-radius:8px;'
        'padding:.8rem 1.1rem;color:#86efac;font-size:.92rem;margin-top:.5rem">'
        '✅ <strong>Resumen en una frase:</strong> estos algoritmos toman una tabla con '
        'muchas columnas y la convierten en un gráfico 2D donde puedes ver con tus propios '
        'ojos si existen grupos o patrones ocultos en los datos.'
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

**En palabras muy simples:** imagina que tienes una hoja de cálculo con 500 columnas.
Nadie puede leer 500 columnas a la vez. Estos algoritmos la convierten en un
**dibujo de puntos** donde los datos similares quedan cerca y los diferentes quedan lejos.
""")

    # Tres analogías cotidianas
    analogias = [
        ("#6C63FF",
         "🎒 La maleta de viaje",
         "No te llevas toda la casa de viaje — empacas sólo lo esencial. "
         "La reducción de dimensionalidad hace lo mismo con los datos: "
         "se queda con las características más importantes y descarta el ruido. "
         "El resultado es una maleta pequeña que contiene casi todo lo que importa."),
        ("#48CAE4",
         "📷 La fotografía de una escultura",
         "Una escultura existe en 3D, pero la foto es 2D. Un buen fotógrafo elige "
         "el ángulo que muestra más detalle en una sola imagen. "
         "PCA hace exactamente eso: encuentra el 'mejor ángulo' para fotografiar "
         "tus datos y que la imagen 2D sea lo más informativa posible."),
        ("#FF6B6B",
         "🗺️ El mapa del metro",
         "El metro de una ciudad existe en 3D (con curvas y desniveles), "
         "pero el mapa es 2D y plano. No es exacto, pero es suficiente para "
         "entender cómo moverte. t-SNE y UMAP construyen ese tipo de mapa para tus datos, "
         "simplificando para que puedas orientarte."),
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
    st.markdown(
        '<div class="read-home">'
        '<div class="rt">📖 Cómo leer esta gráfica — paso a paso</div>'
        '<ul>'
        '<li><strong>Cada punto = una flor real</strong> medida en el campo. '
        'Hay 150 puntos porque el dataset tiene 150 flores.</li>'
        '<li><strong>El color = la especie</strong> de esa flor '
        '(setosa, versicolor o virginica). PCA no sabía los colores — '
        'los ponemos nosotros para ver si el algoritmo las separó bien.</li>'
        '<li><strong>El eje X (CP1)</strong> es la primera "dirección comprimida". '
        f'Captura el <strong>{v[0]*100:.0f}%</strong> de toda la variación de las 4 variables originales. '
        'No representa ninguna variable física concreta.</li>'
        '<li><strong>El eje Y (CP2)</strong> es la segunda dirección. '
        f'Captura el <strong>{v[1]*100:.0f}%</strong> adicional.</li>'
        '<li><strong>¿Qué buscar?</strong> Si los tres colores forman nubes separadas → '
        '¡PCA ha revelado que estas especies son distinguibles con solo 2 números! '
        'Si las nubes se mezclan → las especies se parecen mucho.</li>'
        '<li><strong>Resultado aquí:</strong> La especie setosa (un color) está completamente '
        'separada. Las otras dos se solapan un poco — coincide con la biología real.</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ¿POR QUÉ IMPORTA?
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">¿Por qué importa?</p>', unsafe_allow_html=True)
st.markdown("## Tres razones por las que todo científico de datos necesita esto")

r1, r2, r3 = st.columns(3, gap="medium")
razones = [
    ("👁️", "Ver lo invisible", "#6C63FF",
     "Convierte una hoja de cálculo con 500 columnas en un diagrama 2D que entiendes "
     "en segundos. Si hay grupos naturales en los datos — razas de animales, tipos de "
     "clientes, enfermedades similares — aparecen como nubes de puntos separadas. "
     "Sin reducción de dimensionalidad, esos grupos son completamente invisibles."),
    ("⚡", "Modelos más rápidos y precisos", "#48CAE4",
     "Los modelos de Machine Learning entrenan **10×–100× más rápido** con datos "
     "comprimidos. Menos dimensiones = menos parámetros que aprender = menos riesgo "
     "de 'memorizar' los datos de entrenamiento en lugar de aprender patrones reales "
     "(lo que se llama sobreajuste u overfitting)."),
    ("🔍", "Descubrir patrones ocultos", "#FF6B6B",
     "Revela grupos y relaciones que son completamente invisibles en tablas de datos crudos. "
     "Por ejemplo: en genómica, t-SNE puede revelar subtipos de células que ningún "
     "investigador había definido previamente. Los datos hablan solos."),
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
    "Usa la **barra lateral** (menú izquierdo) para navegar entre secciones. "
    "Cada página tiene teoría, una demo interactiva y un quiz. "
    "**¿No sabes por dónde empezar? Empieza siempre por 🧩 PCA** — es el más sencillo."
)

# ── Qué encontrarás en cada página ─────────────────────────────────────────
with st.expander("🗺️ ¿Qué encontrarás en cada página? (guía para nuevos usuarios)", expanded=False):
    pasos_pantalla = [
        ("📖", "Pestaña '¿Cómo funciona?'",
         "Explicación del algoritmo con analogías cotidianas y diagramas animados. "
         "No hay fórmulas matemáticas. Está pensada para que cualquier persona entienda "
         "la idea intuitiva detrás del método."),
        ("🎯", "Pestaña 'Demo interactiva'",
         "Aquí aplicas el algoritmo a datos reales con un clic. Puedes elegir el dataset "
         "(flores, vinos o dígitos), ajustar los parámetros con deslizadores y ver "
         "inmediatamente cómo cambia el gráfico de puntos. Cada gráfico tiene debajo "
         "un panel azul que explica cómo leerlo."),
        ("📊", "Pestaña '¿Cuántas componentes?' (solo en PCA)",
         "El 'Scree Plot' — un gráfico de barras que te dice cuántas dimensiones necesitas "
         "conservar para no perder demasiada información. Tiene una explicación paso a paso."),
        ("🧠", "Pestaña 'Quiz'",
         "4 preguntas de opción múltiple con retroalimentación inmediata y explicación "
         "detallada de cada respuesta. No penaliza los errores — es para aprender."),
        ("⚔️", "Página 'Comparar'",
         "Los tres algoritmos aplicados al mismo dataset simultáneamente. "
         "Ideal para ver las diferencias de un vistazo y decidir cuál usar."),
        ("🎮", "Página 'Laboratorio'",
         "Modo libre: elige cualquier combinación de dataset + algoritmo + parámetros. "
         "Incluye un glosario completo de todos los términos técnicos."),
    ]
    for icono, titulo, desc in pasos_pantalla:
        st.markdown(
            f'<div class="pantalla-paso">'
            f'<div class="pantalla-icono">{icono}</div>'
            f'<div class="pantalla-texto"><h4>{titulo}</h4><p>{desc}</p></div>'
            f'</div>',
            unsafe_allow_html=True,
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
# SECCIÓN: LOS DATASETS — origen, variables, por qué se eligieron
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Los datos que usamos</p>', unsafe_allow_html=True)
st.markdown("## 📂 ¿De dónde vienen los datasets?")
st.markdown(
    '<div class="callout-info">'
    '🎓 <strong>¿Por qué importa saber el origen de los datos?</strong><br>'
    'Los tres datasets que usa esta app son <em>benchmarks clásicos</em> del Machine Learning: '
    'llevan décadas usándose para enseñar y comparar algoritmos porque tienen propiedades '
    'muy concretas (tamaño manejable, clases bien definidas, variables interpretables). '
    'No son datos aleatorios — cada uno fue elegido para demostrar algo diferente de la '
    'reducción de dimensionalidad.'
    '</div>',
    unsafe_allow_html=True,
)

# ── Cargar los tres datasets para los previews ─────────────────────────────
@st.cache_data
def _load_all():
    iris   = load_iris()
    wine   = load_wine()
    digits = load_digits()
    return iris, wine, digits

_iris, _wine, _digits = _load_all()

ds_tab1, ds_tab2, ds_tab3 = st.tabs([
    "🌸 Iris — el clásico de clásicos",
    "🍷 Vino — datos químicos reales",
    "✏️ Dígitos — imágenes a mano",
])

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1 — IRIS
# ─────────────────────────────────────────────────────────────────────────────
with ds_tab1:
    col_iris_info, col_iris_prev = st.columns([1.1, 1], gap="large")

    with col_iris_info:
        st.markdown("""
### 🌸 Iris — El "Hola Mundo" del Machine Learning

**¿Quién lo creó?**
El botánico y estadístico británico **Ronald A. Fisher** recopiló este dataset en 1936
para ilustrar el análisis discriminante lineal. Es el dataset más citado en la historia
del Machine Learning y aparece en casi todos los libros de texto.

**¿Qué contiene?**
Medidas físicas de **150 flores de iris** de tres especies distintas, tomadas en el
Gaspé Peninsula de Canadá. Cada flor fue medida con un calibre:
""")
        variables = {
            "Variable": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
            "En español": ["largo del sépalo", "ancho del sépalo", "largo del pétalo", "ancho del pétalo"],
            "Rango típico": ["4.3 – 7.9 cm", "2.0 – 4.4 cm", "1.0 – 6.9 cm", "0.1 – 2.5 cm"],
        }
        st.dataframe(pd.DataFrame(variables), hide_index=True, use_container_width=True)

        st.markdown("""
**Las 3 especies (clases):**
- 🌺 *Iris setosa* — muy diferente a las otras dos; fácil de separar
- 🌷 *Iris versicolor* — se superpone parcialmente con virginica
- 🌹 *Iris virginica* — la más grande; difícil de distinguir de versicolor

**¿Por qué lo elegimos para esta app?**
""")
        st.markdown(
            '<div class="why-box" style="border-left-color:#6C63FF">'
            '<strong style="color:#6C63FF">✅ Ideal para aprender porque…</strong>'
            '<p>Solo tiene 4 dimensiones → fácil de entender qué se está comprimiendo. '
            'Las clases son visualmente separables en 2D con PCA. '
            'Es tan conocido que cualquier resultado puede verificarse en miles de referencias. '
            'Con sólo 150 filas, los algoritmos corren en milisegundos.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "🔗 **Fuente oficial:** "
            "[UCI Machine Learning Repository — Iris Dataset]"
            "(https://archive.ics.uci.edu/dataset/53/iris)  \n"
            "📄 Paper original: Fisher, R.A. (1936) *The use of multiple measurements in taxonomic problems*"
        )

    with col_iris_prev:
        st.markdown("#### 👀 Primeras 10 filas del dataset")
        df_iris_show = pd.DataFrame(
            _iris.data[:10],
            columns=["largo sépalo", "ancho sépalo", "largo pétalo", "ancho pétalo"]
        )
        df_iris_show["especie"] = [_iris.target_names[t] for t in _iris.target[:10]]
        st.dataframe(df_iris_show, hide_index=True, use_container_width=True)

        st.markdown("#### 📊 Distribución de clases")
        species_counts = pd.Series(_iris.target).map(
            {i: n for i, n in enumerate(_iris.target_names)}
        ).value_counts()
        fig_bar = px.bar(
            x=species_counts.index, y=species_counts.values,
            labels={"x": "Especie", "y": "Cantidad de flores"},
            color=species_counts.index,
            color_discrete_sequence=["#6C63FF", "#48CAE4", "#FF6B6B"],
            template="plotly_dark",
            height=220,
        )
        fig_bar.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=30))
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("50 flores por especie — dataset perfectamente balanceado.")

        st.markdown("#### 🔢 Estadísticas básicas")
        df_stats = pd.DataFrame(_iris.data,
            columns=["largo sépalo", "ancho sépalo", "largo pétalo", "ancho pétalo"])
        st.dataframe(df_stats.describe().round(2), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2 — VINO
# ─────────────────────────────────────────────────────────────────────────────
with ds_tab2:
    col_wine_info, col_wine_prev = st.columns([1.1, 1], gap="large")

    with col_wine_info:
        st.markdown("""
### 🍷 Vino — Química que distingue bodegas

**¿Quién lo creó?**
Fue donado al UCI Repository en **1991** por M. Forina (Universidad de Génova, Italia).
Contiene el resultado de análisis químicos realizados sobre vinos de tres productores
diferentes de la región de Piamonte (Italia del norte).

**¿Qué contiene?**
Resultados de **13 análisis químicos** sobre 178 muestras de vino. Un sommelier
distingue vinos por olfato; estos datos los distinguen por laboratorio:
""")
        vars_wine = {
            "Variable (en inglés)": [
                "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
                "magnesium", "total_phenols", "flavanoids",
                "nonflavanoid_phenols", "proanthocyanins",
                "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
            ],
            "Qué mide": [
                "% de alcohol", "acidez málica (g/l)", "cenizas minerales",
                "alcalinidad de las cenizas", "magnesio (mg/l)", "fenoles totales",
                "flavonoides (antioxidantes)", "fenoles no flavonoides",
                "proantocianinas", "intensidad del color", "matiz del color",
                "relación proteínas/color", "prolina (aminoácido)"
            ],
        }
        st.dataframe(pd.DataFrame(vars_wine), hide_index=True, use_container_width=True)

        st.markdown("""
**Los 3 productores (clases):**
Simplemente llamados Clase 0, 1 y 2 — son tres bodegas del mismo viñedo italiano.

**¿Por qué lo elegimos para esta app?**
""")
        st.markdown(
            '<div class="why-box" style="border-left-color:#48CAE4">'
            '<strong style="color:#48CAE4">✅ Ideal para demostrar PCA porque…</strong>'
            '<p>Tiene <strong>13 dimensiones</strong> — demasiadas para visualizar directamente. '
            'Muchas variables están altamente correlacionadas entre sí '
            '(por ejemplo, alcohol y flavonoides tienden a subir juntos). '
            'PCA colapsa esas correlaciones y extrae los ejes reales de variación. '
            'Es el dataset perfecto para mostrar que 2 componentes pueden capturar '
            'más del 55 % de toda la varianza de 13 variables.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "🔗 **Fuente oficial:** "
            "[UCI Machine Learning Repository — Wine Dataset]"
            "(https://archive.ics.uci.edu/dataset/109/wine)  \n"
            "📍 Origen geográfico: Piamonte, Italia — tres bodegas distintas."
        )

    with col_wine_prev:
        st.markdown("#### 👀 Primeras 10 filas del dataset")
        col_names_wine = [
            "alcohol", "ac. málica", "cenizas", "alc. cenizas", "magnesio",
            "fenoles", "flavonoides", "f. no flav.", "proantoc.",
            "color", "matiz", "od280/315", "prolina"
        ]
        df_wine_show = pd.DataFrame(_wine.data[:10], columns=col_names_wine).round(2)
        df_wine_show["productor"] = [f"Clase {t}" for t in _wine.target[:10]]
        st.dataframe(df_wine_show, hide_index=True, use_container_width=True)

        st.markdown("#### 📊 Distribución de clases")
        wine_counts = pd.Series(_wine.target).value_counts().sort_index()
        fig_w = px.bar(
            x=[f"Clase {i}" for i in wine_counts.index],
            y=wine_counts.values,
            labels={"x": "Productor", "y": "Muestras"},
            color=[f"Clase {i}" for i in wine_counts.index],
            color_discrete_sequence=["#48CAE4", "#6C63FF", "#FF6B6B"],
            template="plotly_dark", height=220,
        )
        fig_w.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=30))
        st.plotly_chart(fig_w, use_container_width=True)
        st.caption("59 / 71 / 48 muestras por clase — ligeramente desbalanceado.")

        st.markdown("#### 🔢 Estadísticas (primeras 6 variables)")
        df_w_stats = pd.DataFrame(_wine.data[:, :6], columns=col_names_wine[:6])
        st.dataframe(df_w_stats.describe().round(2), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET 3 — DÍGITOS
# ─────────────────────────────────────────────────────────────────────────────
with ds_tab3:
    col_dig_info, col_dig_prev = st.columns([1.1, 1], gap="large")

    with col_dig_info:
        st.markdown("""
### ✏️ Dígitos — Imágenes escritas a mano

**¿Quién lo creó?**
Este dataset es una versión reducida del famoso **MNIST** (Mixed National Institute of
Standards and Technology), preparada para scikit-learn por el equipo de investigación.
Las imágenes originales provienen de escribanos del NIST (agencia del gobierno de EE.UU.)
y de estudiantes universitarios que escribieron dígitos del 0 al 9 en papel cuadriculado.

**¿Qué contiene?**
**1.797 imágenes** de dígitos escritos a mano, cada una de **8×8 píxeles**
(en escala de grises, valores 0–16). Cada imagen se "aplana" en un vector de
**64 números** — esas son las 64 dimensiones del dataset.
""")
        st.markdown("""
```
Ejemplo de la imagen del dígito "0" como matriz 8×8:
 0  0  5 13  9  1  0  0
 0  0 13 15 10 15  5  0
 0  3 15  2  0 11  8  0
 0  4 12  0  0  8  8  0
 0  5  8  0  0  9  8  0
 0  4 11  0  1 12  7  0
 0  2 14  5 10 12  0  0
 0  0  6 13 10  0  0  0
↓ Se convierte en un vector de 64 números → [0,0,5,13,9,1,0,0,0,0,13,...]
```
""")

        st.markdown("""
**Las 10 clases:** dígitos del 0 al 9 — aproximadamente 180 imágenes por dígito.

**¿Por qué lo elegimos para esta app?**
""")
        st.markdown(
            '<div class="why-box" style="border-left-color:#FF6B6B">'
            '<strong style="color:#FF6B6B">✅ Ideal para demostrar t-SNE y UMAP porque…</strong>'
            '<p>Tiene <strong>64 dimensiones</strong> (cada píxel es una variable). '
            'Es completamente imposible visualizar 64 dimensiones directamente. '
            'Pero t-SNE y UMAP son capaces de crear un mapa 2D donde los 10 dígitos '
            'forman 10 islas perfectamente separadas — un resultado visualmente impactante. '
            'Además, todos entendemos los números, así que el resultado es intuitivo.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "🔗 **Fuente oficial (versión completa):** "
            "[MNIST Database — Yann LeCun]"
            "(http://yann.lecun.com/exdb/mnist/)  \n"
            "🔗 **Versión scikit-learn:** "
            "[sklearn.datasets.load_digits]"
            "(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)"
        )

    with col_dig_prev:
        st.markdown("#### 🖼️ Muestra visual: un dígito de cada clase")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_digits, axes = plt.subplots(2, 5, figsize=(8, 3.5))
        fig_digits.patch.set_facecolor("#0E1117")
        for digit in range(10):
            idx = np.where(_digits.target == digit)[0][0]
            ax = axes[digit // 5][digit % 5]
            ax.imshow(_digits.images[idx], cmap="Blues", interpolation="nearest")
            ax.set_title(str(digit), color="white", fontsize=13, fontweight="bold", pad=4)
            ax.axis("off")
            ax.set_facecolor("#1E1E2E")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig_digits, use_container_width=True)
        plt.close()
        st.caption("Cada cuadro es una imagen 8×8 píxeles → 64 números → 64 dimensiones.")

        st.markdown("#### 📊 Distribución de clases")
        digit_counts = pd.Series(_digits.target).value_counts().sort_index()
        fig_d = px.bar(
            x=[str(i) for i in digit_counts.index],
            y=digit_counts.values,
            labels={"x": "Dígito", "y": "Imágenes"},
            color=[str(i) for i in digit_counts.index],
            color_discrete_sequence=px.colors.qualitative.Vivid,
            template="plotly_dark", height=220,
        )
        fig_d.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=30))
        st.plotly_chart(fig_d, use_container_width=True)
        st.caption(f"Total: {len(_digits.target)} imágenes · ~178 por dígito.")

        st.markdown("#### 🔢 ¿Cuántas dimensiones tiene cada dataset?")
        df_dims = pd.DataFrame({
            "Dataset": ["🌸 Iris", "🍷 Vino", "✏️ Dígitos"],
            "Muestras": [150, 178, 1797],
            "Dimensiones originales": [4, 13, 64],
            "Tras reducir a 2D": [2, 2, 2],
            "Info. conservada (PCA)": ["~97 %", "~56 %", "~29 %"],
        })
        st.dataframe(df_dims, hide_index=True, use_container_width=True)
        st.caption(
            "A más dimensiones, más difícil preservar toda la información en 2D. "
            "Por eso Dígitos necesita t-SNE/UMAP para obtener una buena visualización."
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
