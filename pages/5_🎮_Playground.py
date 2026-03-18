"""Playground — experimenta libremente con todos los parámetros."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import load_dataset, apply_pca, apply_tsne, apply_umap, scatter_2d, render_watermark

st.set_page_config(page_title="Playground", page_icon="🎮", layout="wide")
render_watermark()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.callout-blue   { background:#0c1a2e; border:1px solid #2563eb; border-radius:8px;
    padding:.9rem 1.1rem; color:#93c5fd; font-size:.93rem; margin-bottom:.6rem; }
.callout-green  { background:#052e16; border:1px solid #16a34a; border-radius:8px;
    padding:.9rem 1.1rem; color:#86efac; font-size:.93rem; margin-bottom:.6rem; }
.callout-yellow { background:#1c1700; border:1px solid #ca8a04; border-radius:8px;
    padding:.9rem 1.1rem; color:#fde047; font-size:.93rem; margin-bottom:.6rem; }
.dataset-pill { display:inline-block; background:#1E1E2E; border-radius:999px;
    padding:.2rem .8rem; font-size:.8rem; color:#9CA3AF; margin:.15rem; border:1px solid #374151; }
.glosario-card { background:#1E1E2E; border-radius:10px; padding:1rem 1.2rem;
    border-left:4px solid #6C63FF; margin-bottom:.6rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🎮 Playground — Tu laboratorio personal")
st.markdown(
    "Aquí tienes **control total**: elige el dataset, el algoritmo y ajusta los parámetros "
    "para ver cómo cambia el resultado. Es el mejor lugar para experimentar libremente."
)

# ── Sidebar de control ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    st.markdown("### 1️⃣ Elige el dataset")
    dataset_name = st.selectbox(
        "Dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="pg_ds",
        help="¿Qué datos quieres explorar?")

    st.markdown("### 2️⃣ Elige el algoritmo")
    algorithm = st.selectbox(
        "Algoritmo", ["PCA 🧩", "t-SNE 🌌", "UMAP 🚀"], key="pg_algo",
        help="Cada algoritmo comprime los datos de una manera diferente.")

    st.markdown("### 3️⃣ Ajusta los parámetros")
    if algorithm == "PCA 🧩":
        n_components = st.radio(
            "Dimensiones de salida", [2, 3], key="pg_pca_nc", horizontal=True,
            help="2D para un gráfico plano, 3D para un gráfico rotable.")
        st.markdown(
            '<div style="background:#1E1E2E;border-radius:8px;padding:.8rem;'
            'font-size:.82rem;color:#9CA3AF">PCA no tiene parámetros difíciles. '
            'Sólo elige cuántas dimensiones quieres de salida.</div>',
            unsafe_allow_html=True)
    elif algorithm == "t-SNE 🌌":
        perp = st.slider(
            "Perplejidad", 5, 100, 30, key="pg_perp",
            help="Cuántos vecinos 've' cada punto. Bajo=clusters pequeños, Alto=clusters grandes. Recomendado: 30.")
        n_iter = st.select_slider(
            "Iteraciones", [250, 500, 750, 1000], value=500, key="pg_iter",
            help="Más iteraciones = resultado más refinado, pero más lento.")
        n_components = 2
        st.markdown(
            '<div style="background:#1E1E2E;border-radius:8px;padding:.8rem;'
            'font-size:.82rem;color:#9CA3AF">💡 Prueba perplejidades de 5, 30 y 80 '
            'para ver cómo cambia el resultado.</div>',
            unsafe_allow_html=True)
    else:
        nn = st.slider(
            "n_neighbors (vecinos)", 2, 100, 15, key="pg_nn",
            help="Cuántos vecinos considera el grafo. Bajo=estructura local, Alto=estructura global.")
        md = st.slider(
            "min_dist (distancia mínima)", 0.0, 0.99, 0.1, 0.05, key="pg_md",
            help="Qué tan apretados quedan los puntos. 0=muy apretados, 0.9=muy dispersos.")
        n_components = 2
        st.markdown(
            '<div style="background:#1E1E2E;border-radius:8px;padding:.8rem;'
            'font-size:.82rem;color:#9CA3AF">💡 Prueba n_neighbors=2 y luego 50 '
            'para ver la diferencia entre estructura local y global.</div>',
            unsafe_allow_html=True)

    st.divider()
    show_raw = st.checkbox("Ver datos originales (tabla)", value=False, key="pg_raw")
    run_btn = st.button("▶️ Aplicar", type="primary", key="pg_run")

# ── Descripción dataset ────────────────────────────────────────────────────────
X, y, df_orig, desc = load_dataset(dataset_name)

dataset_info = {
    "Iris 🌸": {
        "origen": "Estudio botánico de Ronald Fisher (1936). El dataset más famoso de ML.",
        "muestras": "150 flores de 3 especies",
        "dimensiones": "4 medidas: largo/ancho de sépalo y pétalo (en cm)",
        "categorias": "Setosa / Versicolor / Virginica",
        "nota": "Setosa es muy diferente de las otras dos. Versicolor y Virginica se parecen más.",
    },
    "Vino 🍷": {
        "origen": "Análisis químicos de vinos de Barolo, Italia. UCI Machine Learning Repository.",
        "muestras": "178 vinos de 3 productores",
        "dimensiones": "13 medidas: alcohol, acidez, fenoles, color, magnesio, etc.",
        "categorias": "Productor 0 / Productor 1 / Productor 2",
        "nota": "Los 3 productores hacen vinos con perfiles químicos muy distintos.",
    },
    "Dígitos ✏️": {
        "origen": "Imágenes de dígitos escaneadas. Versión simplificada del dataset MNIST.",
        "muestras": "1.797 imágenes de dígitos del 0 al 9",
        "dimensiones": "64 píxeles (cuadrícula 8×8, cada píxel vale 0–16)",
        "categorias": "Dígitos 0, 1, 2, 3, 4, 5, 6, 7, 8, 9",
        "nota": "El más impresionante: 64 dimensiones comprimidas a 2, y los 10 dígitos se separan.",
    },
}

info = dataset_info[dataset_name]
with st.expander(f"📦 ¿De dónde vienen los datos de '{dataset_name}'?", expanded=False):
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown(f"**Origen:** {info['origen']}")
        st.markdown(f"**Muestras:** {info['muestras']}")
        st.markdown(f"**Dimensiones:** {info['dimensiones']}")
    with col_i2:
        st.markdown(f"**Categorías:** {info['categorias']}")
        st.markdown(f"**Nota:** {info['nota']}")
        st.markdown(f"**Dimensiones originales:** `{X.shape[1]}` → reducidas a `{n_components}`")

col_plot, col_info_panel = st.columns([2, 1])

# ── Ejecución ──────────────────────────────────────────────────────────────────
cache_key = (
    dataset_name, algorithm,
    locals().get("n_components", 2),
    locals().get("perp", None),
    locals().get("n_iter", None),
    locals().get("nn", None),
    locals().get("md", None),
)

if run_btn or "pg_result" not in st.session_state or st.session_state.get("pg_cfg") != cache_key:
    with st.spinner("🔄 Calculando… puede tardar unos segundos para t-SNE"):
        if algorithm == "PCA 🧩":
            nc = n_components
            X_r, pca_model = apply_pca(X, n_components=nc)
            y_r = y
        elif algorithm == "t-SNE 🌌":
            if dataset_name == "Dígitos ✏️" and X.shape[0] > 500:
                idx = np.random.RandomState(42).choice(X.shape[0], 500, replace=False)
                X_r = apply_tsne(X[idx], perplexity=perp, n_iter=n_iter)
                y_r = y[idx]
            else:
                X_r = apply_tsne(X, perplexity=perp, n_iter=n_iter)
                y_r = y
            nc = 2
        else:
            X_r = apply_umap(X, n_neighbors=nn, min_dist=md)
            y_r = y
            nc = 2
        st.session_state["pg_result"] = (X_r, y_r, nc)
        st.session_state["pg_cfg"] = cache_key
        if algorithm == "PCA 🧩":
            st.session_state["pg_pca_model"] = pca_model

if "pg_result" not in st.session_state:
    st.info("👈 Configura los parámetros en el panel izquierdo y pulsa **▶️ Aplicar**.")
    st.stop()

X_r, y_r, nc = st.session_state["pg_result"]

with col_plot:
    if nc == 3:
        from utils.helpers import scatter_3d
        fig = scatter_3d(X_r, y_r, title=f"{algorithm} — {dataset_name}")
    else:
        if algorithm == "PCA 🧩":
            fig = scatter_2d(X_r, y_r,
                             title=f"{algorithm} — {dataset_name}",
                             x_label="Componente Principal 1 (CP1) — máxima varianza →",
                             y_label="Componente Principal 2 (CP2) →")
        elif algorithm == "t-SNE 🌌":
            fig = scatter_2d(X_r, y_r,
                             title=f"{algorithm} — {dataset_name}",
                             x_label="Dimensión t-SNE 1 (sin unidades concretas) →",
                             y_label="Dimensión t-SNE 2 (sin unidades concretas) →")
        else:
            fig = scatter_2d(X_r, y_r,
                             title=f"{algorithm} — {dataset_name}",
                             x_label="UMAP Dimensión 1 →",
                             y_label="UMAP Dimensión 2 →")
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    # Guía de lectura contextual
    if algorithm == "PCA 🧩":
        st.markdown(
            '<div class="callout-blue">📖 <strong>Cómo leer este gráfico:</strong> '
            'El eje X (CP1) es la dirección donde los datos varían más. '
            'El eje Y (CP2) es la segunda dirección de mayor variación. '
            'Si los grupos de colores están bien separados → PCA funciona muy bien para estos datos. '
            'Si se solapan → las diferencias entre grupos son sutiles o no lineales.</div>',
            unsafe_allow_html=True)
    elif algorithm == "t-SNE 🌌":
        st.markdown(
            '<div class="callout-yellow">📖 <strong>Cómo leer este gráfico:</strong> '
            'Los ejes NO tienen significado físico — son coordenadas inventadas por el algoritmo. '
            'Lo que importa: ¿los puntos del mismo color forman "islas" compactas? '
            '⚠️ La distancia ENTRE grupos no es interpretable, sólo la cohesión DENTRO de cada grupo.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="callout-blue">📖 <strong>Cómo leer este gráfico:</strong> '
            'Similar a t-SNE, pero aquí la posición relativa entre grupos SÍ importa: '
            'grupos cercanos en el mapa son más parecidos en los datos originales. '
            'UMAP preserva tanto la estructura local (dentro de cada grupo) '
            'como la global (relación entre grupos).</div>',
            unsafe_allow_html=True)

with col_info_panel:
    st.markdown("### 📊 Resumen del resultado")

    n_clases = len(np.unique(y_r))
    st.metric("Dimensiones originales", f"{X.shape[1]}")
    st.metric("Dimensiones reducidas", f"{nc}")
    st.metric("Muestras visualizadas", f"{len(y_r)}")
    st.metric("Categorías (colores)", f"{n_clases}")

    if algorithm == "PCA 🧩" and "pg_pca_model" in st.session_state:
        pca_model = st.session_state["pg_pca_model"]
        var = pca_model.explained_variance_ratio_ * 100
        total = var.sum()
        st.markdown("---")
        st.markdown("**Varianza conservada por componente:**")
        st.caption("Cuánta información del dataset original captura cada dimensión.")
        for i, v in enumerate(var):
            st.progress(int(v), text=f"CP{i+1}: {v:.1f}% de la información")
        st.metric("Total conservado", f"{total:.1f}%",
                  delta=f"Se pierde el {100-total:.1f}%")
        if total >= 90:
            st.success(f"✅ Excelente: conservamos el {total:.1f}% de la información con sólo {nc} dimensiones.")
        else:
            st.warning(f"⚠️ Con {nc} componentes conservamos el {total:.1f}%. Considera usar más componentes.")

    st.markdown("---")
    st.markdown("**Primeras filas del resultado:**")
    st.caption("Estos son los valores numéricos de las nuevas dimensiones reducidas.")
    df_res = pd.DataFrame(X_r, columns=[f"Dim {i+1}" for i in range(nc)])
    df_res.insert(0, "Categoría", y_r)
    st.dataframe(df_res.head(10), use_container_width=True, hide_index=True)

# ── Vista de datos crudos ──────────────────────────────────────────────────────
if show_raw:
    st.divider()
    st.markdown("### 🗃️ Datos originales — primeras 10 filas")
    st.markdown(
        f"Esto es lo que el algoritmo recibe como entrada: una tabla con "
        f"**{X.shape[0]} filas** (muestras) y **{X.shape[1]} columnas** (variables). "
        f"El algoritmo transforma esto en sólo **{nc} columnas** manteniendo la estructura."
    )
    st.dataframe(df_orig.head(10), use_container_width=True)
    st.caption(
        "Las columnas numéricas son las variables originales (ya estandarizadas). "
        "La columna 'etiqueta' es la categoría real de cada muestra."
    )

# ── Glosario completo ──────────────────────────────────────────────────────────
st.divider()
st.markdown("## 📚 Glosario — Todos los términos explicados sin tecnicismos")
st.markdown("Haz clic en cualquier término para ver su explicación completa.")

glosario = [
    (
        "📐 Dimensión",
        "Una **dimensión** es simplemente una variable o característica de los datos.\n\n"
        "**Ejemplo concreto:** Si tienes datos de personas con altura, peso, edad y salario, "
        "tienes 4 dimensiones.\n\n"
        "Una imagen de 8×8 píxeles tiene 64 dimensiones (un número por cada píxel). "
        "Una imagen de 1000×1000 tendría 1.000.000 dimensiones.\n\n"
        "**¿Por qué es un problema tener muchas?** Porque no podemos dibujar gráficas "
        "de más de 3 ejes. Necesitamos reducir a 2 ó 3 para visualizar.",
    ),
    (
        "📊 Varianza",
        "La **varianza** mide cuánto varían los datos — qué tan dispersos están.\n\n"
        "**Ejemplo:** Si todos los estudiantes de una clase sacan exactamente 7 en un examen, "
        "la varianza es 0 (no hay variación). Si sacan entre 2 y 10, la varianza es alta.\n\n"
        "**En PCA:** buscamos las direcciones de MÁXIMA varianza porque ahí está la información "
        "más útil. Las direcciones de poca varianza suelen ser ruido.",
    ),
    (
        "🧭 Componente Principal (CP)",
        "Una **componente principal** es una nueva dirección en el espacio de datos "
        "que PCA inventa para capturar la máxima información posible.\n\n"
        "**CP1** = la dirección donde los datos varían MÁS (tiene más información)\n"
        "**CP2** = la segunda dirección de mayor variación (siempre perpendicular a CP1)\n"
        "**CP3** = la tercera... y así sucesivamente.\n\n"
        "**Analogía:** Es como encontrar el mejor ángulo para fotografiar una escultura — "
        "el que muestra más detalles en una sola imagen.",
    ),
    (
        "🔑 Loading (contribución de variable)",
        "Un **loading** indica cuánto contribuye cada variable original a una componente principal.\n\n"
        "**Ejemplo:** Si el 'largo del pétalo' tiene un loading de +0.93 en CP1, significa que "
        "las flores con pétalos largos tienen valores altos en CP1.\n\n"
        "**Rango:** de -1 a +1. Cercano a ±1 = variable muy influyente. Cercano a 0 = casi no influye. "
        "Negativo = relación inversa.",
    ),
    (
        "🌐 Manifold (variedad)",
        "Un **manifold** es una superficie de baja dimensión que está 'enrollada' dentro "
        "de un espacio de alta dimensión.\n\n"
        "**Ejemplo:** La Tierra es una esfera 2D (superficie) que existe en el espacio 3D. "
        "Si aplanas un mapa de la Tierra, estás 'desenvolviendo' un manifold.\n\n"
        "**En ML:** Los datos de alta dimensión suelen vivir en un manifold de baja dimensión. "
        "UMAP y t-SNE lo detectan y lo 'desenrollan'.",
    ),
    (
        "🎛️ Perplejidad (t-SNE)",
        "La **perplejidad** controla cuántos vecinos considera cada punto en t-SNE.\n\n"
        "**Piénsalo así:** Es como decidir si una persona define su 'entorno' mirando sólo "
        "a sus 5 vecinos más próximos (perplejidad baja) o a toda su ciudad (perplejidad alta).\n\n"
        "- **5–10:** Cada punto sólo mira a los más cercanos → clusters muy pequeños y separados\n"
        "- **30:** Balance óptimo → **valor recomendado**\n"
        "- **80–100:** Cada punto mira muy lejos → clusters grandes y conectados",
    ),
    (
        "👥 n_neighbors (UMAP)",
        "**n_neighbors** es el número de vecinos que cada punto usa para construir el grafo en UMAP.\n\n"
        "**Analogía:** Si un punto sólo conoce a sus 2 vecinos más cercanos (n_neighbors=2), "
        "sólo ve su entorno inmediato. Si conoce a 100 personas (n_neighbors=100), "
        "tiene una visión mucho más amplia del vecindario.\n\n"
        "- **Bajo (2–5):** Estructura muy local → clusters muy fragmentados\n"
        "- **15:** Valor recomendado → buen balance\n"
        "- **Alto (50–100):** Estructura global → clusters más grandes y conectados",
    ),
    (
        "↔️ min_dist (UMAP)",
        "**min_dist** controla qué tan apretados quedan los puntos dentro de cada cluster en UMAP.\n\n"
        "**Analogía:** Es como el zoom de una cámara al fotografiar un grupo de personas. "
        "Zoom máximo (min_dist=0) → todos muy juntos, difícil ver quién es quién. "
        "Gran angular (min_dist=0.9) → todos muy separados, fácil distinguir pero se pierde la estructura.\n\n"
        "- **0.0:** Puntos ultra-compactos, clusters muy densos\n"
        "- **0.1:** Valor recomendado → clusters compactos con estructura visible\n"
        "- **0.9:** Puntos muy dispersos, sin clusters claros",
    ),
    (
        "🏝️ Cluster (grupo)",
        "Un **cluster** es un grupo natural de puntos que son similares entre sí.\n\n"
        "**Ejemplo:** En el dataset Iris, las flores Setosa forman un cluster porque "
        "todas tienen pétalos y sépalos de tamaños parecidos.\n\n"
        "La reducción de dimensionalidad NO crea los clusters — los revela. "
        "Si los datos tienen grupos naturales, el algoritmo los hace visibles.",
    ),
    (
        "🗺️ Embedding (representación reducida)",
        "Un **embedding** es el resultado de la reducción de dimensionalidad: "
        "la representación de los datos en el nuevo espacio de baja dimensión.\n\n"
        "**Ejemplo:** Si tienes 1.000 canciones descritas por 100 características musicales "
        "(tempo, tono, volumen...) y aplicas PCA para reducir a 2D, el resultado es un "
        "embedding — 1.000 puntos en un plano 2D donde canciones similares están cerca.\n\n"
        "Los ejes del embedding no tienen interpretación directa — son coordenadas abstractas.",
    ),
    (
        "📉 Scree plot",
        "El **scree plot** es un gráfico de barras que muestra cuánta información "
        "captura cada componente principal en PCA.\n\n"
        "**Cómo leerlo:** Las barras van de mayor a menor. La primera barra (CP1) es siempre "
        "la más alta. Busca el 'codo' — el punto donde las barras dejan de baer bruscamente. "
        "Las componentes después del codo aportan muy poco.\n\n"
        "La línea acumulada te dice: 'con X componentes conservo el Y% de la información'.",
    ),
    (
        "📏 Estandarización (z-score)",
        "**Estandarizar** significa transformar cada variable para que tenga media 0 y desviación estándar 1.\n\n"
        "**¿Por qué es necesario para PCA?** Si una variable mide salario en euros (0–50.000) "
        "y otra mide altura en metros (1.5–2.0), PCA sin estandarizar pensaría que el salario "
        "es miles de veces más importante — sólo por la escala numérica.\n\n"
        "Estandarizar nivela el campo de juego: todas las variables parten con la misma importancia.",
    ),
]

cols = st.columns(2)
for i, (titulo, contenido) in enumerate(glosario):
    with cols[i % 2]:
        with st.expander(titulo):
            st.markdown(contenido)

st.markdown("---")
st.markdown(
    '<div class="callout-green">💡 <strong>¿Quieres profundizar más?</strong> '
    'Cada una de las páginas de PCA, t-SNE y UMAP tiene una sección "¿Cómo funciona?" '
    'con explicaciones más detalladas, visualizaciones interactivas y un quiz.</div>',
    unsafe_allow_html=True)
