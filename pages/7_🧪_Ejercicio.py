"""Ejercicio práctico guiado — Dataset de cáncer de mama (Breast Cancer Wisconsin)."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

st.set_page_config(
    page_title="Ejercicio Práctico — Cáncer de Mama",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2; margin-bottom: .3rem;
}
.section-label {
    font-size: .72rem; font-weight: 700; letter-spacing: .11em;
    text-transform: uppercase; color: #10B981; margin-bottom: .3rem;
}
.step-card {
    background: #111827; border: 1px solid #1f2937;
    border-left: 5px solid; border-radius: 14px;
    padding: 1.4rem 1.7rem; margin-bottom: 1rem;
}
.step-card h3 { margin: 0 0 .4rem 0; font-size: 1.05rem; font-weight: 700; }
.step-card p  { margin: 0; color: #9CA3AF; font-size: .93rem; line-height: 1.65; }
.step-num-big {
    font-size: 2rem; font-weight: 800; line-height: 1;
    margin-bottom: .3rem;
}
.task-box {
    background: #052e16; border: 1px solid #16a34a; border-radius: 8px;
    padding: .75rem 1rem; color: #86efac; font-size: .91rem;
    line-height: 1.6; margin-top: .8rem;
}
.warn-box {
    background: #1c1700; border: 1px solid #ca8a04; border-radius: 8px;
    padding: .75rem 1rem; color: #fde047; font-size: .88rem;
    line-height: 1.6; margin-top: .6rem;
}
.info-box {
    background: #0c1a2e; border: 1px solid #2563eb; border-radius: 8px;
    padding: .85rem 1.1rem; color: #93c5fd; font-size: .9rem;
    line-height: 1.65; margin-bottom: .8rem;
}
.answer-box {
    background: #1a1a2e; border: 1px solid #4B5563; border-radius: 10px;
    padding: 1rem 1.2rem; margin-top: .5rem;
    color: #D1D5DB; font-size: .9rem; line-height: 1.7;
}
.answer-box strong { color: #F9FAFB; }
.ds-pill {
    display: inline-block; background: #0f172a; border: 1px solid #334155;
    border-radius: 999px; padding: .2rem .75rem;
    font-size: .78rem; color: #94A3B8; font-weight: 600;
    margin-right: .3rem; margin-bottom: .3rem;
}
</style>
""", unsafe_allow_html=True)

# ── Cargar dataset ────────────────────────────────────────────────────────────
@st.cache_data
def _load():
    bc = load_breast_cancer()
    return bc

bc = _load()
X_raw = bc.data
y     = bc.target          # 0 = maligno, 1 = benigno
names = bc.target_names    # ['malignant', 'benign']
feat  = bc.feature_names

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ══════════════════════════════════════════════════════════════════════════════
# CABECERA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="hero-title">🧪 Ejercicio Práctico Guiado</p>', unsafe_allow_html=True)
st.markdown(
    "**Dataset: Breast Cancer Wisconsin** — un caso real de medicina. "
    "Ningún código que escribir. Solo observar, ajustar y razonar."
)
st.markdown(
    '<span class="ds-pill">569 muestras</span>'
    '<span class="ds-pill">30 dimensiones</span>'
    '<span class="ds-pill">2 clases: maligno / benigno</span>'
    '<span class="ds-pill">Datos reales de biopsias</span>',
    unsafe_allow_html=True,
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PRESENTACIÓN DEL DATASET
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">El dataset</p>', unsafe_allow_html=True)
st.markdown("## ¿Con qué datos vamos a trabajar?")

col_desc, col_prev = st.columns([1.1, 1], gap="large")

with col_desc:
    st.markdown("""
El **Breast Cancer Wisconsin Dataset** fue recopilado en la Universidad de Wisconsin-Madison
en 1993 a partir de imágenes digitalizadas de biopsias de nódulos mamarios.

Cada fila representa **una biopsia** (una muestra de tejido). A partir de la imagen
se midieron 10 características del núcleo celular, y para cada una se calcularon
3 estadísticos (media, error estándar y peor valor), dando **30 variables** en total.

**¿Qué queremos saber?**
¿Pueden estas 30 medidas separar automáticamente los tumores **malignos** de los **benignos**?
¿Qué algoritmo lo hace mejor visualmente?

**¿Por qué es interesante para este ejercicio?**
""")
    st.markdown(
        '<div class="info-box">'
        '🏥 <strong>Contexto real:</strong> en la práctica clínica, una biopsia tarda días '
        'en analizarse. Un modelo que aprenda de estas 30 variables podría ayudar a '
        'priorizar casos urgentes.<br><br>'
        '📐 <strong>30 dimensiones</strong> es demasiado para visualizar directamente, '
        'pero suficiente para que PCA y t-SNE muestren resultados muy diferentes.<br><br>'
        '🎯 <strong>Solo 2 clases</strong> (maligno / benigno) hace que el resultado sea '
        'fácil de interpretar: si el algoritmo funciona, verás <em>dos nubes</em> bien separadas.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Variables más importantes
    st.markdown("#### 📋 Muestra de las 30 variables")
    df_feat = pd.DataFrame({
        "Variable": feat[:10],
        "Qué mide": [
            "Radio medio del núcleo",
            "Textura media (variación de escala de grises)",
            "Perímetro medio",
            "Área media",
            "Suavidad media (variación local del radio)",
            "Compacidad media",
            "Concavidad media (severidad de partes cóncavas)",
            "Puntos cóncavos medios",
            "Simetría media",
            "Dimensión fractal media",
        ],
    })
    st.dataframe(df_feat, hide_index=True, use_container_width=True)
    st.caption("Cada una de estas 10 medidas tiene además su error estándar y su valor máximo → 30 columnas.")

with col_prev:
    st.markdown("#### 👀 Primeras 8 filas (6 variables)")
    cols_show = list(feat[:6])
    df_show = pd.DataFrame(X_raw[:8, :6], columns=cols_show).round(2)
    df_show.insert(0, "diagnóstico", ["maligno" if t == 0 else "benigno" for t in y[:8]])
    st.dataframe(df_show, hide_index=True, use_container_width=True)

    st.markdown("#### 📊 ¿Cuántas muestras de cada clase?")
    counts = pd.Series(y).map({0: "🔴 Maligno", 1: "🟢 Benigno"}).value_counts()
    fig_bar = px.bar(
        x=counts.index, y=counts.values,
        labels={"x": "Diagnóstico", "y": "Muestras"},
        color=counts.index,
        color_discrete_map={"🔴 Maligno": "#FF6B6B", "🟢 Benigno": "#10B981"},
        template="plotly_dark", height=230,
    )
    fig_bar.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=30))
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("212 malignos (37 %) · 357 benignos (63 %) — dataset ligeramente desbalanceado.")

    st.markdown("#### 🔢 Estadísticas de las primeras 5 variables")
    df_stats = pd.DataFrame(X_raw[:, :5], columns=list(feat[:5]))
    st.dataframe(df_stats.describe().round(2), use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# EJERCICIOS — 4 PASOS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Los ejercicios</p>', unsafe_allow_html=True)
st.markdown("## 🎯 Cuatro pasos, cada uno con su misión")
st.markdown(
    '<div class="info-box">'
    '⏱️ <strong>Tiempo estimado: 20–25 minutos en pareja.</strong> '
    'Cada paso tiene un gráfico interactivo aquí mismo — no hace falta salir de esta página. '
    'Leed el enunciado, moved los controles y responded las preguntas en voz alta o por escrito.'
    '</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Paso 1 — PCA básico",
    "Paso 2 — ¿Cuántas dimensiones?",
    "Paso 3 — t-SNE",
    "Paso 4 — Comparar",
    "✅ Respuestas",
])

# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — PCA básico
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🧩 Paso 1 — PCA: ¿se pueden separar los tumores en 2D?")
    st.markdown(
        "PCA comprime las 30 dimensiones en 2 números (componentes principales). "
        "Vamos a ver si esos 2 números bastan para separar los tumores malignos de los benignos."
    )

    col_ctrl1, col_plot1 = st.columns([1, 2], gap="large")
    with col_ctrl1:
        st.markdown("#### ⚙️ Controles")
        n_comp_show = st.slider(
            "Número de componentes a calcular (para el Scree Plot del Paso 2)",
            2, 30, 10, key="p1_ncomp",
        )
        point_size = st.slider("Tamaño de los puntos", 4, 14, 8, key="p1_ps")
        opacity    = st.slider("Opacidad", 0.3, 1.0, 0.8, step=0.05, key="p1_op")

        st.markdown(
            '<div class="task-box">'
            '<strong>🎯 Tu misión:</strong><br>'
            '1. Observa el gráfico. ¿Ves dos nubes de puntos claramente separadas?<br>'
            '2. ¿Cuánta varianza acumulada muestran CP1 + CP2?<br>'
            '3. ¿Hay zonas donde los colores se mezclan? ¿Qué crees que significa?'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="warn-box">'
            '⚠️ <strong>Recuerda:</strong> PCA no conoce las etiquetas (maligno/benigno). '
            'Sólo comprime. Los colores los añadimos nosotros para ver si funcionó.'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_plot1:
        pca2 = PCA(n_components=2, random_state=42)
        Xp   = pca2.fit_transform(X)
        v    = pca2.explained_variance_ratio_

        fig1 = go.Figure()
        for cls, label, color in [(0, "Maligno", "#FF6B6B"), (1, "Benigno", "#10B981")]:
            m = y == cls
            fig1.add_trace(go.Scatter(
                x=Xp[m, 0], y=Xp[m, 1], mode="markers",
                name=label,
                marker=dict(color=color, size=point_size, opacity=opacity,
                            line=dict(width=0.5, color="white")),
            ))
        fig1.update_layout(
            template="plotly_dark", height=420,
            xaxis_title=f"CP1 — {v[0]*100:.1f}% de varianza",
            yaxis_title=f"CP2 — {v[1]*100:.1f}% de varianza",
            legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(size=13)),
            margin=dict(l=30, r=20, t=20, b=40),
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.caption(
            f"PCA 2D · {len(X)} biopsias · varianza capturada: "
            f"**{(v[0]+v[1])*100:.1f}%** de las 30 variables originales."
        )

# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — Scree Plot
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Paso 2 — ¿Cuántas componentes necesitamos realmente?")
    st.markdown(
        "Con 30 variables originales, PCA puede calcular hasta 30 componentes. "
        "El **Scree Plot** nos muestra cuánta información aporta cada una. "
        "El objetivo es encontrar el número mínimo de componentes que conservan el 90% de la varianza."
    )

    col_ctrl2, col_plot2 = st.columns([1, 2], gap="large")
    with col_ctrl2:
        umbral = st.slider("Umbral de varianza acumulada (%)", 70, 99, 90, key="p2_umb")
        st.markdown(
            '<div class="task-box">'
            '<strong>🎯 Tu misión:</strong><br>'
            '1. Mueve el deslizador al umbral que prefieras (90 % es habitual).<br>'
            '2. Observa la línea naranja del gráfico. '
            '¿En qué componente cruza el umbral?<br>'
            '3. ¿Cuántas de las 30 dimensiones originales necesitamos realmente?<br>'
            '4. Compara con el dataset Iris (4D) — ¿es más o menos eficiente comprimir este?'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_plot2:
        pca_all  = PCA(random_state=42).fit(X)
        evr      = pca_all.explained_variance_ratio_
        cum_evr  = np.cumsum(evr)
        n_needed = int(np.searchsorted(cum_evr, umbral / 100) + 1)
        nc_show  = min(n_comp_show, 30)

        fig2 = go.Figure()
        fig2.add_bar(
            x=list(range(1, nc_show + 1)),
            y=evr[:nc_show] * 100,
            name="Varianza individual",
            marker_color="#6C63FF",
            opacity=0.8,
        )
        fig2.add_scatter(
            x=list(range(1, nc_show + 1)),
            y=cum_evr[:nc_show] * 100,
            name="Varianza acumulada",
            mode="lines+markers",
            line=dict(color="#F59E0B", width=2.5),
            marker=dict(size=6),
        )
        fig2.add_hline(
            y=umbral,
            line_dash="dash", line_color="#10B981", line_width=1.5,
            annotation_text=f"{umbral}%",
            annotation_position="right",
        )
        if n_needed <= nc_show:
            fig2.add_vline(
                x=n_needed,
                line_dash="dot", line_color="#FF6B6B", line_width=1.5,
                annotation_text=f"  CP {n_needed}",
                annotation_position="top right",
            )
        fig2.update_layout(
            template="plotly_dark", height=380,
            xaxis_title="Número de componentes",
            yaxis_title="Varianza (%)",
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=30, r=40, t=20, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.info(
            f"Con **{n_needed} componentes** se captura el {umbral}% de la varianza "
            f"(de 30 dimensiones originales → reducción {30/n_needed:.1f}×)."
        )

# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — t-SNE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🌌 Paso 3 — t-SNE: ¿mejora la separación?")
    st.markdown(
        "t-SNE es un algoritmo no lineal que intenta colocar puntos similares cerca en 2D. "
        "A diferencia de PCA, **no tiene en cuenta las distancias globales** — "
        "sólo le importa que los vecinos cercanos queden cerca."
    )

    col_ctrl3, col_plot3 = st.columns([1, 2], gap="large")
    with col_ctrl3:
        st.markdown("#### ⚙️ Controles")
        perp = st.slider("Perplejidad", 5, 80, 30, step=5, key="p3_perp")
        n_iter_val = st.select_slider(
            "Iteraciones", options=[250, 500, 750, 1000], value=500, key="p3_iter"
        )
        run_tsne = st.button("▶ Ejecutar t-SNE", type="primary", key="p3_run",
                             use_container_width=True)
        st.markdown(
            '<div class="task-box">'
            '<strong>🎯 Tu misión:</strong><br>'
            '1. Ejecuta con perplejidad 30. ¿Las dos nubes están más separadas que en PCA?<br>'
            '2. Prueba perplejidad 5. ¿Qué pasa con los grupos?<br>'
            '3. Prueba perplejidad 70. ¿Y ahora?<br>'
            '4. Ejecuta dos veces con los mismos parámetros. '
            '¿El gráfico es idéntico? ¿Por qué?'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="warn-box">'
            '⚠️ t-SNE es lento: puede tardar 10–20 segundos. '
            'Es normal — está haciendo miles de cálculos de vecindario.'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_plot3:
        key_tsne = f"tsne_{perp}_{n_iter_val}"
        if run_tsne or key_tsne not in st.session_state:
            with st.spinner("Calculando t-SNE…"):
                tsne_model = TSNE(
                    n_components=2, perplexity=perp,
                    max_iter=n_iter_val, random_state=42, init="pca",
                )
                Xt = tsne_model.fit_transform(X)
            st.session_state[key_tsne] = Xt
        else:
            Xt = st.session_state[key_tsne]

        fig3 = go.Figure()
        for cls, label, color in [(0, "Maligno", "#FF6B6B"), (1, "Benigno", "#10B981")]:
            m = y == cls
            fig3.add_trace(go.Scatter(
                x=Xt[m, 0], y=Xt[m, 1], mode="markers",
                name=label,
                marker=dict(color=color, size=8, opacity=0.8,
                            line=dict(width=0.4, color="white")),
            ))
        fig3.update_layout(
            template="plotly_dark", height=420,
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(size=13)),
            margin=dict(l=30, r=20, t=20, b=40),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            f"t-SNE · perplejidad={perp} · {n_iter_val} iteraciones · "
            f"{len(X)} biopsias"
        )

# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — Comparar PCA vs t-SNE
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### ⚔️ Paso 4 — PCA vs t-SNE cara a cara")
    st.markdown(
        "Ahora ponemos los dos gráficos uno al lado del otro con el mismo dataset. "
        "Esto permite ver directamente qué diferencias introduce el método no lineal."
    )

    st.markdown(
        '<div class="task-box">'
        '<strong>🎯 Tu misión (debatid en pareja):</strong><br>'
        '1. ¿Cuál de los dos métodos separa mejor las clases visualmente?<br>'
        '2. ¿Hay muestras que ambos métodos clasifican igual de mal (zona de mezcla)?<br>'
        '3. Si tuvieras que usar uno de los dos para presentar este análisis a un médico, ¿cuál usarías? Justifica.<br>'
        '4. PCA conserva el 44% de la varianza en 2D. ¿Es suficiente para tomar decisiones clínicas?'
        '</div>',
        unsafe_allow_html=True,
    )

    cA, cB = st.columns(2, gap="large")

    # PCA (ya calculado)
    with cA:
        st.markdown("#### 🧩 PCA")
        fig4a = go.Figure()
        for cls, label, color in [(0, "Maligno", "#FF6B6B"), (1, "Benigno", "#10B981")]:
            m = y == cls
            fig4a.add_trace(go.Scatter(
                x=Xp[m, 0], y=Xp[m, 1], mode="markers", name=label,
                marker=dict(color=color, size=7, opacity=0.75,
                            line=dict(width=0.4, color="white")),
            ))
        fig4a.update_layout(
            template="plotly_dark", height=380,
            xaxis_title=f"CP1 ({v[0]*100:.0f}%)", yaxis_title=f"CP2 ({v[1]*100:.0f}%)",
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=20, r=10, t=20, b=40),
        )
        st.plotly_chart(fig4a, use_container_width=True)
        st.caption(f"Varianza total capturada: **{(v[0]+v[1])*100:.1f}%**")

    # t-SNE (perplejidad 30, 500 iter — precalculada o recalculada)
    with cB:
        st.markdown("#### 🌌 t-SNE (perplejidad 30)")
        key_cmp = "tsne_30_500"
        if key_cmp not in st.session_state:
            with st.spinner("Calculando t-SNE…"):
                Xt_cmp = TSNE(n_components=2, perplexity=30, max_iter=500,
                              random_state=42, init="pca").fit_transform(X)
            st.session_state[key_cmp] = Xt_cmp
        else:
            Xt_cmp = st.session_state[key_cmp]

        fig4b = go.Figure()
        for cls, label, color in [(0, "Maligno", "#FF6B6B"), (1, "Benigno", "#10B981")]:
            m = y == cls
            fig4b.add_trace(go.Scatter(
                x=Xt_cmp[m, 0], y=Xt_cmp[m, 1], mode="markers", name=label,
                marker=dict(color=color, size=7, opacity=0.75,
                            line=dict(width=0.4, color="white")),
            ))
        fig4b.update_layout(
            template="plotly_dark", height=380,
            xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=20, r=10, t=20, b=40),
        )
        st.plotly_chart(fig4b, use_container_width=True)
        st.caption("No hay % de varianza — t-SNE no la reporta.")

# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — Respuestas guiadas
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### ✅ Respuestas y reflexiones guiadas")
    st.markdown(
        "Comprueba tus razonamientos. No hay una única respuesta correcta — "
        "lo importante es que puedas justificar tu elección."
    )

    preguntas = [
        (
            "¿Ves dos nubes claramente separadas en PCA?",
            "**Sí, aunque con solapamiento.** CP1 separa bien las dos clases: los tumores malignos "
            "tienden a tener valores más altos en CP1 (mayor tamaño nuclear). Sin embargo, "
            "hay una zona de mezcla en el centro. PCA captura ~44% de la varianza en 2D — "
            "significa que el 56% restante queda 'aplastado' y algunas muestras parecen "
            "más parecidas de lo que realmente son."
        ),
        (
            "¿Cuántas componentes necesita este dataset para llegar al 90% de varianza?",
            "Aproximadamente **7–8 componentes** (frente a las 2 que necesita Iris para el 97%). "
            "Esto refleja que el cáncer de mama tiene más estructura compleja: "
            "hay 30 variables con correlaciones no tan perfectas entre sí. "
            "En la práctica, 7D es mucho más manejable que 30D para entrenar un clasificador."
        ),
        (
            "¿Por qué t-SNE da un gráfico diferente cada vez que se ejecuta?",
            "Porque t-SNE tiene inicialización **aleatoria** (aunque usamos `init='pca'` "
            "para reducirlo) y su algoritmo de optimización introduce variabilidad. "
            "Los grupos aparecen siempre, pero su posición y orientación cambian. "
            "Por eso nunca se debe usar t-SNE para comparar dos ejecuciones diferentes."
        ),
        (
            "¿Qué método separa mejor visualmente: PCA o t-SNE?",
            "**t-SNE** separa las dos clases de forma más nítida, con menos solapamiento. "
            "Esto es esperable: al ser no lineal, puede capturar relaciones curvas entre "
            "las variables que PCA (lineal) no puede expresar. Sin embargo, t-SNE es lento "
            "y no puede transformar datos nuevos — no sirve para producción."
        ),
        (
            "¿Usarías la visualización 2D de PCA para tomar una decisión clínica?",
            "**No directamente.** Una visualización 2D es exploratoria — muestra tendencias, "
            "no certezas. En un contexto clínico real se necesitaría: (1) un modelo de "
            "clasificación entrenado con todas las dimensiones o con las 7-8 componentes "
            "relevantes, (2) validación externa, y (3) un umbral de confianza claro. "
            "La visualización ayuda a comunicar y detectar outliers, no a diagnosticar."
        ),
    ]

    for i, (pregunta, respuesta) in enumerate(preguntas, 1):
        with st.expander(f"❓ Pregunta {i}: {pregunta}"):
            st.markdown(
                f'<div class="answer-box">{respuesta}</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### 🏁 Reflexión final")
    reflexion = st.radio(
        "Después de este ejercicio, si tuvieras un dataset médico de 50 variables y "
        "quisieras hacer una primera exploración visual, ¿qué harías?",
        [
            "Intentaría visualizar directamente las 50 variables",
            "Aplicaría PCA primero para tener una idea rápida, luego t-SNE para afinar ✅",
            "Solo usaría t-SNE porque da mejores gráficos",
            "Esperaría a tener más datos antes de visualizar nada",
        ],
        index=None,
        key="quiz_ejercicio",
    )
    if reflexion:
        if "PCA primero" in reflexion:
            st.success(
                "🎉 ¡Exacto! PCA es rápido y reproducible — perfecto para una primera exploración. "
                "Si los grupos no son claros en PCA, entonces usas t-SNE o UMAP. "
                "Este flujo de trabajo es el estándar en ciencia de datos."
            )
        elif "50 variables" in reflexion:
            st.error(
                "❌ Con 50 variables es imposible hacer una visualización directa significativa. "
                "Necesitarías elegir 2 variables cada vez — perderías el contexto del resto."
            )
        elif "Solo usaría t-SNE" in reflexion:
            st.warning(
                "⚠️ t-SNE es potente pero lento y no reproducible. "
                "Para datos grandes o cuando necesites transformar datos nuevos, "
                "PCA sigue siendo imprescindible como primer paso."
            )
        else:
            st.warning(
                "⚠️ La cantidad de datos no suele ser la limitante para visualizar. "
                "Con 50 dimensiones incluso 100 muestras justifican usar reducción de dimensionalidad."
            )

st.divider()
st.markdown(
    "<p style='text-align:center;color:#4B5563;font-size:.88rem'>"
    "Dataset: Breast Cancer Wisconsin (Diagnostic) · UCI Machine Learning Repository · "
    "Wolberg, W.H. et al. (1993)"
    "</p>",
    unsafe_allow_html=True,
)
