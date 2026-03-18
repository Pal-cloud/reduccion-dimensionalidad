"""Ejercicio práctico guiado con código — Dataset Breast Cancer Wisconsin."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import traceback
import io
import contextlib
import re as _re

st.set_page_config(
    page_title="Ejercicio Práctico — Breast Cancer",
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
.task-box {
    background: #052e16; border: 1px solid #16a34a; border-radius: 8px;
    padding: .8rem 1.1rem; color: #86efac; font-size: .91rem;
    line-height: 1.65; margin: .6rem 0;
}
.hint-box {
    background: #1c1f2e; border: 1px solid #4338ca; border-radius: 8px;
    padding: .7rem 1rem; color: #a5b4fc; font-size: .88rem;
    line-height: 1.6; margin-top: .4rem;
}
.info-box {
    background: #0c1a2e; border: 1px solid #2563eb; border-radius: 8px;
    padding: .85rem 1.1rem; color: #93c5fd; font-size: .9rem;
    line-height: 1.65; margin-bottom: .8rem;
}
.answer-box {
    background: #111827; border: 1px solid #374151; border-radius: 10px;
    padding: 1rem 1.2rem; color: #D1D5DB; font-size: .9rem; line-height: 1.7;
}
.answer-box strong { color: #F9FAFB; }
.ds-pill {
    display: inline-block; background: #0f172a; border: 1px solid #334155;
    border-radius: 999px; padding: .2rem .75rem;
    font-size: .78rem; color: #94A3B8; font-weight: 600;
    margin-right: .3rem; margin-bottom: .3rem;
}
textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
    font-size: .87rem !important;
}
.respuesta-box {
    background: #0a1628; border: 2px solid #2563eb;
    border-left: 6px solid #3b82f6; border-radius: 10px;
    padding: 1rem 1.3rem; margin-top: .8rem;
}
.respuesta-box .rb-title {
    font-size: .72rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: #60a5fa; margin-bottom: .6rem;
}
.respuesta-box table {
    width: 100%; border-collapse: collapse; font-size: .88rem;
}
.respuesta-box td {
    padding: .3rem .6rem; vertical-align: top; color: #cbd5e1;
    border-bottom: 1px solid #1e3a5f;
}
.respuesta-box td:first-child {
    color: #93c5fd; font-family: monospace; font-weight: 600;
    white-space: nowrap; width: 38%;
}
.respuesta-box tr:last-child td { border-bottom: none; }
</style>
""", unsafe_allow_html=True)

# ── Dataset global (cacheado) ──────────────────────────────────────────────────
@st.cache_data
def _load():
    bc = load_breast_cancer()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(bc.data)
    return bc, X_scaled

bc, X = _load()
X_raw = bc.data
y     = bc.target        # 0 = maligno, 1 = benigno
feat  = bc.feature_names

# ── Ejecutor de código seguro ─────────────────────────────────────────────────
def _has_blanks(code: str) -> bool:
    """True si quedan ___ sin rellenar (exactamente 3 guiones bajos)."""
    return bool(_re.search(r'(?<![_\w])___(?![_\w])', code))

def _run(code: str, extra: dict = {}) -> tuple:
    """Ejecuta código. Devuelve (stdout, error_str, locals_dict)."""
    local_ns: dict = {}
    buf = io.StringIO()
    globs = {
        "np": np, "pd": pd,
        "PCA": PCA, "TSNE": TSNE, "StandardScaler": StandardScaler,
        "X": X, "X_raw": X_raw, "y": y, "feat": feat, "bc": bc,
        **extra,
    }
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<ejercicio>", "exec"), globs, local_ns)  # noqa: S102
        return buf.getvalue(), "", local_ns
    except Exception:
        return buf.getvalue(), traceback.format_exc(), local_ns

def run_user_code(code: str, solution: str, extra: dict = {}) -> tuple:
    """Ejecuta el código del usuario o la solución si hay ___ sin rellenar.

    Devuelve (stdout, error_str, locals_dict, used_solution: bool).
    """
    if _has_blanks(code):
        stdout, err, lns = _run(solution, extra)
        return stdout, err, lns, True   # <-- solución automática
    stdout, err, lns = _run(code, extra)
    return stdout, err, lns, False      # <-- código del usuario

# ══════════════════════════════════════════════════════════════════════════════
# CABECERA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="hero-title">🧪 Ejercicio Práctico con Código</p>',
            unsafe_allow_html=True)
st.markdown(
    "**Dataset: Breast Cancer Wisconsin** · Completa los huecos `___` en cada bloque, "
    "pulsa ▶ Ejecutar y observa el resultado. Sin instalaciones ni configuración."
)
st.markdown(
    '<span class="ds-pill">569 muestras</span>'
    '<span class="ds-pill">30 dimensiones</span>'
    '<span class="ds-pill">2 clases: maligno / benigno</span>'
    '<span class="ds-pill">Datos ya cargados y escalados</span>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="info-box">'
    '📦 <strong>Variables disponibles en todos los bloques de código:</strong><br>'
    '• <code>X</code> — datos escalados (shape 569 × 30)<br>'
    '• <code>X_raw</code> — datos originales sin escalar<br>'
    '• <code>y</code> — etiquetas (0 = maligno · 1 = benigno)<br>'
    '• <code>feat</code> — nombres de las 30 variables<br>'
    '• <code>np</code>, <code>pd</code>, <code>PCA</code>, <code>TSNE</code>, '
    '<code>StandardScaler</code> ya importados'
    '</div>',
    unsafe_allow_html=True,
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📋 El dataset",
    "🧩 Paso 1 — PCA",
    "📊 Paso 2 — Scree Plot",
    "🌌 Paso 3 — t-SNE",
    "✅ Soluciones",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 — Exploración inicial
# ─────────────────────────────────────────────────────────────────────────────
with tab0:
    st.markdown("### 📋 Antes de nada: entiende con qué datos trabajas")

    c_info, c_prev = st.columns([1.1, 1], gap="large")
    with c_info:
        st.markdown("""
**Breast Cancer Wisconsin (Diagnostic)**

Cada fila = una biopsia de tejido mamario. A partir de una imagen microscópica
se midieron 10 propiedades del núcleo celular (radio, textura, perímetro, área…)
y para cada una se calcularon **3 estadísticos** (media, error estándar y valor máximo)
→ **30 variables** en total.

**Objetivo:** ¿pueden esas 30 medidas separar automáticamente
los tumores malignos de los benignos?
""")
        st.markdown(
            '<div class="info-box">'
            '🏥 Este dataset se usa en investigación clínica real. '
            'Los modelos entrenados con él alcanzan >95 % de precisión. '
            'La reducción de dimensionalidad nos ayuda a <em>visualizar por qué</em>.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "🔗 **Fuente oficial:** "
            "[UCI ML Repository — Breast Cancer Wisconsin (Diagnostic)]"
            "(https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  \n"
            "🔗 **En scikit-learn:** "
            "[sklearn.datasets.load\\_breast\\_cancer]"
            "(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)  \n"
            "📄 **Paper original:** Wolberg, W.H. et al. (1993) — *Breast cytology diagnoses "
            "via digital image analysis*, Analytical and Quantitative Cytology and Histology."
        )

    with c_prev:
        df_prev = pd.DataFrame(X_raw[:6, :5], columns=list(feat[:5])).round(2)
        df_prev.insert(0, "diagnóstico",
                       ["maligno" if t == 0 else "benigno" for t in y[:6]])
        st.markdown("#### Primeras filas (5 variables)")
        st.dataframe(df_prev, hide_index=True, use_container_width=True)
        counts = pd.Series(y).map({0: "🔴 Maligno", 1: "🟢 Benigno"}).value_counts()
        fig0 = px.bar(x=counts.index, y=counts.values,
                      color=counts.index,
                      color_discrete_map={"🔴 Maligno": "#FF6B6B", "🟢 Benigno": "#10B981"},
                      template="plotly_dark", height=200,
                      labels={"x": "", "y": "muestras"})
        fig0.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=20))
        st.plotly_chart(fig0, use_container_width=True)

    st.divider()
    st.markdown("#### ✏️ Ejercicio 0 — Explora el dataset con código")
    st.markdown(
        '<div class="task-box">'
        '<strong>Tu misión:</strong> reemplaza cada <code>___</code> para:<br>'
        '1. Imprimir el shape (dimensiones) de <code>X</code><br>'
        '2. Contar cuántos malignos y benignos hay<br>'
        '3. Comparar la media de la primera variable entre las dos clases'
        '</div>',
        unsafe_allow_html=True,
    )

    code_e0 = st.text_area(
        "📝 Código — rellena los ___",
        """\
# 1. Dimensiones del dataset
print("Shape de X:", ___.shape)

# 2. Recuento por clase
print("Malignos (y==0):", (y == ___).sum())
print("Benignos (y==1):", (y == ___).sum())

# 3. Media de la primera variable en cada clase
var0 = X_raw[:, 0]   # primera columna
print(f"Media '{feat[0]}' — malignos: {var0[y==0].mean():.2f}")
print(f"Media '{feat[0]}' — benignos: {var0[y==___].mean():.2f}")
""",
        height=220, key="code_e0",
    )

    _SOL_E0 = """\
print("Shape de X:", X.shape)
print("Malignos (y==0):", (y == 0).sum())
print("Benignos (y==1):", (y == 1).sum())
var0 = X_raw[:, 0]
print(f"Media '{feat[0]}' — malignos: {var0[y==0].mean():.2f}")
print(f"Media '{feat[0]}' — benignos: {var0[y==1].mean():.2f}")
"""
    if st.button("▶ Ejecutar", key="run_e0", type="primary"):
        stdout, err, _, used_sol = run_user_code(code_e0, solution=_SOL_E0)
        if err:
            st.error("Error inesperado — revisa el código:\n```\n" + err + "\n```")
        else:
            if used_sol:
                st.info("💡 Aún hay huecos sin rellenar — aquí tienes el resultado correcto:")
            else:
                st.success("✅ ¡Correcto! Tu código funciona perfectamente.")
            if stdout:
                st.code(stdout, language="text")

    st.markdown(
        '<div class="respuesta-box">'
        '<div class="rb-title">🔑 Respuestas correctas — Ejercicio 0</div>'
        '<table>'
        '<tr><td>___.shape</td><td>→ <strong>X</strong> (los datos escalados, 569×30)</td></tr>'
        '<tr><td>y == ___  (malignos)</td><td>→ <strong>0</strong> &nbsp;·&nbsp; resultado: 212</td></tr>'
        '<tr><td>y == ___  (benignos)</td><td>→ <strong>1</strong> &nbsp;·&nbsp; resultado: 357</td></tr>'
        '<tr><td>y == ___ (última línea)</td><td>→ <strong>1</strong> &nbsp;·&nbsp; media ~12.15</td></tr>'
        '</table>'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PCA
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🧩 Paso 1 — Aplica PCA y visualiza en 2D")
    st.markdown(
        "PCA comprime las 30 dimensiones en 2 componentes principales. "
        "Tu trabajo es escribir el código que lo hace y ver si las dos clases quedan separadas."
    )

    col_task1, col_hint1 = st.columns([1, 1], gap="large")
    with col_task1:
        st.markdown(
            '<div class="task-box">'
            '<strong>Tu misión:</strong><br>'
            '1. Crear <code>PCA(n_components=___)</code> con 2 componentes<br>'
            '2. Ajustar y transformar con <code>fit_transform(___)</code><br>'
            '3. Imprimir el % de varianza de cada componente<br>'
            '4. Ver el gráfico — ¿se separan las dos nubes?'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_hint1:
        with st.expander("💡 Pista"):
            st.markdown(
                "- `pca = PCA(n_components=2, random_state=42)`\n"
                "- `Xp = pca.fit_transform(X)` → aplica PCA a los datos escalados\n"
                "- `pca.explained_variance_ratio_` → array con el % de varianza por componente\n"
                "- `Xp.shape` → debería ser `(569, 2)`"
            )

    code_p1 = st.text_area(
        "📝 Código",
        """\
# 1. Crear PCA con 2 componentes y transformar X
pca = PCA(n_components=___, random_state=42)
Xp  = pca.___(X)           # fit_transform

# 2. Varianza explicada por cada componente
evr = pca.explained_variance_ratio_
print(f"CP1: {evr[0]*100:.1f}%")
print(f"CP2: {evr[1]*100:.1f}%")
print(f"Total 2 componentes: {(evr[0]+evr[1])*100:.1f}%")

# 3. Dimensiones del resultado
print("Shape de Xp:", Xp.___)    # shape
""",
        height=240, key="code_p1",
    )

    _SOL_P1 = """\
pca = PCA(n_components=2, random_state=42)
Xp  = pca.fit_transform(X)
evr = pca.explained_variance_ratio_
print(f"CP1: {evr[0]*100:.1f}%")
print(f"CP2: {evr[1]*100:.1f}%")
print(f"Total 2 componentes: {(evr[0]+evr[1])*100:.1f}%")
print("Shape de Xp:", Xp.shape)
"""
    run_p1 = st.button("▶ Ejecutar y graficar", key="run_p1", type="primary")

    if run_p1:
        stdout, err, lns, used_sol = run_user_code(code_p1, solution=_SOL_P1)
        if err:
            st.error("Error inesperado — revisa el código:\n```\n" + err + "\n```")
        else:
            if used_sol:
                st.info("💡 Aún hay huecos sin rellenar — aquí tienes el resultado correcto:")
            else:
                st.success("✅ ¡Correcto! Tu código funciona perfectamente.")
            if stdout:
                st.code(stdout, language="text")
            if "Xp" in lns:
                st.session_state["pca_Xp"]  = lns["Xp"]
                st.session_state["pca_evr"] = lns["evr"]

    if "pca_Xp" in st.session_state:
        Xp_s = st.session_state["pca_Xp"]
        evr_s = st.session_state["pca_evr"]
        fig1 = go.Figure()
        for cls, label, col in [(0, "Maligno", "#FF6B6B"), (1, "Benigno", "#10B981")]:
            m = y == cls
            fig1.add_trace(go.Scatter(
                x=Xp_s[m, 0], y=Xp_s[m, 1], mode="markers", name=label,
                marker=dict(color=col, size=7, opacity=0.82,
                            line=dict(width=0.4, color="white")),
            ))
        fig1.update_layout(
            template="plotly_dark", height=420,
            xaxis_title=f"CP1 — {evr_s[0]*100:.1f}% varianza",
            yaxis_title=f"CP2 — {evr_s[1]*100:.1f}% varianza",
            legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(size=13)),
            margin=dict(l=30, r=20, t=20, b=40),
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(
            '<div class="hint-box">'
            '🔍 <strong>¿Qué ves?</strong> CP1 separa bastante bien las clases: '
            'los malignos (rojo) tienen núcleos más grandes → valores más altos en CP1. '
            'Hay una zona de solapamiento en el centro — esas biopsias son ambiguas '
            'incluso con las 30 variables. PCA captura ~63% de la varianza total en 2D.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="respuesta-box">'
        '<div class="rb-title">🔑 Respuestas correctas — Paso 1</div>'
        '<table>'
        '<tr><td>n_components=___</td><td>→ <strong>2</strong></td></tr>'
        '<tr><td>pca.___(X)</td><td>→ <strong>fit_transform</strong>(X)</td></tr>'
        '<tr><td>Xp.___</td><td>→ Xp.<strong>shape</strong> &nbsp;·&nbsp; resultado: (569, 2)</td></tr>'
        '<tr><td>CP1 varianza</td><td>→ ~<strong>44.3 %</strong></td></tr>'
        '<tr><td>CP2 varianza</td><td>→ ~<strong>19.0 %</strong></td></tr>'
        '<tr><td>Total 2 comp.</td><td>→ ~<strong>63.3 %</strong> de varianza conservada</td></tr>'
        '</table>'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Scree Plot
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Paso 2 — ¿Cuántas componentes necesitamos?")
    st.markdown(
        "En el Paso 1 usamos 2 componentes para visualizar. "
        "Pero para entrenar un modelo de ML quizás necesitemos más. "
        "El **Scree Plot** muestra cuánta información aporta cada componente."
    )

    col_task2, col_hint2 = st.columns([1, 1], gap="large")
    with col_task2:
        st.markdown(
            '<div class="task-box">'
            '<strong>Tu misión:</strong><br>'
            '1. Ajustar PCA <strong>sin límite</strong> de componentes (calcula las 30)<br>'
            '2. Calcular la varianza <strong>acumulada</strong> con <code>np.cumsum()</code><br>'
            '3. Encontrar cuántas componentes superan el 90 % de varianza<br>'
            '4. Imprimir ese número y el factor de reducción'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_hint2:
        with st.expander("💡 Pista"):
            st.markdown(
                "- `PCA()` sin argumentos calcula todas las componentes posibles (hasta 30 aquí)\n"
                "- `np.cumsum(arr)` → suma acumulada: `[0.44, 0.63, 0.72, ...]`\n"
                "- `np.searchsorted(cum_evr, 0.90)` → primer índice donde la suma supera 0.90\n"
                "- Suma `+1` porque los índices empiezan en 0"
            )

    code_p2 = st.text_area(
        "📝 Código",
        """\
# 1. PCA con todas las componentes (sin n_components)
pca_full = PCA(random_state=42)
pca_full.fit(___)                  # ajustar a X

# 2. Varianza individual y acumulada
evr_all = pca_full.explained_variance_ratio_
cum_evr = np.___(evr_all)          # cumsum → suma acumulada

# 3. ¿Con cuántas componentes llegamos al 90%?
n_needed = int(np.searchsorted(___, 0.90) + 1)
print(f"Componentes para 90% varianza: {n_needed}")
print(f"Reducción: 30 → {n_needed} dimensiones  ({30/n_needed:.1f}× menos)")
print(f"Varianza acumulada con {n_needed} componentes: {cum_evr[n_needed-1]*100:.1f}%")
""",
        height=230, key="code_p2",
    )

    _SOL_P2 = """\
pca_full = PCA(random_state=42)
pca_full.fit(X)
evr_all = pca_full.explained_variance_ratio_
cum_evr = np.cumsum(evr_all)
n_needed = int(np.searchsorted(cum_evr, 0.90) + 1)
print(f"Componentes para 90% varianza: {n_needed}")
print(f"Reduccion: 30 -> {n_needed} dimensiones  ({30/n_needed:.1f}x menos)")
print(f"Varianza acumulada con {n_needed} componentes: {cum_evr[n_needed-1]*100:.1f}%")
"""
    run_p2 = st.button("▶ Ejecutar y graficar", key="run_p2", type="primary")

    if run_p2:
        stdout, err, lns, used_sol = run_user_code(code_p2, solution=_SOL_P2)
        if err:
            st.error("Error inesperado — revisa el código:\n```\n" + err + "\n```")
        else:
            if used_sol:
                st.info("💡 Aún hay huecos sin rellenar — aquí tienes el resultado correcto:")
            else:
                st.success("✅ ¡Correcto! Tu código funciona perfectamente.")
            if stdout:
                st.code(stdout, language="text")
            if "cum_evr" in lns:
                st.session_state["scree_cum"] = lns["cum_evr"]
                st.session_state["scree_evr"] = lns["evr_all"]
                st.session_state["scree_n"]   = lns.get("n_needed")

    if "scree_cum" in st.session_state:
        cum_s  = st.session_state["scree_cum"]
        evr_s2 = st.session_state["scree_evr"]
        n_s    = st.session_state["scree_n"]
        nc = min(20, len(evr_s2))
        fig2 = go.Figure()
        fig2.add_bar(
            x=list(range(1, nc + 1)), y=evr_s2[:nc] * 100,
            name="Varianza individual", marker_color="#6C63FF", opacity=0.85,
        )
        fig2.add_scatter(
            x=list(range(1, nc + 1)), y=cum_s[:nc] * 100,
            name="Varianza acumulada",
            mode="lines+markers",
            line=dict(color="#F59E0B", width=2.5),
            marker=dict(size=7),
        )
        fig2.add_hline(y=90, line_dash="dash", line_color="#10B981",
                       annotation_text="90%", annotation_position="right")
        if n_s and n_s <= nc:
            fig2.add_vline(x=n_s, line_dash="dot", line_color="#FF6B6B",
                           annotation_text=f"  CP {n_s}",
                           annotation_position="top right")
        fig2.update_layout(
            template="plotly_dark", height=380,
            xaxis_title="Número de componentes",
            yaxis_title="Varianza (%)",
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=30, r=55, t=20, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)
        if n_s:
            st.success(
                f"✅ Con **{n_s} componentes** se captura el 90% de la varianza "
                f"— de 30 a {n_s} dimensiones ({30/n_s:.1f}× de reducción)."
            )
        st.markdown(
            '<div class="hint-box">'
            '🔍 <strong>Compara con Iris:</strong> Iris necesitaba solo 2 componentes '
            'para el 97%. Aquí se necesitan ~7–8 — los datos son más complejos. '
            '<strong>Implicación práctica:</strong> si entrenas un clasificador con 7 columnas en lugar de 30, '
            'entrenas 4× más rápido con casi la misma información.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="respuesta-box">'
        '<div class="rb-title">🔑 Respuestas correctas — Paso 2</div>'
        '<table>'
        '<tr><td>pca_full.fit(___)</td><td>→ <strong>X</strong> (datos escalados)</td></tr>'
        '<tr><td>np.___(evr_all)</td><td>→ np.<strong>cumsum</strong>(evr_all)</td></tr>'
        '<tr><td>np.searchsorted(___, 0.90)</td><td>→ <strong>cum_evr</strong></td></tr>'
        '<tr><td>Componentes para 90% varianza</td><td>→ <strong>7</strong> componentes</td></tr>'
        '<tr><td>Factor de reducción</td><td>→ <strong>4.3×</strong> menos dimensiones (30 → 7)</td></tr>'
        '<tr><td>Varianza acumulada</td><td>→ ~<strong>90.6 %</strong></td></tr>'
        '</table>'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — t-SNE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🌌 Paso 3 — t-SNE: ¿mejora la separación?")
    st.markdown(
        "t-SNE es no lineal y suele revelar clústeres más nítidos que PCA. "
        "Vas a escribir el código, elegir la perplejidad y comparar los resultados lado a lado."
    )

    col_ctrl3, col_task3 = st.columns([1, 1], gap="large")
    with col_ctrl3:
        perp = st.slider("Perplejidad a usar en el código", 5, 80, 30, step=5, key="p3_perp")
        n_iter_val = st.select_slider(
            "Iteraciones", options=[250, 500, 750, 1000], value=500, key="p3_iter"
        )
        st.markdown(
            '<div class="hint-box">'
            '💡 <strong>Perplejidad</strong> = nº de vecinos que cada punto considera. '
            'Baja (5–15): grupos compactos y separados artificialmente. '
            'Alta (50–80): visión más global, grupos más difusos.'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_task3:
        st.markdown(
            '<div class="task-box">'
            '<strong>Tu misión:</strong><br>'
            '1. Crear <code>TSNE(n_components=___)</code> con 2 componentes<br>'
            '2. Llamar a <code>.fit_transform(___)</code> sobre <code>X</code><br>'
            '3. Comparar el resultado con el PCA del Paso 1'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.expander("💡 Pista"):
            st.markdown(
                "- `TSNE(n_components=2, perplexity=30, max_iter=500, random_state=42, init='pca')`\n"
                "- `Xt = tsne.fit_transform(X)` — ojo: t-SNE solo tiene `fit_transform`, no `transform`\n"
                "- El resultado tiene shape `(569, 2)` igual que PCA"
            )

    code_p3 = st.text_area(
        "📝 Código",
        f"""\
# Crear y ejecutar t-SNE (tarda ~15 s, es normal)
tsne = TSNE(
    n_components=___,
    perplexity={perp},
    max_iter={n_iter_val},
    random_state=42,
    init="pca",
)
Xt = tsne.___(X)          # fit_transform

print("Shape Xt:", Xt.___)    # shape
print("Rango eje 1:", round(Xt[:,0].min(), 1), "a", round(Xt[:,0].max(), 1))
""",
        height=230, key="code_p3",
    )

    run_p3 = st.button("▶ Ejecutar t-SNE", key="run_p3", type="primary")
    st.caption("⏱️ Puede tardar 10–20 segundos.")

    cache_key = f"tsne_{perp}_{n_iter_val}"
    _SOL_P3 = f"""\
tsne = TSNE(n_components=2, perplexity={perp}, max_iter={n_iter_val}, random_state=42, init="pca")
Xt = tsne.fit_transform(X)
print("Shape Xt:", Xt.shape)
print("Rango eje 1:", round(Xt[:,0].min(), 1), "a", round(Xt[:,0].max(), 1))
"""
    if run_p3:
        with st.spinner("Calculando t-SNE…"):
            stdout, err, lns, used_sol = run_user_code(
                code_p3, solution=_SOL_P3, extra={"perp": perp, "n_iter_val": n_iter_val}
            )
        if err:
            st.error("Error inesperado — revisa el código:\n```\n" + err + "\n```")
        else:
            if used_sol:
                st.info("💡 Aún hay huecos sin rellenar — aquí tienes el resultado correcto:")
            else:
                st.success("✅ ¡Correcto! Tu código funciona perfectamente.")
            if stdout:
                st.code(stdout, language="text")
            if "Xt" in lns:
                st.session_state[cache_key] = lns["Xt"]

    if cache_key in st.session_state:
        Xt_s = st.session_state[cache_key]
        c_left, c_right = st.columns(2, gap="medium")

        with c_left:
            st.markdown(f"#### 🌌 t-SNE (perp={perp})")
            fig_t = go.Figure()
            for cls, label, col in [(0, "Maligno", "#FF6B6B"), (1, "Benigno", "#10B981")]:
                m = y == cls
                fig_t.add_trace(go.Scatter(
                    x=Xt_s[m, 0], y=Xt_s[m, 1], mode="markers", name=label,
                    marker=dict(color=col, size=7, opacity=0.82,
                                line=dict(width=0.4, color="white")),
                ))
            fig_t.update_layout(
                template="plotly_dark", height=370,
                xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
                legend=dict(bgcolor="rgba(0,0,0,0.3)"),
                margin=dict(l=20, r=10, t=20, b=40),
            )
            st.plotly_chart(fig_t, use_container_width=True)
            st.caption(f"perp={perp} · {n_iter_val} iter")

        with c_right:
            st.markdown("#### 🧩 PCA (referencia Paso 1)")
            pca_ref = PCA(n_components=2, random_state=42)
            Xp_ref  = pca_ref.fit_transform(X)
            v_ref   = pca_ref.explained_variance_ratio_
            fig_p = go.Figure()
            for cls, label, col in [(0, "Maligno", "#FF6B6B"), (1, "Benigno", "#10B981")]:
                m = y == cls
                fig_p.add_trace(go.Scatter(
                    x=Xp_ref[m, 0], y=Xp_ref[m, 1], mode="markers", name=label,
                    marker=dict(color=col, size=7, opacity=0.82,
                                line=dict(width=0.4, color="white")),
                ))
            fig_p.update_layout(
                template="plotly_dark", height=370,
                xaxis_title=f"CP1 ({v_ref[0]*100:.0f}%)",
                yaxis_title=f"CP2 ({v_ref[1]*100:.0f}%)",
                legend=dict(bgcolor="rgba(0,0,0,0.3)"),
                margin=dict(l=20, r=10, t=20, b=40),
            )
            st.plotly_chart(fig_p, use_container_width=True)
            st.caption(f"Varianza total: {(v_ref[0]+v_ref[1])*100:.1f}%")

        st.markdown(
            '<div class="hint-box">'
            '🔍 <strong>¿Qué diferencias ves?</strong> t-SNE crea dos nubes más compactas '
            'y separadas que PCA. Pero sus ejes no tienen interpretación física '
            '(no hay % de varianza) y el resultado cambia con distintas semillas. '
            '<strong>PCA para pipelines de ML, t-SNE para exploración visual.</strong>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="respuesta-box">'
        '<div class="rb-title">🔑 Respuestas correctas — Paso 3</div>'
        '<table>'
        '<tr><td>n_components=___</td><td>→ <strong>2</strong></td></tr>'
        '<tr><td>tsne.___(X)</td><td>→ tsne.<strong>fit_transform</strong>(X)</td></tr>'
        '<tr><td>Xt.___</td><td>→ Xt.<strong>shape</strong> &nbsp;·&nbsp; resultado: (569, 2)</td></tr>'
        '<tr><td>Perplejidad recomendada</td><td>→ <strong>30</strong> (valor por defecto equilibrado)</td></tr>'
        '<tr><td>¿t-SNE puede transformar datos nuevos?</td><td>→ <strong>No</strong> — solo <code>fit_transform</code>, sin <code>transform</code></td></tr>'
        '<tr><td>¿Separa mejor que PCA?</td><td>→ <strong>Sí visualmente</strong>, pero sin interpretación cuantitativa de ejes</td></tr>'
        '</table>'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Soluciones
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### ✅ Soluciones completas y explicaciones")
    st.markdown(
        "Despliega cada apartado **solo cuando hayas intentado el ejercicio**. "
        "No hay trampa en mirar las soluciones — lo importante es entender el razonamiento."
    )

    with st.expander("📋 Solución Ejercicio 0 — Explorar el dataset"):
        st.code("""\
# 1. Shape
print("Shape de X:", X.shape)            # (569, 30)

# 2. Recuento por clase
print("Malignos (y==0):", (y == 0).sum()) # 212
print("Benignos (y==1):", (y == 1).sum()) # 357

# 3. Media de la primera variable
var0 = X_raw[:, 0]
print(f"Media '{feat[0]}' — malignos: {var0[y==0].mean():.2f}")  # ~17.46
print(f"Media '{feat[0]}' — benignos: {var0[y==1].mean():.2f}")  # ~12.15
""", language="python")
        st.markdown(
            '<div class="answer-box">'
            '💬 <strong>Por qué importa:</strong> el radio medio es ~44% mayor en malignos. '
            'Eso significa que CP1 de PCA (la dirección de máxima varianza) '
            'estará muy correlacionada con el tamaño del núcleo — '
            'de ahí que separe tan bien las clases en el gráfico.'
            '</div>',
            unsafe_allow_html=True,
        )

    with st.expander("🧩 Solución Paso 1 — PCA"):
        st.code("""\
pca = PCA(n_components=2, random_state=42)
Xp  = pca.fit_transform(X)

evr = pca.explained_variance_ratio_
print(f"CP1: {evr[0]*100:.1f}%")          # ~44.3%
print(f"CP2: {evr[1]*100:.1f}%")          # ~19.0%
print(f"Total: {(evr[0]+evr[1])*100:.1f}%") # ~63.3%

print("Shape de Xp:", Xp.shape)           # (569, 2)
""", language="python")
        st.markdown(
            '<div class="answer-box">'
            '<strong>Interpretación:</strong> con 2 números por biopsia capturamos el ~63% '
            'de la información de 30 variables — suficiente para ver la tendencia, '
            'pero no para diagnosticar. CP1 sola captura el 44%: hay una dirección '
            'dominante de variación (el tamaño del núcleo).'
            '</div>',
            unsafe_allow_html=True,
        )

    with st.expander("📊 Solución Paso 2 — Scree Plot"):
        st.code("""\
pca_full = PCA(random_state=42)
pca_full.fit(X)

evr_all = pca_full.explained_variance_ratio_
cum_evr = np.cumsum(evr_all)

n_needed = int(np.searchsorted(cum_evr, 0.90) + 1)
print(f"Componentes para 90% varianza: {n_needed}")       # 7
print(f"Reducción: 30 → {n_needed} ({30/n_needed:.1f}× menos)")
print(f"Varianza acumulada: {cum_evr[n_needed-1]*100:.1f}%")
""", language="python")
        st.markdown(
            '<div class="answer-box">'
            '<strong>Resultado:</strong> ~7 componentes para el 90% '
            '(Iris solo necesitaba 2 para el 97%). '
            'Los datos de cáncer son más complejos y las 30 variables '
            'tienen correlaciones más débiles entre sí. '
            '<strong>Aplicación:</strong> entrena tu clasificador con 7 columnas en vez de 30 '
            '→ 4× más rápido, casi la misma precisión.'
            '</div>',
            unsafe_allow_html=True,
        )

    with st.expander("🌌 Solución Paso 3 — t-SNE"):
        st.code("""\
tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=500,
    random_state=42,
    init="pca",
)
Xt = tsne.fit_transform(X)

print("Shape Xt:", Xt.shape)       # (569, 2)
print("Rango eje 1:", round(Xt[:,0].min(),1), "a", round(Xt[:,0].max(),1))
""", language="python")
        st.markdown(
            '<div class="answer-box">'
            '<strong>Conclusión:</strong> t-SNE crea una separación más nítida que PCA '
            'porque captura relaciones no lineales. Pero no reporta % de varianza, '
            'no es reproducible sin fijar seed, y <strong>no puede transformar datos nuevos</strong>. '
            'Regla: PCA para pipelines de producción, t-SNE para exploración.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### 🏁 Reflexión final")
    reflexion = st.radio(
        "Tienes un dataset clínico de 50 variables y quieres entrenar un clasificador rápido "
        "y visualizar los datos. ¿Cuál sería tu flujo de trabajo?",
        [
            "Entrenar directamente con las 50 variables sin reducir nada",
            "PCA → reducir a ~10 componentes → entrenar + t-SNE para visualizar ✅",
            "Solo t-SNE para todo — visualizar y también entrenar",
            "Eliminar variables manualmente hasta quedarse con 5",
        ],
        index=None,
        key="quiz_final",
    )
    if reflexion:
        if "PCA →" in reflexion:
            st.success(
                "🎉 ¡Exacto! Este es el flujo estándar en ciencia de datos: "
                "PCA reduce dimensiones de forma reproducible para el modelo, "
                "y t-SNE genera una visualización exploratoria potente. "
                "Son complementarios, no excluyentes."
            )
        elif "50 variables" in reflexion:
            st.warning(
                "⚠️ Funciona, pero con 50 variables hay mucha redundancia. "
                "Reducir a ~10 componentes PCA conserva el 90% de la información "
                "y entrena mucho más rápido."
            )
        elif "Solo t-SNE" in reflexion:
            st.error(
                "❌ t-SNE no puede transformar datos nuevos — no sirve para entrenar modelos. "
                "Su uso es exclusivamente exploratorio."
            )
        else:
            st.error(
                "❌ Eliminar variables manualmente introduce sesgo y pierde correlaciones. "
                "PCA lo hace objetivamente, basado en los propios datos."
            )

st.divider()
st.markdown(
    "<p style='text-align:center;color:#4B5563;font-size:.88rem'>"
    "Dataset: Breast Cancer Wisconsin (Diagnostic) · UCI ML Repository · "
    "Wolberg, W.H. et al. (1993)"
    "</p>",
    unsafe_allow_html=True,
)
