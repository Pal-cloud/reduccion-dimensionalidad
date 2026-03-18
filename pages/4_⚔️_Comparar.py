"""Página de comparación — PCA vs t-SNE vs UMAP lado a lado."""
import numpy as np
import pandas as pd
import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import load_dataset, apply_pca, apply_tsne, apply_umap, scatter_2d, render_watermark

st.set_page_config(page_title="Comparar", page_icon="⚔️", layout="wide")
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
.callout-purple { background:#1a0f2e; border:1px solid #7c3aed; border-radius:8px;
    padding:.9rem 1.1rem; color:#c4b5fd; font-size:.93rem; margin-bottom:.6rem; }
.method-card { background:#1E1E2E; border-radius:12px; padding:1rem 1.2rem;
    border-top:3px solid; margin-bottom:.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("# ⚔️ Los 3 algoritmos cara a cara")
st.markdown(
    "Esta página aplica **PCA, t-SNE y UMAP al mismo dataset** y muestra los tres resultados "
    "juntos. Así puedes ver con tus propios ojos en qué se parecen y en qué se diferencian."
)

with st.expander("📖 ¿Cómo leer estas gráficas? (lee esto primero)", expanded=True):
    st.markdown("""
Las tres gráficas muestran **los mismos datos** procesados de tres maneras distintas.

**¿Qué es cada punto?** Cada punto representa una muestra del dataset
(una flor, un vino, un dígito escrito a mano).

**¿Qué significa el color?** El color indica la categoría real a la que pertenece
cada muestra. Por ejemplo: azul = especie Setosa, verde = Versicolor, rojo = Virginica.

**¿Qué miden los ejes X e Y?** Nada concreto. Son dimensiones "inventadas" por el
algoritmo para comprimir la información. Lo que importa NO es qué mide cada eje,
sino **si los puntos del mismo color forman grupos separados**.

**¿Qué buscamos?**
- 🎯 ¿Hay nubes de puntos del mismo color agrupadas? → **El algoritmo funcionó bien**
- 🔀 ¿Los colores están mezclados sin orden? → Las categorías son difíciles de separar
- 🏝️ ¿Hay "islas" bien definidas? → **Estructura de grupos muy clara**
""")

# ── Configuración ──────────────────────────────────────────────────────────────
st.markdown("---")
col_left, col_right = st.columns([1, 3])
with col_left:
    dataset_name = st.selectbox("Elige el dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="cmp_ds")

    with st.expander("⚙️ Ajustes avanzados (opcional)"):
        st.markdown("*Puedes dejar los valores por defecto — funcionan bien para todos los datasets*")
        perp = st.slider("t-SNE · Perplejidad", 5, 80, 30, key="cmp_perp",
                         help="Cuántos vecinos considera cada punto. 30 es un buen valor.")
        n_iter = st.select_slider("t-SNE · Iteraciones", [250, 500, 1000], value=500, key="cmp_iter",
                                  help="Más iteraciones = resultado más preciso pero más lento.")
        nn = st.slider("UMAP · n_neighbors", 2, 80, 15, key="cmp_nn",
                       help="Tamaño del vecindario. 15 es un buen valor por defecto.")
        md = st.slider("UMAP · min_dist", 0.0, 0.99, 0.1, 0.05, key="cmp_md",
                       help="Qué tan apretados quedan los puntos en el mapa.")

    run_btn = st.button("▶️ Comparar los 3 métodos", type="primary", key="cmp_run")

X, y, _, desc = load_dataset(dataset_name)

with col_right:
    if dataset_name == "Iris 🌸":
        st.markdown("""
**📦 Sobre este dataset: Iris (flores)**

Creado en 1936 por el estadístico Ronald Fisher. Es el dataset más famoso del mundo en
Machine Learning, usado para enseñar desde hace 90 años.

**¿Qué contiene?** Medidas de **150 flores** de 3 especies distintas de iris (una planta):
- 📏 Largo del sépalo, ancho del sépalo, largo del pétalo, ancho del pétalo (todo en cm)

**Las 3 categorías (colores en el gráfico):** Setosa · Versicolor · Virginica

**¿Por qué es ideal para aprender?**
La especie Setosa es muy distinta de las otras dos → aparece siempre muy separada.
Versicolor y Virginica se parecen más entre sí → a veces se solapan un poco.
""")
    elif dataset_name == "Vino 🍷":
        st.markdown("""
**📦 Sobre este dataset: Vino (química)**

Análisis químicos de vinos producidos en la región de Barolo, Italia.

**¿Qué contiene?** **178 muestras** de vino de 3 productores distintos, con
**13 medidas químicas** por muestra: nivel de alcohol, acidez, contenido de
magnesio, fenoles totales, flavonoides, intensidad del color, tono, y más.

**Las 3 categorías (colores):** Productor 0 · Productor 1 · Productor 2

**¿Por qué es interesante?**
Con 13 dimensiones es imposible visualizar directamente. Aquí verás cómo
la reducción de dimensionalidad revela que los 3 productores hacen vinos
con características químicas claramente distintas.
""")
    else:
        st.markdown("""
**📦 Sobre este dataset: Dígitos escritos a mano**

Creado para investigación en reconocimiento de escritura. Es una versión
simplificada del famoso dataset MNIST.

**¿Qué contiene?** **1.797 imágenes** de dígitos del 0 al 9, escaneadas
en cuadrículas de 8×8 píxeles. Cada píxel tiene un valor de 0 (blanco)
a 16 (negro).

**Las 10 categorías (colores):** Los dígitos 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

**¿Por qué es impresionante?**
Cada imagen tiene **64 píxeles = 64 dimensiones**. ¡Y aun así t-SNE y UMAP
consiguen separar los 10 dígitos en sólo 2 dimensiones!
""")

cfg_key = (dataset_name, perp, n_iter, nn, md)

if run_btn or "cmp_results" not in st.session_state or st.session_state.get("cmp_cfg") != cfg_key:
    progress = st.progress(0, text="⏳ Paso 1/3 — PCA (instantáneo)…")
    X_pca, _ = apply_pca(X, n_components=2)

    if dataset_name == "Dígitos ✏️" and X.shape[0] > 500:
        idx = np.random.RandomState(42).choice(X.shape[0], 500, replace=False)
        X_s, y_s = X[idx], y[idx]
        X_pca_s = X_pca[idx]
        st.caption("ℹ️ Para Dígitos se usan 500 muestras aleatorias para que t-SNE no tarde demasiado.")
    else:
        X_s, y_s = X, y
        X_pca_s = X_pca

    progress.progress(33, text="⏳ Paso 2/3 — t-SNE (puede tardar ~15 segundos)…")
    X_tsne = apply_tsne(X_s, perplexity=perp, n_iter=n_iter)
    progress.progress(66, text="⏳ Paso 3/3 — UMAP…")
    X_umap = apply_umap(X_s, n_neighbors=nn, min_dist=md)
    progress.progress(100, text="✅ ¡Listo!")
    progress.empty()
    st.session_state["cmp_results"] = (X_pca_s, X_tsne, X_umap, y_s)
    st.session_state["cmp_cfg"] = cfg_key

if "cmp_results" not in st.session_state:
    st.info("👆 Pulsa **Comparar los 3 métodos** para ver los resultados.")
    st.stop()

X_pca_r, X_tsne_r, X_umap_r, y_r = st.session_state["cmp_results"]

st.markdown("---")
st.markdown("## 📊 Resultados")
st.markdown(
    f"El dataset **{dataset_name}** tiene **{X.shape[1]} variables originales**. "
    f"Cada algoritmo las comprime a **2 dimensiones** para poder dibujarlas. "
    f"Los **{len(np.unique(y_r))} colores** representan las {len(np.unique(y_r))} categorías reales."
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        '<div class="method-card" style="border-color:#6C63FF">'
        '<strong style="color:#6C63FF; font-size:1.1rem">🧩 PCA</strong><br>'
        '<span style="color:#9CA3AF;font-size:.85rem">Lineal · Instantáneo · Interpretable</span>'
        '</div>', unsafe_allow_html=True)
    fig_pca = scatter_2d(X_pca_r, y_r, "PCA",
                         x_label="Componente Principal 1 (CP1) →",
                         y_label="Componente Principal 2 (CP2) →")
    fig_pca.update_layout(height=380)
    st.plotly_chart(fig_pca, use_container_width=True)
    st.markdown(
        '<div class="callout-blue">'
        '<strong>📖 Cómo leer este gráfico:</strong><br>'
        'El eje X (CP1) es la dirección donde los datos varían MÁS. '
        'El eje Y (CP2) es la segunda dirección de mayor variación. '
        'PCA sólo hace "cortes rectos" en los datos — si los grupos '
        'se separan aquí, su diferencia es <em>lineal y clara</em>.'
        '</div>', unsafe_allow_html=True)

with c2:
    st.markdown(
        '<div class="method-card" style="border-color:#38BDF8">'
        f'<strong style="color:#38BDF8; font-size:1.1rem">🌌 t-SNE</strong><br>'
        f'<span style="color:#9CA3AF;font-size:.85rem">No lineal · Clusters compactos · Lento</span>'
        '</div>', unsafe_allow_html=True)
    fig_tsne = scatter_2d(X_tsne_r, y_r, f"t-SNE  (perplejidad={perp})",
                          x_label="Dimensión t-SNE 1 →",
                          y_label="Dimensión t-SNE 2 →")
    fig_tsne.update_layout(height=380)
    st.plotly_chart(fig_tsne, use_container_width=True)
    st.markdown(
        '<div class="callout-yellow">'
        '<strong>📖 Cómo leer este gráfico:</strong><br>'
        'Los ejes no tienen significado — sólo importa quién está cerca de quién. '
        't-SNE forma "islas" muy compactas de puntos similares. '
        '⚠️ Importante: que dos islas estén cerca o lejos <strong>NO significa</strong> '
        'que esas categorías sean parecidas. Sólo el tamaño y compacidad de cada isla importa.'
        '</div>', unsafe_allow_html=True)

with c3:
    st.markdown(
        '<div class="method-card" style="border-color:#A78BFA">'
        f'<strong style="color:#A78BFA; font-size:1.1rem">🚀 UMAP</strong><br>'
        f'<span style="color:#9CA3AF;font-size:.85rem">No lineal · Rápido · Preserva estructura global</span>'
        '</div>', unsafe_allow_html=True)
    fig_umap = scatter_2d(X_umap_r, y_r, f"UMAP  (vecinos={nn}, min_dist={md})",
                          x_label="UMAP Dimensión 1 →",
                          y_label="UMAP Dimensión 2 →")
    fig_umap.update_layout(height=380)
    st.plotly_chart(fig_umap, use_container_width=True)
    st.markdown(
        '<div class="callout-purple">'
        '<strong>📖 Cómo leer este gráfico:</strong><br>'
        'Similar a t-SNE pero con una ventaja: aquí la <strong>posición relativa '
        'entre grupos SÍ tiene significado</strong>. Grupos cercanos en el mapa '
        'son más parecidos entre sí en los datos originales. '
        'También es mucho más rápido que t-SNE.'
        '</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("## 🗺️ Guía rápida: ¿cuándo usar cada método?")

df_guide = pd.DataFrame({
    "¿Cuál es tu situación?": [
        "Quiero ver rápidamente si mis datos tienen grupos",
        "Voy a entrenar un modelo ML y necesito preprocesar",
        "Mi dataset tiene más de 50.000 filas (muy grande)",
        "Necesito aplicar la reducción a datos nuevos en el futuro",
        "Quiero la visualización más bonita de clusters",
        "Mis datos tienen formas curvas o no lineales",
        "Necesito resultados 100% reproducibles siempre",
        "Quiero entender qué variables originales importan más",
    ],
    "🧩 PCA": ["✅ Sí", "✅✅ Ideal", "✅✅ Sí", "✅ Sí", "⚠️ Regular", "❌ No", "✅✅ Sí", "✅✅ Sí"],
    "🌌 t-SNE": ["✅✅ Sí", "❌ No sirve", "❌ Muy lento", "❌ No puede", "✅✅✅ Ideal", "✅✅✅ Sí", "⚠️ Variable", "❌ No"],
    "🚀 UMAP": ["✅✅ Sí", "✅✅ Sí", "✅✅✅ Ideal", "✅✅ Sí", "✅✅ Muy bueno", "✅✅✅ Sí", "✅✅ Con semilla fija", "❌ No"],
})
st.dataframe(df_guide, use_container_width=True, hide_index=True)

st.markdown(
    '<div class="callout-green">💡 <strong>Consejo para empezar:</strong> '
    'Prueba siempre PCA primero — es instantáneo y te da una idea rápida. '
    'Si los grupos no se separan bien con PCA, pasa a UMAP. '
    'Usa t-SNE cuando quieras la visualización más impactante y no te importa esperar unos segundos.'
    '</div>', unsafe_allow_html=True)
