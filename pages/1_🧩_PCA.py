"""Página PCA — explicación interactiva y visual en español."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import load_dataset, apply_pca, scatter_2d, scatter_3d

st.set_page_config(page_title="PCA", page_icon="🧩", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-label { font-size:.75rem; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:#6C63FF; margin-bottom:.2rem; }
.step-block { display:flex; gap:1rem; align-items:flex-start; margin-bottom:1.1rem; }
.step-num { background:#6C63FF; color:#fff; font-weight:800; border-radius:50%;
    width:2.2rem; height:2.2rem; min-width:2.2rem;
    display:flex; align-items:center; justify-content:center; font-size:.95rem; }
.step-body { color:#D1D5DB; font-size:.94rem; line-height:1.6; padding-top:.15rem; }
.callout-green  { background:#052e16; border:1px solid #16a34a; border-radius:8px;
    padding:.9rem 1.1rem; color:#86efac; font-size:.93rem; margin-bottom:.6rem; }
.callout-yellow { background:#1c1700; border:1px solid #ca8a04; border-radius:8px;
    padding:.9rem 1.1rem; color:#fde047; font-size:.93rem; margin-bottom:.6rem; }
.callout-blue   { background:#0c1a2e; border:1px solid #2563eb; border-radius:8px;
    padding:.9rem 1.1rem; color:#93c5fd; font-size:.93rem; margin-bottom:.6rem; }
.big-metric { background:#1E1E2E; border-radius:12px; padding:1.2rem;
    text-align:center; border:1px solid #374151; }
.big-metric .val { font-size:2.5rem; font-weight:800; color:#6C63FF; }
.big-metric .lbl { font-size:.85rem; color:#9CA3AF; margin-top:.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🧩 PCA — Análisis de Componentes Principales")
st.markdown(
    "> *Como encontrar el mejor ángulo para fotografiar una escultura 3D: "
    "elige la perspectiva que captura el MÁXIMO detalle posible.*"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📖 ¿Cómo funciona?",
    "🎯 Demo interactiva",
    "📊 ¿Cuántas componentes?",
    "🧠 Quiz",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — TEORÍA
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">Fundamento conceptual</p>', unsafe_allow_html=True)
    st.markdown("## ¿Qué hace PCA exactamente?")

    col_text, col_plot = st.columns([1, 1], gap="large")

    with col_text:
        st.markdown("""
Imagina una nube de puntos flotando en el espacio 3D, como un enjambre de abejas.
Quieres tomar una **fotografía** del enjambre que muestre la mayor parte posible.

PCA encuentra la **mejor superficie plana** sobre la que proyectar esos puntos,
de modo que la "sombra" resultante sea lo más **grande y dispersa** posible.
Cuanto más grande y dispersa es la sombra, más información conserva.

> **En una frase:** PCA rota el sistema de coordenadas para que los nuevos ejes
> apunten en las direcciones donde los datos varían más.
""")
        st.markdown("### 📌 La receta en 4 pasos")
        pasos = [
            ("Centrar los datos",
             "Resta la media de cada variable para que la nube quede centrada en el origen. "
             "Esto elimina el efecto de las distintas escalas y posiciones."),
            ("Calcular la matriz de covarianza",
             "Mide cuánto varían conjuntamente cada par de variables. "
             "Covarianza alta = información redundante = se pueden comprimir."),
            ("Hallar los eigenvectores (componentes principales)",
             "Son las direcciones de máxima varianza. CP1 = dirección de máxima varianza. "
             "CP2 es perpendicular a CP1 y captura la siguiente mayor varianza."),
            ("Proyectar (comprimir)",
             "Multiplica los datos por las primeras N componentes. "
             "Obtienes un dataset con N columnas conservando la mayor parte de la información."),
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
            '<div class="callout-green">✅ <strong>Idea clave:</strong> PCA conserva las '
            'direcciones de máxima varianza. Las de poca varianza (ruido) se descartan.'
            '</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="callout-yellow">⚠️ <strong>Limitación:</strong> '
            'PCA sólo captura relaciones lineales. Para datos en superficies curvas '
            'usa t-SNE o UMAP.</div>', unsafe_allow_html=True)

    with col_plot:
        st.markdown("### 📐 Las componentes principales en vivo")
        st.markdown(
            "**CP1** (roja) apunta hacia la máxima dispersión. "
            "**CP2** (azul) es perpendicular a CP1."
        )
        np.random.seed(7)
        cov = [[3, 2.5], [2.5, 3]]
        raw = np.random.multivariate_normal([0, 0], cov, 250)
        pca_demo = PCA(n_components=2)
        pca_demo.fit(raw)
        v1 = pca_demo.components_[0] * 3.5
        v2 = pca_demo.components_[1] * 1.6

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=raw[:, 0], y=raw[:, 1], mode="markers",
            marker=dict(color="#6C63FF", opacity=0.4, size=7), name="Datos"))
        fig.add_annotation(ax=0, ay=0, x=float(v1[0]), y=float(v1[1]),
                           axref="x", ayref="y", xref="x", yref="y",
                           arrowhead=3, arrowwidth=3, arrowcolor="#FF6B6B",
                           font=dict(color="#FF6B6B", size=13), text="  CP1")
        fig.add_annotation(ax=0, ay=0, x=float(v2[0]), y=float(v2[1]),
                           axref="x", ayref="y", xref="x", yref="y",
                           arrowhead=3, arrowwidth=2, arrowcolor="#48CAE4",
                           font=dict(color="#48CAE4", size=13), text="  CP2")
        fig.update_layout(template="plotly_dark", height=300, showlegend=True,
                          xaxis_title="Variable X", yaxis_title="Variable Y",
                          margin=dict(l=30, r=20, t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

        proj_1d = raw @ pca_demo.components_[0]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=proj_1d, y=np.zeros_like(proj_1d), mode="markers",
            marker=dict(color="#FF6B6B", size=8, opacity=0.55,
                        symbol="line-ns", line=dict(width=2, color="#FF6B6B"))))
        fig2.update_layout(template="plotly_dark", height=120,
                           xaxis_title="Coordenada en CP1",
                           yaxis=dict(visible=False),
                           margin=dict(l=30, r=20, t=10, b=40), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            f"CP1 sola explica **{pca_demo.explained_variance_ratio_[0]*100:.1f}%** "
            "de la varianza total."
        )
        st.markdown(
            '<div class="callout-blue">💡 <strong>¿Por qué estandarizar?</strong> '
            'Si una variable está en mm (0-2000) y otra en kg (0-100), PCA pensará '
            'que la primera es 20x más importante sólo por la escala. '
            'Estandarizar (media=0, std=1) nivela el campo de juego.</div>',
            unsafe_allow_html=True)

    st.divider()

    st.markdown("## 🎭 Analogía interactiva: la sombra del enjambre")
    st.markdown(
        "Mueve el ángulo de proyección y observa cómo cambia la sombra. "
        "PCA encuentra automáticamente el ángulo que **maximiza** el tamaño de esa sombra."
    )
    angulo = st.slider("Ángulo de proyección (grados)", 0, 180, 45, 5, key="pca_angle")
    theta = np.radians(angulo)
    direccion = np.array([np.cos(theta), np.sin(theta)])
    proyecciones = raw @ direccion
    puntos_proy = np.outer(proyecciones, direccion)
    varianza_actual = float(np.var(proyecciones))
    varianza_max = float(np.var(raw @ pca_demo.components_[0]))
    pct_actual = min(varianza_actual / varianza_max * 100, 100.0)

    fig_shadow = go.Figure()
    idx_show = np.linspace(0, len(raw) - 1, 60, dtype=int)
    for idx in idx_show:
        fig_shadow.add_trace(go.Scatter(
            x=[raw[idx, 0], puntos_proy[idx, 0]],
            y=[raw[idx, 1], puntos_proy[idx, 1]],
            mode="lines", line=dict(color="#33335A", width=0.8), showlegend=False))
    fig_shadow.add_trace(go.Scatter(
        x=raw[:, 0], y=raw[:, 1], mode="markers",
        marker=dict(color="#6C63FF", opacity=0.45, size=6), name="Datos originales"))
    fig_shadow.add_trace(go.Scatter(
        x=puntos_proy[:, 0], y=puntos_proy[:, 1], mode="markers",
        marker=dict(color="#FF6B6B", opacity=0.7, size=6), name="Sombra"))
    fig_shadow.add_annotation(
        ax=0, ay=0, x=float(direccion[0]) * 4, y=float(direccion[1]) * 4,
        axref="x", ayref="y", xref="x", yref="y",
        arrowhead=3, arrowwidth=3, arrowcolor="#F59E0B",
        text="  Eje", font=dict(color="#F59E0B", size=12))
    fig_shadow.update_layout(template="plotly_dark", height=320,
                             xaxis_title="X", yaxis_title="Y",
                             margin=dict(l=30, r=20, t=20, b=40),
                             legend=dict(bgcolor="rgba(0,0,0,0.3)"))
    st.plotly_chart(fig_shadow, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="big-metric"><div class="val">{angulo}°</div>'
                    f'<div class="lbl">Ángulo actual</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="big-metric"><div class="val">{varianza_actual:.2f}</div>'
                    f'<div class="lbl">Varianza capturada</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="big-metric"><div class="val">{pct_actual:.0f}%</div>'
                    f'<div class="lbl">% del máximo posible</div></div>', unsafe_allow_html=True)

    angulo_opt = int(np.degrees(np.arctan2(
        pca_demo.components_[0, 1], pca_demo.components_[0, 0]))) % 180
    if pct_actual >= 98:
        st.success(f"Encontraste el ángulo óptimo (~{angulo_opt}°). PCA lo encuentra automáticamente.")
    elif pct_actual >= 75:
        st.info(f"Buen ángulo ({pct_actual:.0f}% del máximo). Prueba ~{angulo_opt}° para el 100%.")
    else:
        st.warning(f"Este ángulo captura sólo el {pct_actual:.0f}% del máximo. Mueve hacia ~{angulo_opt}°.")

    st.divider()
    st.markdown("## 📉 ¿Qué información se pierde al comprimir?")
    col_loss1, col_loss2 = st.columns([1, 1.2])
    with col_loss1:
        st.markdown("""
PCA es **óptimo**: de entre todas las proyecciones lineales posibles a 2D,
PCA es la que **menos información pierde**.

- **90%+ de varianza** → excelente compresión
- **70–90%** → aceptable para visualización
- **<70%** → la proyección distorsiona mucho

Para Iris (4D), 2 componentes retienen ~97% de la varianza.
""")
    with col_loss2:
        X_iris, _, _, _ = load_dataset("Iris 🌸")
        pca_iris_demo = PCA(n_components=4, random_state=42)
        pca_iris_demo.fit(X_iris)
        var_acum_iris = np.cumsum(pca_iris_demo.explained_variance_ratio_) * 100
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Bar(
            x=["CP1", "CP2", "CP3", "CP4"],
            y=pca_iris_demo.explained_variance_ratio_ * 100,
            name="Varianza individual",
            marker_color=["#6C63FF", "#48CAE4", "#F59E0B", "#6B7280"]))
        fig_loss.add_trace(go.Scatter(
            x=["CP1", "CP2", "CP3", "CP4"], y=var_acum_iris,
            name="Varianza acumulada",
            mode="lines+markers", line=dict(color="#FF6B6B", width=3),
            marker=dict(size=10)))
        fig_loss.add_hline(y=90, line_dash="dash", line_color="#22C55E",
                           annotation_text=" 90%", annotation_position="right")
        fig_loss.update_layout(template="plotly_dark", height=270,
                               title="Dataset Iris — varianza por componente",
                               yaxis_title="Varianza (%)",
                               margin=dict(l=30, r=60, t=40, b=30),
                               legend=dict(bgcolor="rgba(0,0,0,0.3)"))
        st.plotly_chart(fig_loss, use_container_width=True)
        st.caption(f"Con 2 componentes conservamos el {var_acum_iris[1]:.1f}% de Iris.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — DEMO INTERACTIVA
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">Manos a la obra</p>', unsafe_allow_html=True)
    st.markdown("## 🎯 Prueba PCA con datos reales")

    col_cfg, col_plot = st.columns([1, 2.2])
    with col_cfg:
        dataset_name = st.selectbox("Dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="pca_ds")
        n_components = st.slider("Dimensiones de salida", 2, 3, 2, key="pca_nc")
        show_loadings = st.checkbox("Mostrar loadings (contribuciones de variables)", key="pca_load")
        show_arrows = st.checkbox("Mostrar vectores de componentes", key="pca_arrows")
        st.markdown("---")
        st.markdown("""
**¿Qué observar?**
- ¿Se forman nubes separadas por clase?
- ¿Cuánta varianza capturan CP1 + CP2?
- Cambia a 3D: ¿aparece algún grupo nuevo?
- Con Dígitos: ¿separa los 10 números desde 64D?
""")

    X, y, df_orig, desc = load_dataset(dataset_name)
    st.info(desc)
    X_pca, pca_model = apply_pca(X, n_components=n_components)

    with col_plot:
        if n_components == 2:
            var_exp = pca_model.explained_variance_ratio_
            fig_demo = scatter_2d(
                X_pca, y,
                title=f"PCA — {dataset_name}",
                x_label=f"CP1 ({var_exp[0]*100:.1f}% varianza)",
                y_label=f"CP2 ({var_exp[1]*100:.1f}% varianza)")
            if show_arrows:
                feature_names = df_orig.drop(columns=["etiqueta", "clase"]).columns.tolist()
                scale = float(X_pca[:, 0].std()) * 2.5
                for j, feat in enumerate(feature_names):
                    lx = float(pca_model.components_[0, j]) * scale
                    ly = float(pca_model.components_[1, j]) * scale
                    fig_demo.add_annotation(
                        ax=0, ay=0, x=lx, y=ly,
                        axref="x", ayref="y", xref="x", yref="y",
                        arrowhead=2, arrowwidth=2, arrowcolor="#F59E0B",
                        font=dict(color="#F59E0B", size=10), text=f"  {feat}")
        else:
            fig_demo = scatter_3d(X_pca, y, title=f"PCA 3D — {dataset_name}")
        st.plotly_chart(fig_demo, use_container_width=True)

        total_var = pca_model.explained_variance_ratio_.sum() * 100
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Varianza total explicada", f"{total_var:.1f}%")
        col_m2.metric("Dimensiones originales", X.shape[1])
        col_m3.metric("Dimensiones reducidas", n_components)

    if show_loadings and n_components == 2:
        st.divider()
        st.markdown("### 🔍 Loadings — ¿qué variables impulsan cada componente?")
        st.markdown("""
Un **loading** cercano a ±1 significa que esa variable influye mucho en la componente.
Cercano a 0 = apenas contribuye. Negativo = relación inversa.
""")
        feature_names = df_orig.drop(columns=["etiqueta", "clase"]).columns.tolist()
        loadings = pd.DataFrame(
            pca_model.components_[:2].T,
            index=feature_names, columns=["CP1", "CP2"])

        col_l, col_r = st.columns(2)
        with col_l:
            fig_load = px.bar(
                loadings.reset_index().melt(id_vars="index"),
                x="index", y="value", color="variable", barmode="group",
                title="Contribución de variables a CP1 y CP2",
                labels={"index": "Variable", "value": "Loading", "variable": "Componente"},
                template="plotly_dark",
                color_discrete_sequence=["#6C63FF", "#48CAE4"])
            fig_load.add_hline(y=0, line_color="#4B5563")
            fig_load.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_load, use_container_width=True)
        with col_r:
            fig_heat = px.imshow(
                loadings.T, text_auto=".2f",
                color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                title="Mapa de calor de loadings", template="plotly_dark")
            fig_heat.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_heat, use_container_width=True)

        with st.expander("Ver tabla completa de loadings"):
            st.dataframe(
                loadings.style.background_gradient(cmap="RdBu", axis=None).format("{:.3f}"),
                use_container_width=True)

    st.divider()
    st.markdown("## 🔬 Reconstrucción: ¿qué se pierde exactamente?")
    st.markdown("Comprime los datos a N componentes y luego **reconstruye** al espacio original.")
    n_rec = st.slider("Componentes para reconstruir", 1, min(X.shape[1], 10), 2, key="pca_rec")
    pca_rec_model = PCA(n_components=n_rec, random_state=42)
    X_rec = pca_rec_model.inverse_transform(pca_rec_model.fit_transform(X))
    error_rec = float(np.mean((X - X_rec) ** 2))
    var_rec = pca_rec_model.explained_variance_ratio_.sum() * 100

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Componentes usadas", n_rec)
    col_r2.metric("Varianza conservada", f"{var_rec:.1f}%")
    col_r3.metric("Error MSE", f"{error_rec:.4f}")

    if dataset_name == "Dígitos ✏️":
        st.markdown("### 🖼️ Reconstrucción de dígitos escritos a mano")
        st.markdown("Arriba: imagen original. Abajo: reconstruida con sólo N componentes.")
        n_show = min(8, X.shape[0])
        cols_dig = st.columns(n_show)
        for ci in range(n_show):
            img_orig = X[ci].reshape(8, 8)
            img_rec_ = X_rec[ci].reshape(8, 8)
            combined = np.vstack([img_orig, np.zeros((1, 8)), img_rec_])
            fig_dig = go.Figure(go.Heatmap(z=combined, colorscale="gray", showscale=False))
            fig_dig.update_layout(
                height=140, margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False), yaxis=dict(visible=False))
            with cols_dig[ci]:
                st.plotly_chart(fig_dig, use_container_width=True)
        st.caption(f"Arriba: original — Abajo: reconstruida con {n_rec} componente(s) de 64")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — SCREE PLOT
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-label">Cuántas componentes necesito</p>', unsafe_allow_html=True)
    st.markdown("## 📊 El Scree Plot — tu brújula para elegir N componentes")

    col_exp, col_scree = st.columns([1, 1.5], gap="large")
    with col_exp:
        st.markdown("""
El **scree plot** es la herramienta estándar para decidir cuántas componentes conservar.

**La regla del codo (elbow):**
Busca el punto donde las barras dejan de caer bruscamente.
Las componentes después del codo aportan muy poca información adicional.

**La regla del 90%:**
Conserva las componentes suficientes para que la línea acumulada supere el 90%.

Las últimas componentes capturan principalmente **ruido**, no señal real.
Menos componentes = modelo más simple = menos sobreajuste.
""")
        st.markdown(
            '<div class="callout-green">Para visualización: 2-3 componentes son suficientes.<br>'
            'Para preprocesamiento ML: apunta al 90-95% de varianza.</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="callout-blue">Regla de Kaiser: descarta componentes '
            'con eigenvalue menor a 1.0.</div>',
            unsafe_allow_html=True)

    with col_scree:
        ds3 = st.selectbox("Dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="pca_var_ds")
        X3, _, _, _ = load_dataset(ds3)
        max_comp = min(X3.shape[1], 20)
        pca_full = PCA(n_components=max_comp, random_state=42)
        pca_full.fit(X3)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
        ind_var = pca_full.explained_variance_ratio_ * 100
        eigenvalues = pca_full.explained_variance_
        n_comps_arr = np.arange(1, max_comp + 1)

        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(
            x=n_comps_arr, y=ind_var, name="Varianza individual",
            marker_color="#6C63FF", opacity=0.75))
        fig_scree.add_trace(go.Scatter(
            x=n_comps_arr, y=cum_var, name="Varianza acumulada",
            mode="lines+markers", line=dict(color="#FF6B6B", width=3),
            marker=dict(size=9)))
        fig_scree.add_hline(y=90, line_dash="dash", line_color="#48CAE4",
                            annotation_text=" 90%", annotation_position="bottom right",
                            annotation_font_color="#48CAE4")
        fig_scree.add_hline(y=95, line_dash="dot", line_color="#22C55E",
                            annotation_text=" 95%", annotation_position="bottom right",
                            annotation_font_color="#22C55E")
        fig_scree.update_layout(
            title=f"Scree Plot — {ds3}",
            xaxis_title="Número de componentes",
            yaxis_title="Varianza explicada (%)",
            template="plotly_dark",
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=30, r=60, t=50, b=40))
        st.plotly_chart(fig_scree, use_container_width=True)

        n90 = int(np.argmax(cum_var >= 90)) + 1
        n95 = int(np.argmax(cum_var >= 95)) + 1
        st.success(
            f"Con solo **{n90} componente(s)** se explica el 90% de la varianza en {ds3} "
            f"— reduciendo de **{X3.shape[1]}** variables originales.")
        col_k1, col_k2 = st.columns(2)
        col_k1.metric("Componentes para 90%", n90)
        col_k2.metric("Componentes para 95%", n95)

        df_var = pd.DataFrame({
            "Componente": [f"CP{i+1}" for i in range(max_comp)],
            "Eigenvalue": [f"{v:.3f}" for v in eigenvalues],
            "Varianza individual (%)": [f"{v:.2f}" for v in ind_var],
            "Varianza acumulada (%)": [f"{v:.2f}" for v in cum_var],
        })
        with st.expander("Ver tabla completa de varianza"):
            st.dataframe(df_var, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — QUIZ
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-label">Comprueba lo que sabes</p>', unsafe_allow_html=True)
    st.markdown("## 🧠 Quiz — pon a prueba tu comprensión de PCA")
    st.markdown("4 preguntas con feedback inmediato y explicación detallada.")
    st.markdown("---")

    preguntas = [
        {
            "q": "¿Qué preserva PCA al reducir dimensiones?",
            "opts": [
                "Las distancias exactas entre todos los pares de puntos",
                "Las direcciones de máxima varianza en los datos",
                "Las relaciones no lineales entre variables",
                "El número de clases del dataset",
            ],
            "ans": 1,
            "exp": "PCA encuentra las direcciones que maximizan la varianza, "
                   "las que más dispersan los puntos. Por eso se llaman componentes Principales.",
        },
        {
            "q": "¿Por qué estandarizamos los datos ANTES de aplicar PCA?",
            "opts": [
                "Para que todos los puntos tengan el mismo color en el gráfico",
                "Para hacer el algoritmo más lento y preciso",
                "Para que variables con escalas grandes no dominen el resultado",
                "La estandarización es opcional y no cambia nada",
            ],
            "ans": 2,
            "exp": "Si altura está en cm y salario en euros, PCA trataría el salario como "
                   "mucho más importante sólo por la escala. "
                   "Estandarizar da a cada variable el mismo peso inicial.",
        },
        {
            "q": "¿Qué es un loading en el contexto de PCA?",
            "opts": [
                "El número de puntos en el dataset",
                "El porcentaje de varianza explicada por una componente",
                "La contribución de una variable original a una componente principal",
                "El eigenvalue de la matriz de covarianza",
            ],
            "ans": 2,
            "exp": "Un loading indica cuánto contribuye cada variable original a una componente. "
                   "Un loading de +0.93 en longitud del pétalo para CP1 significa que "
                   "flores con pétalo largo tienen valores altos en CP1.",
        },
        {
            "q": "¿Cuándo NO es PCA la mejor herramienta?",
            "opts": [
                "Cuando el dataset tiene muchas dimensiones",
                "Cuando necesitas que la reducción sea reproducible",
                "Cuando las relaciones entre variables son no lineales (curvas, espirales)",
                "Cuando quieres visualizar rápidamente en 2D",
            ],
            "ans": 2,
            "exp": "PCA sólo captura relaciones lineales. Para datos en superficies curvas, "
                   "PCA los aplana y distorsiona. Usa t-SNE o UMAP en esos casos.",
        },
    ]

    score = 0
    answered = 0
    for i, item in enumerate(preguntas):
        st.markdown(f"**Pregunta {i+1} de {len(preguntas)}:** {item['q']}")
        choice = st.radio("", item["opts"], key=f"pca_q{i}", index=None)
        if choice is not None:
            answered += 1
            if item["opts"].index(choice) == item["ans"]:
                score += 1
                st.success(f"Correcto! {item['exp']}")
            else:
                correcta = item["opts"][item["ans"]]
                st.error(f"No del todo. La respuesta correcta es: **{correcta}**\n\n{item['exp']}")
        st.markdown("---")

    if answered == len(preguntas):
        pct = int(score / len(preguntas) * 100)
        if pct == 100:
            st.balloons()
            st.success(f"Perfecto! {score}/{len(preguntas)} — Dominas los fundamentos de PCA!")
        elif pct >= 50:
            st.info(f"Bien: {score}/{len(preguntas)}. Repasa y vuelve a intentarlo.")
        else:
            st.warning(f"{score}/{len(preguntas)}. Vuelve a la pestana Como funciona.")
