"""Página PCA — explicación interactiva paso a paso."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import load_dataset, apply_pca, scatter_2d, scatter_3d

st.set_page_config(page_title="PCA", page_icon="🧩", layout="wide")

st.markdown("# 🧩 PCA — Análisis de Componentes Principales")
st.markdown(
    "> *Como encontrar el ángulo perfecto para fotografiar un objeto en 3D: "
    "elige la perspectiva que captura MÁS información posible.*"
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📖 ¿Cómo funciona?", "🎯 Demo interactiva", "📊 Varianza explicada", "🧠 Quiz"]
)

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## ¿Qué hace PCA exactamente?")

    col_text, col_steps = st.columns([1, 1])
    with col_text:
        st.markdown(
            """
            Imagina que tienes una **nube de puntos en 3D** (como estrellas en el espacio).
            PCA busca el **mejor plano 2D** donde proyectar esas estrellas,
            de forma que se pierda la menor información posible.

            ### Los pasos son:
            1. **Centrar los datos** — restar la media para que el origen esté en el centro.
            2. **Calcular la covarianza** — ¿qué variables se mueven juntas?
            3. **Hallar eigenvectores** — las *direcciones principales* de la nube.
            4. **Proyectar** — comprimir los datos a esas direcciones.
            """
        )
        st.success(
            "✅ **Clave:** PCA conserva las direcciones de **máxima varianza**, "
            "es decir, las que más diferencian a los puntos entre sí."
        )
        st.warning(
            "⚠️ **Limitación:** PCA sólo captura relaciones **lineales**. "
            "Si los datos tienen curvas o espirales, no es el mejor método."
        )

    with col_steps:
        # Animación visual sencilla con un scatter + vectores
        np.random.seed(7)
        N = 150
        angle = np.pi / 4
        cov = [[3, 2.5], [2.5, 3]]
        raw = np.random.multivariate_normal([0, 0], cov, N)
        pca_demo = PCA(n_components=2)
        pca_demo.fit(raw)
        v1 = pca_demo.components_[0] * 3
        v2 = pca_demo.components_[1] * 1.2

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=raw[:, 0], y=raw[:, 1],
            mode="markers",
            marker=dict(color="#6C63FF", opacity=0.5, size=7),
            name="Datos originales",
        ))
        # PC1
        fig.add_annotation(ax=0, ay=0, x=v1[0], y=v1[1],
                            axref="x", ayref="y", xref="x", yref="y",
                            arrowhead=3, arrowwidth=3, arrowcolor="#FF6B6B",
                            text="PC1 (máx. varianza)")
        # PC2
        fig.add_annotation(ax=0, ay=0, x=v2[0], y=v2[1],
                            axref="x", ayref="y", xref="x", yref="y",
                            arrowhead=3, arrowwidth=2, arrowcolor="#48CAE4",
                            text="PC2")
        fig.update_layout(
            title="Componentes principales sobre los datos",
            template="plotly_dark",
            height=370,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🎯 Pruébalo con datos reales")

    col_cfg, col_plot = st.columns([1, 2])
    with col_cfg:
        dataset_name = st.selectbox(
            "Dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="pca_ds"
        )
        n_components = st.slider("Número de componentes a visualizar", 2, 3, 2, key="pca_nc")
        show_loadings = st.checkbox("Mostrar loadings (contribución de variables)", key="pca_load")

    X, y, df_orig, desc = load_dataset(dataset_name)
    st.info(desc)

    X_pca, pca_model = apply_pca(X, n_components=n_components)

    with col_plot:
        if n_components == 2:
            var_explained = pca_model.explained_variance_ratio_
            fig = scatter_2d(
                X_pca, y,
                title=f"PCA — {dataset_name}",
                x_label=f"PC1 ({var_explained[0]*100:.1f}% varianza)",
                y_label=f"PC2 ({var_explained[1]*100:.1f}% varianza)",
            )
        else:
            fig = scatter_3d(X_pca, y, title=f"PCA 3D — {dataset_name}")
        st.plotly_chart(fig, use_container_width=True)

    if show_loadings and n_components == 2:
        st.markdown("### Loadings — ¿qué variables influyen en cada componente?")
        feature_names = df_orig.drop(columns=["etiqueta", "clase"]).columns.tolist()
        loadings = pd.DataFrame(
            pca_model.components_[:2].T,
            index=feature_names,
            columns=["PC1", "PC2"],
        )
        fig_load = px.bar(
            loadings.reset_index().melt(id_vars="index"),
            x="index", y="value", color="variable",
            barmode="group",
            title="Contribución de cada variable a los componentes",
            labels={"index": "Variable", "value": "Loading", "variable": "Componente"},
            template="plotly_dark",
            color_discrete_sequence=["#6C63FF", "#48CAE4"],
        )
        fig_load.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_load, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📊 ¿Cuántas componentes necesito?")
    st.markdown(
        "El gráfico de **varianza explicada acumulada** (o *scree plot*) te dice "
        "cuántas componentes son suficientes para representar los datos."
    )

    ds3 = st.selectbox("Dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="pca_var_ds")
    X3, y3, _, _ = load_dataset(ds3)
    max_comp = min(X3.shape[1], 20)
    pca_full = PCA(n_components=max_comp, random_state=42)
    pca_full.fit(X3)

    cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
    ind_var = pca_full.explained_variance_ratio_ * 100
    n_comps = np.arange(1, max_comp + 1)

    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(
        x=n_comps, y=ind_var, name="Varianza individual",
        marker_color="#6C63FF", opacity=0.7,
    ))
    fig_scree.add_trace(go.Scatter(
        x=n_comps, y=cum_var, name="Varianza acumulada",
        mode="lines+markers", line=dict(color="#FF6B6B", width=3),
        marker=dict(size=8),
    ))
    fig_scree.add_hline(y=90, line_dash="dash", line_color="#48CAE4",
                        annotation_text="90% varianza", annotation_position="right")
    fig_scree.update_layout(
        title="Scree Plot — varianza explicada por componente",
        xaxis_title="Número de componentes",
        yaxis_title="Varianza explicada (%)",
        template="plotly_dark",
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
    )
    st.plotly_chart(fig_scree, use_container_width=True)

    n90 = int(np.argmax(cum_var >= 90)) + 1
    st.success(
        f"✅ Con sólo **{n90} componente(s)** se explica el 90% de la varianza "
        f"del dataset *{ds3}* (de {X3.shape[1]} variables originales)."
    )


# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🧠 Comprueba lo que has aprendido")

    questions = [
        {
            "q": "¿Qué conserva PCA al reducir dimensiones?",
            "opts": [
                "La distancia entre todos los pares de puntos",
                "La varianza máxima en las nuevas direcciones",
                "Las relaciones no lineales entre variables",
                "El número de clases del dataset",
            ],
            "ans": 1,
            "exp": "PCA elige las direcciones (componentes) que **maximizan la varianza**, "
                   "es decir, las que más separan los puntos entre sí.",
        },
        {
            "q": "¿Por qué se estandarizan los datos antes de aplicar PCA?",
            "opts": [
                "Para que todos los puntos tengan el mismo color",
                "Para que el algoritmo sea más lento y preciso",
                "Para que variables con escalas grandes no dominen el resultado",
                "No es necesario estandarizar",
            ],
            "ans": 2,
            "exp": "Si una variable mide alturas en centímetros (0-200) y otra en milímetros (0-2000), "
                   "sin estandarizar la segunda dominaría el PCA aunque no fuera más informativa.",
        },
        {
            "q": "¿Cuándo NO es recomendable usar PCA?",
            "opts": [
                "Cuando los datos tienen muchas dimensiones",
                "Cuando las relaciones entre variables son no lineales (curvas, espirales)",
                "Cuando queremos visualizar en 2D",
                "Cuando el dataset tiene pocos puntos",
            ],
            "ans": 1,
            "exp": "PCA sólo captura relaciones **lineales**. Para estructuras curvas o "
                   "manifolds complejos, t-SNE o UMAP son mejores opciones.",
        },
    ]

    for i, item in enumerate(questions):
        st.markdown(f"**Pregunta {i+1}:** {item['q']}")
        choice = st.radio("", item["opts"], key=f"pca_q{i}", index=None)
        if choice is not None:
            if item["opts"].index(choice) == item["ans"]:
                st.success(f"✅ ¡Correcto! {item['exp']}")
            else:
                st.error(f"❌ No exactamente. {item['exp']}")
        st.markdown("---")
