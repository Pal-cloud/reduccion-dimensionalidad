import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ── Paleta de colores consistente ─────────────────────────────────────────────
PALETTE = px.colors.qualitative.Vivid


@st.cache_data
def load_dataset(name: str):
    """Carga y estandariza un dataset conocido."""
    if name == "Iris 🌸":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["etiqueta"] = [data.target_names[i] for i in data.target]
        df["clase"] = data.target
        desc = (
            "📦 **Dataset Iris** (botánica, 1936) — "
            "150 flores · 4 dimensiones · 3 especies · "
            "Origen: estudio de Ronald Fisher. "
            "Variables: largo/ancho de sépalo y pétalo (cm). "
            "Ideal para aprender: Setosa es fácilmente separable de las otras dos especies."
        )
    elif name == "Vino 🍷":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["etiqueta"] = [data.target_names[i] for i in data.target]
        df["clase"] = data.target
        desc = (
            "📦 **Dataset Vino** (química, Italia) — "
            "178 muestras · 13 dimensiones · 3 productores · "
            "Origen: análisis químicos de vinos de Barolo, Italia (UCI ML Repository). "
            "Variables: alcohol, acidez, fenoles, color, magnesio y 8 más. "
            "Interesante porque muchas variables están correlacionadas entre sí."
        )
    elif name == "Dígitos ✏️":
        data = load_digits()
        df = pd.DataFrame(data.data)
        df.columns = [f"pixel_{i}" for i in range(df.shape[1])]
        df["etiqueta"] = data.target.astype(str)
        df["clase"] = data.target
        desc = (
            "📦 **Dataset Dígitos** (visión por computador) — "
            "1.797 imágenes · 64 dimensiones · 10 dígitos (0–9) · "
            "Origen: imágenes de dígitos escritos a mano, escaneadas en cuadrículas 8×8 píxeles. "
            "Cada píxel es una dimensión (valor 0–16). "
            "El más impresionante: 64 dimensiones → 2D con grupos perfectamente separados."
        )
    else:
        raise ValueError(f"Dataset desconocido: {name}")

    X = df.drop(columns=["etiqueta", "clase"]).values
    y = df["etiqueta"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, df, desc


def apply_pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca


def apply_tsne(X: np.ndarray, perplexity: int = 30, n_iter: int = 1000) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=42,
        init="pca",
    )
    return tsne.fit_transform(X)


def apply_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    try:
        import umap  # noqa: PLC0415
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
        )
        return reducer.fit_transform(X)
    except ImportError:
        st.warning("UMAP no está instalado. Ejecuta: `pip install umap-learn`")
        return np.zeros((X.shape[0], 2))


def scatter_2d(
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    x_label: str = "Componente 1",
    y_label: str = "Componente 2",
) -> go.Figure:
    df_plot = pd.DataFrame(
        {"x": X_2d[:, 0], "y": X_2d[:, 1], "Clase": labels}
    )
    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="Clase",
        title=title,
        labels={"x": x_label, "y": y_label},
        color_discrete_sequence=PALETTE,
        template="plotly_dark",
        hover_data={"Clase": True, "x": ":.3f", "y": ":.3f"},
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.5, color="white")))
    fig.update_layout(
        legend=dict(title="Clase", bgcolor="rgba(0,0,0,0.3)"),
        title_font_size=16,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


def scatter_3d(X_3d: np.ndarray, labels: np.ndarray, title: str) -> go.Figure:
    df_plot = pd.DataFrame(
        {"x": X_3d[:, 0], "y": X_3d[:, 1], "z": X_3d[:, 2], "Clase": labels}
    )
    fig = px.scatter_3d(
        df_plot,
        x="x", y="y", z="z",
        color="Clase",
        title=title,
        color_discrete_sequence=PALETTE,
        template="plotly_dark",
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return fig
