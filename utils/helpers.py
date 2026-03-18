import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import base64
import os


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


def _logo_base64() -> str:
    """Devuelve el logo personal codificado en base64 para embeber en HTML.

    Prueba varias rutas candidatas para ser compatible con ejecución local,
    Streamlit Cloud (/mount/src/...) y cualquier otro entorno.
    """
    candidates = [
        # 1) Relativo al propio helpers.py  →  utils/../assets/images/
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "assets", "images", "logo-pal.png"),
        # 2) Relativo al directorio de trabajo actual (Streamlit Cloud usa cwd = repo root)
        os.path.join(os.getcwd(), "assets", "images", "logo-pal.png"),
        # 3) Ruta hardcoded conocida en Streamlit Cloud
        "/mount/src/reduccion-dimensionalidad/assets/images/logo-pal.png",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return ""


def render_watermark(opacity: float = 1.0, size_px: int = 52) -> None:
    """Renderiza el logo personal en un footer fijo en la parte inferior de la página.

    El footer tiene fondo claro (#e8eef1) para que el logo oscuro (#021017)
    se vea con sus colores originales sin ningún filtro CSS.
    Llama esta función UNA vez por página, justo después de st.set_page_config().
    Si el logo no se encuentra o hay cualquier error, falla silenciosamente.
    """
    try:
        b64 = _logo_base64()
    except Exception:
        return
    if not b64:
        return
    st.markdown(
        f"""
        <style>
        /* ── Footer con logo ───────────────────────────────────────────── */
        .site-footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: #e8eef1;
            border-top: 1px solid #c8d6dc;
            padding: 6px 24px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 10px;
            z-index: 9999;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.18);
        }}
        .site-footer img {{
            width: {size_px}px;
            height: auto;
            opacity: {opacity};
            display: block;
            /* sin filtro: colores originales del logo */
        }}
        .site-footer .footer-text {{
            font-size: .75rem;
            color: #3a4a50;
            font-family: 'Inter', sans-serif;
            letter-spacing: .03em;
        }}
        .site-footer .footer-link {{
            font-size: .75rem;
            color: #1a5276;
            font-family: 'Inter', sans-serif;
            text-decoration: none;
            letter-spacing: .03em;
        }}
        .site-footer .footer-link:hover {{
            text-decoration: underline;
        }}
        /* Empuja el contenido principal para que no quede tapado por el footer */
        .main .block-container {{
            padding-bottom: 70px !important;
        }}
        </style>
        <div class="site-footer">
            <img src="data:image/png;base64,{b64}" alt="Logo personal" />
            <span class="footer-text">© 2026 Paloma Gómez</span>
            <a class="footer-link" href="https://www.linkedin.com/in/palomagsal" target="_blank" rel="noopener">&#128279; LinkedIn</a>
            <a class="footer-link" href="https://github.com/Pal-cloud" target="_blank" rel="noopener">&#128025; GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
