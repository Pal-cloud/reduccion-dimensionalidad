"""Home page — English introduction and navigation hub."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

st.set_page_config(
    page_title="Dimensionality Reduction — Interactive Guide",
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
.compare-table th { color: #6C63FF !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="hero-title">🔭 Dimensionality Reduction</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">An interactive guide to PCA, t-SNE and UMAP — '
    'no equations required</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-badge">'
    '<span class="badge">🐍 Python</span>'
    '<span class="badge">📊 scikit-learn</span>'
    '<span class="badge">🚀 UMAP</span>'
    '<span class="badge">📈 Plotly</span>'
    '<span class="badge">🎛️ Streamlit</span>'
    '<span class="badge">🆓 Open Source</span>'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# WHAT IS IT?  (two columns)
# ══════════════════════════════════════════════════════════════════════════════
col_what, col_live = st.columns([1.1, 1], gap="large")

with col_what:
    st.markdown('<p class="section-label">The core idea</p>', unsafe_allow_html=True)
    st.markdown("## What is Dimensionality Reduction?")
    st.markdown("""
Modern datasets are enormous. A single 28×28 grayscale image has **784 dimensions**.
A gene-expression dataset can have **tens of thousands**. Our brains can only see 3.

**Dimensionality reduction** compresses high-dimensional data into 2D or 3D maps
while preserving the structure that matters most — letting you *see* patterns
that were completely invisible in a spreadsheet.
""")

    # The three everyday analogies
    analogies = [
        ("#6C63FF",
         "🎒 The Suitcase",
         "You don't pack your entire house for a trip. You keep only the essentials. "
         "DR does the same: keeps the most informative features and throws away noise."),
        ("#48CAE4",
         "📷 The Photograph",
         "A sculpture exists in 3D, but a photo is 2D. A great photographer picks the "
         "angle that shows the most detail. PCA finds that exact angle for your data."),
        ("#FF6B6B",
         "🗺️ The City Map",
         "A map is not the city — it's a 2D compression of 3D reality. "
         "t-SNE and UMAP build maps like this for your data, minimising distortion."),
    ]
    for color, title, text in analogies:
        st.markdown(
            f'<div class="concept-box" style="border-left-color:{color}">'
            f'<strong style="color:{color}">{title}</strong>'
            f'<p style="margin:.4rem 0 0 0;color:#D1D5DB;font-size:.92rem">{text}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

with col_live:
    st.markdown('<p class="section-label">Live preview</p>', unsafe_allow_html=True)
    st.markdown("## Iris dataset: 4D → 2D with PCA")
    st.markdown(
        "Below is a real PCA projection of the Iris flower dataset. "
        "150 flowers described by **4 measurements** compressed into **2 dimensions**. "
        "Notice how the three species naturally cluster apart."
    )

    # ── live mini-demo ─────────────────────────────────────────────────────
    iris = load_iris()
    X_scaled = (iris.data - iris.data.mean(axis=0)) / iris.data.std(axis=0)
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X_scaled)
    colors = ["#6C63FF", "#48CAE4", "#FF6B6B"]
    names  = iris.target_names

    fig_home = go.Figure()
    for i, (name, color) in enumerate(zip(names, colors)):
        mask = iris.target == i
        fig_home.add_trace(go.Scatter(
            x=Xp[mask, 0], y=Xp[mask, 1],
            mode="markers",
            name=name.capitalize(),
            marker=dict(color=color, size=9, opacity=0.85,
                        line=dict(width=0.5, color="white")),
        ))
    v = pca.explained_variance_ratio_
    fig_home.update_layout(
        template="plotly_dark",
        height=340,
        xaxis_title=f"PC1 — {v[0]*100:.1f}% variance",
        yaxis_title=f"PC2 — {v[1]*100:.1f}% variance",
        legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(size=12)),
        margin=dict(l=30, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_home, use_container_width=True)
    st.caption(
        f"✅ Just 2 numbers per flower capture "
        f"**{(v[0]+v[1])*100:.1f}%** of all the information in 4 variables."
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# WHY DOES IT MATTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Why it matters</p>', unsafe_allow_html=True)
st.markdown("## Three reasons every data scientist needs this")

r1, r2, r3 = st.columns(3, gap="medium")
reasons = [
    ("👁️", "Visualisation", "#6C63FF",
     "Turn a 500-column spreadsheet into a 2D scatter plot you can understand "
     "in seconds. Spot clusters, outliers and trends instantly."),
    ("⚡", "Speed", "#48CAE4",
     "Machine Learning models train **10×–100× faster** on compressed data. "
     "Less dimensions = fewer parameters = less overfitting."),
    ("🔍", "Discovery", "#FF6B6B",
     "Uncover hidden groups and relationships that are completely invisible "
     "in raw high-dimensional tables. Let the data surprise you."),
]
for col, (icon, title, color, text) in zip([r1, r2, r3], reasons):
    with col:
        st.markdown(
            f'<div class="nav-card" style="border-left: 4px solid {color};">'
            f'<div class="icon">{icon}</div>'
            f'<h3 style="color:{color}">{title}</h3>'
            f'<p>{text}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Explore the app</p>', unsafe_allow_html=True)
st.markdown("## Pick a topic and dive in")
st.markdown(
    "Use the **sidebar** or click any card below to jump to a section. "
    "Each page has theory, an interactive demo, and a quiz."
)

pages = [
    ("🧩", "PCA", "Principal Component Analysis",
     "The classic. Rotate your data to find the directions of maximum variance. "
     "Best for speed, reproducibility, and preprocessing ML pipelines.",
     "Linear", "Fast", "Reproducible", "#6C63FF", "pages/1_🧩_PCA"),
    ("🌌", "t-SNE", "t-distributed Stochastic Neighbor Embedding",
     "Place similar points close together in 2D. Reveals hidden clusters "
     "even in highly non-linear data structures.",
     "Non-linear", "Best for clusters", "Slow on large data", "#48CAE4", "pages/2_🌌_t-SNE"),
    ("🚀", "UMAP", "Uniform Manifold Approximation and Projection",
     "Faster and more scalable than t-SNE. Preserves both local clusters "
     "AND global structure. Can transform new data.",
     "Non-linear", "Fast", "Scalable", "#FF6B6B", "pages/3_🚀_UMAP"),
    ("⚔️", "Compare", "PCA vs t-SNE vs UMAP side by side",
     "Apply all three algorithms to the same dataset simultaneously "
     "and see the differences with your own eyes.",
     "All methods", "Same dataset", "Side by side", "#F59E0B", "pages/4_⚔️_Comparar"),
    ("🎮", "Playground", "Your personal lab",
     "Full control over every parameter. Explore freely, "
     "download results, and read the built-in glossary.",
     "Any algorithm", "Any dataset", "Free exploration", "#22C55E", "pages/5_🎮_Playground"),
]

col_a, col_b = st.columns(2, gap="medium")
for i, (icon, name, full, desc, t1, t2, t3, color, _) in enumerate(pages):
    tag_html = (
        f'<span class="tag" style="background:{color}22;color:{color}">{t1}</span>'
        f'<span class="tag" style="background:#ffffff11;color:#9CA3AF">{t2}</span>'
        f'<span class="tag" style="background:#ffffff11;color:#9CA3AF">{t3}</span>'
    )
    card_html = (
        f'<div class="nav-card" style="border-top: 3px solid {color};">'
        f'<div class="icon">{icon}</div>'
        f'<h3 style="color:{color}">{name} — <span style="font-weight:400;color:#9CA3AF;font-size:.9rem">{full}</span></h3>'
        f'<p>{desc}</p>'
        f'<div class="tags">{tag_html}</div>'
        f'</div>'
    )
    target = col_a if i % 2 == 0 else col_b
    with target:
        st.markdown(card_html, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# HOW TO USE THIS APP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Getting started</p>', unsafe_allow_html=True)
st.markdown("## Suggested learning path")

col_steps, col_datasets = st.columns([1, 1], gap="large")

with col_steps:
    steps = [
        ("Start here", "Read this Home page to understand the big picture."),
        ("Learn PCA", "Go to 🧩 PCA. Read how it works, then try the demo with the Iris dataset."),
        ("Learn t-SNE", "Go to 🌌 t-SNE. Adjust the perplexity slider and watch the clusters change."),
        ("Learn UMAP", "Go to 🚀 UMAP. Compare it mentally with t-SNE — what's different?"),
        ("Compare all three", "Go to ⚔️ Compare and run all three on the same dataset at once."),
        ("Experiment freely", "Go to 🎮 Playground. Try every combination. Check the glossary."),
    ]
    for num, (title, text) in enumerate(steps, 1):
        st.markdown(
            f'<div class="step-row">'
            f'<div class="step-num">{num}</div>'
            f'<div class="step-text"><strong style="color:#E5E7EB">{title}:</strong> {text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

with col_datasets:
    st.markdown("### 📊 Datasets available throughout the app")
    datasets = [
        ("🌸", "Iris", "150 flowers · 4 dimensions · 3 species",
         "The 'Hello World' of ML. Great for beginners — classes are almost linearly separable.",
         "#6C63FF"),
        ("🍷", "Wine", "178 wines · 13 dimensions · 3 producers",
         "Italian wine chemistry. Many correlated variables — PCA shines here.",
         "#48CAE4"),
        ("✏️", "Digits", "1,797 images · 64 dimensions · 10 classes",
         "Handwritten digits (0–9). t-SNE and UMAP separate all 10 groups perfectly from 64D.",
         "#FF6B6B"),
    ]
    for emoji, name, meta, why, color in datasets:
        st.markdown(
            f'<div class="concept-box" style="border-left-color:{color};margin-bottom:.7rem">'
            f'<strong style="color:{color}">{emoji} {name}</strong> '
            f'<span style="color:#6B7280;font-size:.82rem">{meta}</span>'
            f'<p style="margin:.35rem 0 0 0;color:#9CA3AF;font-size:.88rem">{why}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ALGORITHM QUICK REFERENCE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-label">Quick reference</p>', unsafe_allow_html=True)
st.markdown("## Algorithm cheat sheet")

import pandas as pd
df_ref = pd.DataFrame({
    "": ["Type", "Speed", "Preserves global structure", "Preserves local structure",
         "Can transform new data", "Best for", "Main parameter"],
    "🧩 PCA": ["Linear", "⚡⚡⚡ Very fast", "✅ Yes", "⚠️ Partially",
               "✅ Yes", "Preprocessing, fast viz", "n_components"],
    "🌌 t-SNE": ["Non-linear", "🐢 Slow", "⚠️ Often lost", "✅✅ Excellent",
                 "❌ No", "Cluster visualisation", "perplexity"],
    "🚀 UMAP": ["Non-linear", "⚡⚡ Fast", "✅ Yes", "✅✅ Very good",
                "✅ Yes", "Large datasets, pipelines", "n_neighbors, min_dist"],
})
st.dataframe(df_ref, use_container_width=True, hide_index=True)

st.divider()
st.markdown(
    "<p style='text-align:center;color:#4B5563;font-size:.88rem'>"
    "Built with ❤️ using Python · Streamlit · scikit-learn · UMAP-learn · Plotly"
    "</p>",
    unsafe_allow_html=True,
)
