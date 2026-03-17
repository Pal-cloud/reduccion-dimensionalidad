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

# ── CSS compartido ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.section-label {
    font-size:.75rem; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:#6C63FF; margin-bottom:.2rem;
}
.concept-box {
    background:#1E1E2E; border-radius:10px; padding:1rem 1.2rem;
    border-left:4px solid; margin-bottom:.8rem;
}
.step-block {
    display:flex; gap:1rem; align-items:flex-start; margin-bottom:1rem;
}
.step-num {
    background:#6C63FF; color:#fff; font-weight:800; border-radius:50%;
    width:2rem; height:2rem; min-width:2rem;
    display:flex; align-items:center; justify-content:center; font-size:.9rem;
}
.step-body { color:#D1D5DB; font-size:.93rem; line-height:1.55; padding-top:.1rem; }
.callout-green  { background:#052e16; border:1px solid #16a34a; border-radius:8px;
                   padding:.9rem 1.1rem; color:#86efac; font-size:.93rem; }
.callout-yellow { background:#1c1700; border:1px solid #ca8a04; border-radius:8px;
                   padding:.9rem 1.1rem; color:#fde047; font-size:.93rem; }
.callout-blue   { background:#0c1a2e; border:1px solid #2563eb; border-radius:8px;
                   padding:.9rem 1.1rem; color:#93c5fd; font-size:.93rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🧩 PCA — Principal Component Analysis")
st.markdown(
    "> *Like finding the perfect angle to photograph a 3D sculpture: "
    "choose the perspective that captures the MOST detail.*"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📖 How it works", "🎯 Interactive demo", "📊 Variance explained", "🧠 Quiz"]
)

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">Conceptual foundation</p>', unsafe_allow_html=True)
    st.markdown("## What does PCA actually do?")

    col_text, col_plot = st.columns([1, 1], gap="large")

    with col_text:
        st.markdown("""
Imagine a cloud of points floating in 3D space — like a swarm of bees.
You want to take a **photograph** of the swarm that shows as much of it as possible.

PCA finds the **best flat surface (plane)** to project those points onto,
so that the resulting shadow is as **large and spread out** as possible.
The bigger and more spread the shadow, the more information it preserves.
""")

        st.markdown("### The 4-step recipe")
        steps = [
            ("Center the data",
             "Subtract the mean from every variable so the cloud is centred on the origin. "
             "This removes the effect of scale offsets. "
             "<br><code>x_centred = x − mean(x)</code>"),
            ("Compute the covariance matrix",
             "Measure how much each pair of variables moves together. "
             "High covariance = they carry redundant information. "
             "This is the matrix PCA will decompose."),
            ("Find the eigenvectors (principal components)",
             "These are the special directions in which the data varies the most. "
             "PC1 points in the direction of maximum variance. "
             "PC2 is perpendicular to PC1 and captures the next most variance. And so on."),
            ("Project (compress)",
             "Multiply the centred data by the first N eigenvectors. "
             "You get a new dataset with N columns instead of the original hundreds — "
             "while keeping most of the useful information."),
        ]
        for i, (title, body) in enumerate(steps, 1):
            st.markdown(
                f'<div class="step-block">'
                f'<div class="step-num">{i}</div>'
                f'<div class="step-body"><strong style="color:#E5E7EB">{title}</strong>'
                f'<br>{body}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="callout-green">✅ <strong>Key insight:</strong> PCA keeps the directions '
            'of <strong>maximum variance</strong> — the ones that differentiate the data points '
            'the most. Directions with low variance (noise) are discarded.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="callout-yellow">⚠️ <strong>Limitation:</strong> PCA only captures '
            '<strong>linear</strong> relationships. If the data lives on a curved surface '
            '(a spiral, a sphere…), PCA will distort it. Use t-SNE or UMAP instead.</div>',
            unsafe_allow_html=True,
        )

    with col_plot:
        st.markdown("### Live example: the principal components")
        st.markdown(
            "The arrows show the two principal components of a 2D dataset. "
            "**PC1** (red) points toward the direction of maximum spread. "
            "**PC2** (blue) is perpendicular to PC1."
        )
        np.random.seed(7)
        cov = [[3, 2.5], [2.5, 3]]
        raw = np.random.multivariate_normal([0, 0], cov, 200)
        pca_demo = PCA(n_components=2)
        pca_demo.fit(raw)
        v1 = pca_demo.components_[0] * 3.2
        v2 = pca_demo.components_[1] * 1.4

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=raw[:, 0], y=raw[:, 1], mode="markers",
            marker=dict(color="#6C63FF", opacity=0.45, size=7),
            name="Original data",
        ))
        fig.add_annotation(ax=0, ay=0, x=v1[0], y=v1[1],
                            axref="x", ayref="y", xref="x", yref="y",
                            arrowhead=3, arrowwidth=3, arrowcolor="#FF6B6B",
                            font=dict(color="#FF6B6B", size=13),
                            text="  PC1 — max variance")
        fig.add_annotation(ax=0, ay=0, x=v2[0], y=v2[1],
                            axref="x", ayref="y", xref="x", yref="y",
                            arrowhead=3, arrowwidth=2, arrowcolor="#48CAE4",
                            font=dict(color="#48CAE4", size=13),
                            text="  PC2")
        fig.update_layout(
            template="plotly_dark", height=330, showlegend=True,
            xaxis_title="Variable X", yaxis_title="Variable Y",
            margin=dict(l=30, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### What happens when we project?")
        st.markdown(
            "Project the same points onto PC1 only. "
            "Each point becomes a single number on a line — 2D → 1D, "
            "but most of the spread is preserved."
        )
        proj_1d = raw @ pca_demo.components_[0]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=proj_1d, y=np.zeros_like(proj_1d),
            mode="markers",
            marker=dict(color="#FF6B6B", size=8, opacity=0.6,
                        symbol="line-ns", line=dict(width=2, color="#FF6B6B")),
            name="Projected onto PC1",
        ))
        fig2.update_layout(
            template="plotly_dark", height=160,
            xaxis_title="PC1 coordinate",
            yaxis=dict(visible=False),
            margin=dict(l=30, r=20, t=10, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            f"PC1 alone explains **{pca_demo.explained_variance_ratio_[0]*100:.1f}%** "
            "of the total variance of this dataset."
        )

        st.markdown("### Why standardise first?")
        st.markdown(
            '<div class="callout-blue">💡 If one variable is measured in millimetres (0–2000) '
            'and another in kilograms (0–100), PCA will <strong>incorrectly</strong> think '
            'the first variable is 20× more important — just because of the scale. '
            'Standardising (mean=0, std=1) levels the playing field.</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">Hands-on</p>', unsafe_allow_html=True)
    st.markdown("## 🎯 Try PCA on real data")

    col_cfg, col_plot = st.columns([1, 2])
    with col_cfg:
        dataset_name = st.selectbox("Dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="pca_ds")
        n_components = st.slider("Output dimensions (2 or 3)", 2, 3, 2, key="pca_nc")
        show_loadings = st.checkbox("Show loadings (variable contributions)", key="pca_load")
        st.markdown("---")
        st.markdown(
            "**💡 What to look for:**\n"
            "- Do the classes form separate clouds?\n"
            "- How much variance do PC1 + PC2 explain together?\n"
            "- If you switch to 3D, does a third cluster appear?"
        )

    X, y, df_orig, desc = load_dataset(dataset_name)
    st.info(desc)
    X_pca, pca_model = apply_pca(X, n_components=n_components)

    with col_plot:
        if n_components == 2:
            var_explained = pca_model.explained_variance_ratio_
            fig = scatter_2d(
                X_pca, y,
                title=f"PCA — {dataset_name}",
                x_label=f"PC1 ({var_explained[0]*100:.1f}% variance)",
                y_label=f"PC2 ({var_explained[1]*100:.1f}% variance)",
            )
        else:
            fig = scatter_3d(X_pca, y, title=f"PCA 3D — {dataset_name}")
        st.plotly_chart(fig, use_container_width=True)

        total_var = pca_model.explained_variance_ratio_.sum() * 100
        st.metric(
            label="Total variance explained",
            value=f"{total_var:.1f}%",
            delta=f"from {X.shape[1]} original dimensions → {n_components}",
        )

    if show_loadings and n_components == 2:
        st.markdown("---")
        st.markdown("### Loadings — which variables drive each component?")
        st.markdown(
            "A **loading** close to ±1 means that variable strongly influences the component. "
            "Close to 0 means it barely contributes. Negative = inverse relationship."
        )
        feature_names = df_orig.drop(columns=["etiqueta", "clase"]).columns.tolist()
        loadings = pd.DataFrame(
            pca_model.components_[:2].T,
            index=feature_names,
            columns=["PC1", "PC2"],
        )
        col_l, col_r = st.columns(2)
        with col_l:
            fig_load = px.bar(
                loadings.reset_index().melt(id_vars="index"),
                x="index", y="value", color="variable",
                barmode="group",
                title="Variable loadings per component",
                labels={"index": "Variable", "value": "Loading", "variable": "Component"},
                template="plotly_dark",
                color_discrete_sequence=["#6C63FF", "#48CAE4"],
            )
            fig_load.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_load, use_container_width=True)
        with col_r:
            st.markdown("**Raw loading values:**")
            st.dataframe(loadings.style.background_gradient(cmap="RdBu", axis=None),
                         use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-label">How many components?</p>', unsafe_allow_html=True)
    st.markdown("## 📊 The Scree Plot — choosing the right number of components")

    col_exp, col_plot3 = st.columns([1, 1.5], gap="large")
    with col_exp:
        st.markdown("""
The **scree plot** is your compass for choosing how many principal components to keep.

### What to look for:

**The "elbow" rule:** find the point where the bars stop dropping steeply.
Components after the elbow add very little new information — you can drop them.

**The 90% rule:** keep enough components so the cumulative line (red) crosses 90%.
That means you're preserving 90% of all the variance in the data.

### Why not keep all components?

Because the last components mostly capture **noise**, not signal.
Fewer components → simpler model → less overfitting → faster training.
""")
        st.markdown(
            '<div class="callout-green">✅ In practice, 2–3 components are enough for '
            '<strong>visualisation</strong>. For ML preprocessing, aim for 90–95% variance '
            'explained.</div>',
            unsafe_allow_html=True,
        )

    with col_plot3:
        ds3 = st.selectbox("Dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="pca_var_ds")
        X3, y3, _, _ = load_dataset(ds3)
        max_comp = min(X3.shape[1], 20)
        pca_full = PCA(n_components=max_comp, random_state=42)
        pca_full.fit(X3)

        cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
        ind_var = pca_full.explained_variance_ratio_ * 100
        n_comps_arr = np.arange(1, max_comp + 1)

        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(
            x=n_comps_arr, y=ind_var, name="Individual variance",
            marker_color="#6C63FF", opacity=0.75,
        ))
        fig_scree.add_trace(go.Scatter(
            x=n_comps_arr, y=cum_var, name="Cumulative variance",
            mode="lines+markers", line=dict(color="#FF6B6B", width=3),
            marker=dict(size=8),
        ))
        fig_scree.add_hline(y=90, line_dash="dash", line_color="#48CAE4",
                            annotation_text="90% threshold",
                            annotation_position="bottom right")
        fig_scree.update_layout(
            title="Scree Plot",
            xaxis_title="Number of components",
            yaxis_title="Variance explained (%)",
            template="plotly_dark",
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=30, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_scree, use_container_width=True)

        n90 = int(np.argmax(cum_var >= 90)) + 1
        st.success(
            f"✅ Only **{n90} component(s)** needed to explain 90% of the variance "
            f"in *{ds3}* — down from {X3.shape[1]} original variables."
        )

        # Summary table
        df_var = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(max_comp)],
            "Individual (%)": [f"{v:.2f}" for v in ind_var],
            "Cumulative (%)": [f"{v:.2f}" for v in cum_var],
        })
        with st.expander("📋 See full variance table"):
            st.dataframe(df_var, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-label">Test yourself</p>', unsafe_allow_html=True)
    st.markdown("## 🧠 Quiz — check your understanding")
    st.markdown("Answer all 4 questions. Instant feedback after each one.")
    st.markdown("---")

    questions = [
        {
            "q": "What does PCA preserve when reducing dimensions?",
            "opts": [
                "The exact distances between all pairs of points",
                "The directions of maximum variance in the data",
                "Non-linear relationships between variables",
                "The number of classes in the dataset",
            ],
            "ans": 1,
            "exp": "PCA finds the **directions (components) that maximise variance** — "
                   "the ones that spread the points out the most. This is why it's called "
                   "'Principal' Components: they capture the most important variation.",
        },
        {
            "q": "Why do we standardise data before applying PCA?",
            "opts": [
                "So all points get assigned the same colour in the plot",
                "To make the algorithm slower and more precise",
                "So that variables with large numeric scales don't dominate the result",
                "Standardisation is optional and makes no difference",
            ],
            "ans": 2,
            "exp": "If height is in cm (0–200) and salary is in euros (0–100,000), "
                   "PCA would incorrectly treat salary as 500× more important — purely because "
                   "of the scale. Standardising (z-score) gives every variable equal weight.",
        },
        {
            "q": "What is a 'loading' in the context of PCA?",
            "opts": [
                "The number of data points in the dataset",
                "The percentage of variance explained by a component",
                "The contribution (weight) of an original variable to a principal component",
                "The eigenvalue of the covariance matrix",
            ],
            "ans": 2,
            "exp": "A loading tells you how much each original variable contributes to "
                   "a principal component. A loading of +0.9 on 'height' for PC1 means "
                   "taller individuals score higher on PC1.",
        },
        {
            "q": "When is PCA NOT the best tool?",
            "opts": [
                "When the dataset has many dimensions",
                "When you need the reduction to be reproducible",
                "When the relationships between variables are non-linear (curves, spirals)",
                "When you want to visualise in 2D quickly",
            ],
            "ans": 2,
            "exp": "PCA only captures **linear** relationships. For data that lives on "
                   "curved surfaces (manifolds), PCA will squash and distort it. "
                   "Use t-SNE or UMAP for non-linear structures.",
        },
    ]

    score = 0
    answered = 0
    for i, item in enumerate(questions):
        st.markdown(f"**Question {i+1} of {len(questions)}:** {item['q']}")
        choice = st.radio("", item["opts"], key=f"pca_q{i}", index=None)
        if choice is not None:
            answered += 1
            if item["opts"].index(choice) == item["ans"]:
                score += 1
                st.success(f"✅ Correct! {item['exp']}")
            else:
                correct_text = item["opts"][item["ans"]]
                st.error(f"❌ Not quite. The correct answer is: **{correct_text}**\n\n{item['exp']}")
        st.markdown("---")

    if answered == len(questions):
        pct = int(score / len(questions) * 100)
        if pct == 100:
            st.balloons()
            st.success(f"🏆 Perfect score! {score}/{len(questions)} — You've mastered PCA basics!")
        elif pct >= 50:
            st.info(f"👍 Good effort: {score}/{len(questions)}. Review the explanations above and try again.")
        else:
            st.warning(f"📚 {score}/{len(questions)}. Go back to the 'How it works' tab and read more carefully.")
