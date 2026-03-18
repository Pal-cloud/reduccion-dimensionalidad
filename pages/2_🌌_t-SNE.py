"""Página t-SNE — explicación interactiva y visual en español."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import load_dataset, apply_tsne, scatter_2d

st.set_page_config(page_title="t-SNE", page_icon="🌌", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-label { font-size:.75rem; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:#38BDF8; margin-bottom:.2rem; }
.step-block { display:flex; gap:1rem; align-items:flex-start; margin-bottom:1.1rem; }
.step-num { background:#38BDF8; color:#0f172a; font-weight:800; border-radius:50%;
    width:2.2rem; height:2.2rem; min-width:2.2rem;
    display:flex; align-items:center; justify-content:center; font-size:.95rem; }
.step-body { color:#D1D5DB; font-size:.94rem; line-height:1.6; padding-top:.15rem; }
.callout-green  { background:#052e16; border:1px solid #16a34a; border-radius:8px;
    padding:.9rem 1.1rem; color:#86efac; font-size:.93rem; margin-bottom:.6rem; }
.callout-yellow { background:#1c1700; border:1px solid #ca8a04; border-radius:8px;
    padding:.9rem 1.1rem; color:#fde047; font-size:.93rem; margin-bottom:.6rem; }
.callout-red    { background:#1c0505; border:1px solid #dc2626; border-radius:8px;
    padding:.9rem 1.1rem; color:#fca5a5; font-size:.93rem; margin-bottom:.6rem; }
.callout-blue   { background:#0c1a2e; border:1px solid #2563eb; border-radius:8px;
    padding:.9rem 1.1rem; color:#93c5fd; font-size:.93rem; margin-bottom:.6rem; }
.big-metric { background:#1E1E2E; border-radius:12px; padding:1.2rem;
    text-align:center; border:1px solid #374151; }
.big-metric .val { font-size:2.5rem; font-weight:800; color:#38BDF8; }
.big-metric .lbl { font-size:.85rem; color:#9CA3AF; margin-top:.2rem; }
.perp-card { background:#1E1E2E; border-radius:10px; padding:1rem;
    border-left:4px solid; text-align:center; }
.read-box {
    background: #0a1a2e; border: 1px solid #0e4d8a;
    border-left: 5px solid #38BDF8; border-radius: 10px;
    padding: 1rem 1.3rem; margin-top: .6rem; margin-bottom: .8rem;
}
.read-box .read-title {
    color: #7dd3fc; font-weight: 700; font-size: .82rem;
    text-transform: uppercase; letter-spacing: .07em; margin-bottom: .5rem;
}
.read-box ul { margin: 0; padding-left: 1.2rem; }
.read-box li { color: #bae6fd; font-size: .9rem; line-height: 1.65; margin-bottom: .2rem; }
.read-box li strong { color: #7dd3fc; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🌌 t-SNE — t-distributed Stochastic Neighbor Embedding")
st.markdown(
    "> *Como organizar una fiesta enorme: sienta juntas a las personas que se parecen "
    "y así los grupos naturales emergen solos.*"
)

tab1, tab2, tab3 = st.tabs([
    "📖 ¿Cómo funciona?",
    "🎯 Demo interactiva",
    "🧠 Quiz",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TEORÍA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">Fundamento conceptual</p>', unsafe_allow_html=True)
    st.markdown("## ¿Qué hace t-SNE exactamente?")

    col_text, col_plot = st.columns([1, 1], gap="large")

    with col_text:
        st.markdown("""
Imagina que tienes una fiesta con **500 invitados** que no se conocen entre sí.
Tu misión: colocarlos en el salón de forma que las personas similares queden cerca.

Mides qué tan parecidos son (misma ciudad, mismo trabajo, mismos gustos) y los distribuyes
en el espacio. Al final, sin que nadie te lo dijera, emergen **grupos naturales** visibles.

Eso es t-SNE: mide similitudes en alta dimensión y redistribuye los puntos en 2D
para que esos grupos queden a la vista.

> **La diferencia clave con PCA:**  
> PCA busca *la mejor proyección lineal*. t-SNE se toma el tiempo de *aprender*
> la estructura local de los datos — aunque sea curva o no lineal.
""")

        st.markdown("### 🪜 El algoritmo en 3 pasos")
        pasos = [
            ("Alta dimensión — medir similitudes",
             "Para cada punto, calcula la probabilidad de que cada vecino sea realmente cercano. "
             "Usa una **campana de Gauss**: vecinos muy cercanos → probabilidad ≈ 1. "
             "Puntos lejanos → probabilidad ≈ 0. "
             "La **perplejidad** controla qué tan amplia es esa campana."),
            ("Baja dimensión — inicio aleatorio",
             "Coloca todos los puntos en posiciones aleatorias en un plano 2D. "
             "Esto es el punto de partida, no el resultado. "
             "Las probabilidades de vecindad en 2D también se calculan, pero con "
             "una **distribución t de Student** (colas más gruesas que la Gaussiana)."),
            ("Optimización — acercar lo similar, alejar lo distinto",
             "Mueve los puntos iterativamente para que las probabilidades de vecindad en 2D "
             "se parezcan lo más posible a las de alta dimensión. "
             "Este proceso se repite **cientos o miles de veces** usando gradiente descendente. "
             "El resultado: clusters bien separados y compactos."),
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
            '<div class="callout-yellow">⚠️ <strong>Trampa frecuente:</strong> '
            'Las distancias <em>entre</em> clusters en t-SNE <strong>no son interpretables</strong>. '
            'Que dos grupos estén cerca o lejos en el mapa 2D no dice nada sobre '
            'si son parecidos entre sí. Sólo la distancia <em>dentro</em> de un cluster tiene sentido.'
            '</div>', unsafe_allow_html=True)

        st.markdown("### 📐 Fórmulas clave (t-SNE)")
        st.markdown(r"""
| Fórmula | Qué calcula | En palabras |
|---------|-------------|-------------|
| $p_{j\|i} = \dfrac{\exp(-\|x_i-x_j\|^2/2\sigma_i^2)}{\sum_{k\neq i}\exp(-\|x_i-x_k\|^2/2\sigma_i^2)}$ | Similitud en alta dimensión | Prob. de que $x_j$ sea vecino de $x_i$. Gauss centrada en cada punto; $\sigma_i$ lo fija la **perplejidad**. |
| $q_{ij} = \dfrac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{k\neq l}(1+\|y_k-y_l\|^2)^{-1}}$ | Similitud en 2D | Lo mismo pero en el espacio reducido. Usa **distribución t** (colas gruesas → clusters más separados). |
| $\mathcal{L} = \sum_{i,j} p_{ij}\log\dfrac{p_{ij}}{q_{ij}}$ | Coste (KL divergence) | Mide diferencia entre las dos distribuciones. t-SNE minimiza esto moviendo los puntos en 2D. |

> 💡 La distribución **t de Student** en 2D (en lugar de Gauss) es el truco que da a t-SNE sus clusters tan nítidos: sus colas gruesas "empujan" los clusters más lejos entre sí.
""")

        st.markdown(
            '<div class="callout-red">🚫 <strong>No uses t-SNE para:</strong> '
            'reducción previa al entrenamiento de un modelo ML, ni para datasets '
            'de más de ~50.000 filas. Es un método de <em>visualización</em>, no de '
            '<em>transformación</em>.</div>', unsafe_allow_html=True)

    with col_plot:
        st.markdown("### 🌀 Datos que PCA no puede separar — pero t-SNE sí")
        st.markdown(
            "Este ejemplo muestra una **espiral doble** (datos no lineales). "
            "PCA sólo puede hacer cortes rectos: mezcla las dos espirales. "
            "t-SNE detecta la estructura curva y las separa."
        )
        np.random.seed(42)
        n_pts = 200
        t_vals = np.linspace(0, 4 * np.pi, n_pts)
        noise = 0.25
        X_s1 = np.column_stack([
            t_vals * np.cos(t_vals) + np.random.randn(n_pts) * noise,
            t_vals * np.sin(t_vals) + np.random.randn(n_pts) * noise,
        ])
        X_s2 = np.column_stack([
            -t_vals * np.cos(t_vals) + np.random.randn(n_pts) * noise,
            -t_vals * np.sin(t_vals) + np.random.randn(n_pts) * noise,
        ])
        X_spiral = np.vstack([X_s1, X_s2])
        y_spiral = np.array(["Espiral A"] * n_pts + ["Espiral B"] * n_pts)

        fig_spiral = go.Figure()
        fig_spiral.add_trace(go.Scatter(
            x=X_s1[:, 0], y=X_s1[:, 1], mode="markers",
            marker=dict(color="#38BDF8", size=5, opacity=0.7),
            name="Espiral A"))
        fig_spiral.add_trace(go.Scatter(
            x=X_s2[:, 0], y=X_s2[:, 1], mode="markers",
            marker=dict(color="#F472B6", size=5, opacity=0.7),
            name="Espiral B"))
        fig_spiral.update_layout(
            template="plotly_dark", height=270,
            title="Espacio original (2D)", title_font_size=13,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(bgcolor="rgba(0,0,0,0.3)"))
        st.plotly_chart(fig_spiral, use_container_width=True)

        st.markdown(
            '<div class="read-box">'
            '<div class="read-title">📖 Cómo leer este gráfico de espirales</div>'
            '<ul>'
            '<li>Este gráfico muestra datos que tienen una <strong>estructura curva</strong>: dos espirales entrelazadas, una azul y una rosa.</li>'
            '<li>Si aplicaras PCA, trazaría una línea recta como eje y <strong>mezclaría los dos colores</strong>: no podría distinguir las espirales porque PCA solo hace cortes rectos.</li>'
            '<li>t-SNE <strong>detecta que los puntos del mismo color son vecinos entre sí</strong> aunque la espiral se curve, y los coloca juntos en el mapa 2D.</li>'
            '<li>Este ejemplo ilustra por qué t-SNE es superior a PCA para datos con <strong>estructuras no lineales</strong> (curvas, anillos, nubes irregulares).</li>'
            '</ul>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="callout-blue">💡 En un dataset real de alta dimensión, '
            'los datos pueden tener esta misma estructura curva y t-SNE la detecta. '
            'PCA simplemente la "aplana" y pierde esa información.</div>',
            unsafe_allow_html=True)

    st.divider()

    # ── Perplejidad explicada visualmente ─────────────────────────────────
    st.markdown("## 🎛️ El parámetro estrella: la Perplejidad")
    st.markdown(
        "La **perplejidad** es el parámetro más importante de t-SNE. "
        "Controla cuántos vecinos considera cada punto al medir similitudes."
    )

    c1, c2, c3 = st.columns(3)
    cards = [
        ("#3B1F5E", "#A78BFA", "Perplejidad baja (5–10)",
         "Cada punto sólo mira a sus 5–10 vecinos más cercanos.",
         "Resultado: muchos clusters pequeños y muy aislados. "
         "Puede crear estructuras falsas que no existen en los datos."),
        ("#0c2a1a", "#34D399", "Perplejidad media (30)",
         "Balance entre vecindad local y estructura global.",
         "✅ El valor recomendado para la mayoría de datasets. "
         "Los clusters emergen con un tamaño natural."),
        ("#1c0a00", "#FB923C", "Perplejidad alta (80–100)",
         "Cada punto considera hasta 100 vecinos.",
         "Resultado: clusters más grandes y conectados. "
         "Puede 'fundir' grupos que en realidad son distintos."),
    ]
    for col, (bg, color, titulo, desc1, desc2) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f'<div style="background:{bg};border-radius:10px;padding:1rem;'
                f'border-left:4px solid {color};height:100%">'
                f'<p style="color:{color};font-weight:700;margin-bottom:.4rem">{titulo}</p>'
                f'<p style="color:#D1D5DB;font-size:.88rem;margin-bottom:.4rem">{desc1}</p>'
                f'<p style="color:#9CA3AF;font-size:.85rem">{desc2}</p>'
                f'</div>',
                unsafe_allow_html=True)

    st.divider()

    # ── Por qué distribución t de Student ─────────────────────────────────
    st.markdown("## 📐 ¿Por qué usar distribución t de Student en 2D?")
    col_t1, col_t2 = st.columns([1, 1.2])
    with col_t1:
        st.markdown("""
Esta es la razón del nombre "**t**-SNE" y es una elección brillante.

En alta dimensión usamos una campana de Gauss (estrecha). En 2D usamos una
distribución **t de Student con 1 grado de libertad** (colas mucho más largas).

**¿Por qué importa esto?**

En alta dimensión, muchos puntos pueden ser "vecinos moderados" de un punto central.
En 2D, el espacio es mucho más pequeño y esos puntos no caben todos cerca.

La distribución t con colas gruesas resuelve esto:
- Permite que los puntos **lejanos se separen MÁS** → clusters bien diferenciados
- Evita que todos los puntos colapsen en el centro
- Genera esas hermosas estructuras de "islas" en los plots de t-SNE
""")
        st.markdown(
            '<div class="callout-green">✅ <strong>Resultado:</strong> '
            'Los clusters en t-SNE son más compactos y mejor separados que '
            'si usáramos Gauss en ambas dimensiones.</div>', unsafe_allow_html=True)

    with col_t2:
        x_dist = np.linspace(-5, 5, 300)
        from scipy.stats import norm, t as t_dist
        y_gauss = norm.pdf(x_dist, 0, 1)
        y_t = t_dist.pdf(x_dist, df=1)

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Scatter(
            x=x_dist, y=y_gauss, mode="lines", name="Gauss (alta dimensión)",
            line=dict(color="#38BDF8", width=3)))
        fig_dist.add_trace(go.Scatter(
            x=x_dist, y=y_t, mode="lines", name="t de Student (2D en t-SNE)",
            line=dict(color="#F472B6", width=3)))
        fig_dist.add_annotation(
            x=2.5, y=0.08, text="Colas más gruesas<br>→ mejor separación",
            font=dict(color="#F472B6", size=11),
            showarrow=True, arrowcolor="#F472B6", ax=40, ay=-40)
        fig_dist.update_layout(
            template="plotly_dark", height=280,
            title="Gauss vs t de Student",
            xaxis_title="Distancia", yaxis_title="Probabilidad",
            legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=20, r=20, t=40, b=30))
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown(
            '<div class="read-box">'
            '<div class="read-title">📖 Cómo leer este gráfico de distribuciones</div>'
            '<ul>'
            '<li>El eje horizontal representa la <strong>distancia</strong> entre dos puntos. El eje vertical, la <strong>probabilidad</strong> de que esa distancia sea "normal" según cada distribución.</li>'
            '<li>La <strong>curva azul (Gaussiana)</strong> cae rápido: asigna probabilidad casi cero a distancias mayores de ±2. Muy "estrecha".</li>'
            '<li>La <strong>curva rosa (t de Student)</strong> tiene las "colas" mucho más anchas: sigue asignando algo de probabilidad a distancias grandes. Por eso se dice que tiene <strong>colas más gruesas</strong>.</li>'
            '<li>En t-SNE se usa la curva rosa en el espacio 2D: esto hace que los puntos que no son vecinos cercanos <strong>se empujen más hacia afuera</strong>, creando los clusters bien separados y compactos que caracterizan a t-SNE.</li>'
            '</ul>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Cuándo usar ────────────────────────────────────────────────────────
    st.markdown("## ✅❌ ¿Cuándo usar t-SNE?")
    col_si, col_no = st.columns(2)
    with col_si:
        st.markdown("### ✅ Úsalo cuando...")
        st.markdown("""
- Quieres **visualizar** si existen grupos naturales en tus datos
- Tienes un dataset de **tamaño medio** (hasta ~50.000 filas)
- Quieres verificar que un **algoritmo de clustering** tiene sentido
- Trabajas en **bioinformática** (scRNA-seq, proteómica)
- Quieres una visualización **impactante** para una presentación
""")
    with col_no:
        st.markdown("### ❌ No lo uses cuando...")
        st.markdown("""
- Necesitas **transformar datos nuevos** (t-SNE no tiene `.transform()`)
- Tu dataset tiene más de **50.000 filas** (demasiado lento)
- Quieres una reducción **antes de entrenar** un modelo ML
- Necesitas que las **distancias entre clusters** sean interpretables
- Necesitas **reproducibilidad perfecta** sin fijar semillas
""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEMO INTERACTIVA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">Manos a la obra</p>', unsafe_allow_html=True)
    st.markdown("## 🎯 Prueba t-SNE con datos reales")

    col_cfg, col_plot = st.columns([1, 2.2])
    with col_cfg:
        dataset_name = st.selectbox("Dataset", ["Iris 🌸", "Vino 🍷", "Dígitos ✏️"], key="tsne_ds")
        perplexity = st.slider("Perplejidad", 5, 80, 30, 5, key="tsne_perp")
        n_iter = st.slider("Iteraciones", 250, 1000, 500, 250, key="tsne_iter")
        st.markdown("---")
        st.markdown("""
**💡 Qué observar:**
- Aumenta la perplejidad: ¿los clusters se fusionan?
- Bájala a 5: ¿aparecen estructuras falsas?
- Prueba con Dígitos: ¿separa los 10 números desde 64D?
- ¿Ves grupos que se solapan? t-SNE los respeta.
""")
        st.markdown(
            '<div class="callout-yellow">⚠️ t-SNE puede tardar unos segundos '
            'en datasets grandes. Los resultados cambian ligeramente cada vez '
            'por el inicio aleatorio.</div>', unsafe_allow_html=True)

    X, y, df_orig, desc = load_dataset(dataset_name)
    st.info(desc)

    with col_plot:
        with st.spinner("🔄 Ejecutando t-SNE... (puede tardar unos segundos)"):
            X_tsne = apply_tsne(X, perplexity=perplexity, n_iter=n_iter)

        fig_tsne = scatter_2d(
            X_tsne, y,
            title=f"t-SNE — {dataset_name} (perp={perplexity}, iter={n_iter})",
            x_label="Dimensión t-SNE 1",
            y_label="Dimensión t-SNE 2")
        st.plotly_chart(fig_tsne, use_container_width=True)

        st.markdown(
            '<div class="read-box">'
            '<div class="read-title">📖 Cómo leer este gráfico t-SNE</div>'
            '<ul>'
            '<li>Cada <strong>punto es una muestra</strong> (flor, vino, imagen). El <strong>color es su clase real</strong> (especie, productor, dígito).</li>'
            '<li><strong>Puntos del mismo color agrupados</strong> = t-SNE ha encontrado que esas muestras son similares entre sí en el espacio original de alta dimensión.</li>'
            '<li>Los <strong>ejes no tienen unidades interpretables</strong>: "Dimensión t-SNE 1" y "Dimensión t-SNE 2" no representan ninguna variable real. Son simplemente coordenadas en el mapa 2D.</li>'
            '<li><strong>⚠️ Importante:</strong> la distancia ENTRE grupos (clusters) no tiene significado. Que dos grupos estén cerca o lejos en este mapa no dice si son parecidos entre sí. Solo la distancia DENTRO de un grupo es interpretable.</li>'
            '<li>Si ves nubes de colores bien separadas → buena separación. Si los colores se mezclan → esas clases son difíciles de distinguir incluso para t-SNE.</li>'
            '</ul>'
            '</div>',
            unsafe_allow_html=True,
        )

        n_clusters = len(np.unique(y))
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Dimensiones originales", X.shape[1])
        col_m2.metric("Dimensiones reducidas", 2)
        col_m3.metric("Clases en el dataset", n_clusters)

    st.divider()

    # ── Comparativa de perplejidades ───────────────────────────────────────
    st.markdown("## 📊 ¿Cómo cambia el resultado con la perplejidad?")
    st.markdown(
        "Compara el mismo dataset con **tres perplejidades distintas** lado a lado. "
        "Observa cómo la estructura cambia radicalmente."
    )

    ds_comp = st.selectbox("Dataset para comparar", ["Iris 🌸", "Vino 🍷"], key="tsne_comp_ds")
    X_c, y_c, _, _ = load_dataset(ds_comp)

    if st.button("🔄 Generar comparativa de perplejidades", key="tsne_comp_btn"):
        perp_vals = [5, 30, 80]
        cols_comp = st.columns(3)
        for col, pv in zip(cols_comp, perp_vals):
            with col:
                with st.spinner(f"Calculando perp={pv}..."):
                    Xr = apply_tsne(X_c, perplexity=pv, n_iter=500)
                fig_c = scatter_2d(Xr, y_c, title=f"Perplejidad = {pv}",
                                   x_label="Dim 1", y_label="Dim 2")
                fig_c.update_layout(height=300, showlegend=False,
                                    margin=dict(l=10, r=10, t=40, b=20))
                st.plotly_chart(fig_c, use_container_width=True)
                if pv == 5:
                    st.caption("⚠️ Muchos clusters pequeños — posibles artefactos")
                elif pv == 30:
                    st.caption("✅ Balance óptimo — recomendado")
                else:
                    st.caption("⚠️ Clusters grandes — puede perder detalle local")
    else:
        st.info("👆 Pulsa el botón para generar la comparativa (tarda ~20 segundos)")

    st.markdown(
        '<div class="read-box">'
        '<div class="read-title">📖 Cómo leer la comparativa de perplejidades</div>'
        '<ul>'
        '<li>Los tres gráficos muestran <strong>exactamente los mismos datos</strong>, procesados con tres valores distintos del parámetro "perplejidad".</li>'
        '<li>Con <strong>perplejidad baja (5)</strong>: cada punto solo mira a sus 5 vecinos más cercanos. Resultado: muchos grupos pequeños y dispersos. Pueden aparecer estructuras que son artefactos, no datos reales.</li>'
        '<li>Con <strong>perplejidad media (30)</strong>: balance entre detalles locales y estructura global. Es el valor más recomendado para comenzar.</li>'
        '<li>Con <strong>perplejidad alta (80)</strong>: cada punto considera a muchos vecinos. Los grupos se "fusionan" y se pierde el detalle interno.</li>'
        '<li><strong>Conclusión práctica:</strong> si ves algo interesante en el mapa, verifica que aparezca también con al menos dos valores distintos de perplejidad. Si solo aparece con uno, puede ser un artefacto.</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Métricas de calidad ───────────────────────────────────────────────
    st.markdown("## 📏 ¿Qué tan buena es la reducción?")
    st.markdown("""
A diferencia de PCA, t-SNE no tiene una métrica directa de "varianza explicada".
Pero podemos evaluar la calidad de forma indirecta:
""")

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown("### 🎯 Preservación de vecindad")
        st.markdown(
            "Calculamos un clasificador KNN en el espacio reducido. "
            "Si t-SNE ha preservado bien la estructura, KNN funcionará bien."
        )
        try:
            knn = KNeighborsClassifier(n_neighbors=5)
            y_num = np.array([list(np.unique(y)).index(yi) for yi in y])
            scores = cross_val_score(knn, X_tsne, y_num, cv=5, scoring="accuracy")
            acc = scores.mean() * 100
            st.markdown(
                f'<div class="big-metric"><div class="val">{acc:.1f}%</div>'
                f'<div class="lbl">Precisión KNN-5 en espacio reducido (CV-5)</div></div>',
                unsafe_allow_html=True)
            if acc >= 90:
                st.success("✅ Excelente preservación de la estructura de clases")
            elif acc >= 70:
                st.info("👍 Buena preservación — clusters bien formados")
            else:
                st.warning("⚠️ Prueba ajustando la perplejidad")
        except Exception as e:
            st.error(f"Error calculando métricas: {e}")

    with col_q2:
        st.markdown("### 📊 Compacidad de clusters")
        st.markdown(
            "Medimos la distancia media intra-cluster vs inter-cluster. "
            "Un buen embedding tiene clusters compactos y bien separados."
        )
        try:
            unique_labels = np.unique(y)
            intra_dists = []
            centroids = []
            for lbl in unique_labels:
                pts = X_tsne[y == lbl]
                centroid = pts.mean(axis=0)
                centroids.append(centroid)
                intra_dists.append(np.mean(np.linalg.norm(pts - centroid, axis=1)))

            centroids = np.array(centroids)
            inter_dist = 0.0
            count = 0
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    inter_dist += np.linalg.norm(centroids[i] - centroids[j])
                    count += 1
            inter_dist /= max(count, 1)
            silhouette_proxy = (inter_dist - np.mean(intra_dists)) / max(inter_dist, 1e-9)

            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Distancia intra-cluster (media)", f"{np.mean(intra_dists):.2f}")
            col_m2.metric("Distancia inter-cluster (media)", f"{inter_dist:.2f}")
            st.metric("Índice de separación (mayor = mejor)", f"{silhouette_proxy:.3f}")
        except Exception as e:
            st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — QUIZ
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-label">Comprueba lo que sabes</p>', unsafe_allow_html=True)
    st.markdown("## 🧠 Quiz — pon a prueba tu comprensión de t-SNE")
    st.markdown("4 preguntas con feedback inmediato y explicación detallada.")
    st.markdown("---")

    preguntas = [
        {
            "q": "¿Qué controla la perplejidad en t-SNE?",
            "opts": [
                "El número de dimensiones del resultado final",
                "La velocidad de convergencia del algoritmo",
                "Cuántos vecinos considera cada punto al medir similitudes",
                "La escala de los ejes en el plot 2D",
            ],
            "ans": 2,
            "exp": "La **perplejidad** define el número efectivo de vecinos que cada punto "
                   "considera. Baja = sólo vecinos muy cercanos. Alta = visión más amplia. "
                   "El rango recomendado es 5–50, con 30 como valor por defecto.",
        },
        {
            "q": "¿Qué significa que dos clusters estén lejos en un plot t-SNE?",
            "opts": [
                "Que esos grupos son muy diferentes entre sí en el espacio original",
                "Que t-SNE necesita más iteraciones para converger",
                "Nada — las distancias entre clusters en t-SNE no son interpretables",
                "Que la perplejidad está demasiado alta",
            ],
            "ans": 2,
            "exp": "¡Trampa clásica! Las distancias **entre** clusters en t-SNE **no tienen "
                   "significado**. t-SNE preserva la estructura local (dentro de cada cluster) "
                   "pero distorsiona las distancias globales. Dos clusters lejanos podrían "
                   "ser tan diferentes o tan parecidos como dos clusters cercanos.",
        },
        {
            "q": "¿Por qué t-SNE usa la distribución t de Student en 2D en lugar de Gaussiana?",
            "opts": [
                "Porque la Gaussiana es demasiado lenta de calcular",
                "Para que los puntos lejanos se separen más, creando clusters mejor definidos",
                "Porque la distribución t es siempre más precisa que la Gaussiana",
                "Para que el algoritmo sea determinista",
            ],
            "ans": 1,
            "exp": "La distribución **t de Student** tiene 'colas más gruesas' que la Gaussiana. "
                   "Esto permite que los puntos que no son vecinos cercanos se separen más "
                   "en el espacio 2D, creando los clusters compactos y bien separados "
                   "característicos de t-SNE.",
        },
        {
            "q": "¿En cuál de estos escenarios NO deberías usar t-SNE?",
            "opts": [
                "Para explorar visualmente si hay grupos naturales en los datos",
                "Para reducir dimensiones antes de entrenar un modelo y aplicarlo a datos nuevos",
                "Para verificar que un clustering tiene sentido visualmente",
                "Para visualizar datos de secuenciación genómica (scRNA-seq)",
            ],
            "ans": 1,
            "exp": "t-SNE **no puede transformar nuevos datos** (no tiene `.transform()`). "
                   "Es un método puramente de visualización. Para preprocesamiento de un "
                   "pipeline ML que deba procesar datos nuevos, usa PCA o UMAP.",
        },
    ]

    score = 0
    answered = 0
    for i, item in enumerate(preguntas):
        st.markdown(f"**Pregunta {i+1} de {len(preguntas)}:** {item['q']}")
        choice = st.radio("", item["opts"], key=f"tsne_q{i}", index=None)
        if choice is not None:
            answered += 1
            if item["opts"].index(choice) == item["ans"]:
                score += 1
                st.success(f"✅ ¡Correcto! {item['exp']}")
            else:
                correcta = item["opts"][item["ans"]]
                st.error(f"❌ No del todo. La respuesta correcta es: **{correcta}**\n\n{item['exp']}")
        st.markdown("---")

    if answered == len(preguntas):
        pct = int(score / len(preguntas) * 100)
        if pct == 100:
            st.balloons()
            st.success(f"🏆 ¡Perfecto! {score}/{len(preguntas)} — ¡Dominas t-SNE!")
        elif pct >= 50:
            st.info(f"👍 Bien: {score}/{len(preguntas)}. Repasa las explicaciones y vuelve a intentarlo.")
        else:
            st.warning(f"📚 {score}/{len(preguntas)}. Vuelve a la pestaña '¿Cómo funciona?' con atención.")
