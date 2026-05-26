import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

from utils.data_loader import load_pair_data
from utils.prediction_utils import FEATURE_COLS

# Configuration visuelle Quant Dark
PLOT_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="DM Sans", color="#c9d1d9", size=12),
    xaxis=dict(gridcolor="#21262d", zeroline=False, linecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", zeroline=False, linecolor="#21262d"),
    margin=dict(l=40, r=20, t=40, b=40),
)

def show_features_pca(pair):
    st.markdown(
        """
        <div class="main-title"><span class="icon">🔬</span> Feature Intelligence & PCA Engine</div>
        <div class="sub-title">Analyse spectrale de l'espace des descripteurs : Stationnarité, colinéarité et décomposition orthogonale.</div>
        """, 
        unsafe_allow_html=True
    )

    df = load_pair_data(pair)
    if df.empty:
        st.error("Données historiques introuvables pour l'analyse des features.")
        st.stop()

    # ── COUCHE 1 : PRÉPARATION & TRANSFORMATION EN SÉRIES STATIONNAIRES ──
    # Calcul de la vraie cible Quant : Le rendement à t+1 (Anti-Leakage)
    df['Return_Target'] = df['Close'].pct_change().shift(-1)
    
    # Nettoyage des données pour l'analyse mathématique
    analysis_df = df[FEATURE_COLS + ['Return_Target']].dropna()
    
    X_raw = analysis_df[FEATURE_COLS].values
    y_target = analysis_df['Return_Target'].values

    # Standardisation rigoureuse avant PCA (Z-Score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    st.markdown('<div class="section-title">📡 Vitesse d\'Information Mutuelle (Features vs Forward Returns)</div>', unsafe_allow_html=True)
    
    # Calcul du score d'information mutuelle non-linéaire
    mi_scores = mutual_info_regression(X_scaled, y_target, random_state=42)
    mi_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Mutual Information': mi_scores
    }).sort_values(by='Mutual Information', ascending=False)

    fig_mi = px.bar(
        mi_df, x='Mutual Information', y='Feature', orientation='h',
        title="Capacité Prédictive Non-Linéaire (MI Score)",
        color='Mutual Information', color_continuous_scale='Viridis'
    )
    fig_mi.update_layout(**PLOT_LAYOUT, height=300)
    st.plotly_chart(fig_mi, use_container_width=True)

    # ── COUCHE 2 : DÉCOMPOSITION ORTHOGONALE (PCA) ──
    st.markdown('<div class="section-title">📐 Analyse Spectrale : Décomposition en Composantes Principales</div>', unsafe_allow_html=True)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    cum_explained_variance = np.cumsum(explained_variance)

    c1, c2 = st.columns([1, 1])
    
    with c1:
        # Graphique de l'éboulis des valeurs propres (Scree Plot)
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(X_raw[0]))], y=explained_variance, name="Variance Individuelle", marker_color="#d29922"))
        fig_scree.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(X_raw[0]))], y=cum_explained_variance, name="Variance Cumulée", line=dict(color="#3fb950", width=2)))
        
        # Correction de l'injection : On applique le layout de base puis on surcharge l'axe y de manière isolée
        fig_scree.update_layout(**PLOT_LAYOUT, title="Variance Expliquée par Axe Orthogonal")
        fig_scree.update_layout(yaxis=dict(tickformat=".1%", gridcolor="#21262d", zeroline=False, linecolor="#21262d"))
        
        st.plotly_chart(fig_scree, use_container_width=True)

    with c2:
        # Analyse de la saturation dimensionnelle
        n_90 = np.argmax(cum_explained_variance >= 0.90) + 1
        st.markdown(f"""
        <div style="background-color:#161b22; padding:20px; border-radius:8px; border:1px solid #21262d; height:275px;">
            <h4 style="margin-top:0; color:#c9d1d9;">📊 Diagnostic de Compression</h4>
            <ul style="padding-left:20px; color:#8b949e; font-size:13px;">
                <li>Nombre de descripteurs injectés : <b style="color:#58a6ff;">{len(FEATURE_COLS)}</b></li>
                <li>Axes nécessaires pour capter 90% de la variance : <b style="color:#3fb950;">{n_90} axes</b></li>
                <li>Variance captée par les 2 premiers axes (PC1 + PC2) : <b style="color:#d29922;">{cum_explained_variance[1]:.1%}</b></li>
            </ul>
            <p style="font-size:12px; color:#8b949e; margin-top:15px;">
                ⚠️ <b>Si PC1 + PC2 > 80% :</b> Vos indicateurs techniques sont fortement colinéaires. Ils répliquent tous la même information de tendance sous-jacente (bruit basse fréquence).
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── COUCHE 3 : CARTOGRAPHIE DU BRUIT ET DU SIGNAL ──
    st.markdown('<div class="section-title">🌌 Cartographie Statistique des Deux Premiers Axes (Top Alpha Space)</div>', unsafe_allow_html=True)
    
    # Création d'un DataFrame de projection pour la visualisation bidimensionnelle
    projection_df = pd.DataFrame(data=X_pca[:, :2], columns=['PC1', 'PC2'])
    projection_df['Target_Sign'] = np.where(y_target >= 0, 'Rendement Positif (t+1)', 'Rendement Négatif (t+1)')
    projection_df['Return_Magnitude'] = np.abs(y_target)

    fig_biplot = px.scatter(
        projection_df.tail(800), x='PC1', y='PC2', # Échantillon récent pour la lisibilité
        color='Target_Sign',
        size='Return_Magnitude',
        title="Projection de l'Espace des Features vs Vrais Rendements Futurs (800 derniers points)",
        color_discrete_map={'Rendement Positif (t+1)': '#3fb950', 'Rendement Négatif (t+1)': '#f85149'},
        opacity=0.6
    )
    fig_biplot.update_layout(**PLOT_LAYOUT, height=450)
    st.plotly_chart(fig_biplot, use_container_width=True)
    
    # Message d'aide à la décision pour le chercheur
    st.markdown("""
    > 💡 **Règle de lecture Quant pour le Biplot :** Si les points verts (Rendements Positifs) et les points rouges (Rendements Négatifs) sont complètement mélangés et superposés sur le graphique, cela démontre graphiquement que ton espace géométrique actuel de caractéristiques est **incapable de séparer le signal de décision**, confirmant ton exactitude directionnelle bloquée à ~46%.
    """)