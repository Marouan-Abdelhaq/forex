import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from utils.data_loader import load_pair_data, compute_technical_indicators

# Configuration centralisée du layout pour garder une cohérence UI/UX
PLOT_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="DM Sans", color="#c9d1d9", size=12),
    xaxis=dict(gridcolor="#21262d", zeroline=False, linecolor="#21262d", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#21262d", zeroline=False, linecolor="#21262d", tickfont=dict(size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#21262d"),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
)

def show_dataset_analysis(sel_pair, pair_label):
    """
    Rendu de l'onglet 'Dataset Analysis'.
    Affiche les statistiques descriptives et les distributions des features de l'IA.
    """
    # ── 1. TITRE DE LA SECTION ──
    st.markdown(
        """
        <div class="main-title">
            <span class="icon">📘</span> Dataset Analysis
        </div>
        <div class="sub-title">
            Visualisation des indicateurs techniques de marché calculés
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown(f"### 📊 Analyse Approfondie — {pair_label}")

    # Chargement et calcul des 7 indicateurs alignés IA
    df = load_pair_data(sel_pair)
    if df.empty:
        st.warning("⚠️ Aucune donnée disponible pour cette paire de devises.")
        st.stop()
        
    df_features = compute_technical_indicators(df)

    # Liste stricte des features d'entraînement pour filtrer le describe()
    features_list = ["Close", "SMA_10", "EMA_10", "RSI", "MACD", "Returns", "Volatility"]
    
    # Sécurité au cas où certaines colonnes manquent à l'appel
    available_features = [col for col in features_list if col in df_features.columns]

    # ── 2. TABLEAU DES STATISTIQUES DESCRIPTIVES ──
    st.markdown("#### 📋 Métriques et Indicateurs de l'Entraînement IA")
    st.caption("Statistiques descriptives des 7 variables réellement lues par ton réseau de neurones LSTM :")
    
    # Calcul du résumé statistique (Pandas .describe())
    stats_df = df_features[available_features].describe()
    
    # Affichage via le composant Streamlit natif configuré pour les gros volumes
    st.dataframe(stats_df, use_container_width=True)

    # ── 3. DOUBLE HISTOGRAMME DE DISTRIBUTION (PRIX VS RENDEMENTS) ──
    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 📐 Distribution du Prix de Clôture (`Close`)")
        fig_close = px.histogram(
            df_features, 
            x="Close", 
            nbins=40,
            color_discrete_sequence=["#58a6ff"],
            labels={"Close": "Prix de Clôture"}
        )
        fig_close.update_layout(**PLOT_LAYOUT, height=350)
        fig_close.update_yaxes(title_text="count")
        st.plotly_chart(fig_close, use_container_width=True)

    with col_right:
        st.markdown("#### 📈 Distribution des Rendements (`Returns`)")
        fig_ret = px.histogram(
            df_features, 
            x="Returns", 
            nbins=50,
            color_discrete_sequence=["#f85149"],
            labels={"Returns": "Rendements Journaliers"}
        )
        fig_ret.update_layout(**PLOT_LAYOUT, height=350)
        fig_ret.update_yaxes(title_text="count")
        st.plotly_chart(fig_ret, use_container_width=True)

    # ── 4. GRAPHIQUE TEMPOREL DE LA VOLATILITÉ ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚡ Évolution de la Volatilité Historique (Fenêtre Mobile)")

    if "Volatility" in df_features.columns and "Date" in df_features.columns:
        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Scatter(
                x=df_features["Date"], 
                y=df_features["Volatility"], 
                name="Volatilité (10d)", 
                line=dict(color="#3fb950", width=1.5)
            )
        )
        
        # 1. On applique d'abord la charte graphique globale
        fig_vol.update_layout(**PLOT_LAYOUT, height=400)
        
        # 2. On met à jour les titres des axes de manière isolée sans conflit
        fig_vol.update_xaxes(title_text="Date")
        fig_vol.update_yaxes(title_text="Volatilité calculée")
        
        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.info("La variable de volatilité n'a pas pu être extraite pour le graphique temporel.")