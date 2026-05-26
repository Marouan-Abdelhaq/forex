import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go

from utils.data_loader import load_pair_data
from utils.model_loader import load_keras_model, load_scaler
from utils.prediction_utils import predict_future, FEATURE_COLS # ── ON IMPORTE LA LISTE STRUCTURÉE !

PLOT_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="DM Sans", color="#c9d1d9", size=12),
    xaxis=dict(gridcolor="#21262d", zeroline=False, linecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", zeroline=False, linecolor="#21262d"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#21262d"),
    margin=dict(l=10, r=10, t=40, b=10),
)

def show_future_forecast(pair, model_name, window_size):
    if window_size != 60:
        window_size = 60

    st.title("🔮 Prévisions Futures (Horizon 2026)")
    st.write("Génération de trajectoires prédictives auto-récursives hors-échantillon.")
    st.info(f"**Configuration active :** Paire : `{pair}` | Modèle : `{model_name}` | Fenêtre historique : `{window_size}` jours")

    with st.spinner("Chargement des composants d'inférence..."):
        model = load_keras_model(pair, model_name)
        scaler = load_scaler(pair)

    if model is None or scaler is None:
        st.warning(f"⚠️ Impossible de charger les structures IA pour {pair}.")
        st.stop()

    # 1. Chargement des données (qui calcule maintenant SMA_10, EMA_10, RSI, MACD, Returns, Volatility)
    df = load_pair_data(pair)
    if df.empty or "Close" not in df.columns:
        st.error("Base de données d'initialisation vide ou invalide.")
        st.stop()

    # 2. Vérification de sécurité pour s'assurer que les features d'entraînement sont présentes
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        st.error(f"⚠️ Le loader n'a pas pu générer toutes les colonnes requises. Manquant : {missing_cols}")
        st.stop()

    st.markdown('<div class="section-title">📆 Définition de l\'Horizon de Prédiction</div>', unsafe_allow_html=True)
    horizon_days = st.slider("Nombre de jours ouvrés à projeter (2026) :", min_value=5, max_value=90, value=30, step=5)

    # ── CALCUL VIA L'ALGORITHME RÉCURSIF MUTIVARIÉ ALIGNÉ ──
    with st.spinner("Génération de la trajectoire par l'IA..."):
        try:
            # On passe directement le DataFrame nettoyé par le data_loader
            future_predictions, future_dates = predict_future(
                model=model,
                scaler=scaler,
                df_historical=df,
                window=window_size,
                n_days=horizon_days
            )
        except Exception as e:
            st.error(f"Erreur lors de la génération récursive : {e}")
            st.stop()

    # ── GRAPHIQUE DE PROJECTION ──
    st.markdown('<div class="section-title">📈 Projections Temporelles Globales</div>', unsafe_allow_html=True)
    
    # Historique récent pour le contexte visuel (60 derniers jours)
    history_slice = 60
    hist_dates = df["Date"].values[-history_slice:]
    hist_prices = df["Close"].values[-history_slice:].flatten()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_dates, y=hist_prices, name="Historique Récent", line=dict(color="#58a6ff", width=2)))
    
    # Raccordement visuel parfait entre le passé et le futur projeté
    conn_dates = np.insert(np.array(future_dates), 0, hist_dates[-1])
    conn_values = np.insert(future_predictions, 0, hist_prices[-1])
    
    fig.add_trace(go.Scatter(x=conn_dates, y=conn_values, name="Projection IA (2026)", line=dict(color="#ff7b72", width=2, dash="dash")))
    fig.update_layout(**PLOT_LAYOUT, height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ── TABLEAU DE BORD DES PRÉVISIONS FUTURES ──
    st.markdown('<div class="section-title">📋 Agenda des cibles de cours estimées</div>', unsafe_allow_html=True)
    forecast_df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
        "Prix Cible Estimé": future_predictions
    })
    st.dataframe(forecast_df.style.format({"Prix Cible Estimé": "{:.5f}"}), use_container_width=True)