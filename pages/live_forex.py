import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go

from utils.data_loader import load_pair_data
from utils.model_loader import load_keras_model, load_scaler
from utils.prediction_utils import FEATURE_COLS # ── ON GARANTIT L'ORDRE STRICT DES 7 FEATURES

PLOT_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="DM Sans", color="#c9d1d9", size=12),
    xaxis=dict(gridcolor="#21262d", zeroline=False, linecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", zeroline=False, linecolor="#21262d"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#21262d"),
    margin=dict(l=10, r=10, t=40, b=10),
)

def show_live_forex(pair):
    st.title("⚡ Flux Live & Signaux Instantanés")
    
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    st.write(f"Flux rafraîchi à : `{now_str}`")

    with st.spinner("Chargement des composants d'inférence..."):
        model = load_keras_model(pair, "LSTM_Simple")
        scaler = load_scaler(pair)

    if model is None or scaler is None:
        st.warning(f"⚠️ Composants IA indisponibles pour {pair}.")
        st.stop()

    # Le loader calcule maintenant SMA_10, EMA_10, RSI, MACD, Returns et Volatility
    df = load_pair_data(pair)
    if df.empty or len(df) < 65:
        st.error("Données historiques insuffisantes pour alimenter le modèle (minimum 60 jours requis).")
        st.stop()

    # 1. Vérification de sécurité pour s'assurer que les features d'entraînement calculées sont là
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        st.error(f"⚠️ Erreur d'alignement. Variables manquantes dans le flux de données : {missing_cols}")
        st.stop()

    # --- SIMULATION INTERFACE LIVE (EXEMPLE BASE SUR LA DERNIÈRE BOUGIE) ---
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    variation = last_row["Close"] - prev_row["Close"]
    pct_var = (variation / prev_row["Close"]) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Prix Spot Live", f"{last_row['Close']:.5f}")
    col2.metric("Variation Session", f"{variation:+.5f}", f"{pct_var:+.2f}%")
    # Note de sécurité : Si tu as besoin d'afficher High/Low journée bruts sans qu'ils ne servent au scaler :
    col3.metric("High / Low Journée", f"{df['Close'].tail(24).max():.5f} / {df['Close'].tail(24).min():.5f}")

    # ---🎯 ANALYSE PRÉDICTIVE INTRADAY RECALIBRÉE ---
    st.markdown("### 🎯 Analyse Prédictive Intraday (H+24)")

    # Extraction des 60 derniers jours d'historique basés STRICTEMENT sur l'entraînement
    # FEATURE_COLS = ["Close", "SMA_10", "EMA_10", "RSI", "MACD", "Returns", "Volatility"]
    window_data = df[FEATURE_COLS].tail(60).values.astype(np.float32)

    # Normalisation propre de la matrice complète (shape: 60, 7)
    scaled_window = scaler.transform(window_data)

    # Reshape au format attendu par ton LSTM [Samples=1, Window=60, Features=7]
    input_tensor = scaled_window.reshape(1, 60, 7)

    # Prédiction du cours normalisé suivant (t+1)
    with st.spinner("Calcul du signal algorithmique..."):
        try:
            pred_scaled = model.predict(input_tensor, verbose=0)[0, 0]
            
            # Dénormalisation via la méthode de la matrice fantôme (7 colonnes)
            # L'index 0 correspond STRICTEMENT à la variable "Close"
            dummy_row = np.zeros((1, 7), dtype=np.float32)
            dummy_row[0, 0] = pred_scaled
            
            pred_real = float(scaler.inverse_transform(dummy_row)[0, 0])
        except Exception as e:
            st.error(f"Erreur d'inférence en direct : {e}")
            st.stop()

    # --- INTERPRÉTATION DU SIGNAL TECHNIQUE RECTIFIÉ ---
    current_price = last_row["Close"]
    difference = pred_real - current_price
    
    st.write(f"**Prix calculé pour la prochaine session :** `{pred_real:.5f}`")

    # Ajustement des thresholds de pips selon les réalités du FOREX (ex: 5 pips = 0.0005)
    if difference > 0.0002:
        st.success(f"🟩 **SIGNAL IMMÉDIAT : ACHAT (BULLISH)** ➔ Hausse estimée de {difference:+.5f} pips")
    elif difference < -0.0002:
        st.error(f"🟥 **SIGNAL IMMÉDIAT : VENTE (BEARISH)** ➔ Baisse estimée de {difference:.5f} pips")
    else:
        st.warning("🟨 **SIGNAL IMMÉDIAT : NEUTRE (CONSOLIDATION)** ➔ Évolution latérale stable")

    # Mini Graphique directionnel de 5 jours
    hist_mini = df["Close"].tail(5).values
    dates_mini = pd.to_datetime(df["Date"].tail(5)).dt.strftime("%d/%m").tolist()
    
    # Ajout du point prédictif
    dates_mini.append("IA Proch")
    prices_display = np.append(hist_mini, pred_real)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_mini[:-1], y=prices_display[:-1], name="Réel", line=dict(color="#58a6ff", width=3)))
    fig.add_trace(go.Scatter(x=dates_mini[-2:], y=prices_display[-2:], name="Prédiction", line=dict(color="#ff7b72", width=3, dash="dot")))
    fig.update_layout(**PLOT_LAYOUT, height=250, title="Tendance ultra-courte terme (Prochaine bougie)")
    st.plotly_chart(fig, use_container_width=True)