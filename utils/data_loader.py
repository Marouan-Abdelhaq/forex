import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import streamlit as st

# ==============================
# CONFIGURATION DES CHEMINS
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
REPORT_PATH = BASE_DIR / "reports" / "lab2_report.json"

# ==============================
# CALCUL DES INDICATEURS (STRICTEMENT ALIGNÉS IA)
# ==============================
def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les indicateurs techniques strictement requis par le modèle LSTM.
    Ordre et features attendus : Close, SMA_10, EMA_10, RSI, MACD, Returns, Volatility
    """
    df = df.copy()
    c = df["Close"].copy()

    # 1. Moyennes Mobiles (Fenêtre de 10 jours)
    df["SMA_10"] = c.rolling(window=10, min_periods=1).mean()
    df["EMA_10"] = c.ewm(span=10, adjust=False).mean()

    # 2. MACD classique (12, 26)
    ema_12 = c.ewm(span=12, adjust=False).mean()
    ema_26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26

    # 3. RSI (14 jours) avec protection contre la division par zéro
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(com=13, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=1).mean()
    
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50.0) # Valeur neutre par défaut

    # 4. Rendements (Returns)
    df["Returns"] = c.pct_change().fillna(0)

    # 5. Volatilité (Écart-type mobile des rendements sur 10 jours)
    df["Volatility"] = df["Returns"].rolling(window=10, min_periods=1).std().fillna(0)

    # Nettoyage final des NaNs pour le LSTM
    df = df.bfill().ffill()

    return df

# ==============================
# CHARGEMENT DES DONNÉES FOREX
# ==============================
@st.cache_data(ttl=3600)
def load_pair_data(pair_code):
    csv_path = DATA_DIR / f"{pair_code}.csv"

    if not csv_path.exists():
        st.error(f"Fichier introuvable : {csv_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)

        # Conversion de la colonne Date
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        return df

    except Exception as e:
        st.error(f"Erreur de lecture CSV : {e}")
        return pd.DataFrame()

# ==============================
# CHARGEMENT DU RAPPORT DE MÉTRIQUES
# ==============================
@st.cache_data(ttl=3600)
def load_report():
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"summary": []}