import json
import joblib
from pathlib import Path
import streamlit as st
import tensorflow as tf
from config import MODELS_DIR, REPORTS_DIR

@st.cache_resource
def load_keras_model(pair, model_name):
    """
    Charge un modèle TensorFlow/Keras en testant plusieurs suffixes 
    et en gérant la désérialisation unsafe de Keras si nécessaire.
    """
    paths = [
        MODELS_DIR / f"{pair}_{model_name}.keras",           # ── AJOUTÉ : Ton format réel actuel !
        MODELS_DIR / f"{pair}_{model_name}_final_lab.keras",
        MODELS_DIR / f"{pair}_{model_name}_best_lab.keras",
        MODELS_DIR / f"{pair}_{model_name}_final.keras",
    ]

    for p in paths:
        if p.exists():
            try:
                return tf.keras.models.load_model(
                    str(p),
                    compile=False
                )
            except Exception:
                import keras
                # Nécessaire pour charger certaines configurations d'attention ou couches custom
                keras.config.enable_unsafe_deserialization()
                return tf.keras.models.load_model(
                    str(p),
                    compile=False
                )
    return None

@st.cache_resource
def load_scaler(pair):
    """
    Charge le scaler joblib (.pkl) correspondant à la paire de devises.
    """
    # Ajout de "" au début pour tester en priorité ton fichier "EURUSD_scaler.pkl" direct
    for suffix in ["", "_lab", "_v2"]:
        p = MODELS_DIR / f"{pair}_scaler{suffix}.pkl"
        if p.exists():
            return joblib.load(p)
    return None

@st.cache_data
def load_report():
    """
    Charge le rapport JSON global contenant les métriques de performance.
    """
    p = REPORTS_DIR / "lab2_report.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}