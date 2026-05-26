import os
import sys

import streamlit as st

# Assure que le dossier du projet est dans sys.path (évite ModuleNotFoundError sur utils/* et pages/*)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.styles import inject_custom_styles


# ── IMPORTS DE TOUTES LES PAGES DU DASHBOARD ──
from pages.overview import show_overview
from pages.dataset_analysis import show_dataset_analysis
from pages.features_pca import show_features_pca          # 🔥 Nouvelle page de diagnostic quantitaire
from pages.model_training import show_model_training
from pages.predictions import show_predictions
from pages.future_forecast import show_future_forecast
from pages.live_forex import show_live_forex
from pages.about import show_about

# 🔥 Configuration globale obligatoire de la page de l'application
st.set_page_config(
    page_title="Forex AI Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injection immédiate du dictionnaire des styles CSS isolés
inject_custom_styles()

FOREX_PAIRS = {
    "EURUSD": "EUR/USD",
    "EURMAD": "EUR/MAD",
    "USDMAD": "USD/MAD",
}

# ── STRUCTURE DU PANNEAU LATÉRAL (SIDEBAR) ──
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
        <div style="font-family:'Space Mono',monospace;font-size:1.4rem;color:#58a6ff;font-weight:700">💹 FOREX AI</div>
        <div style="color:#484f58;font-size:0.75rem;margin-top:4px">Lab Session 2 · 2025/2026</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    # Intégration de l'onglet "🔬 Features & PCA" au niveau de la navigation
    page = st.radio("Navigation", [
        "🏠 Overview",
        "📊 Dataset Analysis",
        "🔬 Features & PCA",
        "🧠 Model Training",
        "🎯 Predictions",
        "🔮 Future Forecast 2026",
        "⚡ Live Forex",
        "ℹ️ About",
    ], label_visibility="collapsed")
    
    st.divider()
    st.markdown("**⚙️ Configuration Globale**")
    sel_pair = st.selectbox("Paire Forex", list(FOREX_PAIRS.keys()), format_func=lambda x: FOREX_PAIRS[x])
    sel_model = st.selectbox("Modèle Sélectionné", ["LSTM_Simple", "BiGRU_MHA"], format_func=lambda x: x.replace("_", " "))
    window_size = st.slider("Window Size (jours)", 20, 120, 60, 5)
    epochs = st.slider("Epochs", 10, 150, 40, 5)
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    st.divider()
    
    st.markdown("**🎛️ Commandes Exécutives**")
    btn_load = st.button("📂 Charge Model", use_container_width=True)
    btn_pred = st.button("▶️ Run Inference", use_container_width=True)
    btn_live = st.button("🔄 Sync Live Feed", use_container_width=True)
    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem;color:#484f58;text-align:center'>USMS · FP Khouribga · SIIA<br>Deep Learning · Pr. Bakkouri</div>", unsafe_allow_html=True)

# ── LOGIQUE DE ROUTAGE VERS LES PAGES CORRESPONDANTES ──
if page == "🏠 Overview":
    show_overview(sel_pair, FOREX_PAIRS[sel_pair])
elif page == "📊 Dataset Analysis":
    show_dataset_analysis(sel_pair, FOREX_PAIRS[sel_pair])
elif page == "🔬 Features & PCA":
    # Appel de la nouvelle couche d'analyse spectrale des descripteurs
    show_features_pca(sel_pair)
elif page == "🧠 Model Training":
    # On passe les hyperparamètres de la sidebar pour piloter l'affichage ou l'entraînement
    show_model_training(sel_pair, sel_model, epochs, batch_size, window_size)
elif page == "🎯 Predictions":
    # Permet d'exécuter l'évaluation sur l'historique de test
    show_predictions(sel_pair, sel_model, window_size)
elif page == "🔮 Future Forecast 2026":
    # Lance le forecasting auto-récursif sur 2026
    show_future_forecast(sel_pair, sel_model, window_size)
elif page == "⚡ Live Forex":
    # Affiche le flux temps réel interconnecté au marché
    show_live_forex(sel_pair)
elif page == "ℹ️ About":
    show_about()