import os
from pathlib import Path

# Chemins absolus du projet
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# ── AJOUT DE LA VARIABLE MANQUANTE POUR LE LOADER ──
REPORTS_DIR = BASE_DIR / "reports" # Tes rapports .json de la session 2 sont ici !

# S'assurer que le dossier des modèles existe
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Liste globale des devises et des modèles pris en charge
PAIRS = ["EURUSD", "EURMAD", "USDMAD"]
MODELS_AVAILABLE = ["LSTM_Simple", "BiGRU_MHA"]