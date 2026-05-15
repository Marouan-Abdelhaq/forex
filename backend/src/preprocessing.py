"""
╔══════════════════════════════════════════════════════════════╗
║         FOREX AI — PHASE 3 : PREPROCESSING                  ║
║                   preprocessing.py                           ║
║                                                              ║
║  Étapes :                                                    ║
║    1. Sélection des features                                 ║
║    2. Normalisation  (MinMaxScaler)                          ║
║    3. Création des séquences temporelles                     ║
║    4. Split Train / Validation / Test                        ║
║                                                              ║
║  Output :                                                    ║
║    X_train, X_val, X_test  → (samples, window, features)    ║
║    y_train, y_val, y_test  → (samples,)                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR         = Path(__file__).resolve().parents[2] / "models"

# ── Fenêtre temporelle ──────────────────────
WINDOW_SIZE = 20        # 20 jours de contexte pour prédire le 21ème

# ── Split ratios ────────────────────────────
TRAIN_RATIO = 0.70      # 70% entraînement
VAL_RATIO   = 0.15      # 15% validation
TEST_RATIO  = 0.15      # 15% test

# ── Features sélectionnées ──────────────────
# On exclut : les prix bruts redondants avec les indicateurs,
# et Volume (= 0 sur le Forex spot)
FEATURE_COLUMNS = [
    # ── Momentum ─────────────────
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Stoch_K",
    "Stoch_D",
    # ── Volatilité ───────────────
    "ATR",
    "BB_Width",
    "BB_Pct",
    # ── Tendance ─────────────────
    "Price_vs_SMA20",
    "EMA_12",
    "EMA_26",
    # ── Prix dérivés ─────────────
    "Returns",
    "Log_Returns",
    "HL_Range",
    "OC_Range",
    # ── Prix OHLC (normalisés) ───
    "Open",
    "High",
    "Low",
    "Close",
]

TARGET_COLUMN = "Signal"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  ÉTAPE 1 — CHARGEMENT
# ─────────────────────────────────────────────

def load_features(pair: str = "EURUSD") -> pd.DataFrame:
    """Charge le CSV enrichi produit par features.py."""
    filepath = PROCESSED_DATA_DIR / f"{pair}_features.csv"

    if not filepath.exists():
        raise FileNotFoundError(
            f"❌ Fichier introuvable : {filepath}\n"
            "   → Lancez d'abord features.py"
        )

    df = pd.read_csv(filepath, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    logger.info(f"📂 Données chargées : {filepath.name}")
    logger.info(f"   {len(df)} lignes | {len(df.columns)} colonnes")
    return df


# ─────────────────────────────────────────────
#  ÉTAPE 2 — SÉLECTION DES FEATURES
# ─────────────────────────────────────────────

def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare les features (X) de la cible (y).
    Vérifie que toutes les colonnes existent.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Colonnes manquantes : {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    logger.info(f"🎯 Features sélectionnées : {len(FEATURE_COLUMNS)}")
    logger.info(f"   {FEATURE_COLUMNS}")
    return X, y


# ─────────────────────────────────────────────
#  ÉTAPE 3 — NORMALISATION
# ─────────────────────────────────────────────

def normalize_features(
    X_train_raw: pd.DataFrame,
    X_val_raw:   pd.DataFrame,
    X_test_raw:  pd.DataFrame,
    pair: str = "EURUSD",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Normalise les features avec MinMaxScaler → [0, 1].

    RÈGLE CRITIQUE : le scaler est FITTÉ uniquement sur X_train
    pour éviter le data leakage (fuite d'information du futur).

    Sauvegarde le scaler dans models/scaler.pkl pour la prédiction.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit UNIQUEMENT sur train
    X_train_scaled = scaler.fit_transform(X_train_raw)

    # Transform sur val et test (pas de fit !)
    X_val_scaled   = scaler.transform(X_val_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    # Sauvegarde du scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = MODELS_DIR / f"{pair}_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    logger.info(f"📏 Normalisation MinMaxScaler [0, 1]")
    logger.info(f"   ⚠️  Scaler fitté UNIQUEMENT sur Train (no data leakage)")
    logger.info(f"   💾 Scaler sauvegardé : {scaler_path}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
#  ÉTAPE 4 — SÉQUENCES TEMPORELLES
# ─────────────────────────────────────────────

def create_sequences(
    X_scaled: np.ndarray,
    y:        np.ndarray,
    window:   int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforme les données 2D en séquences 3D pour LSTM/GRU.

    Principe :
      Pour chaque instant t, on prend les 'window' jours précédents
      comme contexte pour prédire le signal au jour t.

    Input  : X_scaled (N, features)
    Output : X_seq    (N - window, window, features)
             y_seq    (N - window,)

    Exemple avec window=20 et 20 features :
      X_seq.shape = (samples, 20, 20)
    """
    X_seq, y_seq = [], []

    for i in range(window, len(X_scaled)):
        X_seq.append(X_scaled[i - window : i])   # fenêtre glissante
        y_seq.append(y[i])                         # label du jour i

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    return X_seq, y_seq


# ─────────────────────────────────────────────
#  ÉTAPE 5 — TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────

def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
) -> tuple:
    """
    Découpe temporelle STRICTE (pas de shuffle !).

    RÈGLE : on respecte l'ordre chronologique pour éviter
    que le modèle "voie le futur" pendant l'entraînement.

    ┌──────────────────────────────────────────────────┐
    │  TRAIN (70%)  │  VALIDATION (15%)  │  TEST (15%) │
    └──────────────────────────────────────────────────┘
    Passé ─────────────────────────────────────► Futur
    """
    n = len(X)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]

    X_val   = X.iloc[n_train : n_train + n_val]
    y_val   = y.iloc[n_train : n_train + n_val]

    X_test  = X.iloc[n_train + n_val:]
    y_test  = y.iloc[n_train + n_val:]

    logger.info(f"📅 Split temporel (NO SHUFFLE) :")
    logger.info(f"   Train      : {len(X_train):,} lignes ({train_ratio*100:.0f}%)")
    logger.info(f"   Validation : {len(X_val):,}  lignes ({val_ratio*100:.0f}%)")
    logger.info(f"   Test       : {len(X_test):,}  lignes ({(1-train_ratio-val_ratio)*100:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
#  RÉSUMÉ
# ─────────────────────────────────────────────

def display_preprocessing_summary(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_val:   np.ndarray,
    y_test:  np.ndarray,
) -> None:
    """Affiche le résumé complet du preprocessing."""

    def _class_dist(y):
        up   = int(y.sum())
        down = len(y) - up
        return f"UP={up} ({up/len(y)*100:.1f}%) | DOWN={down} ({down/len(y)*100:.1f}%)"

    print("\n" + "═" * 62)
    print("  ⚙️   RÉSUMÉ — PREPROCESSING")
    print("═" * 62)
    print(f"  Window size     : {WINDOW_SIZE} jours")
    print(f"  Features        : {X_train.shape[2]}")
    print("─" * 62)
    print(f"  X_train.shape   : {X_train.shape}")
    print(f"  X_val.shape     : {X_val.shape}")
    print(f"  X_test.shape    : {X_test.shape}")
    print("─" * 62)
    print(f"  y_train dist    : {_class_dist(y_train)}")
    print(f"  y_val   dist    : {_class_dist(y_val)}")
    print(f"  y_test  dist    : {_class_dist(y_test)}")
    print("─" * 62)
    print(f"  Total séquences : {len(X_train) + len(X_val) + len(X_test):,}")
    print("═" * 62 + "\n")


# ─────────────────────────────────────────────
#  SAUVEGARDE DES ARRAYS
# ─────────────────────────────────────────────

def save_arrays(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    pair: str = "EURUSD",
) -> None:
    """Sauvegarde les arrays NumPy pour réutilisation directe."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    arrays = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }

    for name, arr in arrays.items():
        path = PROCESSED_DATA_DIR / f"{pair}_{name}.npy"
        np.save(path, arr)

    logger.info(f"💾 Arrays NumPy sauvegardés dans {PROCESSED_DATA_DIR}")


# ─────────────────────────────────────────────
#  PIPELINE PRINCIPALE
# ─────────────────────────────────────────────

def run_preprocessing(
    pair:        str   = "EURUSD",
    window_size: int   = WINDOW_SIZE,
    save:        bool  = True,
) -> dict:
    """
    Pipeline complète de preprocessing.

    Retourne un dictionnaire contenant tous les arrays
    prêts pour l'entraînement du modèle.
    """
    logger.info("=" * 55)
    logger.info("  ⚙️   PHASE 3 — PREPROCESSING")
    logger.info("=" * 55)

    # 1. Chargement
    df = load_features(pair)

    # 2. Sélection features / target
    X, y = select_features(df)

    # 3. Split temporel (AVANT normalisation pour éviter leakage)
    logger.info("📅 Découpe temporelle ...")
    X_train_raw, X_val_raw, X_test_raw, \
    y_train_raw, y_val_raw, y_test_raw = temporal_split(X, y)

    # 4. Normalisation (fit sur train uniquement)
    logger.info("📏 Normalisation ...")
    X_train_sc, X_val_sc, X_test_sc, scaler = normalize_features(
        X_train_raw, X_val_raw, X_test_raw, pair
    )

    # Conversion y en numpy
    y_train_np = y_train_raw.values
    y_val_np   = y_val_raw.values
    y_test_np  = y_test_raw.values

    # 5. Création des séquences temporelles
    logger.info(f"🔄 Création des séquences (window={window_size}) ...")
    X_train, y_train = create_sequences(X_train_sc, y_train_np, window_size)
    X_val,   y_val   = create_sequences(X_val_sc,   y_val_np,   window_size)
    X_test,  y_test  = create_sequences(X_test_sc,  y_test_np,  window_size)

    logger.info(f"   X_train : {X_train.shape}")
    logger.info(f"   X_val   : {X_val.shape}")
    logger.info(f"   X_test  : {X_test.shape}")

    # 6. Résumé
    display_preprocessing_summary(
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )

    # 7. Sauvegarde optionnelle
    if save:
        save_arrays(X_train, X_val, X_test, y_train, y_val, y_test, pair)

    logger.info("✅  Phase 3 terminée avec succès.\n")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler":  scaler,
        "features": FEATURE_COLUMNS,
        "window":   window_size,
    }


# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    data = run_preprocessing(pair="EURUSD", window_size=20, save=True)

    # Vérification rapide
    print("📦 Arrays disponibles dans 'data' :")
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            print(f"   data['{key}'].shape = {val.shape}")