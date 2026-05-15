"""
╔══════════════════════════════════════════════════════════════╗
║           FOREX AI — PHASE 1 : DATA COLLECTION              ║
║                     data_loader.py                           ║
║                                                              ║
║  Description : Téléchargement automatique des données Forex  ║
║  Source      : Yahoo Finance via yfinance                    ║
║  Output      : CSV brut dans backend/data/raw/               ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

# Dossier de sortie (relatif à la racine du projet)
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Paires Forex disponibles
FOREX_PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
}

# Paramètres par défaut
DEFAULT_PAIR      = "EURUSD"
DEFAULT_START     = "2015-01-01"
DEFAULT_END       = datetime.today().strftime("%Y-%m-%d")
DEFAULT_INTERVAL  = "1d"          # daily


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  FONCTIONS
# ─────────────────────────────────────────────

def ensure_directories() -> None:
    """Crée les dossiers de données si inexistants."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Dossier de données : {RAW_DATA_DIR}")


def download_forex(
    pair: str     = DEFAULT_PAIR,
    start: str    = DEFAULT_START,
    end: str      = DEFAULT_END,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """
    Télécharge les données OHLCV d'une paire Forex depuis Yahoo Finance.

    Parameters
    ----------
    pair     : str  — clé de la paire (ex: "EURUSD")
    start    : str  — date de début  (ex: "2015-01-01")
    end      : str  — date de fin    (ex: "2024-12-31")
    interval : str  — fréquence      ("1d" | "1h" | "1wk")

    Returns
    -------
    pd.DataFrame avec colonnes : Open | High | Low | Close | Volume
    """
    if pair not in FOREX_PAIRS:
        raise ValueError(
            f"❌ Paire '{pair}' inconnue. Choix disponibles : {list(FOREX_PAIRS.keys())}"
        )

    ticker_symbol = FOREX_PAIRS[pair]

    logger.info(f"🌐 Téléchargement de {pair} ({ticker_symbol}) ...")
    logger.info(f"   Période   : {start} → {end}")
    logger.info(f"   Fréquence : {interval}")

    # ── Téléchargement ──────────────────────────────────────────────
    raw_df = yf.download(
        tickers=ticker_symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,       # Ajustement automatique des prix
        progress=False,
    )

    # ── Validation ──────────────────────────────────────────────────
    if raw_df.empty:
        raise RuntimeError(
            f"❌ Aucune donnée reçue pour {pair}. "
            "Vérifiez la connexion ou la plage de dates."
        )

    logger.info(f"✅ {len(raw_df)} lignes reçues.")

    # ── Nettoyage ───────────────────────────────────────────────────
    df = _clean_dataframe(raw_df, pair)

    return df


def _clean_dataframe(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Nettoie et normalise le DataFrame brut de yfinance.

    - Aplatit les colonnes MultiIndex si nécessaire
    - Supprime les lignes avec des valeurs nulles
    - Renomme et réorganise les colonnes
    - Ajoute la colonne 'Pair'
    """
    # ── Aplatir MultiIndex (yfinance peut en créer un) ───────────────
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # ── Colonnes attendues ────────────────────────────────────────────
    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Colonnes manquantes : {missing}")

    df = df[expected_cols].copy()

    # ── Index → colonne Datetime ──────────────────────────────────────
    df.index.name = "Datetime"
    df = df.reset_index()

    # ── Supprimer les doublons ────────────────────────────────────────
    df = df.drop_duplicates(subset="Datetime")

    # ── Supprimer les lignes entièrement nulles ───────────────────────
    before = len(df)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"⚠️  {dropped} lignes supprimées (valeurs nulles).")

    # ── Ajouter colonne Pair ──────────────────────────────────────────
    df.insert(1, "Pair", pair)

    # ── Trier par date ────────────────────────────────────────────────
    df = df.sort_values("Datetime").reset_index(drop=True)

    logger.info(f"🧹 Nettoyage terminé : {len(df)} lignes valides.")

    return df


def save_to_csv(df: pd.DataFrame, pair: str) -> Path:
    """
    Sauvegarde le DataFrame en CSV dans data/raw/.

    Nommage : EURUSD_daily_2015-01-01_2024-12-31.csv
    """
    ensure_directories()

    start_str = str(df["Datetime"].min().date())
    end_str   = str(df["Datetime"].max().date())
    filename  = f"{pair}_daily_{start_str}_{end_str}.csv"
    filepath  = RAW_DATA_DIR / filename

    df.to_csv(filepath, index=False)
    logger.info(f"💾 Fichier sauvegardé : {filepath}")
    logger.info(f"   Taille : {filepath.stat().st_size / 1024:.1f} Ko")

    return filepath


def display_summary(df: pd.DataFrame) -> None:
    """Affiche un résumé clair des données téléchargées."""
    print("\n" + "═" * 55)
    print("  📊  RÉSUMÉ DES DONNÉES TÉLÉCHARGÉES")
    print("═" * 55)
    print(f"  Paire         : {df['Pair'].iloc[0]}")
    print(f"  Lignes totales: {len(df):,}")
    print(f"  Début         : {df['Datetime'].min()}")
    print(f"  Fin           : {df['Datetime'].max()}")
    print(f"  Colonnes      : {list(df.columns)}")
    print("─" * 55)
    print(f"  Close min     : {df['Close'].min():.5f}")
    print(f"  Close max     : {df['Close'].max():.5f}")
    print(f"  Close moyen   : {df['Close'].mean():.5f}")
    print(f"  Valeurs nulles: {df.isnull().sum().sum()}")
    print("═" * 55)
    print("\n📋 Aperçu des 5 premières lignes :")
    print(df.head().to_string(index=False))
    print("\n📋 Aperçu des 5 dernières lignes :")
    print(df.tail().to_string(index=False))
    print()


# ─────────────────────────────────────────────
#  PIPELINE PRINCIPALE
# ─────────────────────────────────────────────

def run_data_collection(
    pair: str     = DEFAULT_PAIR,
    start: str    = DEFAULT_START,
    end: str      = DEFAULT_END,
    interval: str = DEFAULT_INTERVAL,
    save: bool    = True,
) -> pd.DataFrame:
    """
    Pipeline complète de collecte des données Forex.

    Parameters
    ----------
    pair     : Paire Forex (ex: "EURUSD", "GBPUSD")
    start    : Date de début
    end      : Date de fin
    interval : Fréquence temporelle
    save     : Sauvegarder en CSV ?

    Returns
    -------
    pd.DataFrame nettoyé et prêt pour la Phase 2
    """
    logger.info("=" * 55)
    logger.info("  🚀  PHASE 1 — DATA COLLECTION")
    logger.info("=" * 55)

    # 1. Téléchargement
    df = download_forex(pair=pair, start=start, end=end, interval=interval)

    # 2. Résumé
    display_summary(df)

    # 3. Sauvegarde
    if save:
        save_to_csv(df, pair)

    logger.info("✅  Phase 1 terminée avec succès.\n")

    return df


# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Exemple : télécharger EUR/USD sur 10 ans ──────────────────────
    df = run_data_collection(
        pair     = "EURUSD",
        start    = "2015-01-01",
        end      = datetime.today().strftime("%Y-%m-%d"),
        interval = "1d",
        save     = True,
    )

    # ── Exemple 2 : télécharger plusieurs paires ─────────────────────
    # for pair in ["EURUSD", "GBPUSD", "USDJPY"]:
    #     run_data_collection(pair=pair, start="2015-01-01", save=True)