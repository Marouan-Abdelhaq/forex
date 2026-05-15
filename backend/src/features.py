"""
╔══════════════════════════════════════════════════════════════╗
║           FOREX AI — PHASE 2 : FEATURE ENGINEERING          ║
║                       features.py                            ║
║                                                              ║
║  Description : Génération des indicateurs techniques         ║
║  Input       : data/raw/EURUSD_daily_*.csv                   ║
║  Output      : data/processed/EURUSD_features.csv            ║
╚══════════════════════════════════════════════════════════════╝
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

RAW_DATA_DIR       = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  CHARGEMENT DES DONNÉES BRUTES
# ─────────────────────────────────────────────

def load_raw_data(pair: str = "EURUSD") -> pd.DataFrame:
    """
    Charge le fichier CSV brut le plus récent pour une paire donnée.
    """
    files = sorted(RAW_DATA_DIR.glob(f"{pair}_daily_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"❌ Aucun fichier trouvé pour {pair} dans {RAW_DATA_DIR}\n"
            "   → Lancez d'abord data_loader.py"
        )

    filepath = files[-1]  # prendre le plus récent
    logger.info(f"📂 Chargement : {filepath.name}")

    df = pd.read_csv(filepath, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    logger.info(f"   {len(df)} lignes chargées.")
    return df


# ─────────────────────────────────────────────
#  INDICATEURS — TENDANCE
# ─────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """SMA et EMA sur plusieurs périodes."""
    close = df["Close"]

    # Simple Moving Averages
    df["SMA_20"] = close.rolling(window=20).mean()
    df["SMA_50"] = close.rolling(window=50).mean()

    # Exponential Moving Averages
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    logger.info("   ✅ Moving Averages : SMA_20, SMA_50, EMA_12, EMA_26")
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    MACD = EMA_12 - EMA_26
    Signal = EMA_9(MACD)
    Histogram = MACD - Signal
    """
    df["MACD"]           = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"]    = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    logger.info("   ✅ MACD : MACD, MACD_Signal, MACD_Histogram")
    return df


# ─────────────────────────────────────────────
#  INDICATEURS — MOMENTUM
# ─────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    RSI (Relative Strength Index) — période 14 jours.
    RSI > 70 → surachat (potentiel retournement baissier)
    RSI < 30 → survente  (potentiel retournement haussier)
    """
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs         = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"]  = 100 - (100 / (1 + rs))

    logger.info(f"   ✅ RSI_{period}")
    return df


def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator %K et %D.
    %K > 80 → surachat | %K < 20 → survente
    """
    low_min  = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()

    df["Stoch_K"] = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(window=d_period).mean()

    logger.info(f"   ✅ Stochastic : Stoch_K({k_period}), Stoch_D({d_period})")
    return df


# ─────────────────────────────────────────────
#  INDICATEURS — VOLATILITÉ
# ─────────────────────────────────────────────

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    ATR (Average True Range) — mesure de la volatilité.
    True Range = max(H-L, |H-Cprev|, |L-Cprev|)
    """
    high      = df["High"]
    low       = df["Low"]
    prev_close = df["Close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(com=period - 1, min_periods=period).mean()

    logger.info(f"   ✅ ATR_{period}")
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands — enveloppe de volatilité autour de la SMA.
    BB_Upper = SMA + 2*std | BB_Lower = SMA - 2*std
    BB_Width = mesure de la contraction/expansion de la volatilité
    BB_Pct   = position du prix dans les bandes (0 à 1)
    """
    sma = df["Close"].rolling(window=period).mean()
    std = df["Close"].rolling(window=period).std()

    df["BB_Upper"]  = sma + std_dev * std
    df["BB_Middle"] = sma
    df["BB_Lower"]  = sma - std_dev * std
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"].replace(0, np.nan)
    df["BB_Pct"]    = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)

    logger.info(f"   ✅ Bollinger Bands({period},{std_dev}) : Upper, Middle, Lower, Width, Pct")
    return df


# ─────────────────────────────────────────────
#  FEATURES — PRIX DÉRIVÉS
# ─────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features dérivées du prix OHLCV :
    - Returns       : variation journalière en %
    - Log_Returns   : log de la variation (plus stable statistiquement)
    - HL_Range      : High - Low (amplitude de la bougie)
    - OC_Range      : Close - Open (direction de la bougie)
    - Price_vs_SMA20: position du prix par rapport à la SMA20
    """
    df["Returns"]        = df["Close"].pct_change()
    df["Log_Returns"]    = np.log(df["Close"] / df["Close"].shift(1))
    df["HL_Range"]       = df["High"] - df["Low"]
    df["OC_Range"]       = df["Close"] - df["Open"]
    df["Price_vs_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"].replace(0, np.nan)

    logger.info("   ✅ Price Features : Returns, Log_Returns, HL_Range, OC_Range, Price_vs_SMA20")
    return df


# ─────────────────────────────────────────────
#  TARGET — SIGNAL UP / DOWN
# ─────────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Variable cible binaire pour la classification :
      Signal = 1  →  UP   📈  (le Close de demain > Close d'aujourd'hui)
      Signal = 0  →  DOWN 📉  (le Close de demain ≤ Close d'aujourd'hui)

    IMPORTANT : on utilise shift(-1) pour regarder le jour SUIVANT.
    La dernière ligne aura NaN → sera supprimée plus tard.
    """
    df["Signal"] = (df["Close"].shift(-1) > df["Close"]).astype(float)

    up_count   = int(df["Signal"].sum())
    down_count = int((df["Signal"] == 0).sum())
    total      = up_count + down_count

    logger.info(
        f"   ✅ Target Signal → UP: {up_count} ({up_count/total*100:.1f}%) "
        f"| DOWN: {down_count} ({down_count/total*100:.1f}%)"
    )
    return df


# ─────────────────────────────────────────────
#  NETTOYAGE FINAL
# ─────────────────────────────────────────────

def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes NaN créées par les fenêtres glissantes.
    La SMA_50 a la fenêtre la plus grande (50 jours) →
    les 50 premières lignes seront supprimées.
    """
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    logger.info(f"   🧹 {dropped} lignes supprimées (fenêtres glissantes + target shift).")
    logger.info(f"   📊 Dataset final : {len(df)} lignes.")
    return df


# ─────────────────────────────────────────────
#  SAUVEGARDE
# ─────────────────────────────────────────────

def save_features(df: pd.DataFrame, pair: str = "EURUSD") -> Path:
    """Sauvegarde le dataset enrichi dans data/processed/."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    filepath = PROCESSED_DATA_DIR / f"{pair}_features.csv"
    df.to_csv(filepath, index=False)

    logger.info(f"💾 Features sauvegardées : {filepath}")
    logger.info(f"   Taille : {filepath.stat().st_size / 1024:.1f} Ko")
    return filepath


# ─────────────────────────────────────────────
#  RÉSUMÉ
# ─────────────────────────────────────────────

def display_features_summary(df: pd.DataFrame) -> None:
    """Affiche le résumé du dataset enrichi."""
    feature_cols = [c for c in df.columns if c not in ["Datetime", "Pair", "Signal"]]

    print("\n" + "═" * 60)
    print("  🧠  RÉSUMÉ — FEATURE ENGINEERING")
    print("═" * 60)
    print(f"  Lignes          : {len(df):,}")
    print(f"  Période         : {df['Datetime'].min().date()} → {df['Datetime'].max().date()}")
    print(f"  Features totales: {len(feature_cols)}")
    print(f"  Target          : Signal (0=DOWN / 1=UP)")
    print("─" * 60)
    print("  📋 Liste des features :")
    for i, col in enumerate(feature_cols, 1):
        print(f"     {i:2d}. {col}")
    print("─" * 60)
    print(f"  Distribution target :")
    print(f"     UP   📈 : {int(df['Signal'].sum()):,} "
          f"({df['Signal'].mean()*100:.1f}%)")
    print(f"     DOWN 📉 : {int((df['Signal']==0).sum()):,} "
          f"({(1-df['Signal'].mean())*100:.1f}%)")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────
#  PIPELINE PRINCIPALE
# ─────────────────────────────────────────────

def run_feature_engineering(pair: str = "EURUSD", save: bool = True) -> pd.DataFrame:
    """
    Pipeline complète de Feature Engineering.

    Étapes :
      1. Chargement des données brutes
      2. Indicateurs de tendance  (SMA, EMA, MACD)
      3. Indicateurs de momentum  (RSI, Stochastic)
      4. Indicateurs de volatilité (ATR, Bollinger Bands)
      5. Features de prix dérivées
      6. Target Signal (UP/DOWN)
      7. Nettoyage des NaN
      8. Sauvegarde

    Returns
    -------
    pd.DataFrame enrichi avec toutes les features
    """
    logger.info("=" * 55)
    logger.info("  🧠  PHASE 2 — FEATURE ENGINEERING")
    logger.info("=" * 55)

    df = load_raw_data(pair)

    logger.info("📐 Calcul des indicateurs :")

    # ── Tendance ─────────────────────────────────────────────────────
    df = add_moving_averages(df)
    df = add_macd(df)

    # ── Momentum ─────────────────────────────────────────────────────
    df = add_rsi(df)
    df = add_stochastic(df)

    # ── Volatilité ───────────────────────────────────────────────────
    df = add_atr(df)
    df = add_bollinger_bands(df)

    # ── Prix dérivés ─────────────────────────────────────────────────
    df = add_price_features(df)

    # ── Target ───────────────────────────────────────────────────────
    df = add_target(df)

    # ── Nettoyage ────────────────────────────────────────────────────
    df = drop_nulls(df)

    # ── Résumé ───────────────────────────────────────────────────────
    display_features_summary(df)

    # ── Sauvegarde ───────────────────────────────────────────────────
    if save:
        save_features(df, pair)

    logger.info("✅  Phase 2 terminée avec succès.\n")

    return df


# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = run_feature_engineering(pair="EURUSD", save=True)