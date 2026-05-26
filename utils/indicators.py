import pandas as pd
import numpy as np

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les indicateurs techniques : RSI, MACD, SMA, Bollinger Bands."""
    df = df.copy()
    c = df["Close"].copy()

    df["SMA_20"] = c.rolling(20).mean()
    df["SMA_50"] = c.rolling(50).mean()
    df["EMA_12"] = c.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Sig"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    std = c.rolling(20).std()
    df["BB_Upper"] = df["SMA_20"] + 2 * std
    df["BB_Lower"] = df["SMA_20"] - 2 * std
    df["Returns"] = c.pct_change() * 100

    return df