import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# L'ordre absolu de l'entraînement
FEATURE_COLS = ["Close", "SMA_10", "EMA_10", "RSI", "MACD", "Returns", "Volatility"]

def make_sequences(data_matrix, window=60):
    X = []
    for i in range(window, len(data_matrix)):
        X.append(data_matrix[i-window:i, :])
    return np.array(X, dtype=np.float32)

def predict_future(model, scaler, df_historical, window=60, n_days=252):
    """
    Inférence itérative multivariée alignée.
    df_historical: DataFrame contenant les colonnes calculées par le data_loader
    """
    # Extraction de la matrice selon l'ordre strict de l'entraînement
    data_matrix = df_historical[FEATURE_COLS].values.astype(np.float32)
    
    # Normalisation complète de la matrice (N, 7)
    scaled_data = scaler.transform(data_matrix)
    
    # Extraction du seed de départ [1, window, 7]
    current_seq = scaled_data[-window:].reshape(1, window, 7).astype(np.float32)
    
    preds_s = []
    fut_dates = []
    
    # Récupération de la dernière vraie date du dataframe pour incrémenter
    last_dt = pd.to_datetime(df_historical["Date"].iloc[-1])
    
    for _ in range(n_days):
        # Prédiction du Close normalisé au pas t+1
        p = float(model.predict(current_seq, verbose=0)[0, 0])
        preds_s.append(p)
        
        # Calendrier FOREX (Saut des weekends)
        next_d = last_dt + timedelta(days=1)
        while next_d.weekday() >= 5:
            next_d += timedelta(days=1)
        last_dt = next_d
        fut_dates.append(next_d)
        
        # --- MISE À JOUR DU VECTEUR DE FEATURES (t+1) ---
        # On duplique le dernier vecteur connu pour conserver la cohérence des autres features
        next_features = np.copy(current_seq[0, -1, :])
        
        # Index 0 = STRICTEMENT LE CLOSE
        next_features[0] = p 
        
        # Glissement de la fenêtre 3D
        new_window = np.vstack([current_seq[0, 1:, :], next_features])
        current_seq = new_window.reshape(1, window, 7)
        
    # --- DÉNORMALISATION ALIGNÉE (INDEX 0) ---
    dummy_matrix = np.zeros((n_days, 7), dtype=np.float32)
    dummy_matrix[:, 0] = preds_s  # Index 0 correspond au Close dans le scaler
    
    fut_real = scaler.inverse_transform(dummy_matrix)[:, 0]
    
    return fut_real, fut_dates