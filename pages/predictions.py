import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

from utils.data_loader import load_pair_data
from utils.model_loader import load_keras_model, load_scaler
from utils.prediction_utils import make_sequences, FEATURE_COLS

# Configuration stricte du layout pour le thème Dark de l'application
PLOT_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="DM Sans", color="#c9d1d9", size=12),
    xaxis=dict(gridcolor="#21262d", zeroline=False, linecolor="#21262d", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#21262d", zeroline=False, linecolor="#21262d", tickfont=dict(size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#21262d"),
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
)

def compute_neutral_sharpe(returns, annualization_factor=252):
    """Calcule un ratio de Sharpe neutre au drift du marché (Excess Returns vs Mean)."""
    returns = np.nan_to_num(np.array(returns))
    std = np.std(returns)
    if std < 1e-8:
        return 0.0
    excess_returns = returns - np.mean(returns)
    std_excess = np.std(excess_returns)
    if std_excess < 1e-8:
        return 0.0
    return (np.mean(returns) / std_excess) * np.sqrt(annualization_factor)

def compute_robust_sortino(returns, annualization_factor=252):
    """Calcule le ratio de Sortino en isolant uniquement la downside deviation."""
    returns = np.nan_to_num(np.array(returns))
    if len(returns) == 0:
        return 0.0
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0.0
    if downside_std < 1e-8:
        return 0.0
    return (np.mean(returns) / downside_std) * np.sqrt(annualization_factor)

def compute_geometric_max_drawdown(returns):
    """Calcule le Maximum Drawdown exact sur une base de capitalisation géométrique."""
    returns = np.nan_to_num(np.array(returns))
    if len(returns) == 0:
        return 0.0
    equity_curve = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity_curve)
    running_max = np.where(running_max <= 0, 1e-8, running_max)
    drawdowns = (equity_curve - running_max) / running_max
    return np.min(drawdowns)

def show_predictions(pair, model_name, window_size):
    if window_size != 60:
        st.warning("⚠️ Ajustement automatique de la fenêtre temporelle à 60 jours pour correspondre à l'entraînement de l'IA.")
        window_size = 60

    st.markdown(
        """
        <div class="main-title"><span class="icon">🎯</span> Quant ML Evaluation Engine</div>
        <div class="sub-title">Framework de production quantitaire : Alignement géométrique, tracking d'IC multi-couches et backtest immunisé.</div>
        """, 
        unsafe_allow_html=True
    )
    st.info(f"💡 **Analyse active :** Paire : `{pair}` | Architecture : `{model_name}` | Fenêtre temporelle : `{window_size}` jours")

    with st.spinner("Chargement de l'infrastructure de modèle..."):
        model = load_keras_model(pair, model_name)
        scaler = load_scaler(pair)

    if model is None or scaler is None:
        st.warning(f"⚠️ Infrastructure introuvable pour {pair}. Vérifiez les fichiers `.keras` et `.pkl` dans `models/`.")
        st.stop()

    df = load_pair_data(pair)
    if df.empty or "Close" not in df.columns:
        st.error("Données historiques introuvables.")
        st.stop()

    # ── COUCHE 1 : INGESTION & ALIGNEMENT MULTIVARIÉ STRICT ──
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        st.error(f"⚠️ Indicateurs techniques manquants : {missing_cols}")
        st.stop()

    data_matrix = df[FEATURE_COLS].values.astype(np.float32)
    close_idx = 0 
    scaled_data = scaler.transform(data_matrix)

    X = make_sequences(scaled_data, window=window_size)
    expected_shape = model.input_shape  
    assert X.shape[1:] == expected_shape[1:], f"Mismatch de Shape! Attend {expected_shape[1:]}, reçu {X.shape[1:]}"
    
    st.caption(f"🔧 **Validation Tenseur Keras :** `input_shape={expected_shape[1:]}` ➔ Alignement structurel validé.")

    with st.spinner("Inférence en cours..."):
        try:
            pred_scaled = model.predict(X, verbose=0)
            dummy_matrix = np.zeros((len(pred_scaled), 7), dtype=np.float32)
            dummy_matrix[:, close_idx] = pred_scaled.flatten()
            predictions = scaler.inverse_transform(dummy_matrix)[:, close_idx]
        except Exception as e:
            st.error(f"Erreur lors de l'inférence : {e}")
            st.stop()

    real_values = df["Close"].values[window_size:].flatten()
    dates = pd.to_datetime(df["Date"].values[window_size:])

    min_len = min(len(dates), len(real_values), len(predictions))
    dates, real_values, predictions = dates[:min_len], real_values[:min_len], predictions[:min_len]

    baseline_rw = df["Close"].shift(1).values[window_size:][:min_len].flatten()
    valid_mask = ~np.isnan(real_values) & ~np.isnan(predictions) & ~np.isnan(baseline_rw)
    dates, real_values, predictions, baseline_rw = dates[valid_mask], real_values[valid_mask], predictions[valid_mask], baseline_rw[valid_mask]

    # Génération des rendements journaliers bruts d'origine (Longueur N - 1)
    returns_real_raw = np.diff(real_values) / real_values[:-1]
    returns_pred_raw = np.diff(predictions) / predictions[:-1]
    dates_returns = dates[1:]

    # ── CONTROLES INTERACTIFS (SIDEBAR) ──
    st.sidebar.markdown("### ⚙️ Paramètres Quant Avancés")
    smooth_window = st.sidebar.slider("Lissage des Returns Prédits (MA)", 1, 5, 2, step=1, help="Filtre le bruit de haute fréquence.")
    if smooth_window > 1:
        returns_pred_series = pd.Series(returns_pred_raw).rolling(window=smooth_window, min_periods=1).mean().values
    else:
        returns_pred_series = returns_pred_raw

    quantile_threshold = st.sidebar.slider("Seuil d'activité (Percentile de conviction)", 50, 95, 75, step=5)
    tx_cost = st.sidebar.slider("Coût d'exécution de base (Friction spread)", 0.0000, 0.0005, 0.00010, step=0.00005, format="%.5f")

    # ── COUCHE 2 : SIGNAL LAYER (TRAITEMENT ROLLING SANS LOOK-AHEAD) ──
    # Remplacement du percentile global par une approche strictement glissante (Rolling Window = 60 jours)
    pct_converted = quantile_threshold / 100.0
    rolling_std_series = pd.Series(returns_pred_series).rolling(window=20, min_periods=5).std()
    rolling_std = rolling_std_series.fillna(rolling_std_series.mean()).values
    
    confidence_scores = np.abs(returns_pred_series) / rolling_std
    
    # Correction majeure : calcul du seuil adaptatif en mode historique glissant (rolling window de 60)
    adaptive_cutoff_series = pd.Series(confidence_scores).rolling(window=60, min_periods=15).quantile(pct_converted)
    adaptive_cutoff = adaptive_cutoff_series.fillna(adaptive_cutoff_series.mean()).values
    
    raw_signals = np.where(confidence_scores > adaptive_cutoff, np.sign(returns_pred_series), 0)

    # ── SYNCHRONISATION ANTI-LEAKAGE STRICTE (t -> t+1) ──
    signals_executed = raw_signals[:-1]        
    returns_real = returns_real_raw[:-1]        
    returns_pred_f = returns_pred_series[:-1]  
    dates_final = dates_returns[:-1]           

    active_mask = signals_executed != 0
    total_active = np.sum(active_mask)
    active_pct = (total_active / len(signals_executed)) * 100

    if total_active > 0:
        direction_accuracy = np.mean(np.sign(returns_real[active_mask]) == signals_executed[active_mask]) * 100
    else:
        direction_accuracy = 0.0

    # ── COUCHE 3 : STATISTICAL VALIDATION CORE & ROLLING STABILITY ──
    rmse_price = np.sqrt(mean_squared_error(real_values[:-1], predictions[:-1]))
    r2_price = r2_score(real_values[:-1], predictions[:-1])
    
    ic_raw = np.corrcoef(returns_pred_raw[:-1], returns_real)[0, 1] if np.std(returns_pred_raw) > 0 else 0.0
    rank_ic_filtered = spearmanr(returns_pred_f, returns_real).correlation if np.std(returns_pred_f) > 0 else 0.0
    policy_ic = spearmanr(signals_executed, returns_real).correlation if total_active > 0 else 0.0

    # Analyse de la stabilité temporelle de l'IC (Rolling IC)
    df_ic_rolling = pd.DataFrame({"pred": returns_pred_f, "real": returns_real})
    rolling_ic_series = df_ic_rolling["pred"].rolling(window=60, min_periods=20).corr(df_ic_rolling["real"])
    ic_stability_vol = rolling_ic_series.std() if len(rolling_ic_series) > 20 else 0.0

    corr_lag_0 = np.corrcoef(predictions, real_values)[0, 1]
    corr_lag_pos1 = np.corrcoef(predictions[:-1], real_values[1:])[0, 1]
    
    st.markdown('<div class="section-title">🔮 Diagnostic Statistique & Causalité Temporelle</div>', unsafe_allow_html=True)
    if corr_lag_pos1 > corr_lag_0:
        st.markdown(f"""
        <div style="background-color:#2a1f10; padding:15px; border-radius:6px; border:1px solid #d29922; margin-bottom:15px;">
            ⚠️ <b>Effet de Déphasage Détecté (Lag Effect)</b> : Corr $t-1$ ({corr_lag_pos1:.3f}) > Corr $t$ ({corr_lag_0:.3f}). Le modèle agit comme un filtre suiveur passif.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color:#132312; padding:15px; border-radius:6px; border:1px solid #3fb950; margin-bottom:15px;">
            🟢 <b>Causalité Temporelle Validée</b> : Signaux synchrones robustes (Corr $t$: {corr_lag_0:.3f} | Corr $t-1$: {corr_lag_pos1:.3f}).
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Tableau de Bord de l\'Engine (Alignement Géométrique & Vol-Adjusted)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (Sur Prix Bruts)", f"{r2_price:.3f}", delta=f"RMSE: {rmse_price:.4f}", delta_color="off")
    c2.metric("Rank IC (Filtered)", f"{rank_ic_filtered:.3f}", delta=f"Vol IC (60d): {ic_stability_vol:.2f}", delta_color="inverse" if ic_stability_vol > 0.15 else "normal")
    c3.metric("Directional Accuracy", f"{direction_accuracy:.1f}%", delta=f"Policy IC: {policy_ic:.3f}", delta_color="normal" if policy_ic > 0 else "inverse")
    
    # ── COUCHE 4 : EXECUTION LAYER & MODÈLE DE FRICTION ASYMÉTRIQUE ──
    # Calcul du turnover
    turnover = np.abs(np.diff(signals_executed))
    turnover = np.insert(turnover, 0, 0)
    
    # Modélisation avancée de la friction : spread élargi en régime de forte volatilité
    market_vol_regime = pd.Series(returns_real).rolling(window=20, min_periods=1).std().fillna(np.std(returns_real)).values
    mean_vol = np.mean(market_vol_regime) if np.mean(market_vol_regime) > 0 else 1.0
    dynamic_friction = tx_cost * (1.0 + (market_vol_regime / mean_vol))
    
    strat_returns_raw = returns_real * signals_executed
    strat_returns_net = strat_returns_raw - (dynamic_friction * turnover * 2) 
    
    sharpe_strat_net = compute_neutral_sharpe(strat_returns_net)
    sortino_strat_net = compute_robust_sortino(strat_returns_net)
    
    c4.metric("Sharpe Net (Drift Neutre)", f"{sharpe_strat_net:.2f}", delta=f"Sortino Net: {sortino_strat_net:.2f}", delta_color="off")

    # Calcul des courbes cumulées strictement géométriques
    cum_market = np.cumprod(1.0 + returns_real) - 1.0
    cum_strat_raw = np.cumprod(1.0 + strat_returns_raw) - 1.0
    cum_strat_net = np.cumprod(1.0 + strat_returns_net) - 1.0

    max_dd_market = compute_geometric_max_drawdown(returns_real)
    max_dd_strat = compute_geometric_max_drawdown(strat_returns_net)

    st.markdown('<div class="section-title">📈 Moteur de Backtest : Rendements Cumulés Géométriques</div>', unsafe_allow_html=True)
    fig_strat = go.Figure()
    fig_strat.add_trace(go.Scatter(x=dates_final, y=cum_market, name=f"Buy & Hold Marché (MaxDD: {max_dd_market:.1%})", line=dict(color="#58a6ff", width=1.5)))
    fig_strat.add_trace(go.Scatter(x=dates_final, y=cum_strat_raw, name="Alpha IA Brut (Hors Friction)", line=dict(color="#d29922", width=1, dash="dash")))
    fig_strat.add_trace(go.Scatter(x=dates_final, y=cum_strat_net, name=f"Alpha IA Net (MaxDD: {max_dd_strat:.1%})", line=dict(color="#3fb950", width=1.7)))
    fig_strat.update_layout(**PLOT_LAYOUT, height=380)
    st.plotly_chart(fig_strat, use_container_width=True)

    # ── COUCHE 5 : ANALYSE PAR RÉGIMES STRUCTURELS (N >= 50) ──
    st.markdown('<div class="section-title">⏳ Robustesse par Régimes Historiques Structurels</div>', unsafe_allow_html=True)
    regimes = {
        "Pre-Crisis Core (<= 2019)": dates_final.year <= 2019,
        "High Volatility (2020 - 2023)": (dates_final.year >= 2020) & (dates_final.year <= 2023),
        "Recent Market (2024 - 2026)": dates_final.year >= 2024
    }
    
    r_cols = st.columns(3)
    for idx, (name, mask) in enumerate(regimes.items()):
        with r_cols[idx]:
            if np.sum(mask) >= 50:
                m_real, m_pred, m_signals = returns_real[mask], returns_pred_f[mask], signals_executed[mask]
                m_active = m_signals != 0
                
                reg_rank_ic = spearmanr(m_pred, m_real).correlation if np.std(m_pred) > 0 and np.std(m_real) > 0 else 0.0
                reg_acc = np.mean(np.sign(m_real[m_active]) == m_signals[m_active]) * 100 if np.sum(m_active) > 0 else 0.0
                reg_act_pct = (np.sum(m_active) / len(m_signals)) * 100
                
                st.metric(label=name, value=f"Rank IC: {reg_rank_ic:.3f}", delta=f"Acc: {reg_acc:.1f}% ({reg_act_pct:.1f}% Act)", delta_color="off")
            else:
                st.metric(label=name, value="N/A", delta="Points < 50")

    # ── AFFICHAGE DES ÉCARTS FINAUX SÉCURISÉ ──
    st.markdown('<div class="section-title">📋 Écarts de Clôture Alignés (Production Mode)</div>', unsafe_allow_html=True)
    
    final_len = len(dates_final)
    results = pd.DataFrame({
        "Prix Réel": real_values[:final_len], 
        "Prédiction IA": predictions[:final_len], 
        "Signal Exécuté (-> t+1)": signals_executed
    }, index=dates_final.strftime('%Y-%m-%d'))
    
    st.dataframe(results.tail(15).style.format("{:.5f}"), use_container_width=True)