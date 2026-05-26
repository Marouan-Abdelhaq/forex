import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from utils.data_loader import load_pair_data, load_report
from utils.data_loader import compute_technical_indicators 
from utils.styles import render_metric_card

PLOT_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="DM Sans", color="#c9d1d9", size=12),
    xaxis=dict(gridcolor="#21262d", zeroline=False, linecolor="#21262d", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#21262d", zeroline=False, linecolor="#21262d", tickfont=dict(size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#21262d"),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
)

def show_overview(sel_pair, pair_label):
    # ── 1. CHARGEMENT SÉCURISÉ DU RAPPORT ──
    report = load_report()
    summary = report.get("summary", [])
    
    # ── 2. LOGIQUE STRICTEMENT CENTRÉE "PAR PAIRE" (FIX ROBUSTE) ──
    current_pair_metrics = next((r for r in summary if r.get("pair") == sel_pair), None)

    # Variables par défaut pour éviter l'affichage de données incohérentes
    best_rmse = None
    best_model = "N/A"
    card_delta = "Données indisponibles"

    if current_pair_metrics:
        # Clés strictes alignées avec le fichier lab2_report.json
        lr = current_pair_metrics.get("LSTM_Simple_rmse")
        br = current_pair_metrics.get("BiGRU_MHA_rmse")

        if lr is not None and br is not None:
            best_rmse = min(lr, br)
            best_model = "LSTM Simple" if lr < br else "BiGRU + MHA"
            card_delta = f"Modèle optimal pour {sel_pair}"
        elif lr is not None:
            best_rmse = lr
            best_model = "LSTM Simple"
            card_delta = "BiGRU_MHA manquant"
        elif br is not None:
            best_rmse = br
            best_model = "BiGRU + MHA"
            card_delta = "LSTM manquant"

    # ── 3. AFFICHAGE DES CARTES DE MÉTRIQUES ──
    col1, col2, col3, col4 = st.columns(4)
    
    # Formatage de la valeur du RMSE
    rmse_display = f"{best_rmse:.5f}" if best_rmse is not None else "N/A"
    
    render_metric_card(
        col1,
        f"Best RMSE ({sel_pair})",
        rmse_display,
        f"Gagnant : {best_model}",
        False
    )
    render_metric_card(col2, "Paires Analysées", "3", "EUR/USD · EUR/MAD · USD/MAD")
    render_metric_card(col3, "Modèles Comparés", "2", "LSTM · BiGRU + MHA")
    render_metric_card(col4, "Fenêtre Historique", "2015–2025", "10 ans de données")

    # ── 4. RÉSUMÉ DE L'ARCHITECTURE ──
    st.markdown('<div class="section-title">📋 Résumé du Projet</div>', unsafe_allow_html=True)
    ci1, ci2 = st.columns(2)
    with ci1:
        st.markdown("""
        <div class="info-box">
            <b>🎯 Tâche :</b> Régression — prédire Close(t+1)<br>
            <b>📐 Fenêtre temporelle :</b> 60 jours d'historique glissant<br>
            <b>📏 Métrique maîtresse :</b> RMSE (Root Mean Squared Error)<br>
            <b>📦 Sources :</b> Yahoo Finance · Daily Forex Framework
        </div>""", unsafe_allow_html=True)
    with ci2:
        st.markdown("""
        <div class="info-box">
            <b>🔵 Modèle 1 :</b> LSTM Simple (Baseline)<br>
            &nbsp;&nbsp;&nbsp;→ 2 couches LSTM séquentielles · Dropout · Dense<br><br>
            <b>🔴 Modèle 2 :</b> BiGRU + Multi-Head Attention (Avancé)<br>
            &nbsp;&nbsp;&nbsp;→ Double bloc BiGRU · MHA (4 têtes) · Connexion résiduelle
        </div>""", unsafe_allow_html=True)

    # ── 5. GRAPHIQUE DES INDICATEURS TECHNIQUES TECHNIQUES REELS ──
    st.markdown(f'<div class="section-title">📈 Historique & Indicateurs Réels {pair_label}</div>', unsafe_allow_html=True)
    
    df = load_pair_data(sel_pair)
    df = compute_technical_indicators(df)

    fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.25, 0.25], shared_xaxes=True, vertical_spacing=0.05)

    # Étage 1 : Prix et Moyennes
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close", line=dict(color="#58a6ff", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_10"], name="SMA 10", line=dict(color="#f0883e", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_10"], name="EMA 10", line=dict(color="#bc8cff", width=1.2, dash="dot")), row=1, col=1)

    # Étage 2 : RSI
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI", line=dict(color="#d2a679", width=1.2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#f85149", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#3fb950", row=2, col=1)
    
    # Étage 3 : Volatilité & Rendements
    if "Volatility" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Volatility"], name="Volatilité (10d)", line=dict(color="#3fb950", width=1.5)), row=3, col=1)
    if "Returns" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Returns"], name="Rendements", line=dict(color="rgba(121, 192, 255, 0.4)", width=0.8), opacity=0.5), row=3, col=1)

    fig.update_layout(**PLOT_LAYOUT, height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # ── 6. TABLEAU COMPARATIF GLOBAL TRADITIONNEL ──
    if summary:
        st.markdown('<div class="section-title">🏆 Comparaison RMSE Globale</div>', unsafe_allow_html=True)
        rows_html = ""
        for row in summary:
            p = row.get("pair", "N/A")
            lr_row = row.get("LSTM_Simple_rmse")
            br_row = row.get("BiGRU_MHA_rmse")
            lp_row = row.get("LSTM_Simple_mape")
            bp_row = row.get("BiGRU_MHA_mape")
            
            l_cls, b_cls, winner = ("", "", "–")
            if lr_row is not None and br_row is not None:
                l_cls, b_cls = ("worse", "best") if br_row < lr_row else ("best", "worse")
                winner = "🥇 BiGRU+MHA" if br_row < lr_row else "🥇 LSTM"
                
            rows_html += f"""
            <tr>
                <td style="color:#58a6ff;font-weight:700">{p}</td>
                <td class="{l_cls}">{f'{lr_row:.6f}' if lr_row is not None else '–'}</td>
                <td class="{l_cls}">{f'{lp_row:.2f}%' if lp_row is not None else '–'}</td>
                <td class="{b_cls}">{f'{br_row:.6f}' if br_row is not None else '–'}</td>
                <td class="{b_cls}">{f'{bp_row:.2f}%' if bp_row is not None else '–'}</td>
                <td style="color:#3fb950">{winner}</td>
            </tr>"""
            
        st.markdown(f"""
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Paire</th><th>LSTM RMSE</th><th>LSTM MAPE</th>
                    <th>BiGRU+MHA RMSE</th><th>BiGRU+MHA MAPE</th><th>Meilleur</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)
