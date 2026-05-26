import streamlit as st
import plotly.graph_objects as go
import json
from pathlib import Path

# Configuration du layout graphique
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

# Fonction de chargement sécurisée de l'historique d'entraînement
def load_training_history(pair, model_name):
    base_path = Path(__file__).resolve().parent.parent / "models"
    history_file = base_path / f"{pair}_{model_name}_history.json"
    
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f), True
    return {}, False

# 👉 FIX DE LA SIGNATURE : Acceptation des 5 arguments envoyés par la sidebar
def show_model_training(sel_pair, sel_model, epochs_val, batch_size_val, window_size_val):
    # ── 1. TITRES DE LA PAGE ──
    st.markdown(
        """
        <div class="main-title"><span class="icon">🧠</span> Model Training</div>
        <div class="sub-title">Entraînement des modèles en temps réel et suivi des courbes de perte (Loss)</div>
        """, 
        unsafe_allow_html=True
    )
    
    # Dynamisation du titre de section avec le modèle choisi
    model_label = sel_model.replace("_", " ")
    st.markdown(f"### 🧠 Suivi de l'Entraînement — {sel_pair} ({model_label})")
    st.caption("Visualisation des courbes d'apprentissage réelles extraites de l'historique TensorFlow.")
    
    # Chargement de l'historique d'apprentissage réel basé sur le modèle sélectionné
    history_data, success = load_training_history(sel_pair, sel_model)
    
    if success:
        st.success(f"✅ Historique d'apprentissage réel chargé avec succès depuis {sel_pair}_{sel_model}_history.json.")
    else:
        st.warning(f"⚠️ Historique réel introuvable pour {sel_pair} ({model_label}). Affichage des données de référence.")
        history_data = {
            "loss": [0.042, 0.031, 0.024, 0.018, 0.014, 0.011, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.0035, 0.003, 0.0028, 0.0025, 0.0022, 0.002, 0.0019, 0.0018, 0.0017, 0.0016, 0.0015, 0.0014, 0.0014, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.001336],
            "val_loss": [0.125, 0.082, 0.054, 0.041, 0.032, 0.021, 0.016, 0.012, 0.011, 0.010, 0.009, 0.0085, 0.008, 0.0078, 0.0075, 0.0072, 0.007, 0.0068, 0.0065, 0.008, 0.0072, 0.0068, 0.0064, 0.0061, 0.0059, 0.0055, 0.0052, 0.0049, 0.0046, 0.0042, 0.0039, 0.0036, 0.0032, 0.0031, 0.0029, 0.0048, 0.0024, 0.0019, 0.0016, 0.001509]
        }

    # ── 2. SECTION HYPERPARAMÈTRES CONNECTÉE AUX SLIDERS ──
    st.markdown("#### ⚙️ Hyperparamètres de la Session")
    
    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    with h_col1:
        st.metric(label="Modèle / Optimiseur", value=model_label, delta="Adam (LR: 0.001)", delta_color="off")
    with h_col2:
        st.metric(label="Fonction de perte (Loss)", value="MSE", delta="Mean Squared Error", delta_color="off")
    with h_col3:
        # Utilisation de la valeur du selectbox de la sidebar
        st.metric(label="Taille de batch (Batch Size)", value=f"{batch_size_val}", delta="Échantillons", delta_color="off")
    with h_col4:
        # Affiche la longueur réelle ou la valeur cible du slider
        total_epochs = len(history_data.get('loss', []))
        st.metric(label="Époques (Réelles / Cibles)", value=f"{total_epochs}", delta=f"Slider: {epochs_val} (W: {window_size_val}d)", delta_color="off")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3. GRAPHIQUE DES COURBES DE CONVERGENCE ──
    st.markdown("#### 📉 Courbes de convergence (Train vs Validation Loss)")
    
    epochs_range = list(range(1, len(history_data.get("loss", [])) + 1))
    
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=epochs_range, y=history_data.get("loss", []),
        name="Entraînement (Train Loss)",
        line=dict(color="#58a6ff", width=2)
    ))
    fig_loss.add_trace(go.Scatter(
        x=epochs_range, y=history_data.get("val_loss", []),
        name="Validation (Val Loss)",
        line=dict(color="#f85149", width=2, dash="dash")
    ))
    
    fig_loss.update_layout(**PLOT_LAYOUT, height=380)
    fig_loss.update_xaxes(title_text="Époques")
    fig_loss.update_yaxes(title_text="MSE (Échelle des prix normalisés)")
    
    st.plotly_chart(fig_loss, use_container_width=True)

    # ── 4. EN-TÊTE ANALYTIQUE DE FIN DE SESSION ──
    final_train = history_data.get("loss", [])[-1] if history_data.get("loss", []) else 0.0
    final_val = history_data.get("val_loss", [])[-1] if history_data.get("val_loss", []) else 0.0
    
    st.info(f"💡 **Rapport analytique :** Perte finale d'entraînement : `{final_train:.6f}` | Perte de généralisation : `{final_val:.6f}`.")