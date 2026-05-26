import streamlit as st

def show_about():
    st.title("ℹ️ À propos du Projet")
    
    # ── PRÉSENTATION ACADÉMIQUE ──
    st.markdown("""
    <div style="background-color: #161b22; padding: 20px; border-radius: 8px; border: 1px solid #21262d; margin-bottom: 25px;">
        <h2 style="color: #58a6ff; margin-top: 0;">💹 Forex AI Dashboard</h2>
        <p style="font-size: 1.1rem; color: #c9d1d9;">
            Plateforme décisionnelle et prédictive basée sur le Deep Learning pour l'analyse des marchés des changes (Forex).
        </p>
        <hr style="border-color: #21262d; margin: 15px 0;">
        <p style="margin-bottom: 5px;">🏫 <b>Université Sultan Moulay Slimane</b></p>
        <p style="margin-bottom: 5px;">🎓 <b>Faculté Polydisciplinaire de Khouribga (FPK)</b></p>
        <p style="margin-bottom: 5px;">🔬 Master / Filière : <b>SIIA</b> (Systèmes Intelligents et Intelligence Artificielle)</p>
        <p style="margin-bottom: 5px;">📅 Année Universitaire : <b>2025/2026</b></p>
        <p style="margin-top: 15px; color: #ff7b72;">👨‍🏫 Sous la haute bienveillance et l'encadrement de : <b>Pr. Ibtissam Bakkouri</b></p>
    </div>
    """, unsafe_allow_html=True)

    # ── DESCRIPTION ET OBJECTIFS ──
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<div class="section-title">📋 Description du Système</div>', unsafe_allow_html=True)
        st.write("""
        Ce tableau de bord applique des architectures de réseaux de neurones récurrents profonds pour modéliser 
        et projeter les dynamiques de prix sur trois paires de devises majeures et régionales :
        - **EUR / USD** (Euro / Dollar Américain)
        - **EUR / MAD** (Euro / Dirham Marocain)
        - **USD / MAD** (Dollar Américain / Dirham Marocain)
        
        Les données sources proviennent de l'API financière Yahoo Finance, garantissant un alignement précis avec les cours interbancaires.
        """)

    with col_right:
        st.markdown('<div class="section-title">🎯 Objectifs Scientifiques</div>', unsafe_allow_html=True)
        st.markdown("""
        1. **Modélisation** de séries temporelles financières non linéaires complexes.
        2. **Implémentation comparative** d'un modèle de référence LSTM.
        3. **Conception avancée** d'un modèle hybride BiGRU couplé à un mécanisme de Multi-Head Attention (MHA).
        4. **Évaluation quantitative** via les métriques standards ($RMSE$, $MAE$, $MAPE$, $R^2$).
        5. **Génération de scénarios** hors-échantillon à horizon 2026 par approches auto-récursives.
        6. **Inférence temps réel** synchronisée sur flux de liquidités.
        """)

    st.markdown("---")

    # ── STACK TECHNIQUE ──
    st.markdown('<div class="section-title">🛠️ Architecture Technique & Stack</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.info("""
        🤖 **AI & Deep Learning**
        - TensorFlow 2.x
        - Keras Architectures
        - Scikit-learn (Preprocessing)
        - NumPy Arrays
        """)
        
    with c2:
        st.info("""
        📊 **Données & Flux**
        - Pandas Engines
        - yfinance API
        - Yahoo Finance Backend
        """)
        
    with c3:
        st.info("""
        🎨 **Visualisation**
        - Streamlit Framework
        - Plotly Graphics
        - Custom Dark CSS
        """)
        
    with c4:
        st.info("""
        💾 **Infrastructure & Sécurisation**
        - Python Core
        - Joblib (Scalers Cache)
        - Native JSON I/O
        """)

    st.markdown("---")

    # ── INFRASTRUCTURE DES MODÈLES ──
    st.markdown('<div class="section-title">🧠 Représentation Topologique des Réseaux</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📝 Architecture 1 : LSTM Standard", "⚡ Architecture 2 : BiGRU + Multi-Head Attention"])
    
    with tab1:
        st.write("**Modèle séquentiel profond adapté aux dépendances à long terme :**")
        st.code("""
Input (Window_Size, 1)
   │
   ├───► LSTM (50 units, return_sequences=True)
   ├───► Dropout (Rate: 0.2)
   │
   ├───► LSTM (50 units, return_sequences=False)
   ├───► Dropout (Rate: 0.2)
   │
   ├───► Dense (25 neurons, Activation: ReLU)
   │
   └───► Dense (1 neuron, Activation: Linear) ──► Output: Close(t+1)
        """, language="text")

    with tab2:
        st.write("**Modèle hybride à haute capacité d'extraction avec fenêtrage d'attention :**")
        st.code("""
Input (Window_Size, 1)
   │
   ├───► Bidirectional(GRU) (64 units, return_sequences=True)
   │
   ├───► MultiHeadAttention (4 Heads, Key_Dim=64) ◄── (Mécanisme d'attention contextuel)
   │
   ├───► Skip Connection (Add / Résiduel)
   ├───► LayerNormalization
   │
   ├───► Bidirectional(GRU) (32 units, return_sequences=False)
   │
   ├───► Dense (32 neurons, Activation: Swish)
   │
   └───► Dense (1 neuron, Activation: Linear) ──► Output: Close(t+1)
        """, language="text")

    st.markdown("---")

    # ── DEPLOYMENT COMMANDS ──
    st.markdown('<div class="section-title">🚀 Déploiement & Initialisation du Serveur</div>', unsafe_allow_html=True)
    
    st.write("**1. Installation de l'environnement virtuel et des dépendances :**")
    st.code("""
pip install streamlit tensorflow yfinance plotly scikit-learn joblib streamlit-autorefresh
    """, language="bash")
    
    st.write("**2. Exécution de l'orchestrateur Streamlit à la racine :**")
    st.code("""
streamlit run streamlit_app.py
    """, language="bash")

    # ── FOOTER ──
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding: 15px; border-top: 1px solid #21262d; color: #8b949e; font-size: 0.8rem;">
        💹 Forex AI Dashboard • Conçu sous TensorFlow & Streamlit • USMS FP Khouribga — Master SIIA — 2025/2026
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_about()