import streamlit as st

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0d1117;
    color: #e6edf3;
}
.main-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: "";
    position: absolute;
    top: -60%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(88,166,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    background: linear-gradient(90deg, #58a6ff, #79c0ff, #58a6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0; padding: 0;
}
.main-header p {
    color: #8b949e;
    margin: 0.4rem 0 0;
    font-size: 0.95rem;
}
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-card .label {
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #58a6ff;
    margin: 0.3rem 0;
}
.metric-card .delta {
    font-size: 0.78rem;
    color: #3fb950;
}
.metric-card .delta.neg { color: #f85149; }
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #58a6ff;
    border-left: 3px solid #58a6ff;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
}
.info-box {
    background: rgba(88,166,255,0.07);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #c9d1d9;
    margin: 0.8rem 0;
}
.comparison-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.comparison-table th {
    background: #21262d;
    color: #8b949e;
    padding: 0.7rem 1rem;
    text-align: center;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
}
.comparison-table td {
    padding: 0.65rem 1rem;
    text-align: center;
    border-bottom: 1px solid #21262d;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}
.comparison-table tr:hover td { background: #161b22; }
.best { color: #3fb950; font-weight: 700; }
.worse { color: #8b949e; }
</style>
"""

def inject_custom_styles():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

def render_metric_card(col, label, value, delta="", neg=False):
    delta_cls = "neg" if neg else ""
    col.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="delta {delta_cls}">{delta}</div>
    </div>""", unsafe_allow_html=True)