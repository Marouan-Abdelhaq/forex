# 💹 Forex AI — Deep Learning Research Dashboard

A research-oriented Deep Learning platform for analyzing and forecasting Forex time series using LSTM and Bidirectional GRU models with Multi-Head Attention.

The project provides an interactive Streamlit dashboard for dataset analysis, feature exploration, model training, historical prediction and future forecasting.

---

## 🎯 Project Objective

The objective of this project is to investigate whether Deep Learning models can learn meaningful patterns from historical Forex time series and generate useful forecasts.

This project answers a key question in quantitative finance:

> Can deep learning models generate robust and tradable alpha after accounting for real-world trading constraints?

The system focuses on three currency pairs:

- EUR/USD
- EUR/MAD
- USD/MAD

Rather than presenting the models as a production trading system, this project is designed as an academic and experimental research platform for studying time-series forecasting with Deep Learning.

---

## 🧠 Machine Learning Approach

The project explores two main neural network architectures:

### LSTM

Long Short-Term Memory networks are used to model temporal dependencies in historical Forex data.

### Bidirectional GRU + Multi-Head Attention

A Bidirectional GRU architecture combined with Multi-Head Attention is used to investigate whether attention mechanisms can improve the representation of temporal patterns.

---

## 📊 Data

The project currently works with:

| Currency Pair | Dataset               |
| ------------- | --------------------- |
| EUR/USD       | Historical Forex data |
| EUR/MAD       | Historical Forex data |
| USD/MAD       | Historical Forex data |

The raw datasets are stored in:

```text
data/raw/
```

---

## 🔬 Features & Analysis

The application provides tools for exploring the financial time series, including:

- Dataset exploration
- Technical indicators
- Feature analysis
- PCA-based feature exploration
- Historical prediction analysis
- Model comparison
- Future forecasting

---

## 🏗️ Project Architecture

```text
forex/
│
├── data/
│   └── raw/
│       ├── EURUSD.csv
│       ├── EURMAD.csv
│       └── USDMAD.csv
│
├── models/
│   ├── LSTM models
│   ├── BiGRU + Multi-Head Attention models
│   └── Scalers
│
├── notebooks/
│   ├── data_processor.ipynb
│   └── models_engine.ipynb
│
├── pages/
│   ├── overview.py
│   ├── dataset_analysis.py
│   ├── features_pca.py
│   ├── model_training.py
│   ├── predictions.py
│   ├── future_forecast.py
│   ├── live_forex.py
│   └── about.py
│
├── utils/
│   ├── data_loader.py
│   ├── indicators.py
│   ├── model_loader.py
│   ├── prediction_utils.py
│   └── styles.py
│
├── reports/
│
├── config.py
├── requirements.txt
└── streamlit_app.py
```

---

## 🖥️ Dashboard

The Streamlit dashboard provides the following sections:

### 🏠 Overview

General information about the selected Forex pair and project.

### 📊 Dataset Analysis

Exploration and visualization of historical Forex data.

### 🔬 Features & PCA

Feature analysis and dimensionality reduction using PCA.

### 🧠 Model Training

Configuration and training of the available Deep Learning models.

### 🎯 Predictions

Evaluation and visualization of model predictions on historical data.

### 🔮 Future Forecast

Recursive forecasting using the trained models.

### ⚡ Live Forex

Interface for working with live Forex market data.

---

## 🛠️ Tech Stack

**Programming**

- Python

**Data Science**

- Pandas
- NumPy
- Scikit-learn
- SciPy
- Joblib

**Deep Learning**

- TensorFlow
- Keras

**Financial Analysis**

- yfinance
- TA

**Visualization**

- Plotly
- Matplotlib

**Application**

- Streamlit

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Marouan-Abdelhaq/forex.git
cd forex
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

The application will then be available locally through Streamlit.

---

## 📈 Models

The project currently includes:

- LSTM
- Bidirectional GRU
- Multi-Head Attention

Pre-trained models are stored in `models/`.

The models are provided for experimentation and demonstration purposes.

---

## 🔮 Future Improvements

Possible extensions include:

- Transformer-based forecasting
- More advanced time-series architectures
- Multi-asset portfolio optimization
- Reinforcement Learning trading agents
- FastAPI model serving
- Docker deployment
- Real-time data pipelines
- Experiment tracking
- Automated model evaluation
- Walk-forward validation
- Backtesting and transaction-cost analysis

---

## 🎓 Academic Context

**Licence d'Excellence Project**
SIIA — Intelligent Systems & Artificial Intelligence
Faculty of Polydisciplinary Studies, Khouribga
Academic Year 2025–2026

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only.

Forecasting financial markets is inherently uncertain. The models presented in this repository should not be considered financial advice or a guaranteed trading strategy.

---

## 👤 Author

**Marouan Abdelhaq**
AI / Data / Software Engineering Student
GitHub: [Marouan-Abdelhaq](https://github.com/Marouan-Abdelhaq)
