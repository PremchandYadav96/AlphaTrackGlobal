# 🌍 AlphaTrack Global Pro

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B)

**AlphaTrack Global Pro** is an advanced stock portfolio tracking and analysis dashboard built with **Python** and **Streamlit**.
It empowers investors to seamlessly monitor holdings, analyze performance, and access AI-driven insights — all in one place.

Track investments across **NASDAQ (USA)**, **NSE (India)**, and **BSE (India)** with real-time prices, multi-currency conversion, and powerful analytics.

---

## 🚀 Quick Start (3 Steps)

Get the dashboard running in under a minute:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/PremchandYadav96/AlphaTrackGlobal.git
   cd AlphaTrackGlobal
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   (If requirements.txt is missing, install manually:)
   ```bash
   pip install streamlit yfinance pandas pandas-ta plotly prophet requests
   ```

3. **Run the App**
   ```bash
   streamlit run portfolio_dashboard.py
   ```

💡 For the news feature, add your free NewsAPI key to line 23 in `portfolio_dashboard.py`.

---

## ✨ Features

- **🌐 Multi-Exchange Portfolio Tracking** – Manage stocks from NASDAQ, NSE, and BSE in one unified dashboard.
- **📊 Real-Time Market Data** – Live prices, portfolio valuation, and P&L updates.
- **💡 AI Advisory** – Buy/Sell/Hold signals powered by RSI & SMA indicators.
- **📈 Interactive Charts** – Candlesticks, moving averages, and time-range filters.
- **🔮 Price Forecasting** – 10-day predictions using Meta’s Prophet with MAPE accuracy score.
- **🏢 Company Profiles** – Official summaries and key financial ratios.
- **📰 Live News Feed** – Stock-specific news via NewsAPI.
- **📌 Portfolio Diversification** – Sector/asset allocation visualization to spot risks.
- **💱 Multi-Currency Support** – Convert portfolio values to USD, INR, EUR, and more.
- **🗂 Transaction Logging** – Persistent auto-generated history of all buy operations.
- **🧮 FlexPrice Simulator** – Demo pay-per-use billing model for AI-powered analyses.
- **🌑 Modern Dark UI** – Sleek, responsive design for clarity and focus.

---

## 🔧 Detailed Installation Guide

### 1. Prerequisites

- Python 3.8+ installed
- NewsAPI Key (free signup at [newsapi.org](https://newsapi.org))

### 2. Setup Steps

**Step 1 – Clone the Repository**
```bash
git clone https://github.com/PremchandYadav96/AlphaTrackGlobal.git
cd AlphaTrackGlobal
```

**Step 2 – Create & Activate Virtual Environment (recommended)**
```bash
# Windows
python -m venv venv
.\\venv\\Scripts\\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**Step 3 – Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 – Add Your NewsAPI Key**
Edit line 23 in `portfolio_dashboard.py`:
```python
NEWS_API_KEY = "YOUR_NEWS_API_KEY_HERE"
```

**Step 5 – Run the Application**
```bash
streamlit run portfolio_dashboard.py
```
Your browser will launch automatically with the AlphaTrack Global Pro dashboard.

---

## 🛠 Tech Stack

- **Framework:** Streamlit
- **Backend & Data:** Python, Pandas
- **Market Data:** yfinance
- **Indicators:** pandas-ta
- **Forecasting:** Prophet (Meta)
- **Visualization:** Plotly (Express & Graph Objects)
- **News Feed:** NewsAPI (via Requests)

---

## 📂 Project Structure

```
AlphaTrackGlobal/
│
├── .streamlit/             # Theme & configuration
│   └── config.toml
│
├── portfolio_dashboard.py   # Main application script
├── portfolio.csv            # Auto-generated: stock holdings
├── transactions.csv         # Auto-generated: transaction history
└── README.md                # Project documentation
```

---

## 👥 Contributors

- V C Premchand Yadav
- P R Kiran Kumar Reddy
- Edupulapati Sai Praneeth
- Liel Stephen
- K Sri Harsha Vardhan
- Suheb Nawab Sheikh

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
