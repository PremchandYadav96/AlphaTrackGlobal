# AlphaTrack Global Pro 🌍

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AlphaTrack Global Pro is a sophisticated, all-in-one stock portfolio tracking and analysis dashboard. Built with Python and Streamlit, it empowers users to monitor their investments across global exchanges, gain AI-driven insights, and make data-backed decisions.**

This application provides a seamless and intuitive interface for tracking holdings from NASDAQ (USA), NSE (India), and BSE (India), complete with real-time price data, multi-currency conversion, and in-depth analytical tools.

---

## ✨ Key Features

-   **Multi-Exchange Portfolio Tracking:** Add and manage stocks from NASDAQ, NSE, and BSE in a single, unified dashboard.
-   **Real-time Data:** Get live stock prices, performance metrics (P&L), and total portfolio value updated dynamically.
-   **Deep Dive Analysis:** For any stock in your portfolio, access:
    -   **💡 AI-Powered Advisory:** Simulated Buy/Sell/Hold signals based on technical indicators (RSI, SMA).
    -   **📈 Interactive Candlestick Charts:** Visualize historical performance with customizable time ranges and moving averages.
    -   **🔮 Predictive Forecasting:** A 10-day price forecast powered by Meta's Prophet model, complete with historical accuracy metrics (MAPE).
    -   **🏢 Company Profile:** View detailed company summaries and key financial ratios.
    -   **📰 Live News Feed:** Stay updated with the latest news for each stock, fetched directly from NewsAPI.
-   **Sector & Diversification Analysis:** Visualize your portfolio's allocation by sector and stock weight to identify concentration risks.
-   **Multi-Currency Support:** View your entire portfolio's value in your preferred currency (USD, INR, EUR, etc.).
-   **Transaction Logging:** Automatically maintains a clear and persistent log of all your buy transactions.
-   **Simulated Billing Model:** Features a "FlexPrice" usage and billing simulator to demonstrate a pay-per-use model for AI analyses and advice generation.
-   **Modern & Responsive UI:** A sleek, dark-themed interface designed for clarity and ease of use.

---

## 🚀 Getting Started

Follow these instructions to set up and run AlphaTrack Global Pro on your local machine.

### 1. Prerequisites

-   **Python:** Ensure you have Python 3.8 or newer installed.
-   **NewsAPI Key:** This project uses the [NewsAPI](https://newsapi.org/) to fetch stock-related news.
    -   Sign up for a free developer account on their website to get your API key.

### 2. Setup Instructions

**Step 1: Clone the Repository**
Open your terminal or command prompt and clone this repository to your local machine.
```bash
git clone https://github.com/PremchandYadav96/AlphaTrackGlobal.git
cd AlphaTrackGlobal
```

**Step 2: Create a Virtual Environment (Recommended)**
It's a best practice to create a virtual environment to manage project-specific dependencies.
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**Step 3: Install Required Libraries**
Install all the necessary Python libraries using the provided `requirements.txt` file.
```bash
pip install -r requirements.txt
```
*(Note: If a `requirements.txt` file is not present, you can install the packages manually: `pip install streamlit yfinance pandas pandas-ta plotly prophet requests`)*

**Step 4: Configure the API Key**
Open the `portfolio_dashboard.py` file in your code editor. Find the following line and replace the placeholder with your actual NewsAPI key:
```python
# Line 23 in portfolio_dashboard.py
NEWS_API_KEY = "YOUR_NEWS_API_KEY_HERE" # Replace with your key```

**Step 5: Run the Streamlit App**
You're all set! Run the following command in your terminal from the project's root directory.
```bash
streamlit run portfolio_dashboard.py
```Your web browser will automatically open with the AlphaTrack Global Pro dashboard running.

---

## 🛠️ Tech Stack

-   **Core Framework:** Streamlit
-   **Data Backend:** Python, Pandas
-   **Financial Data Source:** `yfinance`
-   **Technical Analysis:** `pandas_ta`
-   **Forecasting:** `prophet` (by Meta)
-   **Data Visualization:** Plotly Express & Graph Objects
-   **News Feed:** `requests` & NewsAPI

---

## 📂 Project Structure

```
AlphaTrackGlobal/
│
├── .streamlit/
│   └── config.toml        # Streamlit theme and configuration
│
├── portfolio_dashboard.py # Main application script
├── portfolio.csv          # Stores your stock holdings (auto-generated)
├── transactions.csv       # Stores your transaction history (auto-generated)
└── README.md              # You are here!
```

---

## 👥 Team

This project was developed by a dedicated team of collaborators:

-   V C Premchand Yadav
-   P R Kiran Kumar Reddy
-   Edupulapati Sai Praneeth
-   Liel Stephen
-   K Sri Harsha Vardhan
-   Suheb Nawab Sheikh

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/licenses/MIT) file for details.
