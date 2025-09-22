import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import os
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
import requests

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AlphaTrack Global Pro", page_icon="🌍")

# --- API Keys ---
# For local development, it's okay to have it here.
# For deployment, use st.secrets for security.
NEWS_API_KEY = "8083462b641b4fc1ae785c4a89c57d06"

# --- Data Persistence ---
PORTFOLIO_FILE = 'portfolio.csv'
TRANSACTIONS_FILE = 'transactions.csv'

def load_portfolio():
    """Loads portfolio data from a CSV file."""
    return pd.read_csv(PORTFOLIO_FILE) if os.path.exists(PORTFOLIO_FILE) else pd.DataFrame(
        columns=['ticker', 'display_ticker', 'exchange', 'shares', 'purchase_date', 'purchase_price']
    )

def save_portfolio(df):
    """Saves portfolio data to a CSV file."""
    df.to_csv(PORTFOLIO_FILE, index=False)

def load_transactions():
    """Loads transaction history from a CSV file."""
    return pd.read_csv(TRANSACTIONS_FILE) if os.path.exists(TRANSACTIONS_FILE) else pd.DataFrame(
        columns=['Date', 'Ticker', 'Exchange', 'Action', 'Shares', 'Price']
    )

def save_transactions(df):
    """Saves transaction history to a CSV file."""
    df.to_csv(TRANSACTIONS_FILE, index=False)

# --- Data Fetching & Caching ---
@st.cache_data
def get_stock_data(ticker, period="1y", interval="1d"):
    """
    Fetches historical stock data and flattens the column index for compatibility
    with technical analysis libraries.
    """
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def get_current_price(ticker):
    """Fetches the most recent price for a given ticker."""
    try:
        price_data = yf.Ticker(ticker).history(period='1d', interval='1m')
        return float(price_data['Close'].iloc[-1]) if not price_data.empty else None
    except Exception:
        return None

@st.cache_data
def get_company_info(ticker):
    """Fetches company information from Yahoo Finance."""
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

@st.cache_data
def get_exchange_rate(base_currency, target_currency):
    """Fetches the exchange rate between two currencies."""
    if base_currency == target_currency:
        return 1.0
    try:
        ticker = f"{base_currency}{target_currency}=X"
        data = yf.Ticker(ticker).history(period='1d')
        return data['Close'].iloc[-1] if not data.empty else 1.0
    except Exception:
        return 1.0

@st.cache_data
def get_stock_news(ticker_symbol, company_name):
    """Fetches the latest news for a stock."""
    try:
        query = f'"{company_name}" OR "{ticker_symbol}"'
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize=10"
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get('articles', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch news: {e}")
        return []

# --- AI/Gemini API Simulation & Analysis ---
def get_gemini_advice(hist_data):
    """
    Simulated function to generate trading advice based on technical indicators.
    """
    if len(hist_data) < 50:
        return "Hold", "Insufficient data for a reliable recommendation. A longer history is needed to identify clear trends."

    last_price = hist_data['Close'].iloc[-1]
    hist_data.ta.rsi(length=14, append=True)
    hist_data.ta.sma(length=20, append=True)
    hist_data.ta.sma(length=50, append=True)

    rsi = hist_data['RSI_14'].iloc[-1]
    sma20 = hist_data['SMA_20'].iloc[-1]
    sma50 = hist_data['SMA_50'].iloc[-1]

    if last_price > sma20 and last_price > sma50 and rsi < 70:
        return "Potential Buy", f"The stock shows a positive momentum, trading at {last_price:.2f} above its 20-day ({sma20:.2f}) and 50-day ({sma50:.2f}) moving averages. The RSI at {rsi:.2f} is bullish but not yet in overbought territory, suggesting potential for further upside."
    elif last_price < sma20 and last_price < sma50 and rsi > 30:
        return "Potential Sell", f"A bearish trend is observed, with the price at {last_price:.2f} below its key moving averages (20-day: {sma20:.2f}, 50-day: {sma50:.2f}). While the RSI at {rsi:.2f} isn't oversold, the trend suggests caution."
    else:
        return "Hold", f"The market signals are currently mixed. The price ({last_price:.2f}) is likely in a consolidation phase, caught between its key moving averages (20-day: {sma20:.2f}, 50-day: {sma50:.2f}). An RSI of {rsi:.2f} confirms this neutral stance."

def get_gemini_portfolio_analysis(portfolio_df):
    """
    Simulated function for a deep portfolio analysis.
    """
    analysis_text = ""
    if portfolio_df.empty:
        return "Your portfolio is empty. No analysis to perform."

    total_value = portfolio_df['current_value'].sum()
    if total_value > 0:
        portfolio_df['weight'] = portfolio_df['current_value'] / total_value
        heavy_stocks = portfolio_df[portfolio_df['weight'] > 0.4]
        if not heavy_stocks.empty:
            for _, row in heavy_stocks.iterrows():
                 analysis_text += f"\\n- **High Concentration Risk:** Your portfolio is heavily weighted in **{row['display_ticker']}** ({row['weight']:.1%}). This concentration increases risk. Consider diversifying into other stocks or sectors to mitigate potential losses if this stock underperforms."

    sector_counts = portfolio_df['sector'].nunique()
    if sector_counts <= 2:
        analysis_text += f"\\n- **Low Sector Diversification:** Your portfolio is only invested in {sector_counts} sector(s). This exposes you to sector-specific downturns. Exploring investments in other sectors like Technology, Healthcare, or Consumer Staples could provide better risk balance."

    if not analysis_text:
        analysis_text = "Your portfolio shows good initial diversification across multiple stocks and sectors. Continue monitoring and rebalancing as market conditions change."

    return "### AI-Powered Portfolio Analysis\\n" + analysis_text


# --- Main App ---
st.title("AlphaTrack Global Pro 🌍")
portfolio_df = load_portfolio()
transactions_df = load_transactions()

if 'advice_count' not in st.session_state: st.session_state.advice_count = 0
if 'analysis_count' not in st.session_state: st.session_state.analysis_count = 0

with st.sidebar:
    st.header("💼 Add to Portfolio")
    exchange = st.radio("Select Exchange", ('NASDAQ (USA)', 'NSE (India)', 'BSE (India)'), horizontal=True, index=0)
    ticker_input = st.text_input("Ticker Symbol", placeholder="e.g., AAPL, RELIANCE, or 500325").upper()
    shares = st.number_input("Number of Shares", min_value=1, value=1, step=1)
    purchase_date_input = st.date_input("Purchase Date")

    if st.button("Add to Portfolio", type="primary"):
        if ticker_input and shares > 0:
            ticker_with_suffix = ticker_input
            if exchange == 'NSE (India)': ticker_with_suffix += ".NS"
            elif exchange == 'BSE (India)': ticker_with_suffix += ".BO"
            try:
                price_data = yf.download(ticker_with_suffix, start=purchase_date_input, end=purchase_date_input + timedelta(days=5), progress=False)
                if price_data.empty: raise ValueError("No data found for the selected date.")
                
                # *** BUG FIX HERE ***
                # Changed .iloc to .iloc[0] to select the first row's value.
                price_value = float(price_data['Close'].iloc[0])

                new_row = pd.DataFrame([{'ticker': ticker_with_suffix, 'display_ticker': ticker_input, 'exchange': exchange, 'shares': shares, 'purchase_date': purchase_date_input.strftime('%Y-%m-%d'), 'purchase_price': price_value}])
                portfolio_df = pd.concat([portfolio_df, new_row], ignore_index=True)
                save_portfolio(portfolio_df)
                info = get_company_info(ticker_with_suffix)
                currency_symbol = '₹' if info.get('currency') == 'INR' else '$'
                new_transaction = pd.DataFrame([{'Date': date.today().strftime("%Y-%m-%d"), 'Ticker': ticker_input, 'Exchange': exchange, 'Action': 'BUY', 'Shares': shares, 'Price': f"{currency_symbol}{price_value:,.2f}"}])
                transactions_df = pd.concat([transactions_df, new_transaction], ignore_index=True)
                save_transactions(transactions_df)
                st.rerun()
            except Exception as e:
                st.error(f"Could not add {ticker_input}. Check the ticker symbol for the selected exchange or the purchase date. Error: {e}")

    st.header("⚙️ Settings")
    selected_currency = st.selectbox("Display Currency", ["USD", "INR", "EUR", "GBP", "CAD", "JPY"])
    currency_symbols = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "CAD": "C$", "JPY": "¥"}

    with st.expander("💳 FlexPrice Billing & Usage", expanded=True):
        st.write("Pay-per-use model simulation.")
        col1, col2 = st.columns(2)
        col1.metric("Advice Generated", st.session_state.advice_count)
        col2.metric("Portfolios Analyzed", st.session_state.analysis_count, help="Deep portfolio analysis runs.")
        advice_cost = st.session_state.advice_count * 0.50
        analysis_cost = st.session_state.analysis_count * 2.00
        total_cost = advice_cost + analysis_cost
        st.info(f"Advice Cost: ${advice_cost:.2f} (@ $0.50/advice)")
        st.info(f"Analysis Cost: ${analysis_cost:.2f} (@ $2.00/analysis)")
        st.success(f"**Total Estimated Bill: ${total_cost:.2f}**")

if portfolio_df.empty:
    st.info("Your portfolio is empty. Add a stock from the sidebar to begin tracking.")
else:
    infos = {ticker: get_company_info(ticker) for ticker in portfolio_df['ticker'].unique()}
    portfolio_df['info'] = portfolio_df['ticker'].map(infos)
    portfolio_df['base_currency'] = portfolio_df['info'].apply(lambda x: x.get('currency', 'USD'))
    rates = {curr: get_exchange_rate(curr, selected_currency) for curr in portfolio_df['base_currency'].unique()}
    portfolio_df['conversion_rate'] = portfolio_df['base_currency'].map(rates)
    live_prices = {ticker: get_current_price(ticker) for ticker in portfolio_df['ticker'].tolist()}
    portfolio_df['current_price_native'] = portfolio_df['ticker'].map(live_prices).fillna(0)
    portfolio_df['purchase_price_converted'] = portfolio_df['purchase_price'] * portfolio_df['conversion_rate']
    portfolio_df['current_price_converted'] = portfolio_df['current_price_native'] * portfolio_df['conversion_rate']
    portfolio_df['current_value'] = portfolio_df['shares'] * portfolio_df['current_price_converted']
    portfolio_df['investment'] = portfolio_df['shares'] * portfolio_df['purchase_price_converted']
    portfolio_df['pnl'] = portfolio_df['current_value'] - portfolio_df['investment']
    portfolio_df['sector'] = portfolio_df['info'].apply(lambda x: x.get('sector', 'N/A'))

    tab1, tab2, tab3 = st.tabs(["📊 Portfolio Overview", "🔬 Deep Dive Analysis", "📜 Transaction History"])

    with tab1:
        st.subheader("Overall Portfolio Performance")
        total_investment = portfolio_df['investment'].sum()
        total_current_value = portfolio_df['current_value'].sum()
        total_pnl = portfolio_df['pnl'].sum()
        pnl_percent = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        display_currency_symbol = currency_symbols.get(selected_currency, "$")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Investment", f"{display_currency_symbol}{total_investment:,.2f}")
        col2.metric("Current Value", f"{display_currency_symbol}{total_current_value:,.2f}")
        col3.metric("Profit & Loss", f"{display_currency_symbol}{total_pnl:,.2f}", delta=f"{pnl_percent:.2f}%")
        if st.button("Run AI Portfolio Analysis"):
            st.session_state.analysis_count += 1
            analysis_result = get_gemini_portfolio_analysis(portfolio_df)
            st.markdown(analysis_result)
        st.subheader("Sector Allocation")
        fig_sunburst = px.sunburst(portfolio_df, path=['sector', 'display_ticker'], values='current_value', title="Portfolio Allocation by Sector and Stock", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_sunburst.update_traces(textinfo="label+percent entry")
        st.plotly_chart(fig_sunburst, use_container_width=True)
        st.subheader("Holdings Details")
        for idx, row in portfolio_df.iterrows():
            with st.container(border=True):
                st.subheader(f"{row['display_ticker']} ({row['info'].get('longName', '')})")
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Current Value", f"{display_currency_symbol}{row['current_value']:,.2f}")
                row_pnl_percent = (row['pnl'] / row['investment'] * 100) if row['investment'] > 0 else 0
                kpi2.metric("Profit/Loss", f"{display_currency_symbol}{row['pnl']:,.2f}", delta=f"{row_pnl_percent:.2f}%")
                kpi3.metric("Shares", f"{row['shares']}")
                kpi4.metric("Avg. Purchase Price", f"{display_currency_symbol}{row['purchase_price_converted']:,.2f}")

    with tab2:
        st.subheader("Individual Stock Deep Dive")
        selected_display_ticker = st.selectbox("Select a stock for analysis:", options=portfolio_df['display_ticker'].unique())
        if selected_display_ticker:
            # *** BUG FIX HERE ***
            # Changed .iloc to .iloc[0] to select the single matching row.
            item = portfolio_df[portfolio_df['display_ticker'] == selected_display_ticker].iloc[0]
            
            ticker, info = item['ticker'], item['info']
            native_currency_symbol = '₹' if info.get('currency') == 'INR' else '$'
            
            # *** BUG FIX HERE ***
            # The tabs must be defined this way to work correctly.
            advisory_tab, chart_tab, profile_tab, forecast_tab, news_tab = st.tabs(["💡 Advisory", "📈 Chart", "🏢 Profile", "🔮 Forecast", "📰 News"])

            with advisory_tab:
                st.subheader(f"Actionable Advisory for {selected_display_ticker}")
                if st.button("Generate AI Advice", key=f"advice_{ticker}", type="primary"):
                    st.session_state.advice_count += 1
                    with st.spinner("Calling AI Advisor..."):
                         hist_data_for_advice = get_stock_data(ticker, period="1y")
                         signal, reason = get_gemini_advice(hist_data_for_advice)
                         if signal == "Potential Buy": st.success(f"**Signal: {signal}**")
                         elif signal == "Potential Sell": st.error(f"**Signal: {signal}**")
                         else: st.info(f"**Signal: {signal}**")
                         st.write("**Reasoning:**")
                         st.write(reason)

            with chart_tab:
                time_range = st.radio("Time Range", ["1M", "3M", "6M", "1Y", "5Y", "All"], horizontal=True, index=3, key=f"chart_{ticker}")
                period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y", "All": "max"}
                hist_data = get_stock_data(ticker, period=period_map[time_range])
                fig = go.Figure()
                if not hist_data.empty:
                    hist_data.ta.sma(length=20, append=True)
                    hist_data.ta.sma(length=50, append=True)
                    fig.add_trace(go.Candlestick(x=hist_data.index, open=hist_data['Open'], high=hist_data['High'], low=hist_data['Low'], close=hist_data['Close'], name='Price'))
                    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['SMA_20'], mode='lines', name='20-Day MA', line=dict(color='yellow', width=1)))
                    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['SMA_50'], mode='lines', name='50-Day MA', line=dict(color='orange', width=1)))
                fig.update_layout(title=f'{selected_display_ticker} Historical Price ({native_currency_symbol})', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            with profile_tab:
                st.subheader(f"{info.get('longName', '')} ({info.get('symbol', '')})")
                st.markdown(f"*Website:* [{info.get('website', 'N/A')}]({info.get('website', 'N/A')})")
                st.markdown(info.get('longBusinessSummary', 'No summary available.'))
                st.subheader("Key Financial Ratios")
                ratios_keys = {'P/E Ratio': 'trailingPE', 'PEG Ratio': 'pegRatio', 'Price to Sales': 'priceToSalesTrailing12Months', 'Price to Book': 'priceToBook', 'Debt to Equity': 'debtToEquity', 'Return on Equity': 'returnOnEquity'}
                formatted_ratios = [{"Ratio": name, "Value": f"{info.get(key, 'N/A'):,.2f}" if isinstance(info.get(key), (int, float)) else "N/A"} for name, key in ratios_keys.items()]
                st.table(pd.DataFrame(formatted_ratios))

            with forecast_tab:
                st.subheader(f"10-Day Price Forecast for {selected_display_ticker}")
                if st.button("Generate Forecast", key=f"forecast_{ticker}"):
                    with st.spinner("Generating forecast... this may take a moment."):
                        try:
                            data_for_prophet = get_stock_data(ticker, period="3y").reset_index()
                            df_train = pd.DataFrame({'ds': pd.to_datetime(data_for_prophet['Date']), 'y': pd.to_numeric(data_for_prophet['Close'])}).dropna()
                            if len(df_train) > 30:
                                model = Prophet()
                                model.fit(df_train)
                                df_cv = cross_validation(model, initial='365 days', period='180 days', horizon='30 days', parallel="processes")
                                df_p = performance_metrics(df_cv)
                                st.metric("Model's Historical Accuracy (MAPE)", f"{df_p['mape'].values[0]:.2%}", help="The Mean Absolute Percentage Error of the forecast model on historical data. Lower is better.")
                                future = model.make_future_dataframe(periods=10)
                                forecast = model.predict(future)
                                st.write("Forecasted Values (next 10 days):")
                                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))
                                fig_forecast = plot_plotly(model, forecast)
                                fig_forecast.update_layout(title="Forecast with Uncertainty Interval")
                                st.plotly_chart(fig_forecast, use_container_width=True)
                            else:
                                st.warning("Not enough historical data available for a reliable forecast.")
                        except Exception as e:
                            st.error(f"Could not generate forecast. Error: {e}")
            
            with news_tab:
                 st.subheader(f"Latest News for {info.get('longName', selected_display_ticker)}")
                 news_articles = get_stock_news(selected_display_ticker, info.get('longName', ''))
                 if news_articles:
                     for article in news_articles:
                         with st.container(border=True):
                             st.markdown(f"**[{article['title']}]({article['url']})**")
                             st.write(f"*Source: {article['source']['name']} | Published: {pd.to_datetime(article['publishedAt']).strftime('%Y-%m-%d')}*")
                             st.write(article.get('description', 'No description available.'))
                 else:
                     st.info("No recent news found.")
                     
    with tab3:
        st.subheader("Transaction Log")
        st.dataframe(transactions_df, use_container_width=True)