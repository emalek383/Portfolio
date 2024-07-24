import yfinance as yf
import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
from setup_forms import setup_stock_selection_form, setup_weights_form, setup_optimise_portfolio_form
from setup_displays import setup_portfolio_display, setup_efficient_frontier_display
from process_forms import process_stock_form

st.set_page_config(layout="wide")

state = st.session_state    
if 'loaded_stocks' not in state:
     state.loaded_stocks = False
    
if 'universe' not in state:
     state.universe = None
     
if 'portfolios' not in state:
     state.portfolios = {}
     
if 'eff_frontier' not in state:
    state.eff_frontier = None

with st.sidebar:    
    st.header("Select stocks for your portfolio")
    stock_selection_form = st.form(border = True, key = "stock_form")
    
    st.header("Adjust your portfolio")
    weights_form = st.container(border = False)
    
    st.header("Optimise your portolfio")
    optimise_portfolio_form = st.container(border = False)

portfolio_display = st.container(border = False)
efficient_frontier_display = st.container(border = False)


if not state.loaded_stocks:
    process_stock_form(stock_selection_form)
    state.loaded_stocks = True
    if state.universe and len(state.universe.stocks) > 1:
        state.eff_frontier = state.universe.calc_efficient_frontier()
    
setup_stock_selection_form(stock_selection_form)
setup_weights_form(weights_form)
setup_optimise_portfolio_form(optimise_portfolio_form)
setup_portfolio_display(portfolio_display)
setup_efficient_frontier_display(efficient_frontier_display)

st.write(f"yfinance version: {yf.__version__}")

def download_data(stocks, start, end):
    """
    Download stock data from Yahoo Finance using Ticker objects.
    """
    all_data = {}
    
    # Convert start and end to datetime objects if they're strings
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    
    st.write(f"Attempting to download data for {len(stocks)} stocks from {start} to {end}")
    
    # Disable yfinance cache
    yf.set_tz_cache_location(None)
    
    progress_bar = st.progress(0)
    for i, stock in enumerate(stocks):
        st.write(f"Processing stock: {stock}")
        try:
            # Use yf.download instead of Ticker.history
            data = yf.download(stock, start=start, end=end, progress=False)
            st.write(f"Downloaded data shape for {stock}: {data.shape}")
            if not data.empty:
                all_data[stock] = data['Close']
                st.write(f"Successfully added data for {stock}")
            else:
                st.warning(f"No data available for {stock}")
        except Exception as e:
            st.error(f"Error downloading data for {stock}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(stocks))
        time.sleep(1)  # Add a small delay to avoid rate limiting
    
    st.write(f"Download process completed. Total stocks with data: {len(all_data)}")
    
    stockData = pd.DataFrame(all_data)
    st.write(f"Final DataFrame shape: {stockData.shape}")
    return stockData

def download_single_day(stock, date):
    try:
        data = yf.download(stock, start=date, end=date + pd.Timedelta(days=1), progress=False)
        st.write(f"Downloaded data shape for {stock} on {date}: {data.shape}")
        return data
    except Exception as e:
        st.error(f"Error downloading data for {stock}: {str(e)}")
        return pd.DataFrame()

stock = 'AAPL'
date = pd.Timestamp('2023-12-29')  # Last trading day of 2023
data = download_single_day(stock, date)
st.write(data)
