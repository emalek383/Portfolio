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

def download_data(stocks, start, end):
    """
    Download stock data from Yahoo Finance using Ticker objects.
    """
    all_data = {}
    
    # Convert start and end to datetime objects if they're strings
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    
    st.write(f"Attempting to download data for {len(stocks)} stocks from {start} to {end}")
    
    progress_bar = st.progress(0)
    for i, stock in enumerate(stocks):
        st.write(f"Processing stock: {stock}")
        ticker = yf.Ticker(stock)
        try:
            chunk_start = start
            stock_data = []
            while chunk_start < end:
                chunk_end = min(chunk_start + timedelta(days=365), end)
                st.write(f"Downloading chunk for {stock}: {chunk_start} to {chunk_end}")
                chunk = ticker.history(start=chunk_start, end=chunk_end, interval="1d")
                st.write(f"Chunk data shape: {chunk.shape}")
                if not chunk.empty:
                    stock_data.append(chunk['Close'])
                    st.write(f"Added chunk data for {stock}")
                else:
                    st.warning(f"Empty chunk for {stock} from {chunk_start} to {chunk_end}")
                chunk_start = chunk_end + timedelta(days=1)
                time.sleep(1)  # Add a small delay to avoid rate limiting
            
            if stock_data:
                all_data[stock] = pd.concat(stock_data)
                st.write(f"Successfully downloaded data for {stock}")
            else:
                st.warning(f"No data available for {stock}")
        except Exception as e:
            st.error(f"Error downloading data for {stock}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(stocks))
    
    st.write(f"Download process completed. Total stocks with data: {len(all_data)}")
    
    stockData = pd.DataFrame(all_data)
    st.write(f"Final DataFrame shape: {stockData.shape}")
    return stockData

# Example usage in your Streamlit app:
stocks = ['AAPL', 'GOOGL', 'MSFT']  # Replace with your actual stock list
start_date = '2023-01-01'  # Replace with your actual start date
end_date = '2023-12-31'  # Replace with your actual end date
data = download_data(stocks, start_date, end_date)
st.write(data)
