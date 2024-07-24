import yfinance as yf
import pandas_datareader as pdr
import streamlit as st
import pandas as pd
import time
import requests
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

ALPHA_VANTAGE_API_KEY = 'OEK1PF1WTFP6O05M'

def get_stock_data_alpha_vantage(symbol, start_date, end_date):
    base_url = 'https://www.alphavantage.co/query'
    function = 'TIME_SERIES_DAILY'
    
    params = {
        'function': function,
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY,
        'outputsize': 'full'
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            st.error(f"Error retrieving data for {symbol}: {data.get('Note', 'Unknown error')}")
            return None
        
        df = pd.DataFrame(data['Time Series (Daily)']).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.loc[start_date:end_date]
        
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        st.write(f"Retrieved data for {symbol}. Shape: {df.shape}")
        return df
    
    except Exception as e:
        st.error(f"Error retrieving data for {symbol}: {str(e)}")
        return None

symbol = 'AAPL'
end_date = datetime.now().date()
start_date = end_date - timedelta(days=365)  # Get 1 year of data
data = get_stock_data_alpha_vantage(symbol, start_date, end_date)
if data is not None:
     st.write(data.head())
     st.line_chart(data['Close'])
