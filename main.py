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

st.write(f"yfinance version: {yf.__version__}")

def get_stock_data_pdr(symbol, start_date, end_date):
    try:
        data = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
        st.write(f"Retrieved data for {symbol}. Shape: {data.shape}")
        return data
    except Exception as e:
        st.error(f"Error retrieving data for {symbol}: {str(e)}")
        return None

symbol = 'AAPL'
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
data = get_stock_data_pdr(symbol, start_date, end_date)
if data is not None:
     st.write(data.head())
     st.line_chart(data['Close'])
