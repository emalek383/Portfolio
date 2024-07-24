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

def get_stock_info(stock):
    try:
        ticker = yf.Ticker(stock)
        info = ticker.info
        st.write(f"Retrieved info for {stock}")
        return info
    except Exception as e:
        st.error(f"Error retrieving info for {stock}: {str(e)}")
        return {}

stock = 'AAPL'
info = get_stock_info(stock)
st.write(info)
