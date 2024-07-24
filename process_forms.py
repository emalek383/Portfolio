# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 09:22:29 2024

@author: emanu
"""
import streamlit as st
import datetime as dt
from dateutil.relativedelta import relativedelta
from stock_universe import stock_universe, portfolio

DEFAULT_STOCKS = "GOOG, NVDA"
now = dt.datetime.today()
DEFAULT_START = now + relativedelta(years = -1)
DEFAULT_END = now

state = st.session_state

def process_stock_form(stocks_form, universe = DEFAULT_STOCKS, start_date = DEFAULT_START, end_date = DEFAULT_END, risk_free_rate = None):
    """
    Process the stock selection form by downloading stock data and setting up the stock universe,
    including max SR portfolio, min vol portfolio and a custom portfolio of uniform weights.
    Updates streamlit session_state automatically.

    Parameters
    ----------
    stocks_form : st.form
        Form that corresponds to the stock selection form..
    universe : str, optional
        Stock universe that was chosen, passed as a comma-separated string. The default is DEFAULT_STOCKS.
    start_date : datetime, optional
        Start date of stocks to be considered, passed as datetime.. The default is DEFAULT_START.
    end_date : datetime, optional
        End date of stocks to be considered, passed as datetime.. The default is DEFAULT_END.
    risk_free_rate : float, optional
        Chosen risk free rate. If None, will use bond data. The default is None.

    Returns
    -------
    universe : stock_universe
        Stock_universe for the chosen stocks (or those which could be downloaded).
    portfolios : dict(portfolio)
        Dictionary of max Sharpe Ratio, min vol and custom (uniform) portfolios of chosen stocks.

    """
    
    if start_date >= end_date:
        stocks_form.error("You must pick a start date before the end date.")
        return
    
    stocks = universe.split(",")
    cleaned_stocks = []
    for stock in stocks:
        stock = stock.strip()
        cleaned_stocks.append(stock)
        
    if len(cleaned_stocks) < 2:
        stocks_form.error("Less than two stocks entered. Need at least two stocks to construct a meaningful portfolio.")
    
    universe = stock_universe(cleaned_stocks, start_date, end_date, risk_free_rate = risk_free_rate)
    if not universe:
        stocks_form.error("Could not build stock uniform")
        return None, None
    
    ignored = universe.get_data()
    
    if len(ignored) > 0:
        stocks_form.error(f"Failed to download {ignored}. Check the tickers. Will try to continue without them.")
        return None, None
            
    if len(universe.stocks) < 2:
        stocks_form.error("Less than two stocks downloaded. Need at least two stocks to construct a meaningful portfolio.")
        return None, None
    else:
        universe.calc_mean_returns_cov()
        if risk_free_rate == None:
            universe.calc_risk_free_rate()
                    
        portfolios = {'max_SR': universe.max_SR_portfolio, 'min_vol': universe.min_vol_portfolio}
        
        # Custom Portfolio: initialise as uniform portflio
        custom_portfolio = portfolio(universe, name = "Custom")
        portfolios['custom'] = custom_portfolio
        
        state.universe = universe
        state.portfolios = portfolios
        
        return universe, portfolios  
    
def recompute_portfolio(weights):
    """
    Recompute the custom portfolio data given new weights.
    Automatically updates portfolios in sreamlit session_state.
    

    Parameters
    ----------
    weights : list(float)
        List of new weights to be used for the custom portfolio..

    Returns
    -------
    None.

    """
        
    universe = state.universe
    custom_portfolio = portfolio(universe, name = "Custom", weights = weights)
    state.portfolios['custom'] = custom_portfolio
    
    return None

def optimise_custom_portfolio(optimiser, target):
    """
    Optimise the custom portfolio according to the optimiser (min_vol or max_returns).
    Automatically updates portfolios in streamlit session_state.
    

    Parameters
    ----------
    optimiser : str
        String corresponding to the chosen optimisation method.
    target : float
        Target to be met while optimising.

    Returns
    -------
    None.

    """
    
    universe = state.universe
    custom_portfolio = universe.optimise_portfolio(optimiser, target)
    custom_portfolio.name = "Custom"
    state.portfolios['custom'] = custom_portfolio