# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:40:43 2024

@author: emanu
"""
import streamlit as st
import datetime as dt
from dateutil.relativedelta import relativedelta
from process_forms import process_stock_form, recompute_portfolio, optimise_custom_portfolio

DEFAULT_STOCKS = "GOOG, NVDA"
now = dt.datetime.today()
DEFAULT_START = now + relativedelta(years = -1)
DEFAULT_END = now
DATE_FORMAT = "DD/MM/YYYY"

state = st.session_state

def setup_stock_selection_form(form):
    """
    Setup the stock selection form.

    Parameters
    ----------
    form : st.form
        Will become the stock selection form.

    Returns
    -------
    None.

    """
    
    universe = form.text_input("Enter your stocks, separated by commas",
                               help = "Enter the stock tickers separated by commas",
                               value = DEFAULT_STOCKS)

    start_date = form.date_input("Choose the start date for the analysis", 
                                 help = "Latest start date is 1 month ago",
                                 value = now + relativedelta(years = -1),
                                 max_value = DEFAULT_START,
                                 format = DATE_FORMAT)
    
    end_date = form.date_input("Choose the end date for the analysis",
                               help = "Latest end date is today", 
                               value = DEFAULT_END,
                               max_value = now,
                               format = DATE_FORMAT)

    risk_free_rate = form.number_input("Enter a custom risk-free-rate, or leave blank to use 13-week T-bill",
                                       value = None,
                                       help = "Enter as percentage")

    submit_button = form.form_submit_button(label = "Analyse")
    
    if submit_button:
        state.loaded_stocks = True
        if risk_free_rate:
            risk_free_rate /= 100
        process_stock_form(form, universe, start_date, end_date, risk_free_rate)
        if len(state.universe.stocks) > 1:
            state.eff_frontier = state.universe.calc_efficient_frontier()
    
    return
    

def setup_weights_form(form):
    """
    Setup the weights selection form.

    Parameters
    ----------
    form : st.form
        Will become the weights selection form.

    Returns
    -------
    form : st.form
        Weights selection form.

    """
    if not state.portfolios or not state.portfolios['custom']:
        form.write("No portfolio has been detected.")
        return form
    
    form.write("Change the relative weights of your portfolio.")
    weights = ask_for_weights(form, state.portfolios['custom'].weights)
    form.button(label = "Recalculate",
                on_click = recompute_portfolio,
                args = (weights, ))

    return form

def ask_for_weights(weights_form, default_weights):
    """
    Create the right number of weights input boxes for the given stock universe.

    Parameters
    ----------
    weights_form : st.form
        Form where the weights input boxes will be displayed.
    default_weights : list(float)
        Default weights to populte the weights input boxes with.

    Returns
    -------
    weights : list(st.number_input)
        List of number input boxes for the weights.

    """
    
    stocks = state.universe.stocks
    num_stocks = len(stocks)
    col_num = min(num_stocks, 3)
    cols = weights_form.columns(col_num)
    
    weights = []
    
    for i, ticker in enumerate(stocks):
        col_index = i % col_num
        with cols[col_index]:
            weights.append(st.number_input(f"{ticker}",
                                        min_value = 0.0,
                                        value = default_weights[i],
                                        step = 0.01))
            
    return weights      

def setup_optimise_portfolio_form(form):
    """
    Setup the form, allowing a user to optimise their portfolio by minimising volatility, given a target return,
    or maximising the return, given a fixed volatility.

    Parameters
    ----------
    form : st.form
        Form that will become optimise portfolio form.

    Returns
    -------
    form : st.form
        Optimise portfolio form.

    """
    if not state.universe or len(state.universe.stocks) < 2:
        return form
    
    options_map = {"Maximise Returns": "max_returns", "Minimise Volatility": "min_vol"}
    
    optimiser = form.radio(label = "Choose how to opimise your portfolio.",
                           options = ["Maximise Returns", "Minimise Volatility"])
    
    chosen_optimiser = options_map[optimiser]

    if chosen_optimiser == "max_returns":
        label = "Choose your desired volatility"
        min_value = state.universe.min_vol
        max_value = state.universe.max_vol
        value = state.portfolios['custom'].vol
        helper = "Adjust the volatility within the possible range."
    else:
        label = "Choose your desired excess returns"
        min_value = state.universe.min_returns
        max_value = state.universe.max_returns
        value = state.portfolios['custom'].excess_returns
        helper = "Adjust the returns within the possible range."
        
    min_val, max_val, current_val = get_safe_bounds(min_value, max_value, value)
        
    target = form.slider(label = label, 
                         min_value = min_value, 
                         max_value = max_value,
                         value = value, 
                         step = 0.01,
                         help = helper)
    
    form.button(label = "Optimise", on_click = optimise_custom_portfolio, args = (chosen_optimiser, target, ))

    return form

def safe_float(value):
    """
    Convert to float and handle potential errors.


    Parameters
    ----------
    value : float
        Value to be converted to float.

    Returns
    -------
    float
        float of the value passed;
        None if value cannot be converted to float.

    """
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def get_safe_bounds(min_val, max_val, current_val):
    """
    Calculate bounds for the slider, so as to include the min_val and max_val without rounding errors.

    Parameters
    ----------
    min_val : float
        Minimum value of slider.
    max_val : float
        Maximum value of slider.
    current_val : float
        Current value of slider.

    Returns
    -------
    min_val : float
        Minimum value of slider
    max_val : float
        Maximum value of slider
    current_val : float
        Current value of slider
        
    """
    
    min_val = safe_float(min_val)
    max_val = safe_float(max_val)
    current_val = safe_float(current_val)
    
    if min_val is None or max_val is None or current_val is None:
        return 0.0, 1.0, 0.5  # Default values if conversion fails
    
    # Ensure min < max
    min_val, max_val = min(min_val, max_val), max(min_val, max_val)
    
    # Ensure current_val is within bounds
    current_val = max(min_val, min(current_val, max_val))
    
    # Add a small buffer
    buffer = (max_val - min_val) * 0.01
    min_val = max(0, min_val - buffer)
    max_val = max_val + buffer
    
    return min_val, max_val, current_val