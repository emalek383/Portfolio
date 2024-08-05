import streamlit as st
import datetime as dt
from dateutil.relativedelta import relativedelta
from process_forms import process_stock_form, recompute_portfolio, optimise_custom_portfolio, process_factor_analysis_form, clear_factor_analysis, impose_factor_constraints, clear_ranges, update_covariance_choice
from helper_functions import get_default_factor_bounds, format_factor_choice, format_covariance_choice, COV_METHODS, COV_METHOD_MAP

DEFAULT_STOCKS = "GOOG, NVDA"
DEFAULT_STOCKS = "GOOG, NVDA, LLY, MRK, MSFT, AAPL, AMZN, TSLA, NFLX, ADBE, UBS, JPM, XOM, GS, BRK-B, A, AMD, MMM, AOS, AES, AFL, APD, ABNB, LNT, ALLE, ARE, A, AMCR, AEE, AEP, AMT, AWK"
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
        errors = process_stock_form(universe, start_date, end_date, risk_free_rate)
        if errors:
            form.error(errors)
            st.write(errors)
    
    return form
    

def setup_weights_form(form, cols_per_row = 6):
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
    weights = ask_for_weights(form, state.portfolios['custom'].weights, cols_per_row)
    form.button(label = "Adjust Weights",
                on_click = recompute_portfolio,
                args = (weights, ))

    return form

def ask_for_weights(weights_form, default_weights, cols_per_row = 6):
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
    col_num = min(num_stocks, cols_per_row)
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

def setup_factor_analysis_form(form):    
    if not state.universe or len(state.universe.stocks) < 2:
        return form
    
    factor_model = form.selectbox(label = "Choose the factor model you want to use", 
                              options = ['ff3', 'ff4', 'ff5', 'ff6'], 
                              format_func = format_factor_choice,
                              )
    
    #col1, col2 = form.columns(2)
    #with col1:
    st.button(label = "Run Factor Model", on_click = process_factor_analysis_form, args = (factor_model, ) )
    #with col2:
    st.button(label = "Clear Factor Model", on_click = clear_factor_analysis)
    
def setup_covariance_form(form):
    if not state.universe or len(state.universe.stocks) < 2 or not state.factor_model:
        return form
    
    cov_type_options = [method['id'] for method in COV_METHODS]
    
    covariance_choice = form.radio(label = "Covariance estimation method:",
                                 options = cov_type_options,
                                 index = cov_type_options.index(state.cov_type) if state.cov_type in cov_type_options else 0,
                                 help = "Choose the method for estimating the covariance matrix.",
                                 format_func = format_covariance_choice
                                 )
    
    form.button(label = "Update covariance estimation method", on_click = update_covariance_choice, args = (covariance_choice, ))
    

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
        label = "Choose your desired maximum volatility"
        min_value = state.universe.min_vol
        max_value = state.universe.max_vol
        value = state.portfolios['custom'].vol
        helper = "Adjust the max volatility within the possible range."
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
    
    factor_bounds = {}
    if state.factor_bounds:
        disabled_checkbox = False
    else:
        disabled_checkbox = True
        
    use_factor_constraints = form.checkbox(label = "Include factor constraints", disabled = disabled_checkbox, help = "Imposing constraints on factor exposure is only possible after a factor analysis has been run.")
    if use_factor_constraints:
        factor_bounds = state.factor_bounds
            
    print(f"Passing factor bounds: {factor_bounds}")
    
    form.button(label = "Optimise", on_click = optimise_custom_portfolio, args = (form, chosen_optimiser, target, state.factor_model, factor_bounds) )
    
    return form

def extract_factor_ranges(factor_list, factor_bounds_values):
        
    factor_bounds = {}
    factor_ranges = get_default_factor_bounds(state.universe)
    print(f"Default factor ranges: {factor_ranges}")
    for idx, factor in enumerate(factor_list):
        factor_bounds_values[idx] = list(factor_bounds_values[idx])
        if factor_bounds_values[idx] == list(factor_ranges[factor]):
            continue
        else:
            if factor_bounds_values[idx][0] == factor_ranges[factor][0]:
                factor_bounds_values[idx][0] = None
            if factor_bounds_values[idx][1] == factor_ranges[factor][1]:
                factor_bounds_values[idx][1] = None
        
        factor_bounds[factor] = factor_bounds_values[idx]
        
    return factor_bounds

def setup_factor_constraints_form(form, factor_model):
    print("\n Setting up factor constraints form")
    factor_list, factor_bounds_values = setup_factor_bounds(form, factor_model)
    print(f"Factor list: {factor_list}")
    print(f"Factor bounds values: {factor_bounds_values}")
    
    form.button(label = "Impose Constraints", on_click = impose_factor_constraints, args = (factor_model, factor_list, factor_bounds_values))
    form.button(label = "Clear Constraints", on_click = clear_ranges)
    
    # form.form_submit_button(label = "Impose Constraints", on_click = impose_factor_constraints, args = (factor_model, factor_list, factor_bounds_values))
    # form.form_submit_button(label = "Clear Constraints", on_click = clear_ranges)

def setup_factor_bounds(form, factor_model):
    factor_map = {'ff3': ['Mkt-RF', 'SMB', 'HML']}
    factor_map['ff4'] = factor_map['ff3'] + ['Mom']
    factor_map['ff5'] = factor_map['ff3'] + ['RMW', 'CMA']
    factor_map['ff6'] = factor_map['ff5'] + ['Mom']
    
    new_factor_bounds = []
    old_factor_bounds = state.factor_bounds
    factor_ranges = get_default_factor_bounds(state.universe)
    for factor in factor_map[factor_model]:
        default_value = factor_ranges[factor].copy()
        if factor in old_factor_bounds and old_factor_bounds[factor]:
            lower = old_factor_bounds[factor][0]
            upper = old_factor_bounds[factor][1]
            if lower:
                default_value[0] = lower
            if upper:
                default_value[1] = upper
            
        new_factor_bounds.append(form.slider(label = factor, #f"Allowed exposure on {factor}",
                                         value = default_value,
                                         min_value = factor_ranges[factor][0],
                                         max_value = factor_ranges[factor][1],
                                         step = 0.1
                                         ) )
        
    return factor_map[factor_model], new_factor_bounds


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