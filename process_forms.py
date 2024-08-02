import numpy as np
import streamlit as st

from StockUniverse import StockUniverse, Portfolio
from data_loader import load_default_stocks, load_default_bonds, load_factor_df
from optimisers import efficient_portfolio, minimise_vol
from helper_functions import convert_to_date

state = st.session_state

def process_stock_form(stock_list = None, start_date = None, end_date = None, risk_free_rate = None):
    """
    Process the stock selection form by downloading stock data and setting up the stock universe,
    including max SR portfolio, min vol portfolio and a custom portfolio of uniform weights.
    If stock_list, start_date or end_date are missing, will load saved stocks.
    Updates streamlit session_state automatically.

    Parameters
    ----------
    stocks_form : st.form
        Form that corresponds to the stock selection form..
    stock_list : str, optional
        Stock universe that was chosen, passed as a comma-separated string. The default is None.
    start_date : datetime, optional
        Start date of stocks to be considered, passed as datetime.. The default is None.
    end_date : datetime, optional
        End date of stocks to be considered, passed as datetime.. The default is None.
    risk_free_rate : float, optional
        Chosen risk free rate. If None, will use bond data. The default is None.

    Returns
    -------
    errors : str
        String of error messages for user.

    """
    
    errors = ""
    
    # If stock_list, start_date or end_date are missing, use saved
    if not stock_list or not start_date or not end_date:
        stock_data = load_default_stocks()
        bonds_data = load_default_bonds()
        stocks = list(stock_data.columns)
        start_date, end_date = convert_to_date(stock_data.index[0]), convert_to_date(stock_data.index[-1])
        universe = StockUniverse(stocks, start_date, end_date)
        universe.stock_data = stock_data
        universe.bonds_data = bonds_data
     
    else:
        # Extract stock list
        stocks = stock_list.split(",")
        cleaned_stocks = set()
        for stock in stocks:
            stock = stock.strip()
            if stock:
                cleaned_stocks.add(stock)
        
        # If stocks, start and end date are the same as loaded, just use the loaded data
        
        if (state.universe and state.universe.stocks and set(state.universe.stocks) == cleaned_stocks and 
            state.universe.start_date and state.universe.start_date == start_date and
            state.universe.end_date and state.universe.end_date == end_date):
            universe = state.universe
            universe.risk_free_rate = risk_free_rate
        
        else: # process form
            if start_date >= end_date:
                errors += "You must pick a start date before the end date."
                return errors
    
            if len(cleaned_stocks) < 2:
                errors += "Less than two stocks entered. Need at least two stocks to construct a meaningful portfolio."
                return errors
        
            universe = StockUniverse(list(cleaned_stocks), start_date, end_date, risk_free_rate = risk_free_rate)
            if not universe:
                errors += "Could not build stock universe. Try again."
                return errors
        
            ignored = universe.get_data()
    
            if len(ignored) > 0:
                if len(ignored) == 1:
                    ignored_str = ignored[0]
                else:
                    ignored_str = ", ".join(ignored)
                    
                errors += f"Failed to download {ignored_str}. Check the tickers. Will try to continue without them.\n"
                
                if len(ignored) == len(cleaned_stocks):
                    errors += "Could not download any stocks. There may be an issue with the Yahoo Finance connection."
                    
                    return errors
            
            if len(universe.stocks) < 2:
                errors += "Less than two stocks downloaded. Need at least two stocks to construct a meaningful portfolio."
                return errors

    universe.calc_mean_returns_cov()
    if risk_free_rate == None:
        universe.calc_risk_free_rate()
                    
    state.eff_frontier = universe.calc_efficient_frontier()
    
    portfolios = {'max_SR': universe.max_SR_portfolio, 'min_vol': universe.min_vol_portfolio}
        
    # Custom Portfolio: initialise as uniform portflio
    custom_portfolio = Portfolio(universe, name = "Custom")
    portfolios['custom'] = custom_portfolio
        
    state.universe = universe
    state.portfolios = portfolios
    state.factor_model = None
    state.factor_bounds = {}
        
    return errors
    
def recompute_portfolio(weights):
    """
    Recompute the custom portfolio data given new weights.
    Automatically updates portfolios in sreamlit session_state.
    

    Parameters
    ----------
    weights : list(float)
        List of new weights to be used for the custom portfolio.

    Returns
    -------
    None.

    """
        
    universe = state.universe
    custom_portfolio = Portfolio(universe, name = "Custom", weights = weights)
    state.portfolios['custom'] = custom_portfolio
    
    return None
 
def optimise_custom_portfolio(optimiser, target, factor_model = None, factor_bounds = None):
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
    
    try:
        custom_portfolio = universe.optimise_portfolio(optimiser, target, factor_bounds)        
    except Exception:
        st.error("Unable to optimise portfolio according to the target and am now optimising without the factor constraints.")
        custom_portfolio = universe.optimise_portfolio(optimiser, target)
        
    custom_portfolio.name = "Custom"
    state.portfolios['custom'] = custom_portfolio
    if factor_bounds:
        if factor_bounds == state.factor_bounds:
            return
        
        state.factor_bounds = factor_bounds

        
        clear_constrained_portfolios()
        
        try:
            min_vol_portfolio = Portfolio(universe, 'Constrained Min Vol')
            min_vol_portfolio = minimise_vol(min_vol_portfolio, factor_bounds = factor_bounds)
            
            state.constrained_eff_frontier, max_SR_portfolio = calculate_constrained_efficient_frontier(universe, factor_bounds)
        
            # Only save the constrained max SR and min vol portfolios if they are actually different (up to numerical tolerance) from unconstrained ones.
            if max_SR_portfolio and not same_weights(state.portfolios['max_SR'].weights, max_SR_portfolio.weights):
                state.portfolios['constrained_max_SR'] = max_SR_portfolio
            if min_vol_portfolio and not same_weights(state.portfolios['min_vol'].weights, min_vol_portfolio.weights):
                state.portfolios['constrained_min_vol'] = min_vol_portfolio
                
        except Exception:
            st.error("Unable to find minimum volatility portfolio while obeying the constraints. This suggests that no portfolios satisfy your factor constraints. Therefore, your factor constraints have been ignored.")
            min_vol_portfolio = None
            state.factor_bounds = factor_bounds = {}
            
    else:
        if 'constrained_max_SR' in state.portfolios:
            del state.portfolios['constrained_max_SR']
        if 'constrained_min_vol' in state.portfolios:
            del state.portfolios['constrained_min_vol']
        if 'constrained_eff_frontier' in state:
            del state['constrained_eff_frontier']
    
def process_factor_analysis_form(factor_model):
    clear_factor_analysis()
    factor_df = load_factor_df(factor_model)
    state.universe.run_factor_analysis(factor_df)
    state.factor_model = factor_model
    
    factor_start_date, factor_end_date = state.universe.factor_analysis.get_date_range()
    date_range = factor_end_date - factor_start_date
    if factor_end_date < state.universe.end_date:
        st.error(f"Note that the Fama-French factor data has a lag and only runs until {factor_end_date.strftime('%d/%m/%Y')}. The remaining days will not be used.")
    if date_range.days < 230:
        st.error(f"Note that the factor analysis only covers {date_range.days} days and may not be reliable!")
    
def clear_ranges():
    state.factor_bounds = {}
    clear_constrained_portfolios()
    if 'constrained_eff_frontier' in state:
        del state['constrained_eff_frontier']
    
def clear_constrained_portfolios():
    if 'constrained_max_SR' in state.portfolios:
        del state.portfolios['constrained_max_SR']
    if 'constrained_min_vol' in state.portfolios:
        del state.portfolios['constrained_min_vol']
    
def clear_factor_analysis():
    state.factor_model = None
    clear_ranges()
        
def calculate_constrained_efficient_frontier(universe, factor_bounds, constraint_set = (0, 1)):
    LOWER = universe.min_returns
    UPPER = universe.max_returns
    
    target_excess_returns = np.linspace(LOWER, UPPER, 500)
    efficient_frontier_vols = []
    constrained_max_SR_portfolio = None
    
    progress_text = "Calculating efficient frontier subject to constraints."
    my_bar = st.progress(0, text = progress_text)
    for iteration, target in enumerate(target_excess_returns):
        # for each efficient portfolio, obtain the portfolio volatility
        eff_portfolio = Portfolio(universe)
        try:
            eff_portfolio = efficient_portfolio(eff_portfolio, target, factor_bounds)
            efficient_frontier_vols.append(eff_portfolio.vol)
            if not constrained_max_SR_portfolio:
                constrained_max_SR_portfolio = eff_portfolio
            elif eff_portfolio.sharpe_ratio > constrained_max_SR_portfolio.sharpe_ratio:
                constrained_max_SR_portfolio = eff_portfolio
                
            constrained_max_SR_portfolio.name = "Constrained Max Sharpe Ratio"
            
        except:
            efficient_frontier_vols.append(None)
            
        percent_complete = (iteration + 1) / len(target_excess_returns)
        my_bar.progress(percent_complete, text = progress_text)
    
    my_bar.empty()
    return ((efficient_frontier_vols, target_excess_returns), constrained_max_SR_portfolio)

def same_weights(weights1, weights2, threshold=1e-4):
    if len(weights1) != len(weights2):
        return False
    
    weights_diff = np.abs(weights1 - weights2)
    
    return np.all(weights_diff < threshold)