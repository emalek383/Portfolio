import numpy as np
import streamlit as st

from StockUniverse import StockUniverse, Portfolio
from data_loader import load_default_stocks, load_default_bonds, load_factor_df
from optimisers import minimise_vol, maximise_returns, maximise_sharpe
from helper_functions import convert_to_date, get_default_factor_bounds, portfolio_satisfies_constraints, format_covariance_choice
from portfolio_state_manager import clear_factor_cov_data, clear_factor_constrained_data, update_efficient_frontier, update_portfolio, get_portfolio, clear_all_portfolio_data, get_efficient_frontier

state = st.session_state

def update_covariance_choice(cov_type):
    state.cov_type = cov_type
    # Recompute the custom portfolio with given weights
    custom_portfolio = get_portfolio('custom')
    custom_portfolio.calc_performance(cov_type = cov_type)
    update_portfolio('custom', custom_portfolio)

def process_stock_form(stock_list = None, start_date = None, end_date = None, risk_free_rate = None):
    """
    Process the stock selection form by downloading stock data and setting up the stock universe,
    including max sharpe portfolio, min vol portfolio and a custom portfolio of uniform weights.
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
    
    clear_all_portfolio_data()
    
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
                    
    eff_frontier_data, max_sharpe_portfolio = universe.calc_efficient_frontier()
    
    update_efficient_frontier(eff_frontier_data, cov_type = 'sample_cov')
    print("\n Upated the efficient frontier")
    print(f"Keys in state: {list(state.keys())}")
    
    update_portfolio('max_sharpe', universe.max_sharpe_portfolio, cov_type = 'sample_cov')
    update_portfolio('min_vol', universe.min_vol_portfolio, cov_type = 'sample_cov')
    
    # Custom Portfolio: initialise as uniform portflio
    custom_portfolio = Portfolio(universe, name = "Custom")
    
    update_portfolio('custom', custom_portfolio, cov_type = 'sample_cov')
    
    state.universe = universe
    state.factor_model = None
    state.factor_bounds = {}
        
    return errors
    
def recompute_portfolio(weights):
    """
    Recompute the custom portfolio data given new weights.
    Automatically updates portfolios in streamlit session_state.
    

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
    update_portfolio('custom', custom_portfolio)
    
    return None


def extract_factor_bounds(factor_list, factor_bounds_values):
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

def impose_factor_constraints(factor_model, factor_list, factor_bounds_values):
    print("\nStarting impose_factor_constraints")
    factor_bounds = extract_factor_bounds(factor_list, factor_bounds_values)
    
    print(f"factor bounds: {factor_bounds}")
    if factor_bounds == state.factor_bounds:
        print("Factor bounds unchanged, returning")
        return 
            
    print("Updating state.factor_bounds")
    state.factor_bounds = factor_bounds
    print("Clearing constrained data")
    clear_factor_constrained_data()
    
    universe = state.universe
    
    if factor_bounds:
        for cov_type in ['sample_cov', 'factor_cov']:
            try:
                min_vol_portfolio = Portfolio(universe, 'Constrained Min Vol')
                min_vol_portfolio = minimise_vol(min_vol_portfolio, factor_bounds = factor_bounds, cov_type = cov_type)
            
                with st.spinner(f"Calculating constrained efficient frontier using {format_covariance_choice(cov_type)}"):
                    constrained_eff_frontier, max_sharpe_portfolio = vol_sweep(universe, factor_bounds = factor_bounds, cov_type = cov_type)
                    
                update_efficient_frontier(constrained_eff_frontier, cov_type, constrained = True)

                # Only save the constrained max sharpe and min vol portfolios if they are actually different (up to numerical tolerance) from unconstrained ones.
                unconstrained_max_sharpe = get_portfolio('max_sharpe', cov_type = cov_type, constrained = False)
                if max_sharpe_portfolio and not same_weights(unconstrained_max_sharpe.weights, max_sharpe_portfolio.weights):
                    # Check whether the max sharpe portfolio actually also saved the constraints but was just missed
                    if not portfolio_satisfies_constraints(unconstrained_max_sharpe, factor_bounds):
                        update_portfolio('max_sharpe', max_sharpe_portfolio, cov_type = cov_type, constrained = True)
                
                unconstrained_min_vol = get_portfolio('min_vol', cov_type = cov_type, constrained = False)
                if min_vol_portfolio and not same_weights(unconstrained_min_vol.weights, min_vol_portfolio.weights):
                    # Check whether the min vol portfolio actually also saved the constraints but was just missed
                    if not portfolio_satisfies_constraints(unconstrained_min_vol, factor_bounds):
                        update_portfolio('min_vol', min_vol_portfolio, cov_type = cov_type, constrained = True)
                    
                st.success(f"Calculated the efficient frontier for {format_covariance_choice(cov_type)} subject to the factor constraints.")
                    
            except Exception:
                st.error(f"Unable to find a minimum volatility portfolio for {format_covariance_choice(cov_type)} while obeying the constraints. This suggests that no portfolios satisfy your factor constraints.")
            
    else:
        clear_ranges()
        
 
def optimise_custom_portfolio(form, optimiser, target, factor_model = None, factor_bounds = None):
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
    cov_type = state.cov_type
    
    try:
        custom_portfolio = universe.optimise_portfolio(optimiser, target, cov_type = cov_type, factor_bounds = factor_bounds)
        form.success("Successfully optimised your custom portfolio.")
    except Exception:
        if state.factor_bounds:
            form.info("Unable to optimise portfolio according to the target and am now optimising without the factor constraints.")
            custom_portfolio = universe.optimise_portfolio(optimiser, target, cov_type = cov_type, factor_bounds = factor_bounds)
        else:
            form.error("Unable to optimise your portfolio.")
            custom_portfolio = get_portfolio('custom')
        
    custom_portfolio.name = "Custom"
    update_portfolio('custom', custom_portfolio)
    
def process_factor_analysis_form(factor_model):
    factor_df = load_factor_df(factor_model)
    state.universe.run_factor_analysis(factor_df)
    state.factor_model = factor_model
    
    initialise_factor_covariance_matrix()
    
    factor_start_date, factor_end_date = state.universe.factor_analysis.get_date_range()
    date_range = factor_end_date - factor_start_date
    if factor_end_date < state.universe.end_date:
        st.info(f"Note that the Fama-French factor data has a lag and only runs until {factor_end_date.strftime('%d/%m/%Y')}. The remaining days will not be used.")
    if date_range.days < 230:
        st.info(f"Note that the factor analysis only covers {date_range.days} days and may not be reliable!")
    
def clear_ranges():
    state.factor_bounds = {}
    clear_factor_constrained_data()
    
def clear_factor_analysis():
    state.factor_model = None
    clear_ranges()
    
def vol_sweep(universe, factor_bounds, cov_type = 'sample_cov', constraint_set = (0, 1), initial_steps = 500, max_refinements = 5, vol_tolerance = 1e-8, max_iters = 1_000):
    max_vol = universe.max_vol
    
    UPPER_VOL = max_vol

    efficient_frontier_vols = []
    efficient_frontier_returns = []
    constrained_max_sharpe_portfolio = None
    
    current_vol = UPPER_VOL
    previous_vol = None
    vol_step = UPPER_VOL / initial_steps
    
    for refinement in range(max_refinements):
        iteration = 0
        while not previous_vol or abs(current_vol - previous_vol) > vol_tolerance and iteration <= max_iters:
            eff_portfolio = Portfolio(universe)
            try:
                eff_portfolio = maximise_returns(eff_portfolio, current_vol, factor_bounds = factor_bounds, cov_type = cov_type)
                efficient_frontier_vols.append(eff_portfolio.vol)
                efficient_frontier_returns.append(eff_portfolio.excess_returns)
            
                if not constrained_max_sharpe_portfolio or eff_portfolio.sharpe_ratio > constrained_max_sharpe_portfolio.sharpe_ratio:
                    constrained_max_sharpe_portfolio = eff_portfolio
                    constrained_max_sharpe_portfolio.name = "Constrained Max Sharpe"
                
                previous_vol = eff_portfolio.vol
            
                current_vol = eff_portfolio.vol - vol_step
  
            except:
                break
            
            iteration += 1
        
        vol_step /= 10
        current_vol = previous_vol - vol_step
  
    efficient_frontier_returns.append(universe.min_returns)
    efficient_frontier_vols.append(previous_vol)
    return ((efficient_frontier_vols, efficient_frontier_returns), constrained_max_sharpe_portfolio) 

def same_weights(weights1, weights2, threshold=1e-3):
    if len(weights1) != len(weights2):
        return False
    
    weights_diff = np.abs(weights1 - weights2)
    
    return np.all(weights_diff < threshold)

def initialise_factor_covariance_matrix():
    clear_factor_cov_data()
    clear_factor_constrained_data()
    
    with st.spinner("Calculating efficient frontier with factor covariance matrix..."):
        factor_eff_frontier, factor_cov_max_sharpe = state.universe.calc_efficient_frontier(cov_type = 'factor_cov')
        update_efficient_frontier(factor_eff_frontier, 'factor_cov', constrained = False)
        
    portfolio = Portfolio(state.universe)
    factor_cov_max_sharpe.name = 'Max Sharpe'
    update_portfolio('max_sharpe', factor_cov_max_sharpe, cov_type = 'factor_cov')

    portfolio = Portfolio(state.universe)
    factor_cov_min_vol = minimise_vol(portfolio, cov_type = 'factor_cov')
    factor_cov_min_vol.name = 'Min Vol'
    update_portfolio('min_vol', factor_cov_min_vol, cov_type = 'factor_cov')