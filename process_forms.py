import streamlit as st
from StockUniverse import StockUniverse, Portfolio
from data_loader import load_default_stocks, load_default_bonds

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
        start_date, end_date = stock_data.index[0], stock_data.index[-1]
        universe = StockUniverse(stocks, start_date, end_date)
        universe.stock_data = stock_data
        universe.bonds_data = bonds_data
     
    else:
        # If stocks, start and end date are the same as loaded, just use the loaded data
        if (state.universe and state.universe.stocks and state.universe.stocks == stock_list and 
            state.universe.start_date and state.universe.start_date == start_date and
            state.universe.end_date and state.universe.end_date == end_date):
            
            state.universe.risk_free_rate = risk_free_rate
        
        else: # process form
            if start_date >= end_date:
                errors += "You must pick a start date before the end date."
                return errors
    
            stocks = stock_list.split(",")
            cleaned_stocks = []
            for stock in stocks:
                stock = stock.strip()
                if stock:
                    cleaned_stocks.append(stock)
        
            if len(cleaned_stocks) < 2:
                errors += "Less than two stocks entered. Need at least two stocks to construct a meaningful portfolio."
                return errors
        
            universe = StockUniverse(cleaned_stocks.copy(), start_date, end_date, risk_free_rate = risk_free_rate)
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
                    
    portfolios = {'max_SR': universe.max_SR_portfolio, 'min_vol': universe.min_vol_portfolio}
        
    # Custom Portfolio: initialise as uniform portflio
    custom_portfolio = Portfolio(universe, name = "Custom")
    portfolios['custom'] = custom_portfolio
        
    state.universe = universe
    state.portfolios = portfolios
        
    return errors
    
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
    custom_portfolio = Portfolio(universe, name = "Custom", weights = weights)
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