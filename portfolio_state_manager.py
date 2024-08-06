"""
Manage the portfolio and efficient frontier data in streamlit.session_state.

"""

import streamlit as st 

state = st.session_state

def initialise_portfolio_state():
    """
    Initialise the portfolio and efficient frontier states in session state if not already present.
    
    The 'portfolios' dict is structured as
        {
            'custom': custom portfolio (StockUniverse.Portfolio),
            {'sample_cov', 'factor_cov'}: dictionary of portfolios corresponding to that covariance estimation method (dict)
                'max_sharpe': Max Sharpe portfolio (StockUniverse.Portfolio)
                'min_vol': Min Vol portfolio (StockUniverse.Portfolio)
                'factor_constrained': Factor-constrained dictionary (dict)
                    'max_sharpe': Max Sharpe Portfolio (StockUniverse.Portfolio)
                    'min_vol': Min Vol Portfolio (StockUniverse.Portfolio)
         }
        
    The 'efficient_frontiers' dict is structured as
        {
            {'sample_cov', 'factor_cov'}: dictionary of efficient frontiers corresponding to that covariance estimation method (dict)
                {'unconstrained', 'factor_constrained'}: unconstrained / factor_constrained efficient frontier data
        }

    Returns
    -------
    None.
    
    """
    
    if 'portfolios' not in state:
        state.portfolios = {
            'custom': None,
            'sample_cov': {
                'max_sharpe': None,
                'min_vol': None,
                'factor_constrained': {
                    'max_sharpe': None,
                    'min_vol': None
                }
            },
            'factor_cov': {
                'max_sharpe': None,
                'min_vol': None,
                'factor_constrained': {
                    'max_sharpe': None,
                    'min_vol': None
                }
            }
        }
    
    if 'efficient_frontiers' not in state:
        state.efficient_frontiers = {
            'sample_cov': {
                'unconstrained': None,
                'factor_constrained': None
            },
            'factor_cov': {
                'unconstrained': None,
                'factor_constrained': None
            }
        }
        
def update_portfolio(portfolio_type, portfolio, cov_type = 'sample_cov', constrained = False):
    """
    Update a specific portfolio in the session state.

    Parameters
    ----------
    portfolio_type : str
        Type of portfolio.
    portfolio : StockUniverse.Portfolio
        The portfolio object to be stored.
    cov_type : str, optional
        Used estimation method for covariance matrix. Default is 'sample_cov'.
    constrained : bool, optional
        Whether the portfolio is factor-constrained. Default is False.

    Raises
    ------
    ValueError
        If covariance type is not specified for non-custom portfolios.

    Returns
    -------
    None.

    """
    
    if portfolio_type == 'custom':
        state.portfolios['custom'] = portfolio
    elif cov_type:
        if constrained:
            state.portfolios[cov_type]['factor_constrained'][portfolio_type] = portfolio
        else:
            state.portfolios[cov_type][portfolio_type] = portfolio
    else:
        raise ValueError("Covariance type must be specified for non-custom portfolios")
        
def get_portfolio(portfolio_type, cov_type = 'sample_cov', constrained = False):
    """
    Get a specific portfolio from the session state.

    Parameters
    ----------
    portfolio_type : str
        Type of portfolio.
    cov_type : str, optional
        Estimation method for covariance matrix. Default is 'sample_cov'.
    constrained : bool, optional
        Whether to get a factor-constrained portfolio. Default is False.

    Raises
    ------
    ValueError
        If covariance type is not specified for non-custom portfolios.
        
    Returns
    -------
    StockUniverse.Portfolio
        The requested portfolio object if it exists.

    """
    
    if portfolio_type == 'custom':
        return state.portfolios['custom']
    elif cov_type:
        if constrained and state.factor_bounds:
            return state.portfolios[cov_type]['factor_constrained'][portfolio_type]
        return state.portfolios[cov_type][portfolio_type]
    else:
        raise ValueError("Covariance type must be specified for non-custom portfolios")
        
def iterate_portfolios(cov_type = 'sample_cov', include_custom = True, include_constrained = True, sort = True):
    """
    Iterate through portfolios stored in the session state.

    Parameters
    ----------
    cov_type : str, optional
        Estimation method for covariance matrix. Default is 'sample_cov'.
    include_custom : bool, optional
        Whether to include custom portfolio. Default is True.
    include_constrained : bool, optional
        Whether to include factor-constrained portfolios. Default is True.
    sort : bool, optional
        Whether to sort the portfolios in a specific order (Max Sharpe, Constrained Max Sharpe, Min Vol, Constrained Min Vol, Custom).
        Default is True.

    Returns
    -------
    list(tuple(str, StockUniverse.Portfolio))
        List of (portfolio name, portfolio object) tuples.
    
    """
    
    portfolios = {}
    if include_custom and state.portfolios['custom'] is not None:
        portfolios['custom'] = state.portfolios['custom']
    
    portfolios.update({
        key: value for key, value in state.portfolios[cov_type].items()
        if key != 'factor_constrained' and value is not None
    })
    
    if include_constrained:
        portfolios.update({
            f"constrained_{key}": value 
            for key, value in state.portfolios[cov_type]['factor_constrained'].items()
            if value is not None
        })
    
    if sort:
        desired_order = ['max_sharpe', 'constrained_max_sharpe', 'min_vol', 'constrained_min_vol', 'custom']
        sorted_portfolios = sorted(
            portfolios.items(),
            key=lambda x: (desired_order.index(x[0]) if x[0] in desired_order else len(desired_order))
        )
        return sorted_portfolios
    else:
        return list(portfolios.items())
    
def update_efficient_frontier(eff_data, cov_type = 'sample_cov', constrained = False):
    """
    Update the efficient frontier data in the session state.

    Parameters
    ----------
    eff_data : tuple(list(float), list(float))
        Efficient frontier data consisting of list of volatilities and list of excess returns.
    cov_type : str, optional
        Estimation method for covariance matrix. . Default is 'sample_cov'.
    constrained : bool, optional
        Whether the efficient frontier is factor-constrained. Default is False.

    Returns
    -------
    None.
    
    """
    
    key = 'factor_constrained' if constrained else 'unconstrained'
    state.efficient_frontiers[cov_type][key] = eff_data

def get_efficient_frontier(cov_type = 'sample_cov', constrained = False):
    """
    Get efficient frontier data from the session state.

    Parameters
    ----------
    cov_type : str, optional
        Estimation method for covariance matrix. Default is 'sample_cov'.
    constrained : bool, optional
        Whether to retrieve factor-constrained efficient frontier. Default is False.

    Returns
    -------
    tuple(list(float), list(float)) or None
        Efficient frontier data (volatilities and excess returns) if it exists, otherwise None.
    
    """
    
    key = 'factor_constrained' if constrained else 'unconstrained'
    return state.efficient_frontiers[cov_type][key]
    
def clear_factor_constrained_data():
    """
    Clear all factor-constrained portfolio and efficient frontier data from the session state.

    Returns
    -------
    None.
    
    """
    
    for cov_type in ['sample_cov', 'factor_cov']:
        state.portfolios[cov_type]['factor_constrained'] = {
            'max_sharpe': None,
            'min_vol': None
            }
        
        state.efficient_frontiers[cov_type]['factor_constrained'] = None
        
def clear_factor_cov_data():
    """
    Clear all factor-covariance based portfolio and efficient frontier data from the session state.

    Returns
    -------
    None.
    
    """
    
    state.portfolios['factor_cov'] = {
        'max_sharpe': None,
        'min_vol': None,
        'factor_constrained': {
            'max_sharpe': None,
            'min_vol': None
            }
        }
    state.efficient_frontiers['factor_cov'] = {
        'unconstrained': None,
        'constrained': None
        }
    
def clear_all_portfolio_data():
    """
    Clear all portfolio and efficient frontier data from the session state.

    Returns
    -------
    None.
    
    """
    
    state.portfolios = {
        'custom': None,
        'sample_cov': {
            'max_sharpe': None,
            'min_vol': None,
            'factor_constrained': {
                'max_sharpe': None,
                'min_vol': None
            }
        },
        'factor_cov': {
            'max_sharpe': None,
            'min_vol': None,
            'factor_constrained': {
                'max_sharpe': None,
                'min_vol': None
            }
        }
    }
    state.efficient_frontiers = {
        'sample_cov': {
            'unconstrained': None,
            'factor_constrained': None
        },
        'factor_cov': {
            'unconstrained': None,
            'factor_constrained': None
        }
    }
    
def clear_sample_cov_data():
    """
    Clear all sample covariance-based portfolio and efficient frontier data from the session state.

    Returns
    -------
    None.
    
    """
    
    state.portfolios['sample_cov'] = {
        'max_sharpe': None,
        'min_vol': None,
        'factor_constrained': {
            'max_sharpe': None,
            'min_vol': None
        }
    }
    state.efficient_frontiers['sample_cov'] = {
        'unconstrained': None,
        'factor_constrained': None
    }