"""
Miscellaneous functions support other modules.

"""


import numpy as np
import pandas as pd
from datetime import datetime

TRADING_DAYS = 252

COV_METHODS = [
    {
     'id': 'sample_cov',
     'name': 'Sample Covariance',
     'description': 'Estimate covariance using historical returns'
     },
    {
     'id': 'factor_cov',
     'name': 'Factor-based Covariance',
     'description': 'Estimate covariance using factor model'
     }
    ]

COV_METHOD_MAP = {method['id']: method for method in COV_METHODS}

def calculate_factor_contributions(portfolio, factor_exposures):
    """
    Calculate the returns contributions of factors for a portfolio.

    Parameters
    ----------
    portfolio : StockUniverse.Portfolio
        Portfolio to be analysed.
    factor_exposures : pd.Series
        Exposure to the factors.
        Index: factor (str)
        Value: beta of that factor (float).

    Returns
    -------
    contributions : dict
        Dictionary containing the returns contributions of the factors.
        Key: factor name (str)
        Value: returns contribution (float).

    """
    
    factor_returns = portfolio.universe.factor_analysis.factor_returns / 100
    factor_returns = factor_returns.loc[portfolio.universe.start_date : portfolio.universe.end_date]
    factor_returns = get_mean_returns(factor_returns) * TRADING_DAYS
    
    total_return = portfolio.excess_returns
    
    contributions = {}
    for factor, exposure in factor_exposures.items():
        factor_contribution = exposure * factor_returns[factor]
        contributions[factor] = (factor_contribution / total_return) * 100
    
    explained_return = sum(contributions.values())
    contributions['Residual'] = 100 - explained_return
    
    return contributions

def get_default_factor_bounds(universe):
    """
    Get the default factor exposure ranges for a stock universe, i.e. the min and max beta for each factor in the universe.

    Parameters
    ----------
    universe : StockUniverse.StockUniverse
        Stock Universe to be studied.

    Returns
    -------
    default_bounds : dict
        Dictionary containing lowest and highest beta for each factor in the universe.
        Key: factor name (str)
        Value: minimum beta, maximum beta (tuple(float, float))

    """
    
    ranges = universe.calc_factor_ranges()
    default_bounds = {}
    for factor in ranges.index:
        min_val = ranges.loc[factor, 'min']
        max_val = ranges.loc[factor, 'max']
        range_width = max_val - min_val
        default_bounds[factor] = [min_val - 0.1*range_width, max_val + 0.1*range_width]
        
    return default_bounds

def portfolio_satisfies_constraints(portfolio, factor_bounds):
    """
    Check whether a portfolio satisfies constraints on factor exposure.

    Parameters
    ----------
    portfolio : StockUniverse.Portfolio
        Portfolio to be studied.
    factor_bounds : dict
        Dictionary containing the constraints on factor exposures as:
        Key: Factor name (str)
        Value: [lower constraint, upper constraint] (tuple(float, float))
                with lower/upper constraint = None indicating no constraint
        If no constraint at all on factor, it won't appear in the dictionary.

    Returns
    -------
    bool
        True if the portfolio satisfies the constraints, False otherwise.

    """
    
    for factor, (lower, upper) in factor_bounds.items():
        factor_betas = portfolio.universe.get_factor_betas(factor).values
        weights = portfolio.weights
        if lower and factor_betas @ weights < lower:
            return False
        if upper and factor_betas @ weights > upper:
            return False
        
    return True

def format_factor_choice(option):
    """
    Format the choice of factor model to plain English.

    Parameters
    ----------
    option : str
        Short name of  factor model.

    Returns
    -------
    str
        Plain English name for the factor model.

    """
    
    format_map = {'ff3': 'Fama-French 3-Factor', 'ff4': 'Fama-French 3-Factor + Momentum', 'ff5': 'Fama-French 5-Factor', 'ff6': 'Fama-French 5-Factor + Momentum'}
    return format_map[option]



def format_covariance_choice(cov_type):
    """
    Format the covariance estimation method to plain English.

    Parameters
    ----------
    cov_type : str
        Short name of the covariance estimation method.

    Returns
    -------
    str
        Plain English name of the covariance estimation method.

    """
    
    return COV_METHOD_MAP[cov_type]['name']

def convert_to_date(date):
    """
    Convert a timestamp or datetime time to a datetime date.

    Parameters
    ----------
    date : pd.Timestamp, datetime time.
        Timestamp to be converted to a date.

    Returns
    -------
    date : datetime.date
        Date of the timestamp.

    """
    
    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()
    
    date = datetime.date(date)
    
    return date

def get_mean_returns(returns):
    """
    Compute mean returns from simple returns.

    Parameters
    ----------
    returns : np.array
        Simple returns to be used.

    Returns
    -------
    mean_returns : np.array
        Mean returns computer from the simple returns.

    """
    
    log_returns = np.log(returns + 1)
    mean_log_returns = log_returns.mean()
    mean_returns = np.exp(mean_log_returns) - 1
    return mean_returns

def same_weights(weights1, weights2, threshold=1e-3):
    """
    Check whether two weight vectors are the same up to a numerical threshold.

    Parameters
    ----------
    weights1, weights2 : array-like
        The weights vectors.
    threshold : float, optional
        Numerical threshold for equality. The default is 1e-3.

    Returns
    -------
    bool
        True if the two weights vectors are the same, otherwise False.

    """
    
    if len(weights1) != len(weights2):
        return False
    
    weights_diff = np.abs(weights1 - weights2)
    
    return np.all(weights_diff < threshold)
