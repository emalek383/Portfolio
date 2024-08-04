import numpy as np
import pandas as pd
from datetime import datetime

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

def format_covariance_choice(cov_type):
    return COV_METHOD_MAP[cov_type]['name']

def convert_to_date(date):
    if isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()
    
    date = datetime.date(date)
    
    return date

def get_default_factor_bounds(universe):
    ranges = universe.calc_factor_ranges()
    default_bounds = {}
    for factor in ranges.index:
        min_val = ranges.loc[factor, 'min']
        max_val = ranges.loc[factor, 'max']
        range_width = max_val - min_val
        default_bounds[factor] = [min_val - 0.1*range_width, max_val + 0.1*range_width]
        
    return default_bounds


def portfolio_satisfies_constraints(portfolio, factor_bounds):
    for factor, (lower, upper) in factor_bounds.items():
        factor_betas = portfolio.universe.get_factor_betas(factor).values
        weights = portfolio.weights
        if lower and factor_betas @ weights < lower:
            return False
        if upper and factor_betas @ weights > upper:
            return False
        
    return True

def format_factor_choice(option):
    format_map = {'ff3': 'Fama-French 3-Factor', 'ff4': 'Fama-French 3-Factor + Momentum', 'ff5': 'Fama-French 5-Factor', 'ff6': 'Fama-French 5-Factor + Momentum'}
    return format_map[option]


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

def portfolio_excess_returns(weights, portfolio):
    """
    Compute the excess returns of a portfolio with given weights.

    Parameters
    ----------
    weights : list(float)
        List of weights to be used for the portfolio.
    portfolio : Portfolio
        Portfolio whose excess returns will be computed.

    Returns
    -------
    portfolio.excess_returns: float
        Excess returns of the portfolio.

    """
    
    portfolio.weights = weights
    portfolio.calc_performance()
    return portfolio.excess_returns

def portfolio_vol(weights, portfolio):
    """
    Compute the volatility of a portfolio with given weights.

    Parameters
    ----------
    weights : list(float)
        List of weights to be used for the portfolio.
    portfolio : Portfolio
        Portfolio whose volatility will be computed.

    Returns
    -------
    portfolio.vol: float
        Volatility of the portfolio.

    """
    
    portfolio.weights = weights
    portfolio.calc_performance()
    return portfolio.vol

def negative_SR(weights, portfolio):
    """
    Compute the negative Sharpe Ratio of a portfolio with given weights.

    Parameters
    ----------
    weights : list(float)
        List of weights to be used for the portfolio.
    portfolio : Portfolio
        Portfolio whose negative Sharpe Ratio we want to compute.

    Returns
    -------
    float
        The negative Sharpe Ratio of the portfolio.

    """
    
    portfolio.weights = weights
    portfolio.calc_performance()
    return - portfolio.sharpe_ratio

def negative_portfolio_excess_returns(weights, portfolio):
    """
    Compute the negative excess returns of a portfolio with given weights.

    Parameters
    ----------
    weights : list(float)
        List of weights to be used for the portfolio.
    portfolio : Portfolio
        Portfolio whose negative excess returns will be computed.

    Returns
    -------
    float
        The portfolio's negative excess returns.

    """
    
    return - portfolio_excess_returns(weights, portfolio)