import numpy as np
import scipy.optimize as sc
from helper_functions import negative_SR, portfolio_vol, portfolio_excess_returns
from helper_functions import negative_portfolio_excess_returns

def maximise_SR(portfolio, constraint_set = (0, 1)):
    """
    Maximise Sharpe Ratio by altering the weights of passed portfolio.
    
    Parameters
    ----------
        portfolio : Portfolio
            Portfolio whose weights will be altered to maximise the Sharpe Ratio.
        constraint_set : list(float, float), optional
            Allowed min and max of weights. The default is (0, 1).
        
    Returns
    -------
        portfolio : Portfolio
            Portfolio whose weights are optimised to maximise the Sharpe Ratio.
            
    """
    
    num_assets = len(portfolio.universe.stocks)
    args = (portfolio)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # require that each asset can only have weights between 0 and 1
    bound = constraint_set
    bounds = tuple(bound for asset in range(num_assets))
    # optimise by minimising the negative Sharpe Ratio, given an initial weight vector of 1 / num_assets for each asset
    result = sc.minimize(negative_SR, [ 1. / num_assets] * num_assets, args = args, \
                        method = 'SLSQP', bounds = bounds, constraints = constraints)
    portfolio.weights = result['x']
    portfolio.calc_performance()
    return portfolio

def minimise_vol(portfolio, constraint_set = (0, 1)):
    """
    Minimise the portfolio variance by altering the weights in passed portfolio.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio whose weights will be altered to minimise the volatility.
    constraint_set : list(float, float), optional
        Allowed min and max of weights. The default is (0, 1).

    Returns
    -------
    portfolio : Portfolio
        Portfolio whose weights are optimised to minimise the volatility.

    """
     
    num_assets = len(portfolio.universe.stocks)
    args = (portfolio)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraint_set
    bounds = tuple(bound for asset in range(num_assets))
    
    result = sc.minimize(portfolio_vol, [1. / num_assets] * num_assets, args = args, \
                        method = 'SLSQP', bounds = bounds, constraints = constraints)
    portfolio.weights = result['x']
    portfolio.calc_performance()
    
    return portfolio
        
def efficient_portfolio(portfolio, excess_returns_target, constraint_set = (0, 1)):
    """
    For a fixed return target, optimise the portfolio for min volatility.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio whose weights will be altered to minimise the volatility.
    returns_target : float
        Target excess return that the portfolio should achieve.
    constraint_set : List(float, float), optional
        Allowed min and max of weights. The default is (0, 1).

    Returns
    -------
    portfolio : Portfolio
        Portfolio whose weights are optimised to minimise the volatility, while reaching the excess returns target.

    """
    
    num_assets = len(portfolio.universe.stocks)
    args = (portfolio)
    constraints = ({'type': 'eq',
                    'fun': lambda x: portfolio_excess_returns(x, portfolio) - excess_returns_target},
                   {'type': 'eq',
                    'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraint_set for asset in range(num_assets))
    eff_portfolio = sc.minimize(portfolio_vol,
                                [1. / num_assets] * num_assets,
                                args = args,
                                method = 'SLSQP', bounds = bounds, constraints = constraints)
    portfolio.weights = eff_portfolio['x']
    portfolio.calc_performance()
    return portfolio

def maximise_returns(portfolio, vol_target, constraint_set = (0, 1)):
    """
    For a fixed volatility target, optimise the portfolio for max returns.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio whose weights will be altered to maximise the returns.
    vol_target : float
        Target volatility that the portfolio can have.
    constraint_set : list(float, float), optional
        Allowed min and max of weights. The default is (0, 1).

    Returns
    -------
    portfolio : Portfolio
        Portfolio with weights optimised to maximise the returns while hitting the volatility target.

    """
 
    num_assets = len(portfolio.universe.stocks)
    args = (portfolio)
    constraints = ({'type': 'eq',
                    'fun': lambda x: portfolio_vol(x, portfolio) - vol_target},
                   {'type': 'eq',
                    'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraint_set for asset in range(num_assets))
    eff_portfolio = sc.minimize(negative_portfolio_excess_returns,
                                [1. / num_assets] * num_assets,
                                args = args,
                                method = 'SLSQP', bounds = bounds, constraints = constraints)
    portfolio.weights = eff_portfolio['x']
    portfolio.calc_performance()
    return portfolio