import numpy as np

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