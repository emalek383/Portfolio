# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:11:33 2024

@author: emanu
"""

import numpy as np

def get_mean_returns(returns):
    """
    Compute mean returns from simple returns.
    
    Args:
        returns: simple returns passed as numpy array (or pd DataFrame).
        
    Returns:
        mean_returns: mean returns as numpy array.
    """
    
    log_returns = np.log(returns + 1)
    mean_log_returns = log_returns.mean()
    mean_returns = np.exp(mean_log_returns) - 1
    return mean_returns

def portfolio_excess_returns(weights, portfolio):
    """
    Compute the excess returns of a portfolio with given weights.
    
    Args:
        weights: list of weights to be used for the portfolio.
        portfolio: portfolio instance whose excess returns will be computed.
        
    Returns:
        portfolio.excess_returns: the portfolio's excess returns.
    """
    
    portfolio.weights = weights
    portfolio.calc_performance()
    return portfolio.excess_returns

def portfolio_vol(weights, portfolio):
    """
    Compute the volatility of a portfolio with given weights.
    
    Args:
        weights: list of weights to be used for the portfolio.
        portfolio: portfolio instance whose volatility will be computed.
        
    Returns:
        portfolio.excess_returns: the portfolio's volatility.
    """
    
    portfolio.weights = weights
    portfolio.calc_performance()
    return portfolio.vol

def negative_SR(weights, portfolio):
    """
    Compute the negative Sharpe Ratio of a portfolio with given weights.
    
    Args:
        weights: list of weights to be used for the portfolio.
        portfolio: portfolio instance whose negative Sharpe Ratio will be computed.
        
    Returns:
        The portfolio's negative Sharpe Ratio.
    """
    
    portfolio.weights = weights
    portfolio.calc_performance()
    return - portfolio.sharpe_ratio

def negative_portfolio_excess_returns(weights, portfolio):
    """
    Compute the negative excess returns of a portfolio with given weights.
    
    Args:
        weights: list of weights to be used for the portfolio.
        portfolio: portfolio instance whose negtive excess returns will be computed.
        
    Returns:
        the portfolio's negative excess returns.
    """
    
    return - portfolio_excess_returns(weights, portfolio)