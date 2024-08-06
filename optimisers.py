"""
Mean-variance optimisation of portfolios, subject to constraints.

Main methods
------------
maximise_sharpe():
    Maximise Sharpe Ratio.
    
minimise_vol():
    Find the minimum volatility portfolio.
    
efficient_portfolio():
    Minimise volatility whilst keeping excess returns fixed to a target.
    
maximise_returns():
    Maximise excess returns whilst the volatility is below a particular target.
    
"""

import numpy as np
import cvxpy as cp

TRADING_DAYS = 252

def clean_weights(weights, threshold=1e-4):
    """
    Clean the weights by setting very small weights to zero and renormalising.

    Parameters
    ----------
    weights : array-like(float)
        Original weights.
    threshold : float, optional
        Numerical threshold below which weights are set to zero. Default is 1e-4.

    Returns
    -------
    cleaned_weights : array-like(float)
        Cleaned and renormalised weights.

    """
    
    if weights is None:
        return None
    
    cleaned_weights = np.where(np.abs(weights) < threshold, 0, weights)
    
    cleaned_weights = cleaned_weights / np.sum(cleaned_weights)
    
    return cleaned_weights

def update_portfolio(portfolio, weights, cov_type):
    """
    Update portfolio with optimised weights, after they are cleaned for numerical errors.

    Parameters
    ----------
    portfolio : StockUniverse.Portfolio
        Portfolio to update.
    weights : list
        Weights corresponding to asset allocation.
    cov_type : str
        Denotes method for estimating covariance matrix.

    Returns
    -------
    portfolio : StockUniverse.Portfolio
        Upated portfolio.

    """
    
    portfolio.weights = clean_weights(weights.value)
    portfolio.calc_performance(cov_type)
    
    return portfolio

def maximise_sharpe(portfolio, cov_type = 'sample_cov', constraint_set = (0, 1)):
    """
    Maximise Sharpe Ratio by altering the weights of passed portfolio, using CVXPY.
    Uses convex optimisation and therefore minimises volatility subject to returns = 1 and using non-normalised weights.
    Then gets correct weights through normalisation.
    
    Parameters
    ----------
        portfolio : StockUniverse.Portfolio
            Portfolio whose weights will be altered to maximise the Sharpe Ratio.
        constraint_set : array-like(float, float), optional
            Allowed min and max of weights. The default is (0, 1).
        
    Returns
    -------
        portfolio : StockUniverse.Portfolio
            Portfolio whose weights are optimised to maximise the Sharpe Ratio.
            
    """
    
    num_assets = len(portfolio.universe.stocks)
    
    weights = cp.Variable(num_assets)
    
    mean_returns = portfolio.universe.mean_returns.values * TRADING_DAYS
    cov_matrix = portfolio.universe.get_covariance_matrix(cov_type) * TRADING_DAYS
    
    portfolio_returns = mean_returns @ weights
    portfolio_var = cp.quad_form(weights, cov_matrix)
    
    objective = cp.Minimize(portfolio_var)
    
    constraints = [
        weights >= constraint_set[0],
        portfolio_returns == 1
        ]
            
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(qcp = True)
    except cp.error.SolverError:
        raise ValueError("Sharpe Ratio Maximisation problem could not be solved.")
        
    if problem.status != cp.OPTIMAL:
        raise ValueError(f"Sharpe Ratio Maximisation did not converge. Status {problem.status}")
        
    portfolio = update_portfolio(portfolio, weights, cov_type)
    
    return portfolio

def minimise_vol(portfolio, cov_type = 'sample_cov', factor_bounds = None, constraint_set = (0, 1)):
    """
    Minimise the portfolio variance by altering the weights in passed portfolio, subject to potential constraints
    on the factor exposure using CVXPY.

    Parameters
    ----------
    portfolio : StockUniverse.Portfolio
        Portfolio whose weights will be altered to minimise the volatility.
    cov_type : str, optional
        Method for estimating covariance matrix. Default is 'sample_cov'.
    factor_bounds : dict, optional
        Dictionary of factor constraints with lower and upper bounds. Default is None.
    constraint_set : array-like(float, float), optional
        Allowed min and max of weights. The default is (0, 1).
        
    Raises
    ------
    ValueError
        If the minimisation procedure did not succeed, typically because of too stringent constraints.
        Also raises error when the solution is inaccurate, typically due to poor scaling of the problem variables.

    Returns
    -------
    portfolio : StockUniverse.Portfolio
        Portfolio whose weights are optimised to minimise the volatility.

    """
    
    num_assets = len(portfolio.universe.stocks)
    
    weights = cp.Variable(num_assets)
    
    cov_matrix = portfolio.universe.get_covariance_matrix(cov_type) * TRADING_DAYS
    
    objective = cp.Minimize(cp.quad_form(weights, cov_matrix))
    
    constraints = [
        cp.sum(weights) == 1,
        weights >= constraint_set[0],
        weights <= constraint_set[1],
        ]
    
    if factor_bounds:
        for factor, (lower, upper) in factor_bounds.items():
            factor_betas = portfolio.universe.get_factor_betas(factor).values
            if lower:
                constraints.append(factor_betas @ weights >= lower)
            if upper:
                constraints.append(factor_betas @ weights <= upper)
            
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve()
    except cp.error.SolverError:
        raise ValueError("Volatility minimisation problem could not be solved.")
        
    if problem.status == cp.OPTIMAL_INACCURATE:
        raise ValueError("Volatility minimisation may be inaccurate.")
    elif problem.status != cp.OPTIMAL:
        raise ValueError(f"Volatility minimisation problem did not converge. Status: {problem.status}")
    
    if weights.value is None:
        raise ValueError("Volatility minimisation resulted in None weights.")
    
    portfolio = update_portfolio(portfolio, weights, cov_type)
    
    return portfolio
        
def efficient_portfolio(portfolio, excess_returns_target, cov_type = 'sample_cov', factor_bounds=None, constraint_set=(0, 1)):
    """
    For a fixed returns target, optimise the portfolio for min volatility using CVXPY, subject to potential constraints on the factor exposure.
    
    Parameters
    ----------
    portfolio : StockUniverse.Portfolio
        Portfolio whose weights will be altered to minimise the volatility.
    excess_returns_target : float
        Target excess return that the portfolio should achieve.
    cov_type : str
        Estimation method for the covariance matrix.
    factor_bounds : dict, optional
        Dictionary of factor constraints with lower and upper bounds. Default is None.
    constraint_set : array-like(float, float), optional
        Allowed min and max of weights. The default is (0, 1).
        
    Raises
    ------
    ValueError
        If the optimisation procedure did not succeed, typically because of too stringent constraints.
        
    Returns
    -------
    portfolio : StockUniverse.Portfolio
        Portfolio whose weights are optimised to minimise the volatility, while reaching the excess returns target.
        
    """
    
    num_assets = len(portfolio.universe.stocks)

    weights = cp.Variable(num_assets)
    weights.value = np.ones(num_assets) / num_assets

    mean_returns = portfolio.universe.mean_returns.values * TRADING_DAYS
    cov_matrix = portfolio.universe.get_covariance_matrix(cov_type) * TRADING_DAYS

    objective = cp.Minimize(cp.quad_form(weights, cov_matrix))

    constraints = [
        cp.sum(weights) == 1,
        weights >= constraint_set[0],
        weights <= constraint_set[1],
        mean_returns @ weights - portfolio.universe.risk_free_rate == excess_returns_target
    ]

    if factor_bounds:
        for factor, (lower, upper) in factor_bounds.items():
            factor_betas = portfolio.universe.get_factor_betas(factor).values
            if lower:
                constraints.append(factor_betas @ weights >= lower)
            if upper:
                constraints.append(factor_betas @ weights <= upper)

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(warm_start = True, verbose = False)
    except cp.error.SolverError:
        raise ValueError("Optimisation problem could not be solved.")
        
    if problem.status == cp.OPTIMAL_INACCURATE:
        pass
    elif problem.status != cp.OPTIMAL:
        raise ValueError(f"Optimisation did not converge. Status: {problem.status}")
    
    if weights.value is None:
        raise ValueError("Optimisation resulted in None weights.")

    portfolio = update_portfolio(portfolio, weights, cov_type)
    
    return portfolio

def maximise_returns(portfolio, vol_target, cov_type = 'sample_cov', factor_bounds = None, constraint_set=(0, 1)):
    """
    For a max allowed volatility target, optimise the portfolio for max returns whilst satisfying constraints on the factor exposure. Uses CVXPY.
    
    Parameters
    ----------
    portfolio : StockUniverse.Portfolio
        Portfolio whose weights will be altered to maximise the returns.
    vol_target : float
        Max volatility that the portfolio can have.
    cov_type : str
        Estimation method used for the covariance matrix.
    factor_bounds : dict, optional
        Dictionary of factor constraints with lower and upper bounds. Default is None.
    constraint_set : array-like(float, float), optional
        Allowed min and max of weights. The default is (0, 1).
        
    Raises
    ------
    ValueError
        If the maximisation procedure did not succeed, typically because of too stringent constraints.
        Also raises error when the solution is inaccurate, typically due to poor scaling of the problem variables.
        
    Returns
    -------
    portfolio : StockUniverse.Portfolio
        Portfolio with weights optimised to maximise the returns while hitting the volatility target.
        
    """
    
    num_assets = len(portfolio.universe.stocks)

    weights = cp.Variable(num_assets)
    weights.value = np.ones(num_assets) / num_assets

    mean_returns = portfolio.universe.mean_returns.values * TRADING_DAYS
    cov_matrix = portfolio.universe.get_covariance_matrix(cov_type) * TRADING_DAYS

    objective = cp.Maximize(mean_returns @ weights - portfolio.universe.risk_free_rate)

    constraints = [
        cp.sum(weights) == 1,
        weights >= constraint_set[0],
        weights <= constraint_set[1],
        cp.quad_form(weights, cov_matrix) <= vol_target**2,
    ]

    if factor_bounds:
        for factor, (lower, upper) in factor_bounds.items():
            factor_betas = portfolio.universe.get_factor_betas(factor).values
            if lower:
                constraints.append(factor_betas @ weights >= lower)
            if upper:
                constraints.append(factor_betas @ weights <= upper)
            
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(warm_start = True, verbose = False)
    except cp.error.SolverError:
        raise ValueError("Return maximisation problem could not be solved.")
        
    if problem.status == cp.OPTIMAL_INACCURATE:
        raise ValueError("Return maximisation may be inaccurate.")
    elif problem.status != cp.OPTIMAL:
        raise ValueError(f"Return maximisation did not converge. Status: {problem.status}")
    
    if weights.value is None:
        raise ValueError("Return maximisation resulted in None weights.")

    portfolio = update_portfolio(portfolio, weights, cov_type)

    return portfolio
