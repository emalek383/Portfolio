import numpy as np
import cvxpy as cp

TRADING_DAYS = 252

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
    
    weights = cp.Variable(num_assets)
    
    mean_returns = portfolio.universe.mean_returns.values * TRADING_DAYS
    cov_matrix = portfolio.universe.cov_matrix.values * TRADING_DAYS
    
    portfolio_returns = mean_returns @ weights #- portfolio.universe.risk_free_rate
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
        
    optimal_weights = weights.value / np.sum(weights.value)
    portfolio.weights = clean_weights(optimal_weights)
    portfolio.calc_performance()
    
    return portfolio

def minimise_vol(portfolio, factor_bounds = None, constraint_set = (0, 1)):
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
    
    weights = cp.Variable(num_assets)
    
    cov_matrix = portfolio.universe.cov_matrix.values * TRADING_DAYS
    
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
        print(f"Got an error at the level of minimise_vol(). Error {cp.error.SolverError}")
        raise ValueError("Volatility minimisation problem could not be solved.")
        
    if problem.status != cp.OPTIMAL:
        raise ValueError(f"Volatility minimisation did not converge. Status: {problem.status}")
        
    portfolio.weights = clean_weights(weights.value)
    portfolio.calc_performance()
    
    return portfolio
        
def efficient_portfolio(portfolio, excess_returns_target, factor_bounds=None, constraint_set=(0, 1)):
    """
    For a fixed return target, optimise the portfolio for min volatility using CVXPY.
    Parameters
    ----------
    portfolio : Portfolio
        Portfolio whose weights will be altered to minimise the volatility.
    excess_returns_target : float
        Target excess return that the portfolio should achieve.
    factor_bounds : dict, optional
        Dictionary of factor constraints with lower and upper bounds.
    constraint_set : List(float, float), optional
        Allowed min and max of weights. The default is (0, 1).
    Returns
    -------
    portfolio : Portfolio
        Portfolio whose weights are optimised to minimise the volatility, while reaching the excess returns target.
    """
    
    num_assets = len(portfolio.universe.stocks)

    # Define optimization variables
    weights = cp.Variable(num_assets)
    weights.value = np.ones(num_assets) / num_assets

    # Access mean returns and covariance matrix
    mean_returns = portfolio.universe.mean_returns.values * TRADING_DAYS
    cov_matrix = portfolio.universe.cov_matrix.values * np.sqrt(TRADING_DAYS)

    # Define objective function (minimize volatility)
    objective = cp.Minimize(cp.quad_form(weights, cov_matrix))

    # Define constraints
    constraints = [
        cp.sum(weights) == 1,  # Weights sum to 1
        weights >= constraint_set[0],  # Lower bound on weights
        weights <= constraint_set[1],  # Upper bound on weights
        mean_returns @ weights - portfolio.universe.risk_free_rate >= excess_returns_target  # Target excess return
    ]

    # Add factor constraints if provided
    if factor_bounds:
        for factor, (lower, upper) in factor_bounds.items():
            factor_betas = portfolio.universe.get_factor_betas(factor).values
            if lower:
                constraints.append(factor_betas @ weights >= lower)
            if upper:
                constraints.append(factor_betas @ weights <= upper)

    # Set up and solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(warm_start = True, verbose = False)
        if problem.status != cp.OPTIMAL:
            raise ValueError(f"Optimization did not converge. Status: {problem.status}")
    #except cp.error.SolverError:
    except Exception as e:
        print(f"Optimisation failed: {e}")
    #raise ValueError("Optimization problem could not be solved. It may be infeasible.")

    # Update portfolio weights and recalculate performance
    portfolio.weights = clean_weights(weights.value)
    portfolio.calc_performance()

    return portfolio

def maximise_returns(portfolio, vol_target, factor_bounds=None, constraint_set=(0, 1)):
    """
    For a fixed volatility target, optimise the portfolio for max returns using CVXPY.
    Parameters
    ----------
    portfolio : Portfolio
        Portfolio whose weights will be altered to maximise the returns.
    vol_target : float
        Target volatility that the portfolio can have.
    factor_bounds : dict, optional
        Dictionary of factor constraints with lower and upper bounds.
    constraint_set : list(float, float), optional
        Allowed min and max of weights. The default is (0, 1).
    Returns
    -------
    portfolio : Portfolio
        Portfolio with weights optimised to maximise the returns while hitting the volatility target.
    """
    
    num_assets = len(portfolio.universe.stocks)

    # Define optimization variables
    weights = cp.Variable(num_assets)
    weights.value = np.ones(num_assets) / num_assets

    # Access mean returns and covariance matrix
    mean_returns = portfolio.universe.mean_returns.values * TRADING_DAYS
    cov_matrix = portfolio.universe.cov_matrix.values * TRADING_DAYS

    # Define objective function (maximize excess returns)
    objective = cp.Maximize(mean_returns @ weights - portfolio.universe.risk_free_rate)

    # Define constraints
    constraints = [
        cp.sum(weights) == 1,  # Weights sum to 1
        weights >= constraint_set[0],  # Lower bound on weights
        weights <= constraint_set[1],  # Upper bound on weights
        cp.quad_form(weights, cov_matrix) <= vol_target**2  # Volatility constraint
    ]

    # Add factor constraints if provided
    if factor_bounds:
        for factor, (lower, upper) in factor_bounds.items():
            factor_betas = portfolio.universe.get_factor_betas(factor).values
            if lower:
                constraints.append(factor_betas @ weights >= lower)
            if upper:
                constraints.append(factor_betas @ weights <= upper)
            
            # constraints.append(factor_betas @ weights >= lower)
            # constraints.append(factor_betas @ weights <= upper)

    # Set up and solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(warm_start = True, verbose = False)
        if problem.status != cp.OPTIMAL:
            raise ValueError(f"Optimization did not converge. Status: {problem.status}")
    #except cp.error.SolverError:
    except Exception as e:
        print(f"Optimisation failed: {e}")
    #raise ValueError("Optimization problem could not be solved. It may be infeasible.")

    # Update portfolio weights and recalculate performance
    portfolio.weights = clean_weights(weights.value)
    portfolio.calc_performance()

    return portfolio

def constraint_factor_exposure(weights, portfolio, factor, lower, upper):
    factor_betas = portfolio.universe.get_factor_betas(factor)
    exposure = np.dot(weights, factor_betas)
    
    return (lower <= exposure <= upper) - 0.5

def clean_weights(weights, threshold=1e-4):
    """
    Clean the weights by setting very small weights to zero and renormalizing.
    
    Parameters:
    weights (np.array): The original weights.
    threshold (float): The threshold below which weights are set to zero.
    
    Returns:
    np.array: The cleaned and renormalized weights.
    """
    if weights is None:
        return None
    
    # Set small weights to zero
    cleaned_weights = np.where(np.abs(weights) < threshold, 0, weights)
    
    # Renormalize to ensure sum is 1
    cleaned_weights = cleaned_weights / np.sum(cleaned_weights)
    
    return cleaned_weights