import numpy as np
import pandas as pd


class RiskMetrics:
    """
    A class for calculating various risk metrics for a given stock universe.

    Attributes
    ----------
    universe : StockUniverse
        The stock universe to analyse.
    num_simulations : int
        The number of Monte Carlo simulations to run.
    historical_returns : dict
        Dictionary of historical returns for different time horizons.
    simulated_returns : dict
        Dictionary of simulated returns for different time horizons.

    Methods
    -------
    _get_time_factor(horizon):
        Get the time factor for a given horizon.
        
    _scale_returns(horizon):
        Scale returns for a given horizon.
        
    precompute_historical_returns(time_horizons):
        Precompute historical returns for given time horizons.
        
    precompute_monte_carlo_simulations(time_horizons):
        Precompute Monte Carlo simulations for given time horizons.
        
    calculate_historical_var_cvar(portfolio, confidence_level, time_horizon):
        Calculate historical VaR and CVaR for a given portfolio.
        
    calculate_monte_carlo_var_cvar(portfolio, confidence_level, time_horizon):
        Calculate Monte Carlo VaR and CVaR for a given portfolio.
    
    """
    
    def __init__(self, universe, num_simulations = 10_000):
        """
        Initialise the RiskMetrics object.

        Parameters
        ----------
        universe : StockUniverse
            The stock universe to analyse.
        num_simulations : int, optional
            The number of Monte Carlo simulations to run. The default is 10,000.

        Returns
        -------
        None.
        
        """
        
        self.universe = universe
        self.num_simulations = num_simulations
        self.historical_returns = {}
        self.simulated_returns = {}
        
    def _get_time_factor(self, horizon):
        """
        Get the time factor for a given horizon.

        Parameters
        ----------
        horizon : str
            The time horizon ('D' for daily, 'W' for weekly, 'M' for monthly).

        Raises
        ------
        ValueError
            If an unsupported time horizon is provided.

        Returns
        -------
        int
            The time factor for the given horizon.
        
        """
        
        factor = {'D': 1, 'W': 5, 'M': 21}
        if horizon in factor:
            return factor[horizon]
        else:
            raise ValueError(f"Unsupported time horizon: {horizon}")
    
    def _scale_returns(self, horizon):
        """
        Scale returns for a given horizon.

        Parameters
        ----------
        horizon : str
            The time horizon to scale returns for.

        Raises
        ------
        ValueError
            If there is not enough data to compute VaR.

        Returns
        -------
        np.array
            Scaled returns for the given horizon.
        
        """
        
        time_factor = self._get_time_factor(horizon)

        time_span = len(self.universe.historical_returns)        
        if time_span >= 50 * self._get_time_factor('W'):
            self.precompute_historical_returns(['W'])
            scaling_factor = np.sqrt(time_factor / self._get_time_factor('W'))
            return scaling_factor * self.historical_returns['W']
            
        elif time_span >= 50 * self._get_time_factor('D'):
            self.precompute_historical_returns(['D'])
            scaling_factor = np.sqrt(time_factor / self._get_time_factor('D'))
            return scaling_factor * self.historical_returns['D']
            
        else:
            raise ValueError("Not enough data to compute VaR.")
    
    def precompute_historical_returns(self, time_horizons = ['D', 'W', 'M']):
        """
        Precompute historical returns for given time horizons.

        Parameters
        ----------
        time_horizons : list of str, optional
            List of time horizons to precompute returns for. The default is ['D', 'W', 'M'].

        Raises
        ------
        ValueError
            If an unsupported time horizon is provided.

        Returns
        -------
        None.
        
        """
        
        daily_returns = self.universe.historical_returns
        for horizon in time_horizons:
            if horizon not in ['D', 'W', 'M']:
                raise ValueError(f"Unsupported time horizon: {horizon}")
                
            time_factor = self._get_time_factor(horizon)
            if len(daily_returns) < 50 * time_factor:
                self.historical_returns[horizon] = self._scale_returns(horizon)
            else:
                if horizon == 'D':
                    self.historical_returns[horizon] = daily_returns
                else:
                    self.historical_returns[horizon] = daily_returns.resample(horizon).agg(lambda x: (1 + x).prod() - 1)
                
    def precompute_monte_carlo_simulations(self, time_horizons = ['D', 'W', 'M']):
        """
        Precompute Monte Carlo simulations for given time horizons.

        Parameters
        ----------
        time_horizons : list of str, optional
            List of time horizons to precompute simulations for. The default is ['D', 'W', 'M'].

        Returns
        -------
        None.
        
        """
        
        for horizon in time_horizons:
            if horizon not in self.historical_returns:
                self.precompute_historical_returns([horizon])
                
            returns = self.historical_returns[horizon]
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            simulated_returns = np.random.multivariate_normal(
                mean_returns,
                cov_matrix,
                self.num_simulations
            )
            
            self.simulated_returns[horizon] = pd.DataFrame(simulated_returns, columns = returns.columns)
            
    def calculate_historical_var_cvar(self, portfolio, confidence_level = 0.95, time_horizon = 'W'):
        """
        Calculate historical VaR and CVaR for a given portfolio.

        Parameters
        ----------
        portfolio : Portfolio
            The portfolio to calculate VaR and CVaR for.
        confidence_level : float, optional
            The confidence level for VaR and CVaR calculation. The default is 0.95.
        time_horizon : str, optional
            The time horizon for VaR and CVaR calculation. The default is 'W' (weekly).

        Returns
        -------
        tuple
            A tuple containing VaR and CVaR values.
        
        """
        
        if time_horizon not in self.historical_returns:
            self.precompute_historical_returns([time_horizon])
            
        historical_returns = self.historical_returns[time_horizon]
        portfolio_returns = historical_returns.dot(portfolio.weights)
        
        var = -np.percentile(portfolio_returns, 100 * (1 - confidence_level) )
        cvar = -portfolio_returns[portfolio_returns <= - var].mean()
        
        return var, cvar
    
    def calculate_monte_carlo_var_cvar(self, portfolio, confidence_level = 0.95, time_horizon = 'W'):
        """
        Calculate Monte Carlo VaR and CVaR for a given portfolio.

        Parameters
        ----------
        portfolio : Portfolio
            The portfolio to calculate VaR and CVaR for.
        confidence_level : float, optional
            The confidence level for VaR and CVaR calculation. The default is 0.95.
        time_horizon : str, optional
            The time horizon for VaR and CVaR calculation. The default is 'W' (weekly).

        Returns
        -------
        tuple
            A tuple containing VaR and CVaR values.
        
        """
        
        if time_horizon not in self.simulated_returns:
            self.precompute_monte_carlo_simulations([time_horizon])
            
        simulated_returns = self.simulated_returns[time_horizon]
        portfolio_returns = simulated_returns.dot(portfolio.weights)
        
        var = - np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        cvar = - portfolio_returns[portfolio_returns <= - var].mean()
        
        return var, cvar