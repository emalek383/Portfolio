import numpy as np
import pandas as pd


class RiskMetrics:
    def __init__(self, universe, num_simulations = 10_000):
        self.universe = universe
        self.num_simulations = num_simulations
        self.historical_returns = {}
        self.simulated_returns = {}
        
    def _get_time_factor(self, horizon):
        factor = {'D': 1, 'W': 5, 'M': 21}
        if horizon in factor:
            return factor[horizon]
        else:
            raise ValueError(f"Unsupported time horizon: {horizon}")
    
    def _scale_returns(self, horizon):
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
        if time_horizon not in self.historical_returns:
            self.precompute_historical_returns([time_horizon])
            
        historical_returns = self.historical_returns[time_horizon]
        portfolio_returns = historical_returns.dot(portfolio.weights)
        
        var = -np.percentile(portfolio_returns, 100 * (1 - confidence_level) )
        cvar = -portfolio_returns[portfolio_returns <= - var].mean()
        
        return var, cvar
    
    def calculate_monte_carlo_var_cvar(self, portfolio, confidence_level = 0.95, time_horizon = 'W'):
        if time_horizon not in self.simulated_returns:
            self.precompute_monte_carlo_simulations([time_horizon])
            
        simulated_returns = self.simulated_returns[time_horizon]
        portfolio_returns = simulated_returns.dot(portfolio.weights)
        
        var = - np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        cvar = - portfolio_returns[portfolio_returns <= - var].mean()
        
        return var, cvar