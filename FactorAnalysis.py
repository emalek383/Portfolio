import numpy as np
import pandas as pd
import statsmodels.api as sm
from helper_functions import convert_to_date

class FactorAnalysis():
    def __init__(self, universe, factor_returns):
        self.universe = universe
        self.factor_returns = factor_returns
        
        if not isinstance(self.factor_returns.index, pd.DatetimeIndex):
            self.factor_returns.set_index('Date', inplace=True)
          
        self.aligned_data = self._align_data()
        self.results = self._run_factor_analysis()
        
    def _align_data(self):
        start_date = pd.Timestamp(self.universe.start_date)
        end_date = pd.Timestamp(self.universe.end_date)
        
        start_date = max(start_date, self.factor_returns.index.min())
        end_date = min(end_date, self.factor_returns.index.max())
        
        stock_data = self.universe.stock_data
        if not isinstance(stock_data.index, pd.DatetimeIndex):
            raise ValueError("stock_data must have a DatetimeIndex")
        
        stock_returns = self.universe.stock_data.pct_change().dropna()
        aligned_data = pd.concat([stock_returns, self.factor_returns], axis=1).loc[start_date:end_date].dropna()
        
        #print(f"Aligned data range: {aligned_data.index[0]} to {aligned_data.index[-1]}")
        
        return aligned_data
    
    def _run_factor_analysis(self):
        results = {}
        
        for ticker in self.universe.stocks:
            y = self.aligned_data[ticker] - self.aligned_data['RF']  # Excess returns
            X = sm.add_constant(self.aligned_data[self.factor_returns.columns.drop('RF')] / 100)
            
            model = sm.OLS(y,X).fit()
            results[ticker] = {
                'alpha': model.params['const'],
                'betas': model.params[1:],
                'r_squared': model.rsquared,
                't_stats': model.tvalues,
                'p_values': model.pvalues
                }
            
        return results
    
    def get_date_range(self):
        start_date = convert_to_date(self.aligned_data.index[0])
        end_date = convert_to_date(self.aligned_data.index[-1])
        
        return start_date, end_date
    
    def get_factor_exposures(self):
        """
        Get factor exposures for all stocks.
        """
        exposures = pd.DataFrame({ticker: result['betas'] for ticker, result in self.results.items()}).T
        exposures.columns = self.factor_returns.columns.drop('RF')
        return exposures

    def get_alphas(self):
        """
        Get alphas for all stocks.
        """
        return pd.Series({ticker: result['alpha'] for ticker, result in self.results.items()})

    def get_r_squared(self):
        """
        Get R-squared values for all stocks.
        """
        return pd.Series({ticker: result['r_squared'] for ticker, result in self.results.items()})

    def get_summary(self):
        """
        Get a summary of the factor analysis results.
        """
        summary = pd.DataFrame({
            'Alpha': self.get_alphas(),
            'R-squared': self.get_r_squared()
        })
        factor_exposures = self.get_factor_exposures()
        return pd.concat([summary, factor_exposures], axis=1)

    def analyse_portfolio(self, portfolio):
        """
        Analyse a given portfolio using the factor analysis results.
        """
        # if not isinstance(portfolio, Portfolio):
        #     raise TypeError("Input must be a Portfolio object")
        
        if portfolio.universe != self.universe:
            raise ValueError("The portfolio's universe does not match the FactorAnalysis universe.")
        
        factor_exposures = self.get_factor_exposures()        
        portfolio_exposures = np.dot(portfolio.weights, factor_exposures)
        portfolio_exposures_df = pd.DataFrame([portfolio_exposures], columns = factor_exposures.columns, index = [portfolio.name if portfolio.name != '' else 'Portfolio'])
        
        portfolio_alpha = np.dot(portfolio.weights, self.get_alphas())
        
        return {
            'Factor Exposures': portfolio_exposures_df,
            'Alpha': portfolio_alpha
        }