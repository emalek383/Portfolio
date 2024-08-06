"""Module creating the FactorAnalysis class to run a factor analysis on a stock universe."""


import numpy as np
import pandas as pd
import statsmodels.api as sm

from helper_functions import convert_to_date

class FactorAnalysis():
    """
    Class to perform factor analysis on a stock universe. Contains methods to align data,
    run factor analysis, compute factor-based covariance matrices, and analyse portfolios.
    
    Attributes
    ----------
    universe : StockUniverse.StockUniverse
        The stock universe on which to perform factor analysis.
    factor_returns : pd.DataFrame
        Returns of the factors used in the analysis.
    aligned_data : pd.DataFrame
        Aligned stock and factor returns data.
    results : dict
        Results of the factor analysis for each stock.
    factor_cov_matrix : pd.DataFrame
        Covariance matrix of factor returns.
    idiosyncratic_var : pd.Series
        Idiosyncratic variances of stocks.
    stock_cov_matrix : pd.DataFrame
        Full covariance matrix of stock returns based on factor analysis.
        
    Methods
    -------
    get_date_range():
        Returns the start and end dates of the aligned data.
        
    get_factor_exposures():
        Returns factor exposures for all stocks.
        
    get_alphas():
        Returns alphas for all stocks.
        
    get_r_squared():
        Returns R-squared values for all stocks.
        
    get_summary():
        Returns a summary of the factor analysis results.
        
    analyse_portfolio(portfolio):
        Analyses a given portfolio using the factor analysis results.
        
    """
    
    def __init__(self, universe, factor_returns):
        """
        Initialise the FactorAnalysis object with a stock universe and factor returns.

        Parameters
        ----------
        universe : StockUniverse.StockUniverse
            The stock universe to analyze.
        factor_returns : pd.DataFrame
            Returns of the factors used in the analysis.
                Index: Date (pd.Timestamp, datetime)
                Columns: Factor (str), must include 'RF' for risk-free-rate to be used

        Returns
        -------
        None.
        
        """
        
        self.universe = universe
        self.factor_returns = factor_returns
        
        if not isinstance(self.factor_returns.index, pd.DatetimeIndex):
            self.factor_returns.set_index('Date', inplace=True)
          
        self.aligned_data = self._align_data()
        self.results = self._run_factor_analysis()
        
        self.factor_cov_matrix, self.idiosyncratic_var, self.stock_cov_matrix = self._compute_factor_based_covariance()
        
    def _align_data(self):
        """
        Align stock return data with factor return data.

        Returns
        -------
        aligned_data : pd.DataFrame
            Index: Date (pd.DatetimeIndex)
            Columns: Stock tickers and factor names (str),
            Values: Daily simple returns (float), in basis points for factor returns
            
            Aligned data containing stock returns and factor returns.
            
        """
        
        start_date = pd.Timestamp(self.universe.start_date)
        end_date = pd.Timestamp(self.universe.end_date)
        
        start_date = max(start_date, self.factor_returns.index.min())
        end_date = min(end_date, self.factor_returns.index.max())
        
        stock_data = self.universe.stock_data
        if not isinstance(stock_data.index, pd.DatetimeIndex):
            raise ValueError("stock_data must have a DatetimeIndex")
        
        stock_returns = self.universe.stock_data.pct_change().dropna()
        aligned_data = pd.concat([stock_returns, self.factor_returns], axis=1).loc[start_date:end_date].dropna()
        
        return aligned_data
    
    def _run_factor_analysis(self):
        """
        Perform factor analysis on each stock in the universe.

        Returns
        -------
        results : dict
            Dictionary containing factor analysis results for each stock.
            
        """
        
        results = {}
        
        for ticker in self.universe.stocks:
            y = self.aligned_data[ticker] - self.aligned_data['RF']
            X = sm.add_constant(self.aligned_data[self.factor_returns.columns.drop('RF')] / 100) # Factor returns data is in basis points
            
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
        """
        Get the date range of the aligned data.

        Returns
        -------
        start_date, end_date : datetime.date
            The start date and end date.
            
        """
        
        start_date = convert_to_date(self.aligned_data.index[0])
        end_date = convert_to_date(self.aligned_data.index[-1])
        
        return start_date, end_date
    
    def get_factor_exposures(self):
        """
        Get factor exposures for all stocks.

        Returns
        -------
        exposures : pd.DataFrame
            Index: Asset tickers (str)
            Columns: Factor names (str)
            Values: Factor exposures (float)
            
            Factor exposures for all stocks.
        
        """
        
        exposures = pd.DataFrame({ticker: result['betas'] for ticker, result in self.results.items()}).T
        exposures.columns = self.factor_returns.columns.drop('RF')
        return exposures

    def get_alphas(self):
        """
        Get alphas for all stocks.

        Returns
        -------
        pd.Series
            Index: Asset tickers (str)
            Values: Alpha values (float)
            
            Alphas for all stocks.
            
        """
        
        return pd.Series({ticker: result['alpha'] for ticker, result in self.results.items()})

    def get_r_squared(self):
        """
        Get R-squared values for all stocks.

        Returns
        -------
        pd.Series
            Index: Asset tickers (str)
            Values: R-squared values (float)
            
            R-squared values for all stocks.
            
        """
        
        return pd.Series({ticker: result['r_squared'] for ticker, result in self.results.items()})

    def get_summary(self):
        """
        Get a summary of the factor analysis results.

        Returns
        -------
        pd.DataFrame
            Index: Asset tickers (str)
            Columns: 'Alpha', 'R-squared', and factor names (str)
            Values: Corresponding values for each stock and factor (float)
            
            Summary of factor analysis results.
            
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

        Parameters
        ----------
        portfolio : Portfolio
            The portfolio to analyse.

        Returns
        -------
        dict
            A dictionary containing:
            - 'Factor Exposures': pd.DataFrame of portfolio factor exposures
            - 'Alpha': float, portfolio alpha
            
        """
        
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
    
    def _compute_factor_based_covariance(self):
        """
        Estimate the covariance matrix of the stock universe using factor analysis.
        
        Returns
        -------
        factor_cov_matrix : pd.DataFrame
            Index: Factor name (str)
            Columns: Factor name (str)
            Value: Covariance of factor returns (float)
            
            Covariance matrix of factor returns
        
        idiosyncratic_var : pd.Series
            Index: Stock ticker (str)
                
            Idiosyncratic variances of stocks
            
        stock_cov_matrix : pd.DataFrame
            Index: Stock ticker (str)
            Columns: Stock ticker (str)
            Value: Covariance of stock returns (float)
                
            Full covariance matrix of stock returns
        
        """
        
        factor_exposures = self.get_factor_exposures()
        
        factor_returns = self.aligned_data[self.factor_returns.columns.drop('RF')] / 100
        factor_cov_matrix = factor_returns.cov()
        
        # Compute idiosyncratic returns
        stock_returns = self.aligned_data[self.universe.stocks]
        predicted_returns = factor_exposures @ factor_returns.T
        # Broadcast RF to match the dimensions of stock_returns
        rf_returns = self.aligned_data['RF'].values[:, np.newaxis] * np.ones_like(stock_returns)
        
        idiosyncratic_returns = stock_returns - predicted_returns.T - rf_returns
        
        idiosyncratic_var = idiosyncratic_returns.var()
        
        # Compute full covariance matrix
        common_cov = factor_exposures @ factor_cov_matrix @ factor_exposures.T
        idiosyncratic_cov = pd.DataFrame(np.diag(idiosyncratic_var), 
                                         index=idiosyncratic_var.index, 
                                         columns=idiosyncratic_var.index)
        stock_cov_matrix = common_cov + idiosyncratic_cov
        
        return factor_cov_matrix, idiosyncratic_var, stock_cov_matrix
