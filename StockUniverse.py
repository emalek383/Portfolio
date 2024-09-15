"""Module creating the StockUniverse and Portfolio classes.

Classes
-------
StockUniverse allows us to study a stock universe.
Portfolio: allows us to study a portfolio of stoks.
"""

import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta

from FactorAnalysis import FactorAnalysis
from RiskMetrics import RiskMetrics
from data_loader import download_data
from helper_functions import get_mean_returns, convert_to_date
from optimisers import maximise_sharpe, minimise_vol, efficient_portfolio, maximise_returns

TRADING_DAYS = 252

class StockUniverse():
    """
    Class to capture a Stock Universe. Contains important info, such as the stocks' mean returns, cov matrix,
    the risk free rate, as well as the max Sharpe Ratio and min vol portfolios.
    
    Attributes
    ----------
    stocks : list(str)
        List of tickers of stocks in the stock universe.
    start_date : datetime
        Start date of stock universe (important for getting data).
    end_date : datetime
        End date of stock universe (important for getting data).
    stock_data: pd.DataFrame
        Stock data.
    bonds_data: pd.DataFrame
        3-months T-Bill data.
    historical_returns : pd.DataFrame
        Historical returns for each stock.
    mean_returns : np.array
        Mean returns of stocks.
    cov_matrix : np.array
        Covariance matrix.
    risk_free_rate : float
        Risk free rate.
    max_sharpe_portfolio : Portfolio
        Max Sharpe Ratio Portfolio.
    min_vol_portfolio : Portfolio
        Min Vol Portfolio.
    min_returns : float
        Minimum annualised returns achievable by a portfolio in the universe.
    max_returns : float
        Maximum annualised returns achievable by a portfolio in the universe.
    min_vol : float
        Minimum annualised volatility achievable by a portfolio in the universe.
    max_vol : float
        Maximum annualised volatility achievable by a portfolio in the universe.
    factor_analysis : Factor_Analysis.Factor_Analysis
        Factor analysis run on the stock universe.
    risk_metrics : RiskMetrics
        Risk metrics calculator for the stock universe.
        
    Methods
    -------
    update_min_max():
        Compute the possible annualised min and max returns and volatility achievable in stock universe.
        Update attributes directly.
        
    optimise_portfolio(optimiser, target):
        Find the best weights in a portfolio in order to reach the target vol or excess returns,
        as specified by optimiser.
        Return the optimised portfolio.
    
    get_data():
        Download stock and 3-months T-Bill data.
        Return the tickers of stocks that could not be downloaded.
    
    calc_mean_returns_cov():
        Calculate the mean returns and covariance matrix from stock data.
        Update `mean_returns` and `cov_matrix`, risk-free-rate, calculate max sharpe portfolio, 
        min vol portfolio and update min/max returns/vol.
    
    calc_risk_free_rate():
        Calculate the risk-free-rate from bonds data. Updates attribute.
        
    get_covariance_matrix(cov_type = 'sample_cov'):
        Get the covariance matrix corresponding to specified estimation method. Returns covariance matrix.
    
    individual_stock_portfolios():
        Calculate the annualised excess returns and volatility of the individual stocks in the universe.
        Return vol and excess returns.
    
    calc_max_sharpe_portfolio():
        Calculate the maximum Sharpe Ratio portfolio. Returns the max Sharpe portfolio.
        
    calc_min_vol_portfolio():
        Calculate the minimum volatility portfolio. Returns the min vol portfoliol.
        
    calc_efficient_frontier(constraint_set = (0,1)):
        Calculate portfolios along the efficient frontier for various values of target excess return,
        and keeps track of max Sharpe portfolio found. 
        Return the efficient portfolios' vols and excess returns and the max Sharpe portfolio found along the way.
        
    run_factor_analysis(factor_returns):
        Run a factor analysis on the stock universe.
        Save the corresponding attribute.
        
    get_factor_betas(factor = None):
        Return the beta of each stock corresponding to the factor, or to all factors if None is passed.
        
    calc_factor_ranges():
        Calculate the min/max betas for each factor amongst all the stocks in the universe.
        Return these as a pd.DataFrame with min/max columns and rows indexed by the factors.
    
    """
    
    def __init__(self, 
                 stocks, 
                 start_date = dt.datetime.today() + relativedelta(years = -1), 
                 end_date = dt.datetime.today(), 
                 stock_data = None,
                 mean_returns = None, 
                 cov_matrix = None,
                 risk_free_rate = None):
        """
        Construct the attributes of the stock universe object.

        Parameters
        ----------
        stocks : list(str)
            List of stock tickers for stocks in universe.
        start_date : datetime, optional
            Start date of universe to be considered. The default is 1 year ago.
        end_date : datetime, optional
            End date of universe to be considered. The default is today.
        mean_returns : pd.Series, optional
                      Index : Asset tickers (str)
                      Values : Daily mean returns (float)
            Daily mean returns of the stock universe. Typically will be downloaded and computed in-class, but can 
            optionally be passed directly. The default is None.
        cov_matrix : pd.DataFrame, optional
                        Index : Asset tickers (str)
                        Columns : Asset tickers (str)
                        Values : Covariance (of daily mean returns) between assets (float)
            Covariance matrix of the stock universe. Typically will be downloaded and computed in-class, but can 
            optionally be passed directly. The default is None.
        risk_free_rate : float, optional
            Risk free rate at the time of the stock universe. Can be computed from bonds data. The default is None.

        Returns
        -------
        None.

        """
        
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = stock_data
        self.bonds_data = None
        self.historical_returns = None
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.factor_analysis = None
        if self.mean_returns and self.cov_matrix:
            self.update_min_max()
            self.max_sharpe_portfolio = self.calc_max_sharpe_portfolio()
            self.min_vol_portfolio = self.calc_min_vol_portfolio()
        else:
            self.min_returns = self.max_returns = self.min_vol = self.max_vol = None
            self.max_sharpe = self.min_vol = 0
        self.risk_metrics = RiskMetrics(self)
        
    def update_min_max(self):
        """
        Calculate the minimum and maximum achievable annualised excess returns and volatility by portfolios
        in the stock universe. Set the risk free rate from bonds data, if no risk free rate has been set yet.

        Returns
        -------
        None.

        """
        
        if not self.risk_free_rate:
            self.calc_risk_free_rate()
            
        self.min_returns = TRADING_DAYS * min(self.mean_returns) - self.risk_free_rate
        self.max_returns = TRADING_DAYS * max(self.mean_returns) - self.risk_free_rate
        
        self.min_vol = self.min_vol_portfolio.vol
        self.max_vol = max(self.individual_stock_portfolios()[0])
    
    def optimise_portfolio(self, optimiser, target, cov_type = 'sample_cov', factor_bounds = None):
        """
        Find the optimal weights in a portfolio in order to reach the target excess returns and minimise vol
        or maximise returns whilst volatility below a target, as specified by optimiser.

        Parameters
        ----------
        optimiser : str
            Describes how we want to optimise: by minimising volatility or maximising returns.
        target : float
            The target we want the portfolio to achieve, as specified by optimised, i.e. the excess returns or volatility.
        cov_type : str, optional
            Estimation method for the covariance matrix. Default is 'sample_cov'.
        factor_bounds: dict, optional
            Dictionary of factor constraints with lower and upper bounds. Default is None.

        Returns
        -------
        optimised_portfolio : Portfolio
            Portfolio with optimal weights, so as to achieve the target excess returns/vol.

        """

        optimised_portfolio = Portfolio(self)
        
        if optimiser == 'min_vol':
            try:
                optimised_portfolio = efficient_portfolio(optimised_portfolio, target, cov_type = cov_type, factor_bounds = factor_bounds)
            except Exception as e:
                raise ValueError(str(e))
        else:
            try:
                optimised_portfolio = maximise_returns(optimised_portfolio, target, cov_type = cov_type, factor_bounds = factor_bounds)
            except Exception as e:
                raise ValueError(str(e))
            
        return optimised_portfolio
    
    def get_data(self):
        """
        Download the stock and bonds data and save it in relevant attributes.

        Returns
        -------
        ignored : list(str)
            List of stock tickers for stocks that could not be donloaded.

        """
        
        stock_data = download_data(self.stocks, self.start_date, self.end_date)
        
        ignored = []
        if not self.stocks or len(stock_data) == 0:
            ignored = self.stocks
        else:
            for ticker in self.stocks:
                if ticker not in stock_data.columns or len(stock_data[ticker]) == 0 or stock_data[ticker].isnull().all():
                    ignored.append(ticker)
                    
            for ticker in ignored:
                self.stocks.remove(ticker)

            stock_data = stock_data[self.stocks]
                
        self.stock_data = stock_data
        
        self.start_date = convert_to_date(self.stock_data.index[0])
        self.end_date = convert_to_date(self.stock_data.index[-1])
        
        bonds_data = download_data(['^IRX'], self.start_date, self.end_date)
        if isinstance(bonds_data, pd.DataFrame):
            self.bonds_data = bonds_data['^IRX']
        else:
            self.bonds_data = bonds_data
        
        return ignored
        
    def calc_mean_returns_cov(self):
        """
        Compute historical returns, mean returns and covariance matrix of stock universe from the (downloaded) stock data.
        Set the risk-free-rate from downloaded bonds data if it has not yet been set.
        Calculate the max Sharpe Ratio and min vol portfolios and update the min/max excess returns/vol.
        Update all these attributes.

        Returns
        -------
        None.

        """
        
        self.historical_returns = self.stock_data.pct_change().dropna()
        self.mean_returns = get_mean_returns(self.historical_returns)
        self.cov_matrix = self.historical_returns.cov()
        
        if not self.risk_free_rate:
            self.calc_risk_free_rate()
        
        self.max_sharpe_portfolio = self.calc_max_sharpe_portfolio()
        self.min_vol_portfolio = self.calc_min_vol_portfolio()
        self.update_min_max()
                
    def calc_risk_free_rate(self):
        """
        Calculate the annualised risk free rate from the downloaded bonds data. Update the attribute `self.risk_free_rate`.

        Returns
        -------
        None.

        """
        
        if self.bonds_data is None or len(self.bonds_data) == 0:
            self.risk_free_rate = 0
            return
        
        R_F_annual = self.bonds_data/100
        self.risk_free_rate = get_mean_returns(R_F_annual)
        
    def get_covariance_matrix(self, cov_type = 'sample_cov'):
        """
        Get the covariance matrix estimated using a specific method.

        Parameters
        ----------
        cov_type : {'sample_cov', 'factor_cov'}
            Estimation method for the covariance matrix. The default is 'sample_cov'.

        Returns
        -------
        pd.DataFrame
            Index : Asset tickers (str)
            Columns : Asset tickers (str)
            Values : Covariance between assets (float)
            
            Covariance matrix.

        """
        
        if cov_type == 'factor_cov' and self.factor_analysis is not None:
            return self.factor_analysis.stock_cov_matrix
        else:
            return self.cov_matrix
            
    def individual_stock_portfolios(self, cov_type = 'sample_cov'):
        """
        Calculate the annualised excess returns and volatility of individual stocks in the stock portfolio.
        
        Parameters
        ----------
        cov_type : str, optional
            Estimation method for the covariance matrix. Default is 'sample_cov'.
        
        Returns
        -------
        vols : list(float)
            List of annualised volatility of individual stocks.
        excess_returns : list(float)
            List of annualised excess returns of the individual stocks.
            
        """
        
        vols = []
        excess_returns = []
        cov_matrix = self.get_covariance_matrix(cov_type)
        for idx, stock in enumerate(self.stocks):
            excess_returns.append( TRADING_DAYS * (self.mean_returns.iloc[idx]) - self.risk_free_rate )
            vols.append( np.sqrt( TRADING_DAYS * cov_matrix[stock][stock]) )
        return vols, excess_returns
    
    def calc_max_sharpe_portfolio(self):
        """
        Calculate the max Sharpe Ratio portfolio.

        Returns
        -------
        max_sharpe_portfolio : Portfolio
            The max Sharpe Ratio portfolio.

        """
        
        max_sharpe_portfolio = Portfolio(self, name = "Max Sharpe")
        max_sharpe_portfolio = maximise_sharpe(max_sharpe_portfolio)
        return max_sharpe_portfolio
        
    def calc_min_vol_portfolio(self):
        """
        Calculate the minimum volatility portfolio.

        Returns
        -------
        min_vol_portfolio : Portfolio
            The mininimum volatility portfolio.

        """
        
        min_vol_portfolio = Portfolio(self, name = "Min Vol")
        min_vol_portfolio = minimise_vol(min_vol_portfolio)
        return min_vol_portfolio
    
    def calc_efficient_frontier(self, cov_type = 'sample_cov', constraint_set = (0, 1)):
        """
        Calculate the portfolios along the efficient frontier for various values of target excess return.
        Calculate the max Sharpe portfolio along the way, update the attribute and return it.

        Parameters
        ----------
        cov_type : str, optional
            Estimation method for the covariance matrix. Default is 'sample_cov'.
        constraint_set : array_like(float, float), optional
            Allowed min and max of weights. The default is (0, 1).

        Returns
        -------
        efficient_frontier_data : tuple(list(float), list(float))
            Tuple containing the list of volatilities and excess returnns of efficient portfolios (i.e. minimising vol given returns).
        cur_max_sharpe_portfolio : Portfolio
            The max Sharpe Portfolio found from the efficient frontier.

        """
        
        LOWER = self.min_returns
        UPPER = self.max_returns
        
        cur_max_sharpe_portfolio = self.max_sharpe_portfolio
        
        target_excess_returns = np.linspace(LOWER, UPPER, 500)
        efficient_frontier_vols = []
        efficient_frontier_returns = []

        for target in target_excess_returns:
            eff_portfolio = Portfolio(self)
            try:
                eff_portfolio = efficient_portfolio(eff_portfolio, target, cov_type = cov_type)
                efficient_frontier_vols.append(eff_portfolio.vol)
                efficient_frontier_returns.append(eff_portfolio.excess_returns)
                
                if cur_max_sharpe_portfolio and cur_max_sharpe_portfolio.sharpe_ratio < eff_portfolio.sharpe_ratio:
                     cur_max_sharpe_portfolio = eff_portfolio
                     cur_max_sharpe_portfolio.name = 'Max Sharpe'
            except:
                continue
                
        if cov_type == 'sample_cov':
            self.max_sharpe_portfolio = cur_max_sharpe_portfolio
            
        efficient_frontier_data = (efficient_frontier_vols, efficient_frontier_returns)
        
        return efficient_frontier_data, cur_max_sharpe_portfolio
            
    def run_factor_analysis(self, factor_returns):
        """
        Run a factor analysis on the stock universe.

        Parameters
        ----------
        factor_returns : pd.DataFrame
            Factor returns of the model to be used.
                Index: Date (pd.Timestamp, datetime)
                Columns: Factor (str), must include 'RF' for risk-free-rate to be used
                Values: Daily simple returns in basis points (float).

        Returns
        -------
        None.

        """
        
        self.factor_analysis = FactorAnalysis(self, factor_returns)
        
        return None
    
    def get_factor_betas(self, factor = None):
        """
        Get the factor betas for each stock in the universe.

        Parameters
        ----------
        factor : str, optional
            Factor whose beta is to be returned. If None, then return all factor betas. The default is None.

        Raises
        ------
        ValueError
            If factor analysis has not yet been set for the universe or the factor passed is not found in the factor model.

        Returns
        -------
        pd.Series, pd.DataFrame
            Index: 
                stock ticker (str)
            Columns:  
                factor (str)
            
            The beta of all stocks with respect to a factor or all factors.

        """
        
        if self.factor_analysis is None:
            raise ValueError("Factor analysis has not been set for this universe.")
        
        factor_exposures = self.factor_analysis.get_factor_exposures()
        
        if factor is None:
            return factor_exposures
        elif factor in factor_exposures.columns:
            return factor_exposures[factor]
        else:
            raise ValueError(f"Factor '{factor}' not found in the factor analysis.")
            
        return None
    
    def calc_factor_ranges(self):
        """
        Calculate the min/max betas for each factor amongst all the stocks in the universe.

        Returns
        -------
        pd.DataFrame
            Index:
                factor, str.
            Columns:
                'min': minimum beta, float
                'max': max_beta, float
                
            The min anx max beta for each factor.

        """
        
        factor_betas = self.get_factor_betas()
        min_betas = factor_betas.min()
        max_betas = factor_betas.max()
        return pd.DataFrame({'min': min_betas, 'max': max_betas})
    
class Portfolio():
    """
    Class to capture a portfolio. Contains the portfolio's name, stock universe, weights, 
    excess returns and volatility.
    
    Attributes
    ----------
    universe : StockUniverse.StockUniverse
        Stock universe containing the stocks of the portfolio.
    name : str
        Name of portfolio.
    returns : float
        Annualised returns of the portfolio.
    vol : float
        Annualised volatility of the portfolio.
    excess_returns : float
        Annualised excess returns of the portfolio.
    weights : np.array
        Weights of the portfolio's stocks.
    sharpe_ratio : float
        Annualised Sharpe Ratio of the portfolio.
    hist_var_cvar : dict
        Dictionary containing historical VaR and CVaR for different time horizons.
    hist_var_confidence_level : float
        Confidence level used for historical VaR and CVaR calculations.
    mc_var_cvar : dict
        Dictionary containing Monte Carlo VaR and CVaR for different time horizons.
    mc_var_confidence_level : float
        Confidence level used for Monte Carlo VaR and CVaR calculations.
        
    Methods
    -------
    normalise_weights():
        Normalise weights so they add up to 1. Update attribute directly.
        
    calc_performance(cov_type = 'sample_cov'):
        Compute the portfolio's annualised returns, excess returns, volatility and sharpe ratio from the Stock Universe 
        mean returns and covariance matrix (using the estimation method 'cov_type').  If the relevant stock data does 
        not exist, save all these quantities as 0. Save attributes directly.
        
    get_performance_df():
        Create a pd.DataFrame of the excess returns, volatility and sharpe ratio.
        
    get_weights_df():
        Create a pd.DataFrame of the weights of the portfolio.
        
    calc_hist_var_cvar(time_horizon, confidence_level):
        Calculate historical VaR and CVaR for the portfolio.
        
    calc_mc_var_cvar(time_horizon, confidence_level):
        Calculate Monte Carlo VaR and CVaR for the portfolio.
        
    """
    
    def __init__(self, universe, name = "", weights = []):
        """
        Construct the attributes of the portfolio object. If no weights are passed, instantiate the portfolio
        with uniform weights.

        Parameters
        ----------
        universe : StockUniverse.StockUniverse
            Stock universe of the portfolio's stocks.
        name : str, optional
            Name of the portfolio, used for print out purposes. The default is "".
        weights : list or np.array, optional
            Weights of the portfolio's stocks. If none passed, portfolio weights will be made uniform. The default is [].

        Returns
        -------
        None.

        """
        
        self.universe = universe
        self.name = name
        self.returns = 0
        self.vol = 0
        self.excess_returns = 0
        
        self.hist_var_cvar = {}
        self.hist_var_confidence_level = None
        self.mc_var_cvar = {}
        self.mc_var_confidence_level = None
        
        if len(weights) == 0:
            self.weights = np.array([1] * len(self.universe.stocks))
        else:
            self.weights = np.array(weights)
            
        self.normalise_weights()
        self.calc_performance()
        
    def normalise_weights(self):
        """
        Normalise weights so they add up to 1. Update attribute directly.

        Returns
        -------
        None.

        """
        
        self.weights = self.weights / self.weights.sum()
    
    def calc_performance(self, cov_type = 'sample_cov'):
        """
        Compute portfolio performance with respect to an estimated covariance matrix: annualised return, excess return, volatility, Sharpe Ratio,
        historical and Monte Carlo VaR and CVaR.
        
        Parameters
        ----------
        cov_type : str, optional
            Estimation method for the covariance matrix. Default is 'sample_cov'.

        Returns
        -------
        float
            Annualised Returns.
        float
            Volatility.
        float
            Annualised Excess Returns.
        float
            Sharpe Ratio.

        """
        
        if self.universe.mean_returns is not None:
            cov_matrix = self.universe.get_covariance_matrix(cov_type)
            self.returns = np.dot(self.weights, self.universe.mean_returns) * TRADING_DAYS
            self.vol = np.sqrt( np.dot(np.dot(self.weights.T, cov_matrix), self.weights) * TRADING_DAYS )
            self.excess_returns = self.returns - self.universe.risk_free_rate
            self.sharpe_ratio = self.excess_returns / self.vol
            try:
                self.calc_hist_var_cvar()
                self.calc_mc_var_cvar()
            except ValueError as e:
                if "Not enough data to compute VaR" in str(e):
                    self.hist_var_cvar = {}
                    self.mc_var_cvar = {}
                else:
                    raise
        
        else:
            self.returns = self.vol = self.excess_returns = self.sharpe_ratio = 0
        
        return self.returns, self.vol, self.excess_returns, self.sharpe_ratio
    
    def get_performance_df(self):
        """
        Create a pd.DataFrame of the excess returns, volatility and sharpe ratio.

        Returns
        -------
        results_df : pd.DataFrame
                Index: 
                    Portfolio name (str)
                Columns:
                    'Excess Returns': annualised excess returns (float)
                    'Volatility': annualised volatility (float)
                    'Sharpe Ratio': annualised Sharpe Ratio (float)
                    'Historical VaR (time_horizon)': historical VaR for different time_horizons
                    'Historical CVaR (time_horizon)': historical CVaR for different time_horizons
                    'Monte Carlo VaR (time_horizon)': Monte Carlo VaR for different time_horizons
                    'Monte Carlo CVaR (time_horizon)': Monte Carlo CVaR for different time_horizons
            Dataframe containing the excess returns, volatility, sharpe ratio and VaR/CVaR.
        format_map : dict
            Dictionary that maps the outputs into a nice format for display.

        """
        
        if not self.returns or not self.vol or not self.excess_returns or not self.sharpe_ratio:
            self.performance()
        
        results = {'Excess Returns': self.excess_returns,
                   'Volatility': self.vol,
                   'Sharpe Ratio': self.sharpe_ratio,}
        
        format_horizon = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}
                
        for horizon, (var, cvar) in self.hist_var_cvar.items():
            results[f'Historical VaR ({format_horizon[horizon]})'] = var
            results[f'Historical CVaR ({format_horizon[horizon]})'] = cvar
        
        for horizon, (var, cvar) in self.mc_var_cvar.items():
            results[f'Monte Carlo VaR ({format_horizon[horizon]})'] = var
            results[f'Monte Carlo CVaR ({format_horizon[horizon]})'] = cvar
        
        results_df = pd.DataFrame(results, index=[self.name + ' Portfolio'])
        format_map = {'Excess Returns': '{:,.1%}'.format, 'Volatility': '{:,.1%}'.format, 'Sharpe Ratio': '{:,.2f}'}
        
        for horizon in format_horizon.values():
            format_map[f'Historical VaR ({horizon})'] = '{:,.1%}'.format
            format_map[f'Historical CVaR ({horizon})'] = '{:,.1%}'.format
            format_map[f'Monte Carlo VaR ({horizon})'] = '{:,.1%}'.format
            format_map[f'Monte Carlo CVaR ({horizon})'] = '{:,.1%}'.format
        
        return results_df, format_map
    
    def get_weights_df(self):
        """
        Create a pd.DataFrame of the weights of the portfolio.

        Returns
        -------
        weights_df : pd.DataFrame
            Dataframe containing the weights of the portfolio's stocks.
                Index: 
                    Portfolio name (str)
                Columns: 
                    Asset tickers (str)
                Values: 
                    Weight of asset in portfolio (float)
        format_map : dict
            Dictionary that maps the outputs into a nice format for display.

        """
        
        weights = {self.universe.stocks[idx]: self.weights[idx] for idx in range(len(self.universe.stocks))}
        
        weights_df = pd.DataFrame(weights, index = [self.name + ' Portfolio'])
        
        format_map = {col: "{:,.0%}".format for col in weights_df.columns}
        
        return weights_df, format_map
    
    def calc_hist_var_cvar(self, time_horizon = None, confidence_level = 0.95):
        """
        Calculate historical VaR and CVaR for the portfolio.

        Parameters
        ----------
        time_horizon : str, optional
            Time horizon for VaR and CVaR calculation. If None, calculates for all available horizons. The default is None.
        confidence_level : float, optional
            Confidence level for VaR and CVaR calculation. The default is 0.95.

        Returns
        -------
        None.
        
        """
        
        if not self.universe.risk_metrics:
            self.universe.risk_metrics = RiskMetrics(self.universe)
            
        if not time_horizon:
            for time_horizon in ['D', 'W', 'M']:
                self.calc_hist_var_cvar(time_horizon = time_horizon, confidence_level = confidence_level)
                
        else:
            rm = self.universe.risk_metrics
            try:
                self.hist_var_cvar[time_horizon] = rm.calculate_historical_var_cvar(self, 
                                                                                    time_horizon = time_horizon,
                                                                                    confidence_level = confidence_level)
                self.hist_var_confidence_level = confidence_level
            except ValueError as e:
                if 'Time period is not long enough' in str(e):
                    pass
                else:
                    raise
    
    def calc_mc_var_cvar(self, time_horizon = None, confidence_level = 0.95):
        """
        Calculate Monte Carlo VaR and CVaR for the portfolio.

        Parameters
        ----------
        time_horizon : str, optional
            Time horizon for VaR and CVaR calculation. If None, calculates for all available horizons. The default is None.
        confidence_level : float, optional
            Confidence level for VaR and CVaR calculation. The default is 0.95.

        Returns
        -------
        None.
        
        """
        
        if not self.universe.risk_metrics:
            self.universe.risk_metrics = RiskMetrics(self.universe)
            
        if not time_horizon:
            for time_horizon in ['D', 'W', 'M']:
                self.calc_mc_var_cvar(time_horizon = time_horizon, confidence_level = confidence_level)
                
        else:
            rm = self.universe.risk_metrics
            try:
                self.mc_var_cvar[time_horizon] = rm.calculate_monte_carlo_var_cvar(self, 
                                                                                   time_horizon = time_horizon,
                                                                                   confidence_level = confidence_level)
                self.mc_var_confidence_level = confidence_level
            except ValueError as e:
                if 'Time period is not long enough' in str(e):
                    pass
                else:
                    raise
                