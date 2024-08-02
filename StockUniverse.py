import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from helper_functions import get_mean_returns, convert_to_date
from optimisers import maximise_SR, minimise_vol, efficient_portfolio, maximise_returns
from data_loader import download_data
from FactorAnalysis import FactorAnalysis

TRADING_DAYS = 252

class StockUniverse():
    """
    Class to capture a Stock Universe. Contains important info, such as the stocks' mean returns, cov matrix,
    the risk free rate, as well as the max Sharpe Ratio and min vol portfolios.
    
    Attributes
    ----------
    stocks : list(str)
        List of tickers of stocks in the stock universe
    start_date : datetime
        Start date of stock universe (important for getting data)
    end_date : datetime
        End date of stock universe (important for getting data)
    stock_data: pd.DataFrame
        Stock data
    bonds_data: pd.DataFrame
        3-months T-Bill data
    mean_returns : np.array
        Mean returns of stocks
    cov_matrix : np.array
        Covariance matrix
    risk_free_rate : float
        Risk free rate
    max_SR_portfolio : Portfolio
        Max Sharpe Ratio Portfolio
    min_vol_portfolio : Portfolio
        Min Vol Portfolio
    min_returns : float
        Minimum annualised returns achievable by a portfolio in the universe
    max_returns : float
        Maximum annualised returns achievable by a portfolio in the universe
    min_vol : float
        Minimum annualised volatility achievable by a portfolio in the universe
    max_vol : float
        Maximum annualised volatility achievable by a portfolio in the universe
        
    Methods
    -------
    update_min_max():
        Computes the possible annualised min and max returns and volatility achievable in stock universe.
        Updates attributes directly.
        
    optimise_portfolio(optimiser, target):
        Finds the best weights in a portfolio in order to reach the target vol or excess returns,
        as specified by optimiser.
        Returns the optimised portfolio.
    
    get_data():
        Downloads stock and 3-months T-Bill data.
        Returns the tickers of stocks that could not be downloaded.
    
    calc_mean_returns_cov():
        Calculates the mean returns and covariance matrix from stock data.
        Updates mean_returns and cov_matrix, risk-free-rate, calculates max SR portfolio, 
        min vol portfolio and updates min/max returns/vol.
    
    calc_risk_free_rate():
        Calculates the risk-free-rate from bonds data. Updates attribute.
    
    individual_stock_portfolios():
        Calculates the annualised excess returns and volatility of the individual stocks in the universe.
        Returns vol and excess returns.
    
    calc_max_SR_portfolio():
        Calculates the maximum Sharpe Ratio portfolio. Updates attributes directly. Returns None.
        
    calc_min_vol_portfolio():
        Calculates the minimum volatility portfolio. Updates attributes directly. Returns None.
        
    calc_efficient_frontier(constraint_set = (0,1)):
        Calculates portfolios along the efficient frontier for various values of target excess return.
        Returns the efficient portfolios' vols and excess returns.
    
    """
    def __init__(self, 
                 stocks, 
                 start_date = dt.datetime.today() + relativedelta(years = -1), 
                 end_date = dt.datetime.today(), 
                 mean_returns = None, 
                 cov_matrix = None,
                 risk_free_rate = None):
        """
        Constructs the attributes of the stock universe object.

        Parameters
        ----------
        stocks : list(str)
            List of stock tickers for stocks in universe.
        start_date : datetime, optional
            Start date of universe to be considered. The default is 1 year ago.
        end_date : datetime, optional
            End date of universe to be considered. The default is today.
        mean_returns : np.array, optional
            Mean returns of the stock universe. Typically will be downloaded and computed in-class, but can 
            optionally be passed directly. The default is None.
        cov_matrix : np.array, optional
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
        self.stock_data = None
        self.bonds_data = None
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.factor_analysis = None
        if self.mean_returns and self.cov_matrix:
            self.update_min_max()
            self.max_SR_portfolio = self.calc_max_SR_portfolio()
            self.min_vol_portfolio = self.calc_min_vol_portfolio()
        else:
            self.min_returns = self.max_returns = self.min_vol = self.max_vol = None
            self.max_SR = self.min_vol = 0
        
    def update_min_max(self):
        """
        Calculates the minimum and maximum achievable annualised excess returns and volatility by portfolios
        in the stock universe. Will set the risk free rate from bonds data, if no risk free rate has been set yet.

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
    
    def optimise_portfolio(self, optimiser, target, factor_bounds = None):
        """
        Finds the best weights in a portfolio in order to reach the target vol or excess returns,
        as specified by optimiser.

        Parameters
        ----------
        optimiser : str
            Describes how we want to optimise: by minimising volatility or maximising returns.
        target : float
            The target we want the portfolio to achieve, as specified by optimised, i.e. the excess returns or volatility.

        Returns
        -------
        optimised_portfolio : Portfolio
            Portfolio with optimal weights, so as to achieve the target excess returns/vol.

        """
        
        
        optimised_portfolio = Portfolio(self)
        
        if optimiser == 'min_vol':
            try:
                optimised_portfolio = efficient_portfolio(optimised_portfolio, target, factor_bounds)
            except Exception as e:
                print(f"Got an error at the level of StockUniverse.optimise_portfolio(). Error {str(e)}")
                raise ValueError(str(e))
        else:
            try:
                optimised_portfolio = maximise_returns(optimised_portfolio, target, factor_bounds)
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
        self.bonds_data = bonds_data.rename('^IRX')
        
        return ignored
        
    def calc_mean_returns_cov(self):
        """
        Compute mean returns and covariance matrix of stock universe from the (downloaded) stock data.
        Will set the risk-free-rate from downloaded bonds data if it has not yet been set.
        Calculate the max Sharpe Ratio and min vol portfolios and upadte the min/max excess returns/vol.
        Update all these attributes.

        Returns
        -------
        None.

        """
        
        returns = self.stock_data.pct_change()
        mean_returns = get_mean_returns(returns)
        cov_matrix = returns.cov()
        
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        
        if not self.risk_free_rate:
            self.calc_risk_free_rate()
        
        self.calc_max_SR_portfolio()
        self.calc_min_vol_portfolio()
        self.update_min_max()
                
    def calc_risk_free_rate(self):
        """
        Calculate the annualised risk free rate from the downloaded bonds data. Update the attribute self.risk_free_rate.

        Returns
        -------
        None.

        """
        if len(self.bonds_data) == 0:
            self.risk_free_rate = 0
            
        R_F_annual = self.bonds_data/100
        self.risk_free_rate = get_mean_returns(R_F_annual)
            
    def individual_stock_portfolios(self):
        """
        Calculate the annualised excess returns and volatility of individual stocks in the stock portfolio.
        
        Returns
        -------
        vols : list(float)
            List of annualised volatility of individual stocks.
        excess_returns : list(float)
            List of annualised excess returns of the individual stocks.
            
        """
        
        vols = []
        excess_returns = []
        for idx, stock in enumerate(self.stocks):
            excess_returns.append( TRADING_DAYS * (self.mean_returns.iloc[idx]) - self.risk_free_rate )
            vols.append( np.sqrt( TRADING_DAYS * self.cov_matrix[stock][stock]) )
        return vols, excess_returns
    
    def calc_max_SR_portfolio(self):
        """
        Calculate the max Sharpe Ratio portfolio, and update the relevant attribute.

        Returns
        -------
        None.

        """
        
        max_SR_portfolio = Portfolio(self, name = "Max Sharpe Ratio")
        max_SR_portfolio = maximise_SR(max_SR_portfolio)
        self.max_SR_portfolio = max_SR_portfolio
        
    def calc_min_vol_portfolio(self):
        """
        Calculate the minimum volatility portfolio, and update the relevant attribute.

        Returns
        -------
        None.

        """
        
        min_vol_portfolio = Portfolio(self, name = "Min Vol")
        min_vol_portfolio = minimise_vol(min_vol_portfolio)
        self.min_vol_portfolio = min_vol_portfolio
    
    def calc_efficient_frontier(self, constraint_set = (0, 1)):
        """
        Calculate the portfolios along the efficient frontier for various values of target excess return.

        Parameters
        ----------
        constraint_set : list(float, float), optional
            Allowed min and max of weights. The default is (0, 1).

        Returns
        -------
        efficient_frontier_vols : list(float)
            List of volatilities of efficient portfolios (i.e. minimising vol) for given target excess returns.
        target_excess_returns : list(float)
            List of target excess returns that we want the portfolios to reach.

        """
        
        LOWER = self.min_returns
        UPPER = self.max_returns
        
        target_excess_returns = np.linspace(LOWER, UPPER, 500)
        efficient_frontier_vols = []
        for target in target_excess_returns:
            # for each efficient portfolio, obtain the portfolio volatility
            eff_portfolio = Portfolio(self)
            try:
                eff_portfolio = efficient_portfolio(eff_portfolio, target,)
                efficient_frontier_vols.append(eff_portfolio.vol)
                if self.max_SR_portfolio and self.max_SR_portfolio.sharpe_ratio < eff_portfolio.sharpe_ratio:
                    self.max_SR_portfolio = eff_portfolio
                    self.max_SR_portfolio.name = 'Max Sharpe Ratio'
            except:
                efficient_frontier_vols.append(None)
        
        return efficient_frontier_vols, target_excess_returns
            
    def run_factor_analysis(self, factor_returns):
        self.factor_analysis = FactorAnalysis(self, factor_returns)
        
        return None
    
    def get_factor_betas(self, factor = None):
        if self.factor_analysis is None:
            raise ValueError("Factor analysis has not been set for this universe.")
        
        factor_exposures = self.factor_analysis.get_factor_exposures()
        
        if factor is None:
            # Return all factor betas
            return factor_exposures
        elif factor in factor_exposures.columns:
            # Return betas of a specific factor
            return factor_exposures[factor]
        else:
            raise ValueError(f"Factor '{factor}' not found in the factor analysis.")
            
        return None
    
    def calc_factor_ranges(self):
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
    universe : StockUniverse
        Stock universe containing the stocks of the portfolio.
    name: str
        Name of portfolio.
    returns: float
        Annualised returns of the portfolio.
    vol: float
        Annualised volatility of the portfolio.
    excess_returns: float
        Annualised excess returns of the portfolio.
    weights: np.array
        Weights of the portfolio's stocks.
    sharpe_ratio: float
        Sharpe Ratio of the portfolio
        
    Methods
    -------
    normalise_weights():
        Normalises weights so they add up to 1. Updates attribute directly.
        
    calc_performance():
        Computes the portfolio's annualised returns, excess returns, volatility and sharpe ratio from the StockUniverse 
        mean returns and covariance matrix. If the relevant stock data, does not exist, all these quantities are
        saved as 0. Saves attributes directly.
        
    get_performance_df():
        Creates a pd.DataFrame of the excess returns, volatility and sharpe ratio.
        
    get_weights_df():
        Creates a pd.DataFrame of the weights of the portfolio.
        
    """
    
    def __init__(self, universe, name = "", weights = []):
        """
        Constructs the attributes of the portfolio object. If no weights are passed, the portfolio will be
        instantiated with uniform weights.

        Parameters
        ----------
        universe : StockUniverse
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
        
        if len(weights) == 0:
            self.weights = np.array([1] * len(self.universe.stocks))
        else:
            self.weights = np.array(weights)
            
        self.normalise_weights()
        self.calc_performance()
        
    def normalise_weights(self):
        """
        Normalises weights so they add up to 1. Updates attribute directly.

        Returns
        -------
        None.

        """
        
        self.weights = self.weights / self.weights.sum()
    
    def calc_performance(self):
        """
        Compute portfolio performance: annualised return, excess return, volatility and Sharpe Ratio.

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
        
        if self.universe.mean_returns is not None and self.universe.cov_matrix is not None:
            self.returns = np.dot(self.weights, self.universe.mean_returns) * TRADING_DAYS
            self.vol = np.sqrt( np.dot(np.dot(self.weights.T, self.universe.cov_matrix), self.weights) * TRADING_DAYS )
            self.excess_returns = self.returns - self.universe.risk_free_rate
            self.sharpe_ratio = self.excess_returns / self.vol
        
        else:
            self.returns = self.vol = self.excess_returns = self.sharpe_ratio = 0
        
        return self.returns, self.vol, self.excess_returns, self.sharpe_ratio
    
    def get_performance_df(self):
        """
        Creates a pd.DataFrame of the excess returns, volatility and sharpe ratio.

        Returns
        -------
        results_df : pd.DataFrame
            Dataframe containing the excess returns, volatility and sharpe ratio.
        format_map : dict
            Dictionary that maps the outputs into a nice format for display.

        """
        
        if not self.returns and self.vol and self.excess_returns and self.sharpe_ratio:
            self.performance()
        
        results = {'Excess Returns': self.excess_returns,
                   'Volatility': self.vol,
                   'Sharpe Ratio': self.sharpe_ratio,}
        
        results_df = pd.DataFrame(results, index=[self.name + ' Porfolio'])
        format_map = {'Excess Returns': '{:,.1%}'.format, 'Volatility': '{:,.1%}'.format, 'Sharpe Ratio': '{:,.2f}'}
        
        return results_df, format_map
    
    def get_weights_df(self):
        """
        Creates a pd.DataFrame of the weights of the portfolio.

        Returns
        -------
        weights_df : pd.DataFrame
            Dataframe containing the weights of the portfolio's stocks.
        format_map : dict
            Dictionary that maps the outputs into a nice format for display..

        """
        weights = {self.universe.stocks[idx]: self.weights[idx] for idx in range(len(self.universe.stocks))}
        
        weights_df = pd.DataFrame(weights, index = [self.name + ' Portfolio'])
        
        format_map = {col: "{:,.0%}".format for col in weights_df.columns}
        
        return weights_df, format_map