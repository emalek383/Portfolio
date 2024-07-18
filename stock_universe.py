# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:55:50 2024

@author: emanu
"""

import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from helper_functions import get_mean_returns
from optimisers import maximise_SR, minimise_vol, efficient_portfolio, maximise_returns
from data_loader import get_data

TRADING_DAYS = 252

class stock_universe():
    def __init__(self, 
                 stocks, 
                 start_date = dt.datetime.today() + relativedelta(years = -1), 
                 end_date = dt.datetime.today(), 
                 mean_returns = [], 
                 cov_matrix = [], 
                 risk_free_rate = None):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        if self.mean_returns and self.cov_matrix:
            self.update_min_max()
            self.max_SR_portfolio = self.calc_max_SR_portfolio()
            self.min_vol_portfolio = self.calc_min_vol_portfolio()
        else:
            self.min_returns = self.max_returns = self.min_vol = self.max_vol = None
            self.max_SR = self.min_vol = 0
        
    def update_min_max(self):
        if not self.risk_free_rate:
            self.calc_risk_free_rate()
            
        self.min_returns = TRADING_DAYS * min(self.mean_returns) - self.risk_free_rate
        self.max_returns = TRADING_DAYS * max(self.mean_returns) - self.risk_free_rate
        
        self.min_vol = self.min_vol_portfolio.vol
        self.max_vol = max(self.individual_stock_portfolios()[0])
        return self.min_returns, self.max_returns, self.min_vol, self.max_vol
        
    def optimise_portfolio(self, optimiser, target):
        optimised_portfolio = portfolio(self)
        if optimiser == 'min_vol':
            optimised_portfolio = efficient_portfolio(optimised_portfolio, target)
        else:
            optimised_portfolio = maximise_returns(optimised_portfolio, target)
            
        return optimised_portfolio
    
    def get_data(self):
        stock_data = get_data(self.stocks, self.start_date, self.end_date)
        ignored = []
        
        for ticker in self.stocks:
            if stock_data[ticker].isnull().all():
                ignored.append(ticker)
                self.stocks.remove(ticker)

        stock_data = stock_data[self.stocks]
                
        self.stock_data = stock_data
        
        self.bonds_data = get_data(['^IRX'], self.start_date, self.end_date)
        
        return ignored
        
    def calc_mean_returns_cov(self):
        '''Compute mean returns and covariance matrix of stock portfolio'''
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
        
        return mean_returns, cov_matrix
        
    def calc_risk_free_rate(self):
        R_F_annual = self.bonds_data/100
        self.risk_free_rate = get_mean_returns(R_F_annual)
        
        return self.risk_free_rate
    
    def individual_stock_portfolios(self):
        vols = []
        excess_returns = []
        for idx, stock in enumerate(self.stocks):
            excess_returns.append( TRADING_DAYS * (self.mean_returns[idx]) - self.risk_free_rate )
            vols.append( np.sqrt( TRADING_DAYS * self.cov_matrix[stock][stock]) )
        return vols, excess_returns
    
    def calc_max_SR_portfolio(self):
        max_SR_portfolio = portfolio(self, name = "Max Sharpe Ratio")
        max_SR_portfolio = maximise_SR(max_SR_portfolio)
        self.max_SR_portfolio = max_SR_portfolio
        
    def calc_min_vol_portfolio(self):
        min_vol_portfolio = portfolio(self, name = "Min Vol")
        min_vol_portfolio = minimise_vol(min_vol_portfolio)
        self.min_vol_portfolio = min_vol_portfolio
    
    def calc_efficient_frontier(self, constraint_set = (0, 1)):
        '''Calculate the Max SR, Min Vol and Efficient Frontier portfolios'''
        LOWER = self.min_returns
        UPPER = self.max_returns
        
        target_excess_returns = np.linspace(LOWER, UPPER, 500)
        efficient_frontier_vols = []
        for target in target_excess_returns:
            # for each efficient portfolio, obtain the portfolio volatility
            eff_portfolio = portfolio(self)
            eff_portfolio = efficient_portfolio(eff_portfolio, target)
            efficient_frontier_vols.append(eff_portfolio.vol)
            #efficient_frontier_vols.append(efficient_portfolio(trial_portfolio, target)['fun'])
        
        return efficient_frontier_vols, target_excess_returns
        
class portfolio():
    def __init__(self, universe, name = "", weights = []):
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
        self.weights = self.weights / self.weights.sum()
    
    def calc_performance(self):
        '''Compute portfolio performance: return and volatility'''
        if self.universe.mean_returns is not None and self.universe.cov_matrix is not None:
            self.returns = np.dot(self.weights, self.universe.mean_returns) * TRADING_DAYS
            self.vol = np.sqrt( np.dot(np.dot(self.weights.T, self.universe.cov_matrix), self.weights) * TRADING_DAYS )
            self.excess_returns = self.returns - self.universe.risk_free_rate
            self.sharpe_ratio = self.excess_returns / self.vol
        
        else:
            self.returns = self.vol = self.excess_returns = self.sharpe_ratio = 0
        
        return self.returns, self.vol, self.excess_returns, self.sharpe_ratio
    
    def get_performance_df(self):
        '''Print Sharpe Ratio, Excess Returns, Volatility and Allocation'''
        if not self.returns and self.vol and self.excess_returns and self.sharpe_ratio:
            self.performance()
        
        results = {'Excess Returns': self.excess_returns,
                   'Volatility': self.vol,
                   'Sharpe Ratio': self.sharpe_ratio,}
        
        results_df = pd.DataFrame(results, index=[self.name + ' Porfolio'])
        format_map = {'Excess Returns': '{:,.1%}'.format, 'Volatility': '{:,.1%}'.format, 'Sharpe Ratio': '{:,.2f}'}
        
        return results_df, format_map
    
    def get_weights_df(self):
        weights = {self.universe.stocks[idx]: self.weights[idx] for idx in range(len(self.universe.stocks))}
        
        weights_df = pd.DataFrame(weights, index = [self.name + ' Portfolio'])
        
        format_map = {col: "{:,.0%}".format for col in weights_df.columns}
        
        return weights_df, format_map
    
    def get_portfolio_results(self):
        '''Print Sharpe Ratio, Excess Returns, Volatility and Allocation'''
        if not self.returns and self.vol and self.excess_returns and self.sharpe_ratio:
            self.performance()
        
        results = {'Allocation ' + self.universe.stocks[idx]: self.weights[idx] for idx in range(len(self.universe.stocks))}
    
        results.update({'Sharpe Ratio': self.sharpe_ratio,
                        'Excess Returns': self.excess_returns,
                        'Volatility': self.vol})
    
        results_df = pd.DataFrame.from_dict(results, orient = 'index', columns = [self.name + ' Portfolio'])
        
        return results_df