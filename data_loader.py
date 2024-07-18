# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 09:30:31 2024

@author: emanu
"""
import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"
import yfinance as yf

def get_data(stocks, start, end):
    """
    Download stock data from yahoo finance.
    
    Args:
        stocks: list of stock tickers
        start: start date in datetime format
        end: end date in datetime format
    
    Returns:
        stockData: pd.DataFrame with close of stock data
    """
    
    stockData = yf.download(stocks, start, end)
    stockData = stockData['Close']
    return stockData