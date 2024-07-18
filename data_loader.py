# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 09:30:31 2024

@author: emanu
"""
#import appdirs as ad
#ad.user_cache_dir = lambda *args: "/tmp"
import yfinance as yf
from contextlib import contextmanager
import sys
import io

yf.pdr_override()

# Disable progress bar
yf.set_tz_cache_location(None)

@contextmanager
def suppress_stdout():
    new_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield
    finally:
        sys.stdout = old_stdout

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
    try:
        # Disable progress bar
        yf.pdr_override()
        yf.set_tz_cache_location(None)
        
        with suppress_stdout():
            stockData = yf.download(stocks, start=start, end=end, progress=False)
        
        if stockData.empty:
            raise ValueError("No data downloaded. Please check your inputs and internet connection.")
    
        #stockData = yf.download(stocks, start, end)
        stockData = stockData['Close']
        return stockData
    except Exception as e:
        print(f"An error occurred while downloading data: {str(e)}")
        # You might want to return None or an empty DataFrame here
        return None