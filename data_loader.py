# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 09:30:31 2024

@author: emanu
"""
import yfinance as yf


def download_data(stocks, start, end):
    """
    Download stock data from yahoo finance.

    Parameters
    ----------
    stocks : list(str)
        List of stock tickers.
    start : datetime
        Start date.
    end : datetime
        End date.

    Returns
    -------
    stockData : pd.DataFrame
        Dataframe containing the closing price of the stock data.

    """
    
    stockData = yf.download(stocks, start=start, end=end, progress=False)
        
    stockData = stockData['Close']
    return stockData