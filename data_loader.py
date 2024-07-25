import yfinance as yf
import pandas as pd

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

def load_default_stocks():
    stock_filepath = "data/default_stock_data.csv"
    stock_data = pd.read_csv(stock_filepath, index_col=0, parse_dates=True)
    
    return stock_data

def load_default_bonds():
    bonds_filepath = "data/default_bonds_data.csv"
    bonds_data = pd.read_csv(bonds_filepath, index_col=0, parse_dates=True)['Close']
    
    return bonds_data