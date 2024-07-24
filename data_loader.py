import yfinance as yf
from pathlib import Path
import appdirs as ad
import pandas as pd

CACHE_DIR = ".cache"

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
    # Force appdirs to say that the cache dir is .cache
    ad.user_cache_dir = lambda *args: CACHE_DIR

    # Create the cache dir if it doesn't exist
    Path(CACHE_DIR).mkdir(exist_ok=True)

    all_data = {}
    
    for stock in stocks:
        ticker = yf.Ticker(stock)
        data = ticker.history(start=start, end=end)
        if not data.empty:
            all_data[stock] = data['Close']
    
    stockData = pd.DataFrame(all_data)    


    return stockData