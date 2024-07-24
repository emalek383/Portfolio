
import yfinance as yf
from pathlib import Path
import appdirs as ad

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

    stockData = yf.download(stocks, start=start, end=end, progress=False)
        
    stockData = stockData['Close']
    return stockData
