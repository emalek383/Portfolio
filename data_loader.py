import yfinance as yf
from pathlib import Path
import appdirs as ad
import pandas as pd

#CACHE_DIR = ".cache"

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
    #ad.user_cache_dir = lambda *args: CACHE_DIR

    # Create the cache dir if it doesn't exist
    #Path(CACHE_DIR).mkdir(exist_ok=True)

    all_data = {}
    
    for stock in stocks:
        ticker = yf.Ticker(stock)
        try:
            # Download data in smaller chunks to avoid timeout issues
            chunk_start = start
            stock_data = []
            while chunk_start < end:
                chunk_end = min(chunk_start + timedelta(days=365), end)
                chunk = ticker.history(start=chunk_start, end=chunk_end, interval="1d")
                if not chunk.empty:
                    stock_data.append(chunk['Close'])
                chunk_start = chunk_end + timedelta(days=1)
                time.sleep(1)  # Add a small delay to avoid rate limiting
            
            if stock_data:
                all_data[stock] = pd.concat(stock_data)
            else:
                print(f"No data available for {stock}")
        except Exception as e:
            print(f"Error downloading data for {stock}: {e}")
    
    stockData = pd.DataFrame(all_data)
    return stockData
