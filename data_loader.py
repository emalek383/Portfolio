
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
    all_data = {}
    
    for stock in stocks:
        ticker = yf.Ticker(stock)
        data = ticker.history(start=start, end=end)
        if not data.empty:
            all_data[stock] = data['Close']
    
    stockData = pd.DataFrame(all_data)
    
    return stockData
