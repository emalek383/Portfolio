"""Functions for loading stock and factor data."""

import yfinance as yf
import pandas as pd
import io
import re
import requests
from datetime import datetime
import os
import zipfile
import json

FACTOR_FILES_LIST = [('data/F-F_Research_Data_Factors_daily.csv', 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip', 'data/F-F_Research_Data_Factors_daily_CSV.zip'),
                     ('data/F-F_Research_Data_5_Factors_2x3_daily.csv', 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip', 'data/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'),
                     ('data/F-F_Momentum_Factor_daily.csv', 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip', 'data/F-F_Momentum_Factor_daily_CSV.zip')]
FACTOR_FILES = {'ff3': FACTOR_FILES_LIST[0], 'ff5': FACTOR_FILES_LIST[1], 'mom': FACTOR_FILES_LIST[2]}

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
    """
    Load default stock data that has been presaved.

    Returns
    -------
    stock_data : pd.DataFrame
        DataFrame with the closing price of the presaved stock data.

    """
    
    stock_filepath = "data/default_stock_data.csv"
    stock_data = pd.read_csv(stock_filepath, index_col=0, parse_dates=True)
    
    return stock_data

def load_default_bonds():
    """
    Load default bonds data that has been presaved.

    Returns
    -------
    bonds_data : pd.DataFrame
        DataFrame with the closing price of the presaved bonds data.

    """
    
    bonds_filepath = "data/default_bonds_data.csv"
    bonds_data = pd.read_csv(bonds_filepath, index_col=0, parse_dates=True)['Close']
    bonds_data = bonds_data.rename('^IRX')
    
    return bonds_data

def is_valid_data_line(line):
    """
    Check if the line starts with a date (8 digits followed by a comma).
    This indicates that it is the first data entry in the factor dataset.

    Parameters
    ----------
    line : str
        line of characters.

    Returns
    -------
    bool
        True if it's a line starting with a date, False otherwise.

    """
    
    # Check if the line starts with a date (8 digits followed by a comma)
    return re.match(r'^\d{8},', line.strip()) is not None

def load_factor_csv(file_path):
    """
    Load factor data from .csv file.

    Parameters
    ----------
    file_path : str
        File path of the csv.

    Raises
    ------
    ValueError
        If no column headers are found in the file, i.e. there seems to be no valid csv data.

    Returns
    -------
    df : pd.DataFrame
        Dataframe of the factor date with columns 'Date' + factor returns for each factor.

    """
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the index of the column headers
    header_index = next((i for i, line in enumerate(lines) if re.match(r'^,[\w-]+', line.strip())), None)
    
    if header_index is None:
        raise ValueError("Could not find column headers in the file.")
    
    # Extract the column names and add 'Date' as the first column
    columns = ['Date'] + [col.strip() for col in lines[header_index].strip().split(',')[1:]]
    
    # Filter out invalid lines and create a string of the data rows
    data_lines = [line for line in lines[header_index+1:] if is_valid_data_line(line)]
    data_string = ''.join(data_lines)
    
    df = pd.read_csv(io.StringIO(data_string), 
                     names=columns,
                     dtype={'Date': str})  # Read 'Date' as string initially
    
    # Convert 'Date' column to datetime after reading
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    
    df = df.set_index('Date')
    
    df = df.replace([-99.99, -999], pd.NA)
    
    return df

def load_factor_df(factor_model):
    """
    Load the factor data required for the respective factor model, e.g. for 'ff4' (Carhart 4-factor model need Fama-French 3-factor model + Momentum).

    Parameters
    ----------
    factor_model : str
        Factor model, current options: 'ff3', 'ff4', 'ff5', 'ff6'.

    Returns
    -------
    factor_df : pd.DataFrame
        Dataframe with the relevant factor returns.

    """
    
    factor_reqs = {'ff3': ['ff3'], 'ff4': ['ff3', 'mom'], 'ff5': ['ff5'], 'ff6': ['ff5', 'mom']} # required files in FACTOR_FILES dict
    
    factor_dfs = []
    for req in factor_reqs[factor_model]:
        factor_data = FACTOR_FILES[req]
        csv_filename = factor_data[0]
        factor_dfs.append(load_factor_csv(csv_filename))
        
    factor_df = pd.concat(factor_dfs, axis = 1).dropna()
        
    return factor_df

def check_and_download_zip(url, zip_filename, csv_filename):
    """
    Check whether the current factor data (in a specific file) is up-to-date and, if not, download the relevant data from online.

    Parameters
    ----------
    url : str
        URL where to download the .zip file with the factor data.
    zip_filename : str
        Filename of the local zip file that we are checking against.
    csv_filename : str
        Filename of the local csv file that we will write to, if we update.

    Returns
    -------
    bool
        Whether we were successfully able to download the zip file.

    """
    
    # Extract the base name of the zip file (without extension)
    base_name = os.path.splitext(os.path.basename(zip_filename))[0]
    
    # File to store the last modified time
    info_filename = f'logs/{base_name}_info.json'
    
    # Send a HEAD request to get the last modified date of the remote file
    response = requests.head(url)
    if response.status_code == 200:
        remote_modified = response.headers.get('Last-Modified')
        
        # Check if we need to download a new file
        download_new = True
        if os.path.exists(info_filename):
            with open(info_filename, 'r') as info_file:
                file_info = json.load(info_file)
                if file_info.get('last_modified') == remote_modified:
                    download_new = False
        
        if download_new:
            response = requests.get(url)
            with open(zip_filename, 'wb') as file:
                file.write(response.content)
            
            # Unzip the file
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(csv_filename))
            
            # Save the remote modification time
            with open(info_filename, 'w') as info_file:
                json.dump({
                    'last_modified': remote_modified,
                    'last_downloaded': datetime.now().isoformat()
                }, info_file)
        else:
            pass
        
        return True
    else:
        print(f"Failed to check remote file for {base_name}. Status code: {response.status_code}")
        return False
    
def update_factors():
    """
    Check whether all the current factor data is up to date. If not, download the new data and extract.

    Returns
    -------
    bool
        Whether we were able to successfully download all the data.

    """
    
    factor_file_data_list = FACTOR_FILES
    errors = []
    for factor_file_data in factor_file_data_list:
        if not check_and_download_zip(factor_file_data[1], factor_file_data[2], factor_file_data[0]):
            errors.append(f"Error with {factor_file_data[0]}")
    if errors:        
        print(errors)
    if errors:
        return False
    return True
