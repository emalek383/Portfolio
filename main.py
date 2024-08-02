import streamlit as st
from setup_forms import setup_stock_selection_form, setup_weights_form, setup_optimise_portfolio_form, setup_factor_analysis_form
from setup_displays import setup_portfolio_display, setup_details_display
from process_forms import process_stock_form
from data_loader import check_latest

import requests
import json
import zipfile
import os
from datetime import datetime
import logging

st.set_page_config(layout="wide")

def load_css(file_name):
    with open(file_name) as f:
        css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

#@st.cache_data
def update_factor_models():
    check_latest()
    
# update_factor_models()

logging.basicConfig(level=logging.INFO)

@st.cache_data
def check_and_download_zip(url, zip_filename, csv_filename):
    try:
        logging.info(f"Processing {url}")
        
        # Extract the base name of the zip file (without extension)
        base_name = os.path.splitext(os.path.basename(zip_filename))[0]
        
        # File to store the last modified time
        info_filename = f'logs/{base_name}_info.json'
        
        # Ensure the logs directory exists
        os.makedirs('logs', exist_ok=True)
        
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
                logging.info(f"Downloading new file for {base_name}...")
                response = requests.get(url)
                with open(zip_filename, 'wb') as file:
                    file.write(response.content)
                logging.info(f"New file downloaded: {zip_filename}")
                
                # Unzip the file
                with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(csv_filename))
                logging.info(f"File extracted to: {csv_filename}")
                
                # Save the remote modification time
                with open(info_filename, 'w') as info_file:
                    json.dump({
                        'last_modified': remote_modified,
                        'last_downloaded': datetime.now().isoformat()
                    }, info_file)
            else:
                logging.info(f"Local file for {base_name} is up to date.")
            
            return True
        else:
            logging.error(f"Failed to check remote file for {base_name}. Status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error while processing {url}: {str(e)}")
        st.error(f"Network error: {str(e)}")
        return False
    except zipfile.BadZipFile:
        logging.error(f"Error extracting zip file: {zip_filename}")
        st.error(f"The downloaded file is not a valid zip file: {zip_filename}")
        return False
    except json.JSONDecodeError:
        logging.error(f"Error reading JSON file: {info_filename}")
        st.error(f"Error reading file information: {info_filename}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error processing {url}: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
        return False
    
url = st.text_input("Enter the URL of the zip file")
zip_filename = st.text_input("Enter the name for the local zip file")
csv_filename = st.text_input("Enter the name for the extracted CSV file")
    
if st.button("Check and Download"):
    if url and zip_filename and csv_filename:
        with st.spinner("Processing..."):
            result = check_and_download_zip(url, zip_filename, csv_filename)
        if result:
            st.success("File processed successfully")
        else:
            st.error("Failed to process file")
    else:
        st.warning("Please fill in all fields")

load_css('styles/style.css')    

state = st.session_state    
if 'loaded_stocks' not in state:
     state.loaded_stocks = False
    
if 'universe' not in state:
     state.universe = None
     
if 'portfolios' not in state:
     state.portfolios = {}
     
if 'eff_frontier' not in state:
    state.eff_frontier = None
    
if 'factor_model' not in state:
    state.factor_model = None
    
if 'factor_bounds' not in state:
    state.factor_bounds = {}
    
if 'constrained_eff_frontier' not in state:
    state.constrained_eff_frontier = None

with st.sidebar:        
    #st.header("Select stocks for your portfolio")
    stock_select_expander = st.expander(label = "Select stocks for your portfolio", expanded = True)
    stock_selection_form = stock_select_expander.form(border = False, key = "stock_form")
    
    #st.header("Run factor analysis")
    factor_analysis_expander = st.expander(label = "Run factor analysis", expanded = True)
    factor_analysis_form = factor_analysis_expander.container(border = False)
    
    #st.header("Optimise your portolfio")
    optimise_expander = st.expander(label = "Optimise your portfolio", expanded = True)
    optimise_portfolio_form = optimise_expander.container(border = False)
    
    #st.header("Manually adjust your portfolio")
    weights_expander = st.expander(label = "Manually adjust your portfolio", expanded = False)
    weights_form = weights_expander.container(border = False)

portfolio_display = st.container(border = False)
details_display = st.container(border = False)

#efficient_frontier_display = st.container(border = False)


if not state.loaded_stocks:
    process_stock_form()
    state.loaded_stocks = True
    if state.universe and len(state.universe.stocks) > 1:
        state.eff_frontier = state.universe.calc_efficient_frontier()
    
setup_stock_selection_form(stock_selection_form)
setup_weights_form(weights_form)
setup_optimise_portfolio_form(optimise_portfolio_form)
setup_factor_analysis_form(factor_analysis_form)

setup_portfolio_display(portfolio_display)
setup_details_display(details_display)
#setup_efficient_frontier_display(efficient_frontier_display)
