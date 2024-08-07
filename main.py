import streamlit as st
from streamlit_javascript import st_javascript
from user_agents import parse

from portfolio_state_manager import initialise_portfolio_state
from setup_forms import setup_covariance_form
from process_forms import process_stock_form


def detect_device():
    """
    Detect whether the page is being viewed on PC.

    Returns
    -------
    bool
        True if being viewed on PC, false otherwise.

    """
    
    if 'is_session_pc' not in st.session_state:
        st.session_state.is_session_pc = True
    try:
        ua_string = st_javascript("navigator.userAgent")
        if ua_string is not None:
            user_agent = parse(ua_string)
            st.session_state.is_session_pc = user_agent.is_pc
        else:
            st.session_state.is_session_pc = True
    except Exception:
        st.session_state.is_session_pc = True
    return st.session_state.is_session_pc


def load_css(file_name):
    """
    Load CSS file into streamlit.

    Parameters
    ----------
    file_name : str
        CSS file filename.

    Returns
    -------
    None.

    """
    
    with open(file_name) as f:
        css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)



st.set_page_config(layout="wide")

is_pc = detect_device()

state = st.session_state
if 'loaded_stocks' not in state:
      state.loaded_stocks = False
         
if 'universe' not in state:
      state.universe = None

if 'factor_model' not in state:
    state.factor_model = None
    
if 'factor_bounds' not in state:
    state.factor_bounds = {}
    
if 'cov_type' not in state:
    state.cov_type = 'sample_cov'

initialise_portfolio_state()

# Load default stocks, if no stocks are loaded yet
if not state.loaded_stocks:
    process_stock_form()
    state.loaded_stocks = True
    
# For displaying info messages at the top of each page
info_container = st.container(border = False)

dashboard = st.Page("my_pages/dashboard.py", title = "Portfolio Analysis")
optimisation = st.Page("my_pages/portfolio_optimisation.py", title = "Portfolio Optimisation")
factor_analysis = st.Page("my_pages/factor_analysis.py", title = "Factor Analysis")
pg = st.navigation([dashboard, optimisation, factor_analysis])
pg.run()

# Display info messages and clear
if 'info_messages' in st.session_state:
        for message in st.session_state.info_messages:
            info_container.info(message)
        st.session_state.info_messages = []

load_css('styles/style.css')

# Create sidebar menu to choose the covariance matrix estimation
if state.factor_model:
    with st.sidebar:
        covariance_method_form = st.container(border = False)
        covariance_method = setup_covariance_form(covariance_method_form)