import streamlit as st
from setup_forms import setup_stock_selection_form, setup_weights_form, setup_optimise_portfolio_form, setup_factor_analysis_form
from setup_displays import setup_portfolio_display, setup_details_display
from process_forms import process_stock_form
from data_loader import check_latest

st.set_page_config(layout="wide")

def load_css(file_name):
    with open(file_name) as f:
        css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

@st.cache_data
def update_factor_models():
    check_latest()
    
update_factor_models()

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
