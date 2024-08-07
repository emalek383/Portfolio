import streamlit as st

from portfolio_state_manager import iterate_portfolios
from setup_forms import setup_factor_analysis_form, setup_factor_constraints_form
from setup_displays import setup_return_comparison_display, setup_exposure_comparison_display, setup_interactive_efficient_frontier_display
from helper_functions import format_factor_choice

state = st.session_state

col1, col2 = st.columns(2)
factor_analysis_form = col1.form('factor_analysis_form', border = False)

setup_factor_analysis_form(factor_analysis_form)

factor_model = state.factor_model
factor_bounds = state.factor_bounds

if factor_model:
    st.metric("Factor model", f"{format_factor_choice(state.factor_model)}")
    
    st.write("You can now estimate the covariance matrix using the factor model, using the control in the sidepanel.")
    
    st.subheader("Set Constraints on the Factor Exposures")
    col1, col2 = st.columns(2)
    factor_constraints_form = col1.container(border = False)
    setup_factor_constraints_form(factor_constraints_form, factor_model)
    
    sorted_portfolios = iterate_portfolios(state.cov_type)
    
    tab1, tab2 = st.tabs(["Factor Exposure", "Factor Return Attribution"])
    
    tab2.subheader("Factor Return Attribution")
    setup_return_comparison_display(tab2, sorted_portfolios, factor_bounds = factor_bounds)
        
    tab1.subheader("Factor Exposure")
    setup_exposure_comparison_display(tab1, sorted_portfolios, factor_bounds = factor_bounds)
    
    if factor_bounds:
        st.subheader("Constrained vs Unconstrained Efficient Frontier")
        st.write("This plot shows how your factor constraints affect the efficient frontier.")
        
        efficient_frontier_display = st.container(border = False)
        
        setup_interactive_efficient_frontier_display(efficient_frontier_display)
