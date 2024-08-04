import streamlit as st
from setup_forms import setup_stock_selection_form, setup_weights_form, setup_optimise_portfolio_form, setup_factor_analysis_form
from setup_displays import setup_dashboard, setup_portfolio_display, sort_portfolios, display_portfolio_weights, display_portfolio_performances, setup_overview_display, setup_efficient_frontier_display
from setup_displays import interactive_efficient_frontier_display
from process_forms import process_stock_form
from portfolio_state_manager import iterate_portfolios

state = st.session_state


if not state.loaded_stocks:
    process_stock_form()
    state.loaded_stocks = True
    if state.universe and len(state.universe.stocks) > 1:
        state.eff_frontier = state.universe.calc_efficient_frontier()

with st.sidebar:        
    #stock_select_expander = st.expander(label = "Select stocks for portfolio", expanded = True)
    st.header("Select stocks for portfolio")
    stock_selection_form = st.form(border = False, key = "stock_form")
    
setup_stock_selection_form(stock_selection_form)

overview_display = st.container(border = False)
performance_display = st.container(border = False)
weights_display = st.container(border = False)
efficient_frontier_display = st.container(border = False)
efficient_frontier_display.subheader("Efficient Frontier")

#setup_dashboard(portfolio_display)

sorted_portfolios = iterate_portfolios(state.cov_type, include_constrained = True)
setup_overview_display(overview_display)
display_portfolio_performances(performance_display, sorted_portfolios)
display_portfolio_weights(weights_display, sorted_portfolios)
interactive_efficient_frontier_display(efficient_frontier_display)