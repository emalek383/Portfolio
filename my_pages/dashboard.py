import streamlit as st

from portfolio_state_manager import iterate_portfolios
from setup_forms import setup_stock_selection_form
from setup_displays import setup_portfolio_weights_display, setup_portfolio_performances_display, setup_overview_display, setup_interactive_efficient_frontier_display

state = st.session_state

with st.sidebar:        
    st.header("Select stocks for portfolio")
    stock_selection_form = st.form(border = False, key = "stock_form")

setup_stock_selection_form(stock_selection_form)


overview_display = st.container(border = False)
performance_display = st.container(border = False)
weights_display = st.container(border = False)
efficient_frontier_display = st.container(border = False)

# Show overview of stock universe and factor analysis, if it has been run
setup_overview_display(overview_display)

# Show portfolio performance and weights and efficient frontier
if state.universe and len(state.universe.stocks) >= 2:
    sorted_portfolios = iterate_portfolios(state.cov_type, include_constrained = True)
    setup_portfolio_performances_display(performance_display, sorted_portfolios)
    setup_portfolio_weights_display(weights_display, sorted_portfolios)
    efficient_frontier_display.subheader("Efficient Frontier")
    setup_interactive_efficient_frontier_display(efficient_frontier_display)