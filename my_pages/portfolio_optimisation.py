import streamlit as st

from portfolio_state_manager import iterate_portfolios, get_portfolio
from setup_forms import setup_optimise_portfolio_form, setup_weights_form
from setup_displays import setup_portfolio_weights_display, setup_portfolio_performances_display, setup_factor_bounds_display, setup_interactive_efficient_frontier_display

state = st.session_state

st.subheader("Optimisation Parameters")
st.write("Optimise your portfolio by maximising returns up to an allowed maximum volatility, or by minimising volatility for a desired excess return.")
st.write("If a factor analysis has been run, you can choose to impose your chosen constraints on the factor exposures.")

col1, col2 = st.columns(2)
with col1:
    optimisation_form = st.container(border = True)
    optimisation_form = setup_optimise_portfolio_form(optimisation_form)
    
with col2:
    if state.factor_bounds:
        setup_factor_bounds_display(st, state.factor_model, state.factor_bounds)
            
    else:
        st.info("No factor constraints have been set. You can set constraints in the factor analysis section.")

st.subheader("Custom Portfolio Adjustment")
with st.expander("Adjust Portfolio Weights"):
    st.write("Adjust your portfolio weights manually.")
    weights_adjustment_form = st.container(border = False)
    setup_weights_form(weights_adjustment_form, cols_per_row = 8)

tab1, tab2 = st.tabs(['Efficient Frontier', 'Portfolio Details'])

tab1.subheader("Efficient Frontier")
efficient_frontier_display = tab1.container(border = False)

setup_interactive_efficient_frontier_display(efficient_frontier_display)

tab2.subheader("Portfolio Details")
portfolio_performances_display = tab2.container(border = False)
portfolio_allocation_display = tab2.container(border = False)

sorted_portfolios = iterate_portfolios(state.cov_type)

# Show only the custom portfolio in the portfolio details display.
# Need to pass custom portfolio as a list.
portfolio_as_list = [['custom', get_portfolio('custom')]]

setup_portfolio_performances_display(portfolio_performances_display, sorted_portfolios)
setup_portfolio_weights_display(portfolio_allocation_display, portfolio_as_list)