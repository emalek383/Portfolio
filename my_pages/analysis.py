import streamlit as st 
from setup_displays import setup_portfolio_display

portfolio_display = st.container(border = False)

setup_portfolio_display(portfolio_display)