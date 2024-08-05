import streamlit as st
from process_forms import process_stock_form
from setup_forms import setup_covariance_form
from portfolio_state_manager import initialise_portfolio_state, update_efficient_frontier
from streamlit_javascript import st_javascript
from user_agents import parse

st.set_page_config(layout="wide")


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

ua_string = st_javascript("""window.navigator.userAgent;""")
user_agent = parse(ua_string)
state.is_session_pc = user_agent.is_pc
print("\n Reloading")
print(f"PC? {state.is_session_pc}")

dashboard = st.Page("my_pages/dashboard.py", title = "Portfolio Analysis")
#analysis = st.Page("my_pages/analysis.py", title = "Portfolio Analysis")
optimisation = st.Page("my_pages/portfolio_optimisation.py", title = "Portfolio Optimisation")
factor_analysis = st.Page("my_pages/factor_analysis.py", title = "Factor Analysis")
#customisation = st.Page("my_pages/customise_portfolio.py", title = "Customise Portfolio")
pg = st.navigation([dashboard, optimisation, factor_analysis])
pg.run()

def load_css(file_name):
    with open(file_name) as f:
        css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

load_css('styles/style.css')

if state.factor_model:
    with st.sidebar:
        covariance_method_form = st.container(border = False)
        covariance_method = setup_covariance_form(covariance_method_form)

if not state.loaded_stocks:
    process_stock_form()
    state.loaded_stocks = True
    if state.universe and len(state.universe.stocks) > 1:
        update_efficient_frontier(state.universe.calc_efficient_frontier(), cov_type = 'sample_cov')
    
# # with st.sidebar:        
# #     stock_select_expander = st.expander(label = "Select stocks for portfolio", expanded = True)
# #     stock_selection_form = stock_select_expander.form(border = False, key = "stock_form")
    
# #     factor_analysis_expander = st.expander(label = "Run factor analysis", expanded = True)
# #     factor_analysis_form = factor_analysis_expander.container(border = False)
    
# #     optimise_expander = st.expander(label = "Optimise portfolio", expanded = True)
# #     optimise_portfolio_form = optimise_expander.container(border = False)
    
# #     weights_expander = st.expander(label = "Manually adjust portfolio", expanded = False)
# #     weights_form = weights_expander.container(border = False)

# # portfolio_display = st.container(border = False)
# # details_display = st.container(border = False)

# # setup_stock_selection_form(stock_selection_form)
# # setup_weights_form(weights_form)
# # setup_optimise_portfolio_form(optimise_portfolio_form)
# # setup_factor_analysis_form(factor_analysis_form)

# # setup_portfolio_display(portfolio_display)
# # setup_details_display(details_display)
