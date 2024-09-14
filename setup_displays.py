"""Setup the streamlit displays."""

import pandas as pd
import streamlit as st

from portfolio_state_manager import iterate_portfolios, get_efficient_frontier
from process_forms import process_stock_form
from plotting_functions import plot_interactive_efficient_frontier, plot_weights_pie_charts, plot_weights_bar_chart, plot_return_comparison, plot_exposure_comparison
from helper_functions import format_factor_choice, format_covariance_choice, calculate_factor_contributions

state = st.session_state

def is_mobile():
    """
    Check whether device being used is a mobile device.

    Returns
    -------
    bool
        True: device is mobile device, False: device is PC.

    """
    
    return not state.is_session_pc

def format_performance_df_for_mobile(df, format_map):
    """
    Format the portfolio performance dataframes for better appearance on mobile devices.
    In particular, remove 'Portfolio' from names, shorten 'Constrained' to 'Constr.'
    and shorten the column headers.

    Parameters
    ----------
    df : pd.DataFrame
        Portfolio performance dataframe to edit.
    format_map : dict
        Dictionary containing map for pandas Dataframe formatting, set for general devices.

    Returns
    -------
    mobile_df : pd.DataFrame
        Portfolio performance dataframe better for appearance on mobile devices.
    mobile_format_map : dict
        Dictionary containing map for pandas Dataframe formatting, now set for mobile devices.

    """
    
    mobile_df = df.copy()
    
    column_map = {
        'Excess Returns': 'Ex. Returns',
        'Volatility': 'Vol',
        'Sharpe Ratio': 'Sharpe',
        }
    
    for column in df.columns:
        if 'CVaR' in column:
            column_map[column] = 'CVaR'
        elif 'VaR' in column:
            column_map[column] = 'VaR'
    
    mobile_df.rename(columns = column_map, inplace = True)
    
    mobile_df.index = mobile_df.index.str.replace(' Portfolio', '')
    mobile_df.index = mobile_df.index.str.replace('Constrained', 'Constr.')
    
    mobile_format_map = {column_map.get(k, k): v for k, v in format_map.items()}
    
    return mobile_df, mobile_format_map

def format_weights_df_for_mobile(df, format_map):
    """
    Format the portfolio allocation dataframes for better appearance on mobile devices.
    In particular, remove 'Portfolio' from names and shorten 'Constrained' to 'Constr.'.
    Parameters
    ----------
    df : pd.DataFrame
        Portfolio allocation dataframe to edit.
    format_map : dict
        Dictionary containing map for pandas Dataframe formatting, set for general devices.

    Returns
    -------
    mobile_df : pd.DataFrame
        Portfolio allocation dataframe better for appearance on mobile devices.
    mobile_format_map : dict
        Dictionary containing map for pandas Dataframe formatting, now set for mobile devices.

    """
    
    mobile_df = df.copy()
    
    mobile_df.index = mobile_df.index.str.replace(' Portfolio', '')
    mobile_df.index = mobile_df.index.str.replace('Constrained', 'Constr.')
    
    mobile_format_map = format_map
    
    return mobile_df, mobile_format_map

def style_table(df, format_map = None, is_mobile = False):
    """
    Style dataframe with portfolio information in a nice way using the format_map and make it mobile-friendly.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to format for nice output.
    format_map : dict, optional
        Dictionary containing map for pandas Dataframe formatting. The default is None.
    is_mobile : bool, optional
        True if on a mobile device, False if on PC. The default is False.

    Returns
    -------
    html_table : str
        Dataframe rendered as a HTML table.

    """
    
    styled = df.style
    if format_map:
        styled = styled.format(format_map)
        
    styled = styled.format_index(None)

    html_table = styled.to_html(index = False)

    if is_mobile:
        html_table = html_table.replace('<th class="blank level0" >&nbsp;', '<th>Portfolio')
    
    html_table = f'<div class="styled-table">{html_table}</div>'
    
    return html_table

def setup_overview_display(display):
    """
    Setup an overview displaying the stock basket and the covariance estimation method, if more than 1.

    Parameters
    ----------
    display : st.container
        Streamlit container where output will be written.

    Returns
    -------
    display : st.container
        Streamlit container with output.

    """
    
    display.header("Overview")
    if not state.universe or len(state.universe.stocks) < 2:
        display.write("You need at least two stocks to build a portfolio.")
        process_stock_form()
        
        return display
    
    col1, col2  = display.columns(2)
    with col1:
        st.metric("Number of stocks", len(state.universe.stocks))
        st.metric("Risk-free Rate", f"{state.universe.risk_free_rate:.2%}")
        if state.factor_model:
            st.metric("Factor model", format_factor_choice(state.factor_model))
    
    with col2:
        st.metric("Analysis Start Date", f"{state.universe.start_date.strftime('%d/%m/%Y')}")
        st.metric("Analysis End Date",f"{state.universe.end_date.strftime('%d/%m/%Y')}")
        if state.factor_model:
            st.metric("Covariance Estimation Method", format_covariance_choice(state.cov_type))

def setup_interactive_efficient_frontier_display(display):
    """
    Setup the display for the interactive efficient frontier. Ensure that there is an efficient frontier, otherwise display an error.

    Parameters
    ----------
    display : st.container
        Where the output will be displayed.

    Returns
    -------
    display : st.container
        Display with output.

    """
    
    eff_frontier_data = get_efficient_frontier(state.cov_type, constrained=False)
    if not eff_frontier_data or len(eff_frontier_data) != 2:
        display.error("Failed to load efficient frontier. Need at least two stocks to have a viable portfolio.")
        return display
    
    factor_bounds = state.factor_bounds
    constrained_eff_frontier_data = get_efficient_frontier(state.cov_type, constrained=True)
    
    if factor_bounds:
        sorted_portfolios = iterate_portfolios(state.cov_type, include_custom=True, include_constrained=True)
    else:
        sorted_portfolios = iterate_portfolios(state.cov_type, include_custom=True, include_constrained=False)
        
    vols, excess_returns = state.universe.individual_stock_portfolios(cov_type=state.cov_type)
    stock_data = (state.universe.stocks, vols, excess_returns)
    
    fig = plot_interactive_efficient_frontier(eff_frontier_data, constrained_eff_frontier_data = constrained_eff_frontier_data, sorted_portfolios = sorted_portfolios, stock_data = stock_data, is_mobile = is_mobile())
    
    st.plotly_chart(fig, use_container_width = True, config = {'responsive': True, 'displayModeBar': False,})
    
    
    return display

def setup_portfolio_performances_display(output, portfolios):
    """
    Setup display for the performances of the portfolios that have been passed.

    Parameters
    ----------
    output : st.container
        Container in which the information will be displayed.
    portfolios : list(tuple(str, StockUniverse.Portfolio))
        List of tuples (short name, portfolio) whose performances are to be displayed.

    Returns
    -------
    None.

    """
        
    output.markdown("#### Performance ####")
    joined_perf = []
    
    if not portfolios:
        return
    
    col1, col2, col3, col4 = output.columns([0.25, 0.25, 0.15, 0.3])
    
    var_option = col1.radio(
        'Choose VaR Estimation Method:',
        ('Historical', 'Monte Carlo'),
        horizontal = True,
        key = "var_option_radio"
        )
    
    var_conf_level = col3.number_input(
        'VaR/CVaR confidence level:',
        min_value = 0, 
        max_value = 100,
        value = 95)

    for _, portfolio in portfolios:
        if var_option == 'Historical':
            freqs = portfolio.hist_var_cvar.keys()
        else:
            freqs = portfolio.mc_var_cvar.keys()
        
        portfolio.calc_hist_var_cvar(confidence_level = var_conf_level / 100)
        portfolio.calc_mc_var_cvar(confidence_level = var_conf_level / 100)
        perf_df, format_map = portfolio.get_performance_df()
        joined_perf.append(perf_df)
            
    horizon_map = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}
    
    def format_horizon(key):
        return horizon_map[key]
    
    var_period = col2.radio(
        'Choose VaR/CVaR period:',
        freqs,
        horizontal = True,
        format_func = format_horizon)
    
    joined_perf_df = pd.concat(joined_perf, axis = 0)
    
    cols = ['Excess Returns', 'Volatility', 'Sharpe Ratio']
        
    old_var_col = f"{var_option} VaR ({horizon_map[var_period]})"
    old_cvar_col = f"{var_option} CVaR ({horizon_map[var_period]})"
    new_var_col = f"{var_conf_level}% {var_option} VaR ({horizon_map[var_period]})"
    new_cvar_col = f"{var_conf_level}% {var_option} CVaR ({horizon_map[var_period]})"
    
    if old_var_col in joined_perf_df.columns:
        joined_perf_df[new_var_col] = joined_perf_df[old_var_col]
        joined_perf_df.drop(old_var_col, axis=1, inplace=True)
        format_map[new_var_col] = format_map.pop(old_var_col)
    
    if old_cvar_col in joined_perf_df.columns:
        joined_perf_df[new_cvar_col] = joined_perf_df[old_cvar_col]
        joined_perf_df.drop(old_cvar_col, axis=1, inplace=True)
        format_map[new_cvar_col] = format_map.pop(old_cvar_col)
    
    cols.append(new_var_col)
    cols.append(new_cvar_col)

    joined_perf_df = joined_perf_df[cols]
    
    if is_mobile():
        joined_perf_df, format_map = format_performance_df_for_mobile(joined_perf_df, format_map)
    
    html_table = style_table(joined_perf_df, format_map, is_mobile())
    wrapped_table = f'<div class="table-wrapper-75">{html_table}</div>'
    output.markdown(wrapped_table, unsafe_allow_html = True)
    
def setup_portfolio_weights_display(output, portfolios, label_threshold = 0.05, group_threshold = 0.05):
    """
    Show the weights of the portfolios passed. User can choose between seing the weights in a
    pie chart, table or bar chart.

    Parameters
    ----------
    output : st.container
        Container in which the information will be placed.
    portfolios : list(tuple(str, StockUniverse.Portfolio))
        List of tuples (short name, portfolio) whose allocations will be displayed.

    Returns
    -------
    None.

    """
    
    output.markdown("#### Allocation ####")
    
    viz_option = output.radio(
        'Choose visualisation type:',
        ('Pie Chart', 'Table', 'Bar Chart'),
        horizontal = True,
        key = "viz_type_radio"
        )
    
    if viz_option == 'Pie Chart':
        fig = plot_weights_pie_charts(portfolios, label_threshold, group_threshold, is_mobile = is_mobile())
        if len(portfolios) == 1:
            col1, col2, col3 = output.columns([0.1, 0.7, 0.2])
            col2.pyplot(fig, use_container_width = True)
        else:
            output.pyplot(fig, use_container_width = True)
        
    elif viz_option == 'Table':
        display_weights_table(output, portfolios)
        
    else:
        display_weights_bar_chart(output, portfolios, group_threshold)
        
def format_factor_bounds(factor_bounds):
    """
    Format the constraints on the factor exposures for nicer viewing in streamlit.

    Parameters
    ----------
    factor_bounds : dict
        Dictionary containing the constraints on factor exposures as 
        key: value = factor: [lower, upper] with lower/upper = None indicating no constraint.
        If no constraint at all on factor, it won't appear in the dictionary.

    Returns
    -------
    formatted_bounds : list(str)
        Better formatted constraints for viewing in streamlit.

    """
    
    formatted_bounds = []
    max_factor_length = max(len(factor) for factor in factor_bounds.keys()) if factor_bounds else 0
    lower_bounds = [len(f"{bounds[0]:+.2f}") for bounds in factor_bounds.values() if bounds[0] is not None]
    if len(lower_bounds) > 0:
        max_lower_bound_length = max(lower_bounds)
    else:
        max_lower_bound_length = 0
    
    for factor, bounds in factor_bounds.items():
        lower, upper = bounds
        if lower is not None and upper is not None:
            bound_str = f"{lower:+.2f}  ≤ {factor:<{max_factor_length}} ≤ {upper:+.2f}"
        elif lower is not None:
            bound_str = f"{lower:+.2f}\u00A0 ≤ {factor:<{max_factor_length}}"
        elif upper is not None:
            bound_str = f"{' ':{max_lower_bound_length}}  {factor:<{max_factor_length}} ≤ {upper:+.2f}"
        else:
            continue  # Skip factors with no constraints

        formatted_bounds.append(bound_str)
    
    return formatted_bounds

def setup_factor_bounds_display(container, factor_model, factor_bounds):
    """
    Show the constraints on the factor exposures.

    Parameters
    ----------
    container : st.container
        Container where the output will be placed.
    factor_model : str
        Name of the factor model used.
    factor_bounds : dict
        Dictionary containing the constraints on factor exposures as 
        key: value = factor: [lower, upper] with lower/upper = None indicating no constraint.
        If no constraint at all on factor, it won't appear in the dictionary.

    Returns
    -------
    None.

    """
    
    formatted_bounds = format_factor_bounds(factor_bounds)
    if formatted_bounds:
        container.markdown("#### Current Factor Constraints ####")
        container.write(f"{format_factor_choice(factor_model)} Model")
        bounds_text = "\n".join(formatted_bounds)
        container.code(bounds_text)
    else:
        container.info("No active factor constraints.")
        

def display_weights_table(output, portfolios):
    """
    Display the portfolio allocation in a table with better formatting for mobile devices.

    Parameters
    ----------
    output : st.container
        Streamlit container where the output will be placed.
    portfolios : list(tuple(str, StockUniverse.Portfolio))
        List of tuples (short name, portfolio) whose allocations will be displayed.

    Returns
    -------
    None.

    """
    
    joined_weights = []
    for _, portfolio in portfolios:
        weights_df, format_map = portfolio.get_weights_df()
        joined_weights.append(weights_df)
        
    joined_weights_df = pd.concat(joined_weights, axis = 0)
    
    if is_mobile():
        joined_weights_df, format_map = format_weights_df_for_mobile(joined_weights_df, format_map)
        
    html_table = style_table(joined_weights_df, format_map, is_mobile())
    wrapped_table_100 = f"<div class = 'table-wrapper-full'>{html_table}</div>"
    output.markdown(wrapped_table_100, unsafe_allow_html = True)
    
def display_weights_bar_chart(output, portfolios, group_threshold):
    """
    Display the portfolio allocation in a bar chart, with a selectbox for the user to choose which portfolio to see,
    or whether to display all. Show only the assets individually above the group_threshold.

    Parameters
    ----------
    output : st.container
        Streamlit container where the output will be displayed.
    portfolios : list(tuple(str, StockUniverse.Portfolio))
        List of tuples (short name, portfolio) whose allocations will be displayed.
    group_threshold : float
        Cutoff for asset allocation %. Underneath this cutoff, assets will only be displayed group in 'Others'.

    Returns
    -------
    None.

    """
    
    if len(portfolios) > 1:
    
        portfolio_options = [portfolio[1].name for portfolio in portfolios] + ['Show all']
        selected_portfolio = output.selectbox("Select portfolio to display:", portfolio_options)
        
    else:
        selected_portfolio = portfolios[0][1].name
        
    fig = plot_weights_bar_chart(selected_portfolio, portfolios, group_threshold)
    output.pyplot(fig, use_container_width = True)

def setup_return_comparison_display(output, portfolios, factor_bounds = None):
    """
    Display the returns attributable to different factor for various portfolios.

    Parameters
    ----------
    output : st.container
        Streamlit container where the output will be displayed.
    portfolios : list(tuple(str, StockUniverse.Portfolio))
        List of tuples (short name, portfolio) to be analysed.
    factor_bounds : dict, optional
        Dictionary containing the constraints on factor exposures as 
        key: value = factor: [lower, upper] with lower/upper = None indicating no constraint.
        If no constraint at all on factor, it won't appear in the dictionary.
        The default is None.

    Returns
    -------
    None.

    """
    
    name_map = {name: portfolio.name for name, portfolio in portfolios}
    
    exposures = {name: p.universe.factor_analysis.analyse_portfolio(p)['Factor Exposures'].iloc[0] for name, p in portfolios}
    contributions = {name: calculate_factor_contributions(portfolio, exposures[name]) for name, portfolio in portfolios}
    
    exposures_data = pd.DataFrame(exposures).T
    fig = plot_return_comparison(exposures_data, contributions, name_map, factor_bounds, is_mobile = is_mobile())
    
    col1, col2, col3 = output.columns([0.05, 0.8, 0.15])
    col2.pyplot(fig, use_container_width = True)

def setup_exposure_comparison_display(output, portfolios, factor_bounds = None):
    """
    Display the exposures of different portfolios to the various factors.
    Include the factor bounds on the display.

    Parameters
    ----------
    output : st.container
        Streamlit container where the output will be displayed.
    portfolios : list(tuple(str, StockUniverse.Portfolio))
        List of tuples (short name, portfolio) to be analysed.
    factor_bounds : dict, optional
        Dictionary containing the constraints on factor exposures as 
        key: value = factor: [lower, upper] with lower/upper = None indicating no constraint.
        If no constraint at all on factor, it won't appear in the dictionary.
        The default is None.

    Returns
    -------
    None.

    """
    
    name_map = {name: portfolio.name for name, portfolio in portfolios}
    
    exposures = {name: p.universe.factor_analysis.analyse_portfolio(p)['Factor Exposures'].iloc[0] for name, p in portfolios}
    exposures_data = pd.DataFrame(exposures).T
    
    fig = plot_exposure_comparison(exposures_data, name_map, factor_bounds)
    
    col1, col2, col3 = output.columns([0.05, 0.8, 0.15])
    col2.pyplot(fig, use_container_width = True)
    