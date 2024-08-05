import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
import streamlit.components.v1 as components
from process_forms import process_stock_form
from helper_functions import get_mean_returns, format_factor_choice, format_covariance_choice
from portfolio_state_manager import iterate_portfolios, get_portfolio, get_efficient_frontier
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TRADING_DAYS = 252

state = st.session_state

def is_mobile():
    return not state.is_session_pc

def format_performance_df_for_mobile(df, format_map):
    mobile_df = df.copy()
    
    column_map = {
        'Excess Returns': 'Ex. Returns',
        'Volatility': 'Vol',
        'Sharpe Ratio': 'Sharpe'
        }
    
    mobile_df.rename(columns = column_map, inplace = True)
    
    mobile_df.index = mobile_df.index.str.replace(' Portfolio', '')
    mobile_df.index = mobile_df.index.str.replace('Constrained', 'Constr.')
    
    mobile_format_map = {column_map.get(k, k): v for k, v in format_map.items()}
    
    return mobile_df, mobile_format_map

def setup_dashboard(display):
    display.header("Overview")
    if not state.universe or len(state.universe.stocks) < 2:
        display.write("You need at least two stocks to build a portfolio.")
        process_stock_form()
        
        return display
    
    col1, col2  = display.columns(2)
    with col1:
        st.metric("Number of stocks", len(state.universe.stocks))
        st.metric("Risk-free Rate", f"{state.universe.risk_free_rate:.2%}")
    
    with col2:
        st.metric("Analysis Start Date", f"{state.universe.start_date.strftime('%d/%m/%Y')}")
        st.metric("Analysis End Date",f"{state.universe.end_date.strftime('%d/%m/%Y')}")

    sorted_portfolios = iterate_portfolios(state.cov_type, include_custom = True, include_constrained = True)

    display_portfolio_performances(display, sorted_portfolios)
    
    display_portfolio_weights(display, sorted_portfolios, label_threshold = 0.05, group_threshold = 0.05)

def setup_overview_display(display):
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

# def old_setup_dashboard(display):
#     display.header("Overview")
#     if not state.universe or len(state.universe.stocks) < 2:
#         display.write("You need at least two stocks to build a portfolio.")
#         process_stock_form()
        
#         return display
    
#     col1, col2  = display.columns(2)
#     with col1:
#         st.metric("Number of stocks", len(state.universe.stocks))
#         st.metric("Risk-free Rate", f"{state.universe.risk_free_rate:.2%}")
    
#     with col2:
#         st.metric("Analysis Start Date", f"{state.universe.start_date.strftime('%d/%m/%Y')}")
#         st.metric("Analysis End Date",f"{state.universe.end_date.strftime('%d/%m/%Y')}")
    
#     portfolio_as_list = [['custom', state.portfolios['custom']]]
    
#     sorted_portfolios = sort_portfolios(state.portfolios)

#     display_portfolio_performances(display, sorted_portfolios)
    
#     #display_weights_pie_charts(display, portfolio_as_list, label_threshold = 0.05, group_threshold = 0.01)
    
#     num_stocks = len(state.universe.stocks)
    
#     if num_stocks > 20 and state.portfolios['custom'].is_equally_weighted():
#         display.write(f"You are using an equally weighted portfolio of {num_stocks} stocks")

#     else:
#         display_weights_pie_charts(display, portfolio_as_list, label_threshold = 0.05, group_threshold = 0.05)
#         display_weights_bar_chart(display, portfolio_as_list, group_threshold = 0.05)

def setup_portfolio_overview(display, label_threshold = 0.05, group_threshold = 0.05):
    portfolio_as_list = [['custom', get_portfolio('custom')]]
    
    sorted_portfolios = iterate_portfolios(st.cov_type, include_custom = True, include_constrained = True)

    display_portfolio_performances(display, sorted_portfolios)
    
    display.markdown("#### Allocation ####")
    viz_option = display.radio(
        'Choose visualisation type:',
        ('Pie Chart', 'Table', 'Bar Chart'),
        horizontal = True,
        key = "viz_type_radio"
        )
    
    if viz_option == 'Pie Chart':
        display_weights_pie_charts(display, portfolio_as_list, label_threshold, group_threshold)
        
    elif viz_option == 'Table':
        display_weights_table(display, portfolio_as_list)
        
    else:
        display_weights_bar_chart(display, portfolio_as_list, group_threshold)
    
def setup_portfolio_display(display):
    """
    Setup the portfolio display panel to show the stocks chosen, the set risk free rate and the performances of the
    Max Sharpe Ratio, Min Volatility and Custom Portfolios.

    Parameters
    ----------
    display : st.container
        Container in which to display portfolio info.

    Returns
    -------
    display : st.container
        Container with the updated display.

    """
        
    display.header("Portfolio Analysis")
    if not state.universe or len(state.universe.stocks) < 2:
        display.write("You need at least two stocks to build a portfolio.")
        process_stock_form()
        
        return display
    
    if len(state.universe.stocks) > 15:
        stock_string = ", ".join(state.universe.stocks[:10])
        display.write(f"Analysing portfolios consisting of {len(state.universe.stocks)} portfolios, including {stock_string}, ... using data from {state.universe.start_date.strftime('%d/%m/%Y')} to {state.universe.end_date.strftime('%d/%m/%Y')}.")
    else:
        stock_string = ", ".join(state.universe.stocks)
        display.write(f"Analysing portfolios consisting of {stock_string} using data from {state.universe.start_date.strftime('%d/%m/%Y')} to {state.universe.end_date.strftime('%d/%m/%Y')}.")
    display.write(f'Risk free rate is {state.universe.risk_free_rate:.2%}.')
    
    display.markdown("### Max Sharpe Ratio, Min Vol and Custom Portfolios ###")
    if state.factor_bounds:
        unchanged = []
        constrained_max_sharpe = get_portfolio('max_sharpe', cov_type = state.cov_type, constrained = True)
        if constrained_max_sharpe == None:
            unchanged.append('Max Sharpe Ratio')
        constrained_min_vol = get_portfolio('min_vol', cov_type = state.cov_type, constrained = True)
        if constrained_min_vol == None:
            unchanged.append('Min Vol')
            
        if len(unchanged) == 1:
            display.write(f"The {unchanged[0]} portfolio is not affected by the factor constraints.")
        if len(unchanged) == 2:
            display.write(f"The {unchanged[0]} and {unchanged[1]} portfolios are not affected by the factor constraints.")
                
    sorted_portfolios = iterate_portfolios(state.cov_type, include_custom = True, include_constrained = True)
    
    display_portfolio_performances(display, sorted_portfolios)
    
    return display

def setup_details_display(display):
    
    tab_names = ['Allocation', 'Efficient Frontier']
    
    if state.factor_model:
        tab_names = ['Factor Analysis'] + tab_names
    
    name_to_id_map = {tab_name: idx for idx, tab_name in enumerate(tab_names)}
    
    tabs = display.tabs(tab_names)
    
    sorted_portfolios = iterate_portfolios(state.cov_type, include_custom = True, include_constrained = True)
    
    display_portfolio_weights(tabs[name_to_id_map['Allocation']], sorted_portfolios)
    setup_efficient_frontier_display(tabs[name_to_id_map['Efficient Frontier']])           
    if state.factor_model:
        display_portfolio_comparison(tabs[name_to_id_map['Factor Analysis']], sorted_portfolios, state.factor_bounds)
    
    return display

def plot_interactive_efficient_frontier(eff_frontier_data):
    eff_vols, eff_excess_returns = eff_frontier_data
    
    factor_bounds = state.factor_bounds
    constrained_eff_frontier = get_efficient_frontier(state.cov_type, constrained=True)
        
    if factor_bounds:
        sorted_portfolios = iterate_portfolios(state.cov_type, include_custom=True, include_constrained=True)
    else:
        sorted_portfolios = iterate_portfolios(state.cov_type, include_custom=True, include_constrained=False)
        
    portfolio_results = {name: (portfolio.vol, portfolio.excess_returns) 
                         for name, portfolio in sorted_portfolios if portfolio}
    portfolio_order = [name for name, portfolio in sorted_portfolios]
    
    vols, excess_returns = state.universe.individual_stock_portfolios(cov_type=state.cov_type)
    
    # Create the plot
    fig = make_subplots()
    
    # Add efficient frontier with hover information
    ef_sharpe_ratios = [er/vol for er, vol in zip(eff_excess_returns, eff_vols)]
    fig.add_trace(go.Scatter(
        x=eff_vols, 
        y=eff_excess_returns, 
        mode='lines', 
        name='Efficient Frontier', 
        line=dict(color='red'),
        hovertemplate='<b>Efficient Frontier</b><br>Volatility: %{x:.2%}<br>Excess Return: %{y:.2%}<br>Sharpe Ratio: %{customdata:.2f}<extra></extra>',
        customdata=ef_sharpe_ratios
    ))
    
    # Add constrained efficient frontier if it exists
    if constrained_eff_frontier:
        constrained_eff_vols, constrained_eff_excess_returns = constrained_eff_frontier
        constrained_ef_sharpe_ratios = [er/vol for er, vol in zip(constrained_eff_excess_returns, constrained_eff_vols)]
        fig.add_trace(go.Scatter(
            x=constrained_eff_vols, 
            y=constrained_eff_excess_returns, 
            mode='lines', 
            name= 'Constr. Eff. Frontier' if is_mobile() else 'Constrained Efficient Frontier', 
            line=dict(color='orange', dash='dash'),
            hovertemplate='<b>Constrained Efficient Frontier</b><br>Volatility: %{x:.2%}<br>Excess Return: %{y:.2%}<br>Sharpe Ratio: %{customdata:.2f}<extra></extra>',
            customdata=constrained_ef_sharpe_ratios
        ))
    
    # Add portfolios
    # portfolio_plot_settings = {
    #     'max_sharpe': {'marker': 'star', 'size': 10, 'color': 'green', 'label': 'Max Sharpe Portfolio'},
    #     'constrained_max_sharpe': {'marker': 'star', 'size': 10, 'color': 'orange', 'label': 'Constrained Max Sharpe Portfolio'},
    #     'min_vol': {'marker': 'x', 'size': 10, 'color': 'green', 'label': 'Min Vol Portfolio'},
    #     'constrained_min_vol': {'marker': 'x', 'size': 10, 'color': 'orange', 'label': 'Constrained Min Vol Portfolio'},
    #     'custom': {'marker': 'x', 'size': 10, 'color': 'blue', 'label': 'Custom Portfolio'}
    # }
    
    portfolio_plot_settings = {
        'max_sharpe': {'marker': 'star', 'size': 15, 'color': 'green', 'label': 'Max Sharpe Portfolio'},
        'constrained_max_sharpe': {'marker': 'star', 'size': 15, 'color': 'orange', 'label': 'Constr. Max Sharpe Portfolio' if is_mobile() else 'Constrained Max Sharpe Portfolio'},
        'min_vol': {'marker': 'x', 'size': 15, 'color': 'green', 'label': 'Min Vol Portfolio'},
        'constrained_min_vol': {'marker': 'x', 'size': 15, 'color': 'orange', 'label': 'Constr. Min Vol Portfolio' if is_mobile() else 'Constrained Min Vol Portfolio'},
        'custom': {'marker': 'x', 'size': 15, 'color': 'blue', 'label': 'Custom Portfolio'}
    }
    
    for name in portfolio_order:
        if name in portfolio_results:
            portfolio_vol, portfolio_excess_returns = portfolio_results[name]
            settings = portfolio_plot_settings[name]
            fig.add_trace(go.Scatter(
                x=[portfolio_vol], 
                y=[portfolio_excess_returns], 
                mode='markers', 
                name=settings['label'],
                marker=dict(symbol=settings['marker'], size=settings['size'], color=settings['color']),
                hovertemplate=f'<b>{settings["label"]}</b><br>Volatility: %{{x:.2%}}<br>Excess Return: %{{y:.2%}}<br>Sharpe Ratio: %{{customdata:.2f}}<extra></extra>',
                customdata=[portfolio_excess_returns/portfolio_vol]
            ))
    
    # Add individual stocks
    fig.add_trace(go.Scatter(
        x=vols, 
        y=excess_returns, 
        mode='markers', 
        name='Stocks',
        marker=dict(size=8, color='blue'),
        text=state.universe.stocks,
        hovertemplate='<b>%{text}</b><br>Volatility: %{x:.2%}<br>Excess Return: %{y:.2%}<br>Sharpe Ratio: %{customdata:.2f}<extra></extra>',
        customdata=[excess_returns[i]/vols[i] for i in range(len(vols))]
    ))
    
    legend_font_size = 10 if is_mobile() else 17
    axis_label_font_size = 10 if is_mobile() else 20
    title_font_size = 16 if is_mobile() else 24
    tickformat = '.2f' if is_mobile() else '.2%'
    margin = dict(l = 0, r = 15, t = 50, b = 50) if is_mobile() else dict(l = 50, r = 50, t = 50, b = 50)
    standoff = 4 if is_mobile() else 8
    
    fig.update_layout(
        title={
            'text': 'Excess Returns vs Volatility for Portfolios',
            'font': {'size': title_font_size, 'color': 'black'}
            },
        #xaxis_title={'text': 'Volatility', 'font': {'size': axis_label_font_size, 'color': 'black'}},
        #yaxis_title={'text': 'Excess Return', 'font': {'size': axis_label_font_size, 'color': 'black'}},
        legend=dict(
            font=dict(size=legend_font_size, color='black'),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        hovermode="closest",
        height=800,
        font=dict(color='black'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title = dict(text = "Volatility", font = {'size': axis_label_font_size, 'color': 'black'}, standoff = standoff),
            tickformat=tickformat, 
            gridcolor='lightgray',
            tickfont=dict(color='black'), 
            title_font=dict(color='black'),
            showline = True,
            linewidth = 2,
            linecolor = 'black',
            mirror = True
        ),
        yaxis=dict(
            title = dict(text = "Excess Returns", font = {'size': axis_label_font_size, 'color': 'black'}, standoff = standoff),
            tickformat=tickformat, 
            gridcolor='lightgray',
            tickfont=dict(color='black'), 
            title_font=dict(color='black'),
            showline = True,
            linewidth = 2,
            linecolor ='black',
            mirror = True
        ),
        margin=margin,
        autosize=True,
    )
    
    
    return fig

def create_modal(plot_html):
    print("\n Running modal")
    modal_html = f"""
    <style>
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.4);
        }}
        .modal-content {{
            background-color: #fefefe;
            margin: 5% auto;
            padding: 20px;
            border: 1px solid #888;
            width: 90%;
            height: 90%;
        }}
        .close {{
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
        }}
        .close:hover,
        .close:focus {{
            color: black;
            text-decoration: none;
            cursor: pointer;
        }}
        @media screen and (max-width: 768px) {{
            .modal-content {{
                width: 100%;
                height: 100%;
                margin: 0;
            }}
        }}
    </style>
    <div id="myModal" class="modal">
        <div class="modal-content">
            <span class="close">&times;</span>
            {plot_html}
        </div>
    </div>
    <script>
        var modal = document.getElementById("myModal");
        var span = document.getElementsByClassName("close")[0];
        span.onclick = function() {{
            modal.style.display = "none";
        }}
        window.onclick = function(event) {{
            if (event.target == modal) {{
                modal.style.display = "none";
            }}
        }}
        function showModal() {{
            modal.style.display = "block";
            if (window.matchMedia("(max-width: 768px)").matches) {{
                try {{
                    screen.orientation.lock('landscape');
                }} catch (error) {{
                    console.error('Unable to lock screen orientation:', error);
                }}
            }}
        }}
    </script>
    """
    return modal_html

def interactive_efficient_frontier_display(display):
    eff_vols, eff_excess_returns = get_efficient_frontier(state.cov_type, constrained=False)
    if len(eff_vols) == 0 or len(eff_excess_returns) == 0:
        display.error("Failed to load efficient frontier. Need at least two stocks to have a viable portfolio.")
        return display
    
    fig = plot_interactive_efficient_frontier((eff_vols, eff_excess_returns))
    
    #inject_screen_detector()
    
    # if is_mobile():
    #     # For mobile, show a button that opens the plot in a modal
    #     print("\n On Mobile!")
    #     if 'modal_created' not in state:
    #         state.modal_created = False

    #     if not state.modal_created:
    #         # Generate the plot HTML
    #         plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            
    #         # Create the modal
    #         modal_html = create_modal(plot_html)
    #         components.html(modal_html, height=0)
    #         state.modal_created = True

    #     if display.button("View Efficient Frontier Plot"):
    #         st.write('<script>showModal();</script>', unsafe_allow_html=True)
    
    #else:
    if True:
        st.plotly_chart(fig, use_container_width = True, config = {'responsive': True, 'displayModeBar': False,})
    
    #display.plotly_chart(fig, use_container_width=True)
    
    return display

def setup_efficient_frontier_display(display):
    """
    Setup the efficient frontier display panel to show a plot of the efficient frontier, the individual stocks
    in the chosen stock universe and the portfolios in session state (max_SR, min_vol and custom).

    Parameters
    ----------
    display : st.container
        Container in which the information will be placed.

    Returns
    -------
    display : st.container
        Container with the display information.

    """
    
    eff_vols, eff_excess_returns = get_efficient_frontier(state.cov_type, constrained = False)
    if len(eff_vols) == 0 or len(eff_excess_returns) == 0:
        display.error("Failed to load efficient frontier. Need at least two stocks to have a viable portfolio.")
        return display
    
    factor_bounds = state.factor_bounds
    
    constrained_eff_frontier = get_efficient_frontier(state.cov_type, constrained = True)
        
    if factor_bounds:
        sorted_portfolios = iterate_portfolios(state.cov_type, include_custom = True, include_constrained = True)
    else:
        sorted_portfolios = iterate_portfolios(state.cov_type, include_custom = True, include_constrained = False)
        
    portfolio_results = {}
    for name, portfolio in sorted_portfolios:
        if portfolio:
          portfolio_results[name] = (portfolio.vol, portfolio.excess_returns)
    
    #{name: (portfolio.vol, portfolio.excess_returns) for name, portfolio in sorted_portfolios}
    portfolio_order = [name for name, portfolio in sorted_portfolios]

    vols, excess_returns = state.universe.individual_stock_portfolios(cov_type = state.cov_type)

    plt.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Excess Return')
    ax.set_title('Excess Returns vs Volatility for Portfolios')

    ax.plot(eff_vols, eff_excess_returns, '-', markersize = 3, color = 'red', label = 'Efficient Frontier')
    
    if constrained_eff_frontier:
        constrained_eff_vols, constrained_eff_excess_returns = constrained_eff_frontier
        ax.plot(constrained_eff_vols, constrained_eff_excess_returns, '--', markersize = 3, color = 'orange', label = 'Factor Constrained Efficient Frontier')

    portfolio_plot_settings = {'max_sharpe': {'marker': '*', 'size': 10, 'color': 'green', 'label': 'Max Sharpe Portfolio'},
                               'constrained_max_sharpe': {'marker': '*', 'size': 10, 'color': 'orange', 'label': 'Constrained Max Sharpe Portfolio'},
                               'min_vol': {'marker': 'x', 'size': 10, 'color': 'green', 'label': 'Min Vol Portfolio'},
                               'constrained_min_vol': {'marker': 'x', 'size': 10, 'color': 'orange', 'label': 'Constrained Min Vol Portfolio'},
                               'custom': {'marker': 'x', 'size': 10, 'color': 'blue', 'label': 'Custom Portfolio'}
                               }

    for name in portfolio_order:
        if name in portfolio_results:
            portfolio_vol, portfolio_excess_returns = portfolio_results[name]
            settings = portfolio_plot_settings[name]
            ax.plot(portfolio_vol, portfolio_excess_returns, settings['marker'], markersize = settings['size'], color = settings['color'], label = settings['label'])
        
    ax.scatter(vols, excess_returns, marker = 'o', s = 15, color = 'blue')
    texts = []
    for idx, ticker in enumerate(state.universe.stocks):
        # texts.append(ax.annotate(ticker, (vols[idx], excess_returns[idx]), xytext = (5,5),
        #                          textcoords = 'offset points', ha = 'left', va = 'bottom',
        #                          bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        #                          arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')))
        
        # adjust_text(texts, arrowprops = dict(arrowstyle = '-', color = 'r', lw = 0.5))
        
        ax.annotate(ticker, (vols[idx], excess_returns[idx]), xytext=(2, 2), 
                 textcoords='offset points', ha='left', va='bottom')
        
        #ax.annotate(ticker, (vols[idx], excess_returns[idx]))
        
    ax.legend()
    
    col1, col2, col3 = display.columns([0.05, 0.8, 0.15])
    
    col2.pyplot(fig, use_container_width = False)
    
    return display

def display_portfolio_performances(output, portfolios):
    """
    Show the performances of the portfolios passed.


    Parameters
    ----------
    output : st.container
        Container in which the information will be placed.
    portfolios : dict(portfolio)
        Dictionary of portfolios to display.

    Returns
    -------
    None.

    """
        
    output.markdown("#### Performance ####")
    joined_perf = []
    
    if not portfolios:
        return
    
    for _, portfolio in portfolios:
        perf_df, format_map = portfolio.get_performance_df()
        joined_perf.append(perf_df)
        
    joined_perf_df = pd.concat(joined_perf, axis = 0)
    
    if is_mobile():
        joined_perf_df, format_map = format_performance_df_for_mobile(joined_perf_df, format_map)
    
    html_table = style_table(joined_perf_df, format_map, is_mobile())
    wrapped_table = f'<div class="table-wrapper-75">{html_table}</div>'
    output.markdown(wrapped_table, unsafe_allow_html = True)
    
def display_portfolio_weights(output, portfolios, label_threshold = 0.05, group_threshold = 0.05):
    """
    Show the weights of the portfolios passed in a pie chart and (optionally) in an expandable table.

    Parameters
    ----------
    output : st.container
        Container in which the information will be placed.
    portfolios : dict(portfolio)
        Dictionary of portfolios whose information will be displayed.
    incl_table : bool, optional
        Boolean of whether to show a table with the weights info (in an expander). The default is False.

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
        if len(portfolios) == 1:
            col1, col2, col3 = output.columns([0.1, 0.7, 0.2])
            display_weights_pie_charts(col2, portfolios, label_threshold, group_threshold)
        else:
            display_weights_pie_charts(output, portfolios, label_threshold, group_threshold)
        
    elif viz_option == 'Table':
        display_weights_table(output, portfolios)
        
    else:
        display_weights_bar_chart(output, portfolios, group_threshold)
        
def format_factor_bounds(factor_bounds):
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


def display_factor_bounds(container, factor_model, factor_bounds):
    formatted_bounds = format_factor_bounds(factor_bounds)
    if formatted_bounds:
        container.markdown("#### Current Factor Constraints ####")
        container.write(f"{format_factor_choice(factor_model)} Model")
        bounds_text = "\n".join(formatted_bounds)
        container.code(bounds_text)
    else:
        container.info("No active factor constraints.")
        
def generate_color_dict(portfolios, all_stocks, group_threshold):
    stocks_above_threshold = set()
    for _, portfolio in portfolios:
        weights = portfolio.weights
        stocks_above_threshold.update([stock for stock, weight in zip(all_stocks, weights) if weight >= group_threshold])
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(stocks_above_threshold) > len(default_colors):
        extended_colors = []
        for i in range(len(stocks_above_threshold)):
            idx = i % len(default_colors)
            color = default_colors[idx]
            if i >= len(default_colors):
                rgb = mcolors.to_rgb(color)
                lighter_rgb = tuple(min(1, c * 1.3) for c in rgb)
                color = lighter_rgb
            extended_colors.append(color)
        colors = extended_colors
    else:
        colors = default_colors[:len(stocks_above_threshold)]
    
    return dict(zip(stocks_above_threshold, colors))
        
def display_weights_pie_charts(output, portfolios, label_threshold, group_threshold):
    num_portfolios = len(portfolios)
    
    container_width = 1730 #st_dimensions()['width']
    fig_width = min(container_width / 96, 20)  # Convert pixels to inches, max width of 20 inches
    fig_height = fig_width * 0.5 * num_portfolios  # Adjust height based on number of portfolios
        
    fig, axes = plt.subplots(1, num_portfolios, figsize=(fig_width, fig_height), dpi = 200)
    if num_portfolios == 1:
        axes = [axes]
    
    all_stocks = portfolios[0][1].universe.stocks
    color_dict = generate_color_dict(portfolios, all_stocks, group_threshold)
    
    # stocks_above_threshold = set()
    
    # for _, portfolio in portfolios:
    #     weights = portfolio.weights
    #     stocks_above_threshold.update([stock for stock, weight in zip(all_stocks, weights) if weight >= group_threshold])
    
    # default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # if len(stocks_above_threshold) > len(default_colors):
    #     extended_colors = []
    #     for i in range(len(stocks_above_threshold)):
    #         idx = i % len(default_colors)
    #         color = default_colors[idx]
    #         if i >= len(default_colors):
    #             rgb = mcolors.to_rgb(color)
    #             lighter_rgb = tuple(min(1, c * 1.3) for c in rgb)
    #             color = lighter_rgb
    #         extended_colors.append(color)
    #     colors = extended_colors
    # else:
    #     colors = default_colors[:len(stocks_above_threshold)]
    
    #color_dict = dict(zip(stocks_above_threshold, colors))
    
    legend_elements = []
    legend_labels = []
    others_element = None
    others_label = None
    
    names_fontsize = 20 if is_mobile() else 12
    pct_fontsize = 18 if is_mobile() else 12
    others_fontsize = 18 if is_mobile() else 12

    for idx, (_, portfolio) in enumerate(portfolios):
        weights = np.array(portfolio.weights)
        stocks = np.array(all_stocks.copy())
        
        non_zero_mask = weights > 0 
        weights = weights[non_zero_mask]
        stocks = stocks[non_zero_mask]
        small_indices = np.where(np.array(weights) < group_threshold)[0]
        if len(small_indices) > 0:
            small_sum = np.sum([weights[i] for i in small_indices])
            large_mask = ~np.isin(np.arange(len(weights)), small_indices)
            weights = list(weights[large_mask])
            stocks = list(stocks[large_mask])
            #weights = [w for i, w in enumerate(weights) if i not in small_indices]
            #stocks = [s for i, s in enumerate(stocks) if i not in small_indices]
            weights.append(small_sum)
            stocks.append(f"Others (<{group_threshold * 100:.1f}%)")
        
        # Sort weights and stocks
        sorted_indices = np.argsort(weights)[::-1]
        weights = [weights[i] for i in sorted_indices]
        stocks = [stocks[i] for i in sorted_indices]
            
        portfolio_colors = [color_dict.get(stock, '#DDDDDD') for stock in stocks]  # Use light grey for 'Others'
        
        if is_mobile():
            pct_formatting = lambda pct: f'{pct:.0f}%' if pct > label_threshold * 100 else ''
        else:
            pct_formatting = lambda pct: f'{pct:.1f}%' if pct > label_threshold * 100 else ''
        
        wedges, texts, autotexts = axes[idx].pie(weights,
                                                 colors = portfolio_colors,
                                                 autopct = pct_formatting,
                                                 #autopct = lambda pct: f'{pct:.1f}%' if pct > label_threshold * 100 else '',
                                                 pctdistance=0.75,
                                                 textprops={'fontsize': pct_fontsize}
                                                 )
        for i, (wedge, autotext) in enumerate(zip(wedges, autotexts)):
            if weights[i] >= label_threshold:
                autotext.set_visible(True)
            else:
                autotext.set_visible(False)
            
            if stocks[i] not in legend_labels and stocks[i] != f"Others (<{group_threshold * 100:.1f}%)":
                legend_elements.append(wedge)
                legend_labels.append(stocks[i])
            elif stocks[i] == f"Others (<{group_threshold * 100:.1f}%)" and others_element is None:
                others_element = wedge
                others_label = stocks[i]
        
        title = portfolio.name + ' Portfolio'
        if is_mobile():
            title = title.replace('Constrained', 'Constr.')
            
        axes[idx].set_title(title, fontsize = names_fontsize)
        
    # Add "Others" to the end of the legend if it exists
    if others_element is not None:
        legend_elements.append(others_element)
        legend_labels.append(others_label)
        
    num_cols = max(1, len(legend_labels) // 10)  # Adjust 10 to change the number of items per column
    fig.legend(legend_elements, legend_labels, loc = 'center left', 
               bbox_to_anchor=(1, 0.5), title="Stocks", fontsize=others_fontsize,
               title_fontsize = others_fontsize, ncol=num_cols, columnspacing=1,
               handletextpad=0.5)
    
    plt.tight_layout()
    plt.subplots_adjust(right=1)
    
    output.pyplot(fig, use_container_width = True)

def display_weights_table(output, portfolios):
    joined_weights = []
    for _, portfolio in portfolios:
        weights_df, format_map = portfolio.get_weights_df()
        joined_weights.append(weights_df)
        
    joined_weights_df = pd.concat(joined_weights, axis = 0)
        
    html_table = style_table(joined_weights_df, format_map)
    wrapped_table_100 = f"<div class = 'table-wrapper-full'>{html_table}</div>"
    output.markdown(wrapped_table_100, unsafe_allow_html = True)
    
def display_weights_bar_chart(output, portfolios, group_threshold):
    if len(portfolios) > 1:
    
        portfolio_options = [portfolio[1].name for portfolio in portfolios] + ['Show all']
        selected_portfolio = output.selectbox("Select portfolio to display:", portfolio_options)
        
    else:
        selected_portfolio = portfolios[0][1].name
    
    all_stocks = portfolios[0][1].universe.stocks
    color_dict = generate_color_dict(portfolios, all_stocks, group_threshold)
    
    # stocks_above_threshold = set()
    # for _, portfolio in portfolios:
    #     weights = portfolio.weights
    #     stocks_above_threshold.update([stock for stock, weight in zip(all_stocks, weights) if weight >= group_threshold])

    # color_dict = dict(zip(stocks_above_threshold, 
    #                       plt.cm.tab20(np.linspace(0, 1, len(stocks_above_threshold)))))
    
    if selected_portfolio == "Show all":
        num_portfolios = len(portfolios)
        fig, axes = plt.subplots(num_portfolios, 1, figsize=(10, 5*num_portfolios))
        if num_portfolios == 1:
            axes = [axes]
            
    else:
        fig, ax = plt.subplots(figsize = (10, 5))
        axes = [ax]

    portfolio_dict = {portfolio.name: portfolio for _, portfolio in portfolios}

    portfolios_to_plot = [portfolio[1] for portfolio in portfolios] if selected_portfolio == 'Show all' else [portfolio_dict[selected_portfolio]]

    for idx, portfolio in enumerate(portfolios_to_plot):
        weights = np.array(portfolio.weights)
        stocks = np.array(all_stocks.copy())
        
        non_zero_mask = weights > 0
        weights = weights[non_zero_mask]
        stocks = stocks[non_zero_mask]
        
        # Group stocks with less than group_threshold allocation
        small_indices = np.where(np.array(weights) < group_threshold)[0]
        if len(small_indices) > 0:
            small_sum = sum([weights[i] for i in small_indices])
            weights = [w for i, w in enumerate(weights) if i not in small_indices]
            stocks = [s for i, s in enumerate(stocks) if i not in small_indices]
            weights.append(small_sum)
            stocks.append(f"Others (<{group_threshold*100:.1f}%)")

        colors = [color_dict.get(stock, '#DDDDDD') for stock in stocks]
        
        axes[idx].bar(stocks, weights, color=colors)
        axes[idx].set_title(f"{portfolio.name} Portfolio")
        axes[idx].set_ylabel('Weight')
        if len(stocks) > 15:
            axes[idx].set_xticklabels(stocks, rotation = 45, ha = 'right')
        else:
            axes[idx].set_xticklabels(stocks)

    plt.tight_layout()
    output.pyplot(fig, use_container_width = True)

def calculate_factor_contributions(portfolio, factor_exposures):
    factor_returns = portfolio.universe.factor_analysis.factor_returns / 100
    factor_returns = factor_returns.loc[portfolio.universe.start_date : portfolio.universe.end_date]
    factor_returns = get_mean_returns(factor_returns) * TRADING_DAYS
    
    total_return = portfolio.excess_returns
    
    contributions = {}
    for factor, exposure in factor_exposures.items():
        factor_contribution = exposure * factor_returns[factor]
        contributions[factor] = (factor_contribution / total_return) * 100
    
    # Calculate residual (unexplained) return
    explained_return = sum(contributions.values())
    contributions['Residual'] = 100 - explained_return
    
    return contributions

def plot_portfolio_comparison(factor_exposures, factor_contributions, name_map = None, factor_bounds = None):
    fig, (ax2, ax1) = plt.subplots(2, 1, figsize = (12, 16))
    
    exposure_data = pd.DataFrame(factor_exposures).T
    portfolio_names = exposure_data.index
    if name_map:
        portfolio_full_names = [name_map[name] for name in portfolio_names]
    else:
        portfolio_full_names = portfolio_names
    factors = exposure_data.columns
    x = np.arange(len(portfolio_names))
    width = 0.8 / len(factors)
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {factor: default_colors[i % len(default_colors)] for i, factor in enumerate(factors)}
    color_map['Residual'] = default_colors[len(factors) % len(default_colors)]
    
    for i, factor in enumerate(factors):
        ax1.bar(x + i*width, exposure_data[factor], width, label = factor, color = color_map[factor])
        if factor_bounds and factor in factor_bounds:
            lower, upper = factor_bounds[factor]
            bounds_added_to_legend = False
            if lower:
                ax1.axhline(y = [lower], color = color_map[factor], linestyle = 'dashed', label = f'{factor} bounds')
                bounds_added_to_legend = True
            if upper:
                if bounds_added_to_legend:
                    ax1.axhline(y = [upper], color = color_map[factor], linestyle = 'dashed')
                else:
                    ax1.axhline(y = [upper], color = color_map[factor], linestyle = 'dashed', label = f'{factor} bounds')
        
    ax1.set_ylabel('Factor Exposure')
    ax1.set_title('Factor Exposure Comparison Across Portfolios')
    ax1.set_xticks(x + width * (len(factors) - 1) / 2)
    ax1.set_xticklabels(portfolio_full_names, rotation = 45, ha = 'right')
    ax1.legend(title = 'Factors', bbox_to_anchor = (1.05, 1), loc = 'upper left')
    
    bar_width = 0.4    
    
    for i, portfolio_name in enumerate(portfolio_names):
        positive_bottom = 0
        negative_bottom = 0
        for factor in factors:
            value = factor_contributions[portfolio_name].get(factor, 0)
            if value >= 0:
                rect = ax2.bar(x[i], value, bar_width, bottom = positive_bottom, label = factor if i == 0 else "", color = color_map[factor])
                positive_bottom += value
                # add label
                height = rect[0].get_height()
                ax2.text(rect[0].get_x() + rect[0].get_width()/2, positive_bottom - height/2,
                         f"{value:.1f}%", ha = "center", va = "center", fontsize = 8)
            else:
                rect = ax2.bar(x[i], value, bar_width, bottom = negative_bottom, label = factor if i == 0 else "", color = color_map[factor])
                # add label
                height = rect[0].get_height()
                ax2.text(rect[0].get_x() + rect[0].get_width()/2, negative_bottom + height/2,
                         f"{value:.1f}%", ha = "center", va = "center", fontsize = 8)
                negative_bottom += value
                
        residual = factor_contributions[portfolio_name].get('Residual', 0)
        if residual >= 0:
            rect = ax2.bar(x[i], residual, bar_width, bottom = positive_bottom, label = 'Residual' if i == 0 else '', color = color_map['Residual'])
            # add label
            height = rect[0].get_height()
            ax2.text(rect[0].get_x() + rect[0].get_width()/2, positive_bottom + height/2,
                     f"{residual:.1f}%", ha = "center", va = "center", fontsize = 8)
        else:
            rect = ax2.bar(x[i], residual, bar_width, bottom = negative_bottom, label = 'Residual' if i == 0 else '', color = color_map['Residual'])
            # add label
            height = rect[0].get_height()
            ax2.text(rect[0].get_x() + rect[0].get_width()/2, negative_bottom + height/2,
                     f"{residual:.1f}%", ha = "center", va = "center", fontsize = 8)
        
    ax2.set_ylabel('Return Attribution (%)')
    ax2.set_title('Percentage of Return Attributable to Factors Across Portfolios')
    ax2.set_xticks(x)
    ax2.set_xticklabels(portfolio_full_names)
    ax2.legend(title = 'Factors', bbox_to_anchor = (1.05, 1), loc = 'upper left')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def plot_return_comparison(factor_exposures, factor_contributions, name_map = None, factor_bounds = None):
    fig, ax = plt.subplots(figsize = (10, 5))
    
    exposure_data = pd.DataFrame(factor_exposures).T
    portfolio_names = exposure_data.index
    if name_map:
        portfolio_full_names = [name_map[name] for name in portfolio_names]
    else:
        portfolio_full_names = portfolio_names
    factors = exposure_data.columns
    x = np.arange(len(portfolio_names))
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {factor: default_colors[i % len(default_colors)] for i, factor in enumerate(factors)}
    color_map['Residual'] = default_colors[len(factors) % len(default_colors)]

    bar_width = 0.4    
    
    for i, portfolio_name in enumerate(portfolio_names):
        positive_bottom = 0
        negative_bottom = 0
        for factor in factors:
            value = factor_contributions[portfolio_name].get(factor, 0)
            if value >= 0:
                rect = ax.bar(x[i], value, bar_width, bottom = positive_bottom, label = factor if i == 0 else "", color = color_map[factor])
                positive_bottom += value
                # add label
                height = rect[0].get_height()
                ax.text(rect[0].get_x() + rect[0].get_width()/2, positive_bottom - height/2,
                         f"{value:.1f}%", ha = "center", va = "center", fontsize = 8)
            else:
                rect = ax.bar(x[i], value, bar_width, bottom = negative_bottom, label = factor if i == 0 else "", color = color_map[factor])
                # add label
                height = rect[0].get_height()
                ax.text(rect[0].get_x() + rect[0].get_width()/2, negative_bottom + height/2,
                         f"{value:.1f}%", ha = "center", va = "center", fontsize = 8)
                negative_bottom += value
                
        residual = factor_contributions[portfolio_name].get('Residual', 0)
        if residual >= 0:
            rect = ax.bar(x[i], residual, bar_width, bottom = positive_bottom, label = 'Residual' if i == 0 else '', color = color_map['Residual'])
            # add label
            height = rect[0].get_height()
            ax.text(rect[0].get_x() + rect[0].get_width()/2, positive_bottom + height/2,
                     f"{residual:.1f}%", ha = "center", va = "center", fontsize = 8)
        else:
            rect = ax.bar(x[i], residual, bar_width, bottom = negative_bottom, label = 'Residual' if i == 0 else '', color = color_map['Residual'])
            # add label
            height = rect[0].get_height()
            ax.text(rect[0].get_x() + rect[0].get_width()/2, negative_bottom + height/2,
                     f"{residual:.1f}%", ha = "center", va = "center", fontsize = 8)
        
    ax.set_ylabel('Return Attribution (%)')
    ax.set_title('Percentage of Return Attributable to Factors Across Portfolios')
    ax.set_xticks(x)
    ax.set_xticklabels(portfolio_full_names, rotation = 45, ha = 'right')
    ax.legend(title = 'Factors', bbox_to_anchor = (1.05, 1), loc = 'upper left')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def plot_exposure_comparison(factor_exposures, name_map = None, factor_bounds = None):
    fig, ax = plt.subplots(figsize = (10, 5))
    
    exposure_data = pd.DataFrame(factor_exposures).T
    portfolio_names = exposure_data.index
    if name_map:
        portfolio_full_names = [name_map[name] for name in portfolio_names]
    else:
        portfolio_full_names = portfolio_names
    factors = exposure_data.columns
    x = np.arange(len(portfolio_names))
    width = 0.8 / len(factors)
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {factor: default_colors[i % len(default_colors)] for i, factor in enumerate(factors)}
    color_map['Residual'] = default_colors[len(factors) % len(default_colors)]
    
    for i, factor in enumerate(factors):
        ax.bar(x + i*width, exposure_data[factor], width, label = factor, color = color_map[factor])
        if factor_bounds and factor in factor_bounds:
            lower, upper = factor_bounds[factor]
            bounds_added_to_legend = False
            if lower:
                ax.axhline(y = [lower], color = color_map[factor], linestyle = 'dashed', label = f'{factor} bounds')
                bounds_added_to_legend = True
            if upper:
                if bounds_added_to_legend:
                    ax.axhline(y = [upper], color = color_map[factor], linestyle = 'dashed')
                else:
                    ax.axhline(y = [upper], color = color_map[factor], linestyle = 'dashed', label = f'{factor} bounds')
        
    ax.set_ylabel('Factor Exposure')
    ax.set_title('Factor Exposure Comparison Across Portfolios')
    ax.set_xticks(x + width * (len(factors) - 1) / 2)
    ax.set_xticklabels(portfolio_full_names, rotation = 45, ha = 'right')
    ax.legend(title = 'Factors', bbox_to_anchor = (1.05, 1), loc = 'upper left')
    
    plt.tight_layout()
    return fig

def display_return_comparison(output, portfolios, factor_bounds = None):
    name_map = {name: portfolio.name for name, portfolio in portfolios}
    
    exposures = {name: p.universe.factor_analysis.analyse_portfolio(p)['Factor Exposures'].iloc[0] for name, p in portfolios}
    contributions = {name: calculate_factor_contributions(portfolio, exposures[name]) for name, portfolio in portfolios}
    
    fig = plot_return_comparison(exposures, contributions, name_map, factor_bounds)
    
    col1, col2, col3 = output.columns([0.05, 0.8, 0.15])
    col2.pyplot(fig, use_container_width = True)
    return

def display_exposure_comparison(output, portfolios, factor_bounds = None):
    name_map = {name: portfolio.name for name, portfolio in portfolios}
    
    exposures = {name: p.universe.factor_analysis.analyse_portfolio(p)['Factor Exposures'].iloc[0] for name, p in portfolios}
    
    fig = plot_exposure_comparison(exposures, name_map, factor_bounds)
    
    col1, col2, col3 = output.columns([0.05, 0.8, 0.15])
    col2.pyplot(fig, use_container_width = True)

def display_portfolio_comparison(output, portfolios, factor_bounds = None):
    name_map = {name: portfolio.name for name, portfolio in portfolios}
    
    exposures = {name: p.universe.factor_analysis.analyse_portfolio(p)['Factor Exposures'].iloc[0] for name, p in portfolios}
    contributions = {name: calculate_factor_contributions(portfolio, exposures[name]) for name, portfolio in portfolios}
    
    fig = plot_portfolio_comparison(exposures, contributions, name_map, factor_bounds)
    
    output.markdown("#### Factor Analysis ####")
    col1, col2, col3 = output.columns([0.05, 0.8, 0.15])
    col2.pyplot(fig, use_container_width = False)
    
def style_table(df, format_map = None, is_mobile = False):
    styled = df.style
    if format_map:
        styled = styled.format(format_map)
        
    styled = styled.format_index(None)

    html_table = styled.to_html(index = False)

    if is_mobile:
        html_table = html_table.replace('<th class="blank level0" >&nbsp;', '<th>Portfolio')
    
    html_table = f'<div class="styled-table">{html_table}</div>'
    
    return html_table

def sort_portfolios(portfolios):
    desired_order = ['max_SR', 'constrained_max_SR', 'min_vol', 'constrained_min_vol', 'custom']
    sorted_portfolios = sorted(
        portfolios.items(),
        key = lambda x: (desired_order.index(x[0]) if x[0] in desired_order else len(desired_order))
        )
    
    return sorted_portfolios