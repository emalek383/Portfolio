import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
# from streamlit_dimensions import st_dimensions
from process_forms import process_stock_form
from helper_functions import get_mean_returns

TRADING_DAYS = 252

state = st.session_state

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
        if 'constrained_max_SR' not in state.portfolios:
            unchanged.append('Max Sharpe Ratio')
        if 'constrained_min_vol' not in state.portfolios:
            unchanged.append('Min Vol')
            
        if len(unchanged) == 1:
            display.write(f"The {unchanged[0]} portfolio is not affected by the factor constraints.")
        if len(unchanged) == 2:
            display.write(f"The {unchanged[0]} and {unchanged[1]} portfolios are not affected by the factor constraints.")
                
    sorted_portfolios = sort_portfolios(state.portfolios)
    
    display_portfolio_performances(display, sorted_portfolios)
    
    return display

def setup_details_display(display):
    
    tab_names = ['Allocation', 'Efficient Frontier']
    
    if state.factor_model:
        tab_names = ['Factor Analysis'] + tab_names
    
    name_to_id_map = {tab_name: idx for idx, tab_name in enumerate(tab_names)}
    
    tabs = display.tabs(tab_names)
    
    sorted_portfolios = sort_portfolios(state.portfolios)
    
    display_portfolio_weights(tabs[name_to_id_map['Allocation']], sorted_portfolios)
    setup_efficient_frontier_display(tabs[name_to_id_map['Efficient Frontier']])           
    if state.factor_model:
        display_portfolio_comparison(tabs[name_to_id_map['Factor Analysis']], sorted_portfolios)
    
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
        
    display.markdown("#### Efficient Frontier ####")
    
    if not state.eff_frontier:
        display.write("Need at least two stocks to have a viable portfolio.")
        return display
        
    eff_vols, eff_excess_returns = state.eff_frontier
    
    factor_bounds = state.factor_bounds
    
    if factor_bounds and state.constrained_eff_frontier:
        constrained_eff_vols, constrained_eff_excess_returns = state.constrained_eff_frontier
        
    my_portfolio_results = [(portfolio.vol, portfolio.excess_returns) for portfolio in state.portfolios.values()]

    vols, excess_returns = state.universe.individual_stock_portfolios()

    plt.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Excess Return')
    ax.set_title('Excess Returns vs Volatility for Portfolios')

    ax.plot(eff_vols, eff_excess_returns, '-', markersize = 3, color = 'red', label = 'Efficient Frontier')
    
    if state.constrained_eff_frontier:
        ax.plot(constrained_eff_vols, constrained_eff_excess_returns, '--', markersize = 3, color = 'orange', label = 'Factor Constrained Efficient Frontier')
        

    markers = ['*', 'x', 'x']
    markersizes = [10, 10, 10]
    colors = ['green', 'green', 'blue']
    labels = ['Max Sharpe Ratio Portfolio', 'Min Vol Portfolio', 'Custom Portfolio']

    for ((portfolio_vol, portfolio_excess_returns), marker, size, color, label) in zip(my_portfolio_results, markers, markersizes, colors, labels):
        ax.plot(portfolio_vol, portfolio_excess_returns, marker, markersize = size, color = color, label = label)
        
    ax.plot(vols, excess_returns, 'o', markersize = 5, color = 'blue')
    for idx, ticker in enumerate(state.universe.stocks):
        ax.annotate(ticker, (vols[idx], excess_returns[idx]))
        
    if factor_bounds:
        if 'constrained_max_SR' in state.portfolios:
            ax.plot(state.portfolios['constrained_max_SR'].vol, state.portfolios['constrained_max_SR'].excess_returns,'*', markersize = 10, color = 'orange', label = 'Constrained Max Sharpe Ratio Portfolio')
        if 'constrained_min_vol' in state.portfolios:
            ax.plot(state.portfolios['constrained_min_vol'].vol, state.portfolios['constrained_min_vol'].excess_returns, 'x', markersize = 10, color = 'orange', label = 'Constrained Min Vol Portfolio')
        
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
    
    for _, portfolio in portfolios:
        perf_df, format_map = portfolio.get_performance_df()
        joined_perf.append(perf_df)
        
    joined_perf_df = pd.concat(joined_perf, axis = 0)
    html_table = style_table(joined_perf_df, format_map)
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
        display_weights_pie_charts(output, portfolios, label_threshold, group_threshold)
        
    elif viz_option == 'Table':
        display_weights_table(output, portfolios)
        
    else:
        pass
        display_weights_bar_chart(output, portfolios, group_threshold)
        
def display_weights_pie_charts(output, portfolios, label_threshold, group_threshold):
    num_portfolios = len(portfolios)
    
    container_width = 1730 #st_dimensions()['width']
    fig_width = min(container_width / 96, 20)  # Convert pixels to inches, max width of 20 inches
    fig_height = fig_width * 0.5 * num_portfolios  # Adjust height based on number of portfolios
        
    fig, axes = plt.subplots(1, num_portfolios, figsize=(fig_width, fig_height), dpi = 200)
    if num_portfolios == 1:
        axes = [axes]
    
    all_stocks = portfolios[0][1].universe.stocks
    
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
    
    color_dict = dict(zip(stocks_above_threshold, colors))
    
    legend_elements = []
    legend_labels = []

    for idx, (_, portfolio) in enumerate(portfolios):
        weights = portfolio.weights
        stocks = all_stocks.copy()
        
        small_indices = np.where(np.array(weights) < group_threshold)[0]
        if len(small_indices) > 0:
            small_sum = sum([weights[i] for i in small_indices])
            weights = [w for i, w in enumerate(weights) if i not in small_indices]
            stocks = [s for i, s in enumerate(stocks) if i not in small_indices]
            weights.append(small_sum)
            stocks.append(f"Others (<{group_threshold * 100:.1f}%)")
            
        portfolio_colors = [color_dict.get(stock, '#DDDDDD') for stock in stocks]  # Use light grey for 'Others'
        
        wedges, texts, autotexts = axes[idx].pie(weights,
                                                 colors = portfolio_colors,
                                                 autopct = lambda pct: f'{pct:.1f}%' if pct > label_threshold * 100 else '',
                                                 pctdistance=0.75,
                                                 textprops={'fontsize': 12}
                                                 )
        for i, (wedge, autotext) in enumerate(zip(wedges, autotexts)):
            if weights[i] >= label_threshold:
                autotext.set_visible(True)
            else:
                autotext.set_visible(False)
                
            if stocks[i] not in legend_labels:
                legend_elements.append(wedge)
                legend_labels.append(stocks[i])
        
        axes[idx].set_title(portfolio.name + ' Portfolio', fontsize = 12)
            
    num_cols = max(1, len(legend_labels) // 10)  # Adjust 10 to change the number of items per column
    fig.legend(legend_elements, legend_labels, loc = 'center left', 
               bbox_to_anchor=(1, 0.5), title="Stocks", fontsize=12,
               title_fontsize = 12, ncol=num_cols, columnspacing=1,
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
    portfolio_options = [portfolio[1].name for portfolio in portfolios] + ['Show all']
    selected_portfolio = output.selectbox("Select portfolio to display:", portfolio_options)
    
    all_stocks = portfolios[0][1].universe.stocks
    stocks_above_threshold = set()
    for _, portfolio in portfolios:
        weights = portfolio.weights
        stocks_above_threshold.update([stock for stock, weight in zip(all_stocks, weights) if weight >= group_threshold])

    color_dict = dict(zip(stocks_above_threshold, 
                          plt.cm.tab20(np.linspace(0, 1, len(stocks_above_threshold)))))
    
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
        weights = portfolio.weights
        stocks = all_stocks.copy()
        
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
        axes[idx].set_xticklabels(stocks)

    plt.tight_layout()
    output.pyplot(fig, use_container_width = False)

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

def plot_portfolio_comparison(factor_exposures, factor_contributions, name_map = None):
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
        
    ax1.set_ylabel('Factor Exposure')
    ax1.set_title('Factor Exposure Comparison Across Portfolios')
    ax1.set_xticks(x + width * (len(factors) - 1) / 2)
    ax1.set_xticklabels(portfolio_full_names)
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

def display_portfolio_comparison(output, portfolios):
    name_map = {name: portfolio.name for name, portfolio in portfolios}
    
    exposures = {name: p.universe.factor_analysis.analyse_portfolio(p)['Factor Exposures'].iloc[0] for name, p in portfolios}
    contributions = {name: calculate_factor_contributions(portfolio, exposures[name]) for name, portfolio in portfolios}
    
    fig = plot_portfolio_comparison(exposures, contributions, name_map)
    
    output.markdown("#### Factor Analysis ####")
    col1, col2, col3 = output.columns([0.05, 0.8, 0.15])
    col2.pyplot(fig, use_container_width = False)
    
def style_table(df, format_map = None):
    styled = df.style
    if format_map:
        styled = styled.format(format_map)
        
    return styled.set_table_attributes("class = 'styled-table'").to_html(index = True)

def sort_portfolios(portfolios):
    desired_order = ['max_SR', 'constrained_max_SR', 'min_vol', 'constrained_min_vol', 'custom']
    sorted_portfolios = sorted(
        portfolios.items(),
        key = lambda x: (desired_order.index(x[0]) if x[0] in desired_order else len(desired_order))
        )
    
    return sorted_portfolios

def generate_distinct_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return list(RGB_tuples)