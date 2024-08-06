"""Functions plotting efficient frontier, portfolio allocations, and factor returns contributions / exposures."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_interactive_efficient_frontier(eff_frontier_data, constrained_eff_frontier_data = None, sorted_portfolios = None, stock_data = None, is_mobile = False):
    """
    Create an interactive plot showing the efficient frontier, the constrained efficient frontier (if it exists),
    the individual portfolios (max sharpe, min vol, custom and also constrained max sharpe, constrained min vol,
    if they exist) and the stocks of the stock basket.
    Include features to make it more mobile user friendly.


    Parameters
    ----------
    eff_frontier_data : tuple(list, list)
        Tuple of the efficient frontier volatilities and excess returns.
    constrained_eff_frontier_data : tuple(list, list), optional
        Tuple of the factor-constrained efficient frontier volatilities and excess returns. The default is None.
    sorted_portfolios : list(tuple(str, StockUniverse.Portfolio)), optional
        List of tuples (short name, portfolio) to be plotted. The default is None.
    stock_data : tuple(list, list, list), optional
        The tickers, volatilities and excess returns of the stocks in the universe. Arranged in a tuple of 3 lists. 
        default is None.
    is_mobile : bool, optional
        Whether the page is being viewed on a mobile device. True if viewed on mobile device, False if viewed on PC. The default is False.

    Returns
    -------
    fig : plotly.fig
        Figure to be plotted.

    """
    
    eff_vols, eff_excess_returns = eff_frontier_data
    
    constrained_eff_frontier = constrained_eff_frontier_data
        
    portfolio_results = {name: (portfolio.vol, portfolio.excess_returns) 
                         for name, portfolio in sorted_portfolios if portfolio}
    portfolio_order = [name for name, portfolio in sorted_portfolios]
    
    stock_names, vols, excess_returns = stock_data
    
    fig = make_subplots()
    
    # Add efficient frontier with hover information: vol, excess returns, sharpe
    ef_sharpe_ratios = [er/vol for er, vol in zip(eff_excess_returns, eff_vols)]
    fig.add_trace(go.Scatter(
        x=eff_vols, 
        y=eff_excess_returns, 
        mode='lines', 
        name='Efficient Frontier', 
        line=dict(color='red'),
        hovertemplate='<b>Efficient Frontier</b><br>Volatility: %{x:.2%}<br>Excess Return: %{y:.2%}<br>Sharpe Ratio: %{customdata:.2f}<extra></extra>',
        customdata=ef_sharpe_ratios,
        showlegend = False if is_mobile else True
    ))
    if is_mobile:
        fig.add_trace(go.Scatter(
            x=[None], 
            y=[None], 
            mode='lines', 
            name = 'Efficient Frontier',
            line=dict(color='red', width = 1),
            showlegend = True
        ))
    
    # Add constrained efficient frontier if it exists with hover information
    if constrained_eff_frontier:
        constrained_eff_vols, constrained_eff_excess_returns = constrained_eff_frontier
        constrained_ef_sharpe_ratios = [er/vol for er, vol in zip(constrained_eff_excess_returns, constrained_eff_vols)]
        fig.add_trace(go.Scatter(
            x=constrained_eff_vols, 
            y=constrained_eff_excess_returns, 
            mode='lines', 
            name= 'Constr. Eff. Frontier' if is_mobile else 'Constrained Efficient Frontier', 
            line=dict(color='orange', dash='dash'),
            hovertemplate='<b>Constrained Efficient Frontier</b><br>Volatility: %{x:.2%}<br>Excess Return: %{y:.2%}<br>Sharpe Ratio: %{customdata:.2f}<extra></extra>',
            customdata=constrained_ef_sharpe_ratios,
            showlegend = False if is_mobile else True
        ))
        if is_mobile:
            fig.add_trace(go.Scatter(
                x=[None], 
                y=[None], 
                mode='lines', 
                name= 'Constr. Eff. Frontier' if is_mobile else 'Constrained Efficient Frontier', 
                line=dict(color='orange', dash='dash', width = 1),
                showlegend = True
            ))
    
    # Settings for portfolios including to make mobile user friendly
    portfolio_plot_settings = {
        'max_sharpe': {'marker': 'star', 'size': 15, 'color': 'green', 'label': 'Max Sharpe Portfolio', 'legend_size': 8},
        'constrained_max_sharpe': {'marker': 'star', 'size': 15, 'color': 'orange', 'label': 'Constr. Max Sharpe Portfolio' if is_mobile else 'Constrained Max Sharpe Portfolio', 'legend_size': 8},
        'min_vol': {'marker': 'x', 'size': 15, 'color': 'green', 'label': 'Min Vol Portfolio', 'legend_size': 8},
        'constrained_min_vol': {'marker': 'x', 'size': 15, 'color': 'orange', 'label': 'Constr. Min Vol Portfolio' if is_mobile else 'Constrained Min Vol Portfolio', 'legend_size': 8},
        'custom': {'marker': 'x', 'size': 15, 'color': 'blue', 'label': 'Custom Portfolio', 'legend_size': 8}
    }
    
    # Add portfolios
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
                customdata=[portfolio_excess_returns/portfolio_vol],
                showlegend = False if is_mobile else True
            ))
            if is_mobile:
                fig.add_trace(go.Scatter(
                    x=[None], 
                    y=[None], 
                    mode='markers', 
                    name=settings['label'],
                    marker=dict(symbol=settings['marker'], size=settings['legend_size'], color=settings['color']),
                    showlegend = True
                ))
    
    # Add individual stocks
    fig.add_trace(go.Scatter(
        x=vols, 
        y=excess_returns, 
        mode='markers', 
        name='Stocks',
        marker=dict(size=8, color='blue'),
        text=stock_names,
        hovertemplate='<b>%{text}</b><br>Volatility: %{x:.2%}<br>Excess Return: %{y:.2%}<br>Sharpe Ratio: %{customdata:.2f}<extra></extra>',
        customdata=[excess_returns[i]/vols[i] for i in range(len(vols))],
        showlegend = False if is_mobile else True
    ))
    if is_mobile:
        fig.add_trace(go.Scatter(
            x=[None], 
            y=[None], 
            mode='markers', 
            name='Stocks',
            marker=dict(size=6, color='blue'),
            showlegend = True
        ))
    
    # Settings for plot including to make mobile friendly
    legend_font_size = 10 if is_mobile else 17
    axis_label_font_size = 10 if is_mobile else 20
    title_font_size = 16 if is_mobile else 24
    tickformat = '.2f' if is_mobile else '.2%'
    margin = dict(l = 0, r = 15, t = 50, b = 50) if is_mobile else dict(l = 50, r = 50, t = 50, b = 50)
    standoff = 4 if is_mobile else 8
    
    legend_settings = dict(
        font=dict(size=legend_font_size, color='black'),
        yanchor="top",
        y=1.01,
        xanchor="left",
        x=0.00,
    )
    if is_mobile:
        legend_settings.update({'bgcolor': 'rgba(0,0,0,0)', 'bordercolor': 'rgba(0,0,0,0)', 'borderwidth': 0})
    
    fig.update_layout(
        title={
            'text': 'Excess Returns vs Volatility for Portfolios',
            'font': {'size': title_font_size, 'color': 'black'}
            },
        legend=legend_settings,
        legend_tracegroupgap=1,
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

def generate_color_dict(portfolios, all_stocks, group_threshold):
    """
    Generate a colour dictionary for the stocks in a set of portfolios that can be used in plots.
    Ensures that when there are many stocks, we still have well-dstinguished colours.
    Stocks which are below the group_threshold in all portfolios are grouped together
    to keep the number of colours needed manageable.

    Parameters
    ----------
    portfolios : list(tuple(str, StockUniverse.Portfolio))
        List of tuples (short name, portfolio) of portfolios for which we will create colours (for a plot).
    all_stocks : list
        Tickers of all the stocks.
    group_threshold : float
        Stocks whose allocation in all portfolio fall below this cutoff only appear grouped 
        and will not get individual colours assigned.

    Returns
    -------
    dict
        Dictionary of colours to use in the format key: value = stock ticker: color.

    """
    
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

def plot_weights_pie_charts(portfolios, label_threshold, group_threshold, is_mobile = False):
    """
    Plot the asset allocations of portfolios using pie charts. Annotate the
    pie charts with the allocation % except for assets below the label_threshold
    and group together assets below group_threshold as "Others". Some special formatting for
    mobile devices. Use a colour scheme fixed to number of assets that will be shown in all
    portfolios and which can be used for other plots too.

    Parameters
    ----------
    portfolios : list(tuple(str, StockUniverse.Portfolio))
        List of tuples (short name, portfolio) of portfolios whose allocations will be displayed.
    label_threshold : float
        Assets with allocation below this threshold will not have their allocation annotated.
    group_threshold : float
        Stocks whose allocation in a portfolio fall below this cutoff will appear as "Others".
    is_mobile : bool, optional
        True if page is being viewed on mobile device, False if on PC. The default is False.

    Returns
    -------
    fig : matplotlib.fig
        Figure of the pie charts.

    """
    
    num_portfolios = len(portfolios)
    
    container_width = 1730
    fig_width = min(container_width / 96, 20)  # Convert pixels to inches, max width of 20 inches
    fig_height = fig_width * 0.5 * num_portfolios  # Adjust height based on number of portfolios
        
    fig, axes = plt.subplots(1, num_portfolios, figsize=(fig_width, fig_height), dpi = 200)
    if num_portfolios == 1:
        axes = [axes]
    
    all_stocks = portfolios[0][1].universe.stocks
    color_dict = generate_color_dict(portfolios, all_stocks, group_threshold)
    
    legend_elements = []
    legend_labels = []
    others_element = None
    others_label = None
    
    names_fontsize = 20 if is_mobile else 12
    pct_fontsize = 18 if is_mobile else 12
    others_fontsize = 18 if is_mobile else 12
    group_format = f"Others (<{group_threshold * 100:.0f}%)" if is_mobile else f"Others (<{group_threshold * 100:.1f}%)"

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
            weights.append(small_sum)
            stocks.append(group_format)
        
        # Sort weights and stocks
        sorted_indices = np.argsort(weights)[::-1]
        weights = [weights[i] for i in sorted_indices]
        stocks = [stocks[i] for i in sorted_indices]
            
        portfolio_colors = [color_dict.get(stock, '#DDDDDD') for stock in stocks]  # Use light grey for 'Others'
        
        if is_mobile:
            pct_formatting = lambda pct: f'{pct:.0f}%' if pct > label_threshold * 100 else ''
        else:
            pct_formatting = lambda pct: f'{pct:.1f}%' if pct > label_threshold * 100 else ''
        
        wedges, texts, autotexts = axes[idx].pie(weights,
                                                 colors = portfolio_colors,
                                                 autopct = pct_formatting,
                                                 pctdistance=0.75,
                                                 textprops={'fontsize': pct_fontsize}
                                                 )
        for i, (wedge, autotext) in enumerate(zip(wedges, autotexts)):
            if weights[i] >= label_threshold:
                autotext.set_visible(True)
            else:
                autotext.set_visible(False)
            
            if stocks[i] not in legend_labels and stocks[i] != group_format:
                legend_elements.append(wedge)
                legend_labels.append(stocks[i])
            elif stocks[i] == group_format and others_element is None:
                others_element = wedge
                others_label = stocks[i]
        
        title = portfolio.name + ' Portfolio'
        if is_mobile:
            title = title = portfolio.name + '\n Portfolio'
            title = title.replace('Constrained', 'Constr.')
        else:
            title = portfolio.name + ' Portfolio'
            
        axes[idx].set_title(title, fontsize = names_fontsize)
        
    # Add "Others" to the end of the legend if it exists
    if others_element is not None:
        legend_elements.append(others_element)
        legend_labels.append(others_label)
        
    num_cols = max(1, len(legend_labels) // 10)
    fig.legend(legend_elements, legend_labels, loc = 'center left', 
               bbox_to_anchor=(1, 0.5), title="Stocks", fontsize=others_fontsize,
               title_fontsize = others_fontsize, ncol=num_cols, columnspacing=1,
               handletextpad=0.5)
    
    plt.tight_layout()
    plt.subplots_adjust(right=1)
    
    return fig

def plot_weights_bar_chart(selected_portfolio, portfolios, group_threshold):
    """
    Plot a bar chart of portfolio asset allocations. Group together assets below group_threshold
    as "Others".  Use a colour scheme fixed to number of assets that will be shown in all 
    portfolios and which can be used for other plots too.

    Parameters
    ----------
    selected_portfolio : str
        Which portfolio to be displayed out of portfolios, or "Show all" to display all portfolios.
    portfolios : list(tuple(str, StockUniverse.Portfolio))
        List of tuples (short name, portfolio) of portfolios whose allocations will be displayed.
    group_threshold : float
        Stocks whose allocation in a portfolio fall below this cutoff will appear as "Others".

    Returns
    -------
    fig : matplotlib.fig
        Figure of plot.

    """
    all_stocks = portfolios[0][1].universe.stocks
    color_dict = generate_color_dict(portfolios, all_stocks, group_threshold)
    
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

    return fig

def add_smart_label(ax, rect, value, bottom, is_residual = False, cutoff = 9):
    """
    Add annotations in a nice way to a stacked bar chart. Don't annotate bars whose values are below a cutoff.

    Parameters
    ----------
    ax : matplotlib.ax
        Axis of the bar chart.
    rect : matplotlib.patches.Rectangle
        Rectangle denoting the bar that will be annotated.
    value : float
        Value corresponding to the bar that will be annotated.
    bottom : float
        Bottom of the bar that will be annotated.
    is_residual : bool, optional
        True if the bar corresponds to residual returns, False otherwise. The default is False.
    cutoff : float, optional
        Bars with values below the cutoff will not be annotated. The default is 9.

    Returns
    -------
    None.

    """
    
    height = rect[0].get_height()
    if abs(height) < cutoff:
        return

    x = rect[0].get_x() + rect[0].get_width() / 2
    if value >= 0:
        if is_residual:
            y = bottom + height / 2
        else:
            y = bottom - height / 2
    else:
        if is_residual:
            y = bottom - height / 2
        else:
            y = bottom + height/2
    label = f"{value:.1f}%"
    
    ax.text(x, y, label, ha='center', va='center', fontsize=8)

def plot_return_comparison(exposure_data, factor_contributions, name_map = None, factor_bounds = None, is_mobile = False):
    """
    Plot return contributions of factors for a set of portfolios as a stacked bar chart.
    Annotate the returns above a threshold. Make plot friendly to mobile users.

    Parameters
    ----------
    exposure_data : pd.DataFrame
        Dataframe containing the factor exposures of each portfolio. Columns = factors, rows = portfolios.
    factor_contributions : dict
        Dictionary containing the returns contributions of the factors for each portfolio.
        Key: Value = portfolio name: returns contribution as dict {factor: return contribution}
    name_map : dict, optional
        Dictionary mapping the short name to the full portfolio name, that can be displayed, for each portfolio.
        Key: Value = short name: full name. The default is None.
    factor_bounds : dict, optional
        Dictionary containing the constraints on factor exposures as 
        key: value = factor: [lower, upper] with lower/upper = None indicating no constraint.
        If no constraint at all on factor, it won't appear in the dictionary.
        The default is None.
    is_mobile : bool, optional
        True if being viewed on mobile device, otherwise false. The default is False.

    Returns
    -------
    fig : matplotlib.fig
        Figure of the returns bar chart.

    """
    
    fig, ax = plt.subplots(figsize = (10, 5))
    
    portfolio_names = exposure_data.index
    if name_map:
        portfolio_full_names = [name_map[name] for name in portfolio_names]
    else:
        portfolio_full_names = portfolio_names
    factors = exposure_data.columns
    x = np.arange(len(portfolio_names))
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {factor: default_colors[i % len(default_colors)] for i, factor in enumerate(factors)}
    color_map['Residual'] = 'darkgrey'

    bar_width = 0.4    
    
    def max_difference_in_inner_dict(nested_dict):
        max_diff = 0
        for inner_dict in nested_dict.values():
            values = list(inner_dict.values())
            diff = max(values) - min(values)
            if diff > max_diff:
                max_diff = diff
        return max_diff

    max_difference = max_difference_in_inner_dict(factor_contributions)
    
    cutoff = max_difference / 15 if is_mobile else max_difference / 15
    
    for i, portfolio_name in enumerate(portfolio_names):
        positive_bottom = 0
        negative_bottom = 0
        for factor in factors:
            value = factor_contributions[portfolio_name].get(factor, 0)
            if value >= 0:
                rect = ax.bar(x[i], value, bar_width, bottom = positive_bottom, label = factor if i == 0 else "", color = color_map[factor])
                positive_bottom += value
                add_smart_label(ax, rect, value, positive_bottom, cutoff = cutoff)
            else:
                rect = ax.bar(x[i], value, bar_width, bottom = negative_bottom, label = factor if i == 0 else "", color = color_map[factor])
                add_smart_label(ax, rect, value, negative_bottom, cutoff = cutoff)
                negative_bottom += value
                
        residual = factor_contributions[portfolio_name].get('Residual', 0)
        if residual >= 0:
            rect = ax.bar(x[i], residual, bar_width, bottom = positive_bottom, label = 'Residual' if i == 0 else '', color = color_map['Residual'])
            add_smart_label(ax, rect, residual, positive_bottom, is_residual = True, cutoff = cutoff)
        else:
            rect = ax.bar(x[i], residual, bar_width, bottom = negative_bottom, label = 'Residual' if i == 0 else '', color = color_map['Residual'])
            add_smart_label(ax, rect, residual, negative_bottom, cutoff = cutoff)
        
    ax.set_ylabel('Return Attribution (%)')
    ax.set_title('Percentage of Return Attributable to Factors Across Portfolios')
    ax.set_xticks(x)
    ax.set_xticklabels(portfolio_full_names, rotation = 45, ha = 'right')
    ax.legend(title = 'Factors', bbox_to_anchor = (1.05, 1), loc = 'upper left')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def plot_exposure_comparison(exposure_data, name_map = None, factor_bounds = None):
    """
    Plot the exposure of different portfolios to a set of factors as a bar chart.

    Parameters
    ----------
    exposure_data : pd.DataFrame
        Dataframe containing the factor exposures of each portfolio. Columns = factors, rows = portfolios.
    name_map : dict, optional
        Dictionary mapping the short name to the full portfolio name, that can be displayed, for each portfolio.
        Key: Value = short name: full name. The default is None.
    factor_bounds : dict, optional
        Dictionary containing the constraints on factor exposures as 
        key: value = factor: [lower, upper] with lower/upper = None indicating no constraint.
        If no constraint at all on factor, it won't appear in the dictionary.
        The default is None.

    Returns
    -------
    fig : matplotlib.fig
        Figure of the plot.

    """
    fig, ax = plt.subplots(figsize = (10, 5))
    
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