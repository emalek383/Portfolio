import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from process_forms import process_stock_form

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
    
    stock_string = ", ".join(state.universe.stocks)
    display.write(f"Analysing portfolios consisting of {stock_string} using data from {state.universe.start_date.strftime('%d/%m/%Y')} to {state.universe.end_date.strftime('%d/%m/%Y')}.")
    display.write(f'Risk free rate is {state.universe.risk_free_rate:.2%}.')
    
    display.markdown("### Max Sharpe Ratio, Min Vol and Custom Portfolios ###")
    display_portfolio_performances(display, state.portfolios)
    display_portfolio_weights(display, state.portfolios, incl_table = True)
    
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
        
    display.markdown("### Efficient Frontier ###")
    
    if not state.eff_frontier:
        display.write("Need at least two stocks to have a viable portfolio.")
        return display
        
    eff_vols = state.eff_frontier[0]
    eff_excess_returns = state.eff_frontier[1]
        
    my_portfolio_results = [(portfolio.vol, portfolio.excess_returns) for portfolio in state.portfolios.values()]

    vols, excess_returns = state.universe.individual_stock_portfolios()

    plt.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots()
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Excess Return')
    ax.set_title('Excess Returns vs Volatility for Portfolios')

    markers = ['*', 'x', 'x']
    markersizes = [12, 10, 10]
    colors = ['green', 'green', 'blue']

    ax.plot(eff_vols, eff_excess_returns, '-', markersize = 3, color = 'red')

    for idx, (portfolio_vol, portfolio_excess_returns) in enumerate(my_portfolio_results):
        ax.plot(portfolio_vol, portfolio_excess_returns, markers[idx], markersize = markersizes[idx], color = colors[idx])
        
    ax.plot(vols, excess_returns, 'o', markersize = 5, color = 'blue')
    for idx, ticker in enumerate(state.universe.stocks):
        ax.annotate(ticker, (vols[idx], excess_returns[idx]))
    ax.legend(['Efficient Frontier', 'Max Sharpe Ratio Portfolio', 'Minimal Variance Portfolio', 'Custom Portfolio'])
        
    display.pyplot(fig)
    
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
    for portfolio in portfolios.values():
        perf_df, format_map = portfolio.get_performance_df()
        joined_perf.append(perf_df)
        
    joined_perf_df = pd.concat(joined_perf, axis = 0)
    output.table(joined_perf_df.style.format(format_map))
    
def display_portfolio_weights(output, portfolios, incl_table = False):
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
    num_portfolios = len(portfolios.keys())
    fig_width = 2.5 * num_portfolios
    fig_height = 2.5
        
    fig, axes = plt.subplots(1, num_portfolios, figsize=(fig_width, fig_height))
    stocks = list(portfolios.values())[0].universe.stocks

    for idx, portfolio in enumerate(portfolios.values()):
        wedges, _, _ = axes[idx].pie(portfolio.weights,
                                     autopct = '%1.1f%%',
                                     pctdistance=0.75,
                                     textprops={'fontsize': 8})
        axes[idx].set_title(portfolio.name + ' Portfolio', fontsize = 8)
            
    fig.legend(wedges, stocks, loc='center left', bbox_to_anchor=(1, 0.5), title="Stocks")
        
    plt.tight_layout()
    plt.subplots_adjust(right=1)
        
    output.pyplot(fig, use_container_width=True)
    
    if incl_table:
        joined_weights = []
        for portfolio in portfolios.values():
            weights_df, format_map = portfolio.get_weights_df()
            joined_weights.append(weights_df)
        
        joined_weights_df = pd.concat(joined_weights, axis = 0)
        
        table_expander = output.expander("Allocation Details", expanded = False)
        table_expander.table(joined_weights_df.style.format(format_map))