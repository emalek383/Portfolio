import streamlit as st 

state = st.session_state

def initialise_portfolio_state():
    if 'portfolios' not in state:
        state.portfolios = {
            'custom': None,
            'sample_cov': {
                'max_sharpe': None,
                'min_vol': None,
                'factor_constrained': {
                    'max_sharpe': None,
                    'min_vol': None
                }
            },
            'factor_cov': {
                'max_sharpe': None,
                'min_vol': None,
                'factor_constrained': {
                    'max_sharpe': None,
                    'min_vol': None
                }
            }
        }
    
    if 'efficient_frontiers' not in state:
        state.efficient_frontiers = {
            'sample_cov': {
                'unconstrained': None,
                'factor_constrained': None
            },
            'factor_cov': {
                'unconstrained': None,
                'factor_constrained': None
            }
        }
        
def update_portfolio(portfolio_type, portfolio, cov_type = 'sample_cov', constrained = False):
    if portfolio_type == 'custom':
        state.portfolios['custom'] = portfolio
    elif cov_type:
        if constrained:
            state.portfolios[cov_type]['factor_constrained'][portfolio_type] = portfolio
        else:
            state.portfolios[cov_type][portfolio_type] = portfolio
    else:
        raise ValueError("Covariance type must be specified for non-custom portfolios")
        
def get_portfolio(portfolio_type, cov_type = 'sample_cov', constrained = False):
    if portfolio_type == 'custom':
        return state.portfolios['custom']
    elif cov_type:
        if constrained and state.factor_bounds:
            return state.portfolios[cov_type]['factor_constrained'][portfolio_type]
        return state.portfolios[cov_type][portfolio_type]
    else:
        raise ValueError("Covariance type must be specified for non-custom portfolios")
        
def iterate_portfolios(cov_type = 'sample_cov', include_custom = True, include_constrained = True, sort = True):
    portfolios = {}
    if include_custom and state.portfolios['custom'] is not None:
        portfolios['custom'] = state.portfolios['custom']
    
    portfolios.update({
        key: value for key, value in state.portfolios[cov_type].items()
        if key != 'factor_constrained' and value is not None
    })
    
    if include_constrained:
        portfolios.update({
            f"constrained_{key}": value 
            for key, value in state.portfolios[cov_type]['factor_constrained'].items()
            if value is not None
        })
    
    if sort:
        desired_order = ['max_sharpe', 'constrained_max_sharpe', 'min_vol', 'constrained_min_vol', 'custom']
        sorted_portfolios = sorted(
            portfolios.items(),
            key=lambda x: (desired_order.index(x[0]) if x[0] in desired_order else len(desired_order))
        )
        return sorted_portfolios
    else:
        return list(portfolios.items())
    
def update_efficient_frontier(eff_data, cov_type = 'sample_cov', constrained = False):
    key = 'factor_constrained' if constrained else 'unconstrained'
    state.efficient_frontiers[cov_type][key] = eff_data

def get_efficient_frontier(cov_type = 'sample_cov', constrained = False):
    key = 'factor_constrained' if constrained else 'unconstrained'
    return state.efficient_frontiers[cov_type][key]
    
def clear_factor_constrained_data():
    for cov_type in ['sample_cov', 'factor_cov']:
        state.portfolios[cov_type]['factor_constrained'] = {
            'max_sharpe': None,
            'min_vol': None
            }
        
        state.efficient_frontiers[cov_type]['factor_constrained'] = None
        
def clear_factor_cov_data():
    state.portfolios['factor_cov'] = {
        'max_sharpe': None,
        'min_vol': None,
        'factor_constrained': {
            'max_sharpe': None,
            'min_vol': None
            }
        }
    state.efficient_frontiers['factor_cov'] = {
        'unconstrained': None,
        'constrained': None
        }
    
def clear_all_portfolio_data():
    state.portfolios = {
        'custom': None,
        'sample_cov': {
            'max_sharpe': None,
            'min_vol': None,
            'factor_constrained': {
                'max_sharpe': None,
                'min_vol': None
            }
        },
        'factor_cov': {
            'max_sharpe': None,
            'min_vol': None,
            'factor_constrained': {
                'max_sharpe': None,
                'min_vol': None
            }
        }
    }
    state.efficient_frontiers = {
        'sample_cov': {
            'unconstrained': None,
            'factor_constrained': None
        },
        'factor_cov': {
            'unconstrained': None,
            'factor_constrained': None
        }
    }
    
def clear_sample_cov_data():
    state.portfolios['sample_cov'] = {
        'max_sharpe': None,
        'min_vol': None,
        'factor_constrained': {
            'max_sharpe': None,
            'min_vol': None
        }
    }
    state.efficient_frontiers['sample_cov'] = {
        'unconstrained': None,
        'factor_constrained': None
    }