# Portfolio
[Web app](https://www.emanuelmalek.com/quant_projects/portfolio_optimiser.html) for optimising portfolios using Modern Portfolio Theory, aka mean-variance optimisation, and Factor Analysis.

## Functionalities: ##
- Optimise portfolios in a stock basket, i.e.
  - minimise volatility for a given excess return,
  or
  - maximise excess returns with volatility below a threshold.
- Find portfolios maximising Sharpe Ratio and mininimising volatility.
- Calculate and display the efficient frontier of optimal portfolios.
- Run a Factor Analysis, currently supporting
  - Fama-French 3-factor model
  - Fama-French 5-factor model
  - Carhart 4-factor model
  - Fama-French 5-factor model + Momentum
- Use Factor Analysis to enhance the estimation of the stocks' covariance matrix.
- Analyse portfolios with the Factor Analysis to find factor exposures (beta) and returns attributable to each factor.
- Impose constraints on the factor exposure of your portfolios and visualise the differences in their performance and the efficient frontier.

## How to run
The web app is built using streamlit. After pip installing streamlit, you can launch the web app by running
```
streamlit run main.py
```
