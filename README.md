# Momentum Strategy Enhancement Project

This project implements and enhances the momentum strategy outlined in the seminal paper "Returns to Buying Winners and Selling Losers" by Jegadeesh and Titman (1993). The strategy is replicated using U.S. stock market data from 2004 to 2024. Additionally, machine learning techniques are applied to improve the performance of the traditional momentum approach, offering insights into potential enhancements in predictive accuracy and return optimization.

## Assumptions

In their study "Market Dynamics and Momentum Profits" by Gloria Yuan Tian et al. document that higher momentum returns are observed when markets remain in the same state compared to transitions to different states. Furthermore, the paper "Enhanced Momentum Strategies",  by Pedro Barroso et al. compares three enhanced momentum strategies based on volatility scaling, which is grounded in the empirical observation that returns tend to be lower during periods of high volatility. Consequently, a strategy has been developed to predict increases in volatility prior to trend transitions.

Based on these findings, four variability features were extracted across five time periods, resulting in a total of 20 features:

- **Open Price**: Monthly standard deviation normalized by its average.
- **Volume**: Monthly standard deviation normalized by its average.
- **Price Fluctuations**: The difference between the closing and opening prices of each day.
- **Sharpe Ratio**: The Sharpe ratio calculated for the relevant time period.

## Enhanced Momentum Strategy

The enhanced momentum strategy builds upon the original momentum strategy by incorporating a decision-making process for buying stocks in the top decile and short-selling stocks in the lower decile. This decision-making process utilizes the aforementioned 20 features to determine whether to buy or sell a stock. 

Two separate models were trained: one for buying and another for selling. The buying model was trained using two types of data that indicated whether the price of a stock designated for purchase would rise (indicating a buy) or fall (indicating not to buy).
