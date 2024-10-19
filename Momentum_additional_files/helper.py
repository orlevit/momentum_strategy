import os
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy import stats
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from config import STOCK_TIME, FORMATION_PERIOD_MONTHS, HOLDING_PERIOD_MONTHS, TOP_DECILE

def collecting_data(start_date, end_date, file_price_loc, file_div_loc):
    def fetch_sp500_companies():
        # Fetch the list of S&P 500 companies from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
    
        # Parse the table containing the S&P 500 companies
        table = soup.find('table', {'class': 'wikitable'})
        df = pd.read_html(str(table))[0]
    
        # Return the list of company symbols
        return df['Symbol'].tolist()
    
    def get_sp500_companies_active_in_year(start_date, end_date):
        # Get the current S&P 500 companies
        sp500_companies = fetch_sp500_companies()
        
        active_companies = []
        for symbol in sp500_companies:
            try:
                # Get historical data for the company
                company = yf.Ticker(symbol)
                historical_data = company.history(start=start_date, end=end_date)
                
                # Check if data exists for that year
                if not historical_data.empty:
                    active_companies.append(symbol)
            except Exception as e:
                # print(f"Could not retrieve data for {symbol}: {e}")
                pass
    
        return active_companies

    if os.path.isfile(file_price_loc) and os.path.isfile(file_div_loc):
        df_price = pd.read_csv(file_price_loc)
        df_div = pd.read_csv(file_div_loc)

        date_index = df_price['Date']
        df_price.drop(['Date'],  axis=1, inplace=True)
        df_div.drop(['Date'],  axis=1, inplace=True)

        df_price.columns = pd.MultiIndex.from_product([[STOCK_TIME], df_price.columns])
        df_div.columns = pd.MultiIndex.from_product([['Dividends'], df_div.columns])
        
        # Concatenate along columns (axis=1)
        result_df = pd.concat([df_price, df_div], axis=1)
        
        result_df['Date'] = date_index
        result_df.set_index('Date', inplace=True)

        return result_df
        
    else:
        # Get the list of S&P 500 companies for the year 2004
        companies_2004 = get_sp500_companies_active_in_year(start_date, end_date)
        price_data = yf.download(companies_2004, start=start_date, end=end_date, interval='1mo', actions =True)[['Open','Dividends']]
        num_columns_with_nan = price_data.isnull().any().sum()
        print(f'The number of columns that have nan values: {num_columns_with_nan}, and are been dropped.')
        price_data = price_data.dropna(axis=1)
        price_data[STOCK_TIME].to_csv(file_price_loc)
        price_data['Dividends'].to_csv(file_div_loc)

        return price_data


def get_relative(df):
    relative_df = pd.DataFrame(columns=df[STOCK_TIME].columns.values)
    
    for i in range(len(df) - FORMATION_PERIOD_MONTHS ):
        block = df.iloc[i:i + FORMATION_PERIOD_MONTHS + 1] # Add row to get the date index
        block_12 = block[:-1]
        date_index = block.index[-1]
        diff = (block_12[STOCK_TIME].iloc[-1] + block_12['Dividends'].sum(axis=0) - block_12[STOCK_TIME].iloc[0]) / block_12[STOCK_TIME].iloc[0]
        row_df = pd.DataFrame(diff).T
        row_df.index = [date_index]
        relative_df = pd.concat([relative_df, row_df])
    return relative_df


def calc_one_sided_test(open_positions):
    portfolios_gain = []
    for k, v in open_positions.items():
        portfolios_gain.append(v['gained_valued'])
    
    null_mean = 0 
    t_statistic, p_value = stats.ttest_1samp(portfolios_gain, 0)
    p_value_one_tailed = p_value / 2
    
    return np.mean(portfolios_gain), p_value_one_tailed

# Annual risk_free rate using the 10 Year Treasury Bond Yield
def get_risk_free_rate(start_date, end_date):
    treasury_bond_yield = yf.download("^TNX", start=start_date, end=end_date, interval='1d')

    # Calculate annual average yield (Risk-Free Rate)
    annual_risk_free_rate = treasury_bond_yield[STOCK_TIME].resample('Y').mean()
        
    return annual_risk_free_rate

def calc_profolios_dates(begin_date, end_date, formation_period, holding_period):
    b_date_obj = datetime.strptime(begin_date, "%Y-%m-%d")
    e_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    b_new_date = b_date_obj + relativedelta(months=formation_period - 1)
    e_new_date = e_date_obj - relativedelta(months=holding_period)
    
    profolios_begin_time = b_new_date.strftime("%Y-%m-%d")
    profolios_end_time = e_new_date.strftime("%Y-%m-%d")
    
    return profolios_begin_time, profolios_end_time
    
def calc_end_profolios_date(date):
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    new_date = date_obj - relativedelta(months=HOLDING_PERIOD_MONTHS + 1)
    profolios_end_time = new_date.strftime("%Y-%m-%d")
    return profolios_end_time


def momentum_strategy(price_data_df, relative_df):
    
    open_positions = {}
    total_gained_valued = 0
    
    
    # Loop over each month in the data
    for ii, date in enumerate(relative_df.index, 1):
    
        if ii <= len(relative_df) - HOLDING_PERIOD_MONTHS: 
            # Get the returns for the past formation period
            past_returns = relative_df.loc[date]
            
            # Rank stocks based on past returns
            ranked_stocks = past_returns.rank(ascending=True)
        
            # Define deciles
            losers_stocks = ranked_stocks[ranked_stocks <= ranked_stocks.quantile(1/TOP_DECILE)].index.values.tolist()
            winners_stocks = ranked_stocks[ranked_stocks >= ranked_stocks.quantile(1 - (1/TOP_DECILE))].index.values.tolist()
            
            winners_stocks_value = price_data_df.loc[date, STOCK_TIME][winners_stocks].sum()
            losers_stocks_value = price_data_df.loc[date, STOCK_TIME][losers_stocks].sum()
    
            balance_losers =  winners_stocks_value / losers_stocks_value
                
            open_positions[ii] = {'date':date,
                                  'w_stocks': winners_stocks, 
                                  'l_stocks': losers_stocks,
                                  'balance_losers':balance_losers,
                                  'ini_stock_val': 2 * winners_stocks_value}

        if HOLDING_PERIOD_MONTHS < ii:    
            prev_loc = ii - HOLDING_PERIOD_MONTHS
            winners_stocks = open_positions[prev_loc]['w_stocks']
            losers_stocks = open_positions[prev_loc]['l_stocks']
            balance_losers = open_positions[prev_loc]['balance_losers']
    
            winners_stocks_value = price_data_df.loc[date,STOCK_TIME][winners_stocks].sum()
            losers_stocks_value = price_data_df.loc[date, STOCK_TIME][losers_stocks].sum()
    
            gained_valued = winners_stocks_value - (losers_stocks_value * balance_losers)
            base_gain = open_positions[prev_loc]['ini_stock_val']
            
            open_positions[prev_loc]['rel_gained'] =  (gained_valued +  base_gain) / base_gain
            open_positions[prev_loc]['gained_valued'] = gained_valued
    
            # print(prev_loc,base_gain,gained_valued,(gained_valued +  base_gain) / base_gain)
            total_gained_valued += gained_valued
            # print(date, total_gained_valued)
    
    return open_positions, total_gained_valued


def plot_profolios(open_positions):
    total_gained_valued_series = pd.Series({v['date']: v['gained_valued'] for v in open_positions.values()})
    total_gained_valued_series.index = pd.to_datetime(total_gained_valued_series.index)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(total_gained_valued_series.index, total_gained_valued_series.values, marker='o', linestyle='-')
    plt.title('profolios gains Over Time')
    plt.xlabel('Date')
    plt.ylabel('Gained Value')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout() 
    plt.show()


def calculate_metrics(strategy_returns, sp500_returns, risk_free_rate):
    # Annualized Return (Assume monthly returns are used)
    annualized_return = np.prod(1 + strategy_returns) ** (12 / len(strategy_returns)) - 1
    
    # Volatility (Annualized)
    volatility = strategy_returns.std() * np.sqrt(12)  # 12 months in a year
    
    # Sharpe Ratio 
    sharpe_ratio = (annualized_return - risk_free_rate.mean()) / volatility
    
    # Max Drawdown
    cumulative_returns = (1 + strategy_returns).cumprod()
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    
    # Value at Risk (VaR) at 1% and 5%
    VaR_1 = np.percentile(strategy_returns, 1)
    VaR_5 = np.percentile(strategy_returns, 5)
    
    # Alpha against S&P 500 (CAPM)
    alpha = ((1 + annualized_return) / (1 + sp500_returns.mean()) - 1) * 12

    return {
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'VaR 1%': VaR_1,
        'VaR 5%': VaR_5,
        'Alpha': alpha}