import os
import pickle
import pandas as pd
import yfinance as yf
import requests
import numpy as np
from bs4 import BeautifulSoup
from config import STOCK_TIME, FORMATION_PERIOD_MONTHS, HOLDING_PERIOD_MONTHS, TOP_DECILE, DATA_POS_LOC


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


def adjust_data(loc, col_name):
    df = pd.read_csv(loc)
    date_index = df['Date']
    df.drop(['Date'],  axis=1, inplace=True)
    df.columns = pd.MultiIndex.from_product([[col_name], df.columns])
    return df


def collecting_data(start_date, end_date, file_open_loc, file_close_loc, file_high_loc, file_low_loc, file_vol_loc, file_div_loc):
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

    if os.path.isfile(file_open_loc):
        date_index = pd.read_csv(file_open_loc)['Date']
    
        df_open = adjust_data(file_open_loc, 'Open')
        df_close = adjust_data(file_close_loc, 'Close')
        df_low = adjust_data(file_low_loc, 'Low')
        df_high = adjust_data(file_high_loc, 'High')
        df_vol = adjust_data(file_vol_loc, 'Volume')
        df_div = adjust_data(file_div_loc, 'Dividends')

        result_df = pd.concat([df_open,df_close,df_low,df_high,df_vol,df_div], axis=1)
        result_df['Date'] = date_index
        result_df.set_index('Date', inplace=True)

        return result_df
        
    else:
        # Get the list of S&P 500 companies for the year 2004
        companies_2004 = get_sp500_companies_active_in_year(start_date, end_date)
        price_data = yf.download(companies_2004, start=start_date, end=end_date, interval='1mo', actions =True)[['Open','Close','Volume','High', 'Low','Dividends']]
        num_columns_with_nan = price_data.isnull().any().sum()
        print(f'The number of columns that have nan values: {num_columns_with_nan}, and are been dropped.')
        price_data = price_data.dropna(axis=1)
        price_data['Dividends'].to_csv(file_div_loc)
        price_data['Open'].to_csv(file_open_loc)
        price_data['Close'].to_csv(file_close_loc)
        price_data['Low'].to_csv(file_low_loc)        
        price_data['High'].to_csv(file_high_loc)
        price_data['Volume'].to_csv(file_vol_loc)


        return price_data


def momentum_strategy_stat(price_data_df, relative_df):
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
                                  'ini_stock_val': 2 * winners_stocks_value,
                                  'w_stocks_lower': {'code': []},
                                  'w_stocks_higher': {'code': []},
                                  'l_stocks_lower': {'code': []},
                                  'l_stocks_higher': {'code': []}                                 
                                 }                                  
    
        if HOLDING_PERIOD_MONTHS < ii:    
            prev_loc = ii - HOLDING_PERIOD_MONTHS
            prev_date = open_positions[prev_loc]['date']            
            winners_stocks = open_positions[prev_loc]['w_stocks']
            losers_stocks = open_positions[prev_loc]['l_stocks']
            balance_losers = open_positions[prev_loc]['balance_losers']

            for stock in winners_stocks:
                curr_price = price_data_df.loc[prev_date, STOCK_TIME][stock]
                prev_price = price_data_df.loc[date, STOCK_TIME][stock]

                if prev_price <= curr_price:
                    open_positions[prev_loc]['w_stocks_higher']['code'].append(stock)
                else:
                    open_positions[prev_loc]['w_stocks_lower']['code'].append(stock)                    

            for stock in losers_stocks:
                curr_price = price_data_df.loc[prev_date, STOCK_TIME][stock]
                prev_price = price_data_df.loc[date, STOCK_TIME][stock]

                if prev_price <= curr_price:
                    open_positions[prev_loc]['l_stocks_higher']['code'].append(stock)
                else:
                    open_positions[prev_loc]['l_stocks_lower']['code'].append(stock)                    
                    
    return open_positions


def calc_variability_in_stocks_per_type(pos, block, stocks_type, add_info_name, risk_free_rate):
        pos[stocks_type][f'sharpe_ratio_{add_info_name}'] = []
        pos[stocks_type][f'open_p_{add_info_name}'] = []
        pos[stocks_type][f'prices_flunc_p_{add_info_name}'] = []
        pos[stocks_type][f'volume_p_{add_info_name}'] = []

        sharpe_ratio_list = []
        open_p_list = []
        prices_flunc_p_list = []
        volume_p_list = []
        
        for stock in pos[stocks_type]['code']:
            annual_return = ((block['Open'][stock].iloc[-1] + block['Dividends'][stock].sum(axis=0) - block['Open'][stock].iloc[0]) / block['Open'][stock].iloc[0]) - 1
            volatility = np.std(block['Open'][stock])
            sharpe_ratio = (annual_return - risk_free_rate) / volatility
            
            prices_flunc_p = np.std(block['Close'][stock] - block['Open'][stock]) / np.mean(block['Close'][stock] - block['Open'][stock])        
            volatility_p = np.std(block['Open'][stock]) / np.mean(block['Open'][stock])
            volume_p = np.std(block['Volume'][stock]) / np.mean(block['Volume'][stock])
            
            sharpe_ratio_list.append(sharpe_ratio)
            open_p_list.append(volatility_p)
            prices_flunc_p_list.append(prices_flunc_p)
            volume_p_list.append(volume_p)
    
        pos[stocks_type][f'sharpe_ratio_{add_info_name}'].append(float(np.mean(sharpe_ratio_list)))
        pos[stocks_type][f'open_p_{add_info_name}'].append(float(np.mean(open_p_list)))
        pos[stocks_type][f'prices_flunc_p_{add_info_name}'].append(float(np.mean(prices_flunc_p_list)))
        pos[stocks_type][f'volume_p_{add_info_name}'].append(float(np.mean(volume_p_list)))


def calc_variability_in_stocks(price_data_df, open_positions, risk_free_rate):
    stocks_types = ['w_stocks_higher', 'w_stocks_lower', 'l_stocks_higher', 'l_stocks_lower']
    
    for stocks_type in stocks_types:
        for i, pos in open_positions.items():
        
            block = price_data_df.iloc[i:i + FORMATION_PERIOD_MONTHS + 1]
            block12 = block[:-1]

            calc_variability_in_stocks_per_type(pos, block12, stocks_type, '12',risk_free_rate)
            for quarter_i in range(4):
                block4 = block12[quarter_i: quarter_i + 4]
                calc_variability_in_stocks_per_type(pos, block4, stocks_type, str(quarter_i + 1), risk_free_rate)


def create_stat(price_data_df, relative_df, risk_free_rate):
    if os.path.isfile(DATA_POS_LOC):
        with open(DATA_POS_LOC, 'rb') as handle:
            positions = pickle.load(handle)

    else:
        positions = momentum_strategy_stat(price_data_df, relative_df)
        calc_variability_in_stocks(price_data_df, positions, risk_free_rate)
                        
        with open(DATA_POS_LOC, 'wb') as handle:
            pickle.dump(positions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return positions

def extract_period_name(metric_name_raw, winner_losers):
    period_name_period = metric_name_raw.split('_')[-1]
    win_lose = winner_losers.split('_')[0]
    
    if win_lose == 'w':
        wl_name = 'Winners'
    if win_lose == 'l':
        wl_name = 'Losers'                
        
    if period_name_period == '12':
        period_name = 'One year'
    if period_name_period == '1':
        period_name =  'First quarter'
    if period_name_period == '2':
        period_name =  'Second quarter'
    if period_name_period == '3':
        period_name =  'Third quarter'
    if period_name_period == '4':
        period_name =  'Fourth quarter'
    
    if metric_name_raw.startswith('sharpe_ratio'):
        metric_name = 'Sharpe Ratio'
    if metric_name_raw.startswith('open_p'):
        metric_name = 'Open Price'
    if metric_name_raw.startswith('prices_flunc_p'):
        metric_name = 'Price Fluctuations'
    if metric_name_raw.startswith('volume_p'):
        metric_name = 'Volume'     
        
    return period_name, wl_name, metric_name

def get_risk_free_rate(start_date, end_date):
    treasury_bond_yield = yf.download("^TNX", start=start_date, end=end_date, interval='1d')

    # Calculate annual average yield (Risk-Free Rate)
    annual_risk_free_rate = treasury_bond_yield[STOCK_TIME].resample('Y').mean()
        
    return annual_risk_free_rate