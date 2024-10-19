import os
import pandas as pd
import yfinance as yf
import requests
import numpy as np

from config import STOCK_TIME, FORMATION_PERIOD_MONTHS, HOLDING_PERIOD_MONTHS, TOP_DECILE


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