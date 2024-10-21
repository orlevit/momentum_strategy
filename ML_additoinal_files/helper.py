import os
import pickle
import pandas as pd
import yfinance as yf
import requests
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from sklearn.metrics import recall_score, precision_score, f1_score

from config import STOCK_TIME, FORMATION_PERIOD_MONTHS, HOLDING_PERIOD_MONTHS, TOP_DECILE, DATA_POS_LOC, IMG_DIR_LOC,GROUP_LABELS, GROUP_LABELS_TO_STR


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


def calc_variability_per_stocks_per_type(run_ind, date, features_dict, pos, block, stocks_type, group_labels, add_info_name, risk_free_rate):
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
            sharpe_ratio = float((annual_return - risk_free_rate) / volatility)
            prices_flunc_p = float(np.std(block['Close'][stock] - block['Open'][stock]) / np.mean(block['Close'][stock] - block['Open'][stock]))
            volatility_p = float(np.std(block['Open'][stock]) / np.mean(block['Open'][stock]))
            volume_p = float(np.std(block['Volume'][stock]) / np.mean(block['Volume'][stock]))

            if run_ind in features_dict:
                features_dict[run_ind][f'sharpe_ratio_{add_info_name}'] = sharpe_ratio
                features_dict[run_ind][f'open_p_{add_info_name}'] = volatility_p
                features_dict[run_ind][f'prices_flunc_p_{add_info_name}'] = prices_flunc_p
                features_dict[run_ind][f'volume_p_{add_info_name}'] = volume_p
        
            else:  
                features_dict[run_ind] = {'date': date,\
                                          'stock': stock,\
                                          f'sharpe_ratio_{add_info_name}':sharpe_ratio, \
                                          f'open_p_{add_info_name}': volatility_p,\
                                          f'prices_flunc_p_{add_info_name}': prices_flunc_p,\
                                          f'volume_p_{add_info_name}': volume_p,\
                                          'label': group_labels[stocks_type]}
            run_ind +=1 
            

def calc_variability_per_stocks(price_data_df, open_positions, risk_free_rate, group_labels):
    stocks_types = group_labels.keys()
    features_dict = {}

    for i, pos in open_positions.items():
        for stocks_type in stocks_types:
            run_ind = len(features_dict)
            date = pos['date']
            
            block = price_data_df.iloc[i:i + FORMATION_PERIOD_MONTHS + 1]
            block12 = block[:-1]
            
            calc_variability_per_stocks_per_type(run_ind, date, features_dict, pos, block12, stocks_type, group_labels, '12',risk_free_rate)
            for quarter_i in range(4):
                block4 = block12[quarter_i: quarter_i + 4]
                calc_variability_per_stocks_per_type(run_ind, date, features_dict, pos, block4, stocks_type, group_labels, str(quarter_i + 1), risk_free_rate)
            
    return features_dict



def get_risk_free_rate(start_date, end_date):
    treasury_bond_yield = yf.download("^TNX", start=start_date, end=end_date, interval='1d')

    # Calculate annual average yield (Risk-Free Rate)
    annual_risk_free_rate = treasury_bond_yield[STOCK_TIME].resample('Y').mean()
        
    return annual_risk_free_rate

def plot_label_frequency(df, group_labels_to_str, name, split_name):
    # Map the label numbers to their corresponding names
    df['label_name'] = df['label'].map(group_labels_to_str)
    
    # Count the frequency of each label
    label_counts = df['label_name'].value_counts()
    
    # Calculate the percentage for each label
    total_count = label_counts.sum()
    percentages = (label_counts / total_count) * 100

    # Create the bar plot
    ax = label_counts.plot(kind='bar', color='skyblue', edgecolor='black', figsize=(10, 6))

    # Add percentages and counts on top of each bar
    for i, count in enumerate(label_counts):
        percentage = percentages[i]
        ax.text(i, count + 0.5, f'{count} ({percentage:.1f}%)', ha='center', va='bottom', fontsize=10)

    # Set plot title and labels
    plt.title(f'{split_name}\nLabel Frequencies and Percentages')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_path = os.path.join(IMG_DIR_LOC, f'{name}_{split_name}.png')
    plt.savefig(save_path, format='png', dpi=300)
    plt.show()


def populate_with_model(model, price_df, date, s_stocks_list, rfr, momentum_group):
    stocks_list = []
    for stock in s_stocks_list:
        stock_features = calc_variability_per_stocks(model, price_df, date, stock, rfr, GROUP_LABELS)
        pred_group = int(best_estimator.predict(stock_features)[0])

        if momentum_group == pred_group:
            stocks_list.append(losers_stock)

    return stocks_list


def split_train_test(df, train_range, name):
    
    train_df = df.iloc[:train_range]
    test_df = df.iloc[train_range:]
    print(f'{name} Data')
    print(f'Train length: {train_df.shape}, Test length: {test_df.shape}')
    
    plot_label_frequency(train_df, GROUP_LABELS_TO_STR, name, 'Train')
    plot_label_frequency(test_df, GROUP_LABELS_TO_STR, name, 'Test')

    return train_df, test_df


def measures_results(model, test_df):
    exclude_columns = ['date', 'label_name','stock', 'label']
    clean_test_df = test_df.loc[:, ~test_df.columns.isin(exclude_columns)]
    
    test_df['prediction'] = model.predict(clean_test_df)
    test_df['label'] = test_df['label'].astype(int)
    test_df['prediction'] = test_df['prediction'].astype(int)  
    
    r_score = recall_score(test_df['prediction'], test_df['label'] , average = 'weighted')
    p_score = precision_score(test_df['prediction'], test_df['label'] , average = 'weighted')
    f1 = f1_score(test_df['prediction'].values, test_df['label'].values, average='weighted')

    print(f'Precision: {p_score}\nRecall: {f1}\nf1_score: {f1}\n')

def print_feature_importance(df, importances, name):
    
    exclude_columns = ['date', 'label_name','stock','label']
    clean_df = df.loc[:, ~df.columns.isin(exclude_columns)]
    feature_importance_df = pd.DataFrame({
            'Feature': clean_df.columns,
            'Importance': importances
        })
        
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(name)
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.gca().invert_yaxis() 
    save_path = os.path.join(IMG_DIR_LOC, f'{name}_model_feature_importance.png')
    plt.savefig(save_path, format='png', dpi=300)
    plt.show()