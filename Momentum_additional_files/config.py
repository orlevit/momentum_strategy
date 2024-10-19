import os 
DATA_DIR_LOC = os.path.join(os.getcwd(), 'data_dir')
DATA_DIV_LOC = os.path.join(DATA_DIR_LOC, 'data_dividends.csv')
DATA_PRICE_LOC = os.path.join(DATA_DIR_LOC, 'data_prices.csv')
START_DATE = '2004-01-01'
END_DATE = '2024-10-01'
STOCK_TIME = 'Open'
HOLDING_PERIOD_MONTHS = 3
FORMATION_PERIOD_MONTHS = 12
TOP_DECILE = 10