import os 

# Data locatopns
DATA_DIR_LOC = os.path.join(os.getcwd(), 'data_dir')
DATA_DIV_LOC = os.path.join(DATA_DIR_LOC, 'data_dividends.csv')
DATA_OPEN_LOC = os.path.join(DATA_DIR_LOC, 'data_open.csv')
DATA_CLOSE_LOC = os.path.join(DATA_DIR_LOC, 'data_close.csv')
DATA_LOW_LOC = os.path.join(DATA_DIR_LOC, 'data_low.csv')
DATA_HIGH_LOC = os.path.join(DATA_DIR_LOC, 'data_high.csv')
DATA_VOL_LOC = os.path.join(DATA_DIR_LOC, 'data_vol.csv')

# Momentum hyperparameters
START_DATE = '2004-01-01'
END_DATE = '2024-10-01'
STOCK_TIME = 'Open'
HOLDING_PERIOD_MONTHS = 3
FORMATION_PERIOD_MONTHS = 12
TOP_DECILE = 10