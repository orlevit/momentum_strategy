import os 

# Data locatopns
DATA_DIR_LOC = os.path.join(os.getcwd(), 'data_dir')
DATA_DIV_LOC = os.path.join(DATA_DIR_LOC, 'data_dividends.csv')
DATA_OPEN_LOC = os.path.join(DATA_DIR_LOC, 'data_open.csv')
DATA_CLOSE_LOC = os.path.join(DATA_DIR_LOC, 'data_close.csv')
DATA_LOW_LOC = os.path.join(DATA_DIR_LOC, 'data_low.csv')
DATA_HIGH_LOC = os.path.join(DATA_DIR_LOC, 'data_high.csv')
DATA_VOL_LOC = os.path.join(DATA_DIR_LOC, 'data_vol.csv')
DATA_POS_LOC = os.path.join(DATA_DIR_LOC, 'ml_stat.pickle')
DATA_ML_VIZ_LOC = os.path.join(DATA_DIR_LOC, 'ml_viz_stat.pickle')

IMG_DIR_LOC = os.path.join(os.getcwd(), 'images')

# Momentum hyperparameters
START_DATE = '2004-01-01'
END_DATE = '2024-10-01'
STOCK_TIME = 'Open'
HOLDING_PERIOD_MONTHS = 3
FORMATION_PERIOD_MONTHS = 12
TOP_DECILE = 10
GROUP_LABELS = {'w_stocks_higher':0, 'w_stocks_lower':1, 'l_stocks_higher':2, 'l_stocks_lower':3}

# Model hyperparameters
RANDOM_STATE = 42
TRAIN_SPLIT = 0.7
