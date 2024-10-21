import xgboost as xgb
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from time import time
from config import RANDOM_STATE
import pandas as pd

pipeline = Pipeline(steps=[('classifier', AdaBoostClassifier())])

# Bootstrap aggregating
RandomForestClassifier_param = {'classifier': [RandomForestClassifier()],
                                'classifier__n_estimators': [100,200],
                                'classifier__max_depth': [5, 10],
                                'classifier__class_weight': ['balanced'],
                                'classifier__min_samples_split': [2],
                                'classifier__random_state': [RANDOM_STATE]}
# # Boosting
# AdaBoost_param = {'classifier': [AdaBoostClassifier()],
#                   'classifier__n_estimators': [100, 200],
#                   'classifier__learning_rate': [0.1, 1],
#                   'classifier__random_state': [RANDOM_STATE]}

GBClassifier_param = {'classifier': [GradientBoostingClassifier()],
                      'classifier__n_estimators': [100, 200],    
                      'classifier__max_depth': [3],
                      'classifier__subsample': [0.1, 1.0],
                      'classifier__learning_rate': [1],
                      'classifier__random_state': [RANDOM_STATE]}

params = [RandomForestClassifier_param, GBClassifier_param]#, AdaBoost_param]

def train_model(train_df):

    gs = GridSearchCV(pipeline, param_grid=params, refit='f1_weighted', return_train_score=True, cv=10, n_jobs=-1, scoring='f1_weighted')
    
    tic = time()
    exclude_columns = ['date', 'label_name','stock']
    clean_train_df = train_df.loc[:, ~train_df.columns.isin(exclude_columns)]
    
    X = clean_train_df.drop(columns=['label'])
    y =  pd.to_numeric(clean_train_df['label'])
    
    print('train_df:',X.shape)
    gs.fit(X, y)
    toc = time()
    print(f'GridSearchCV time(mintues): {round((toc-tic)/60,2)}')
     
    return pd.DataFrame(gs.cv_results_), gs.best_estimator_, gs.best_estimator_[0].feature_importances_