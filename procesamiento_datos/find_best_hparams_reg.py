import pandas as pd
import numpy as np
from features import one_hot_encoding
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run.run_reg import add_regresion_features, drop_columns_regresion


def main():
    file_path = r'procesamiento_datos/data_dev.csv'
    data_dev = pd.read_csv(file_path)
    categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final', 'Tipo de vendedor']
    data_dev = add_regresion_features(data_dev, categorical_columns, mode='train')
    data_dev = drop_columns_regresion(data_dev)
    X_dev = data_dev.drop(['Precio'], axis=1)
    Y_dev = data_dev['Precio']
    param_grid_important = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
    model = Lasso(max_iter=10000, random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid_important, cv=3, scoring='neg_mean_squared_error', verbose=3)
    grid_search.fit(X_dev, Y_dev)
    best_score = grid_search.best_score_
    best_hparams = grid_search.best_params_   
    print(f'best score: {best_score}')
    print(f'best hparams: {best_hparams}')

# def test():
       
#     file_path = r'data_test.csv'
#     data_test = pd.read_csv(file_path)
#     categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final']
#     data_test = one_hot_encoding(data_test, categorical_columns, mode='test')
#     data_test.drop(['Versión'], axis=1, inplace=True)
#     data_test.drop(['Motor'], axis=1, inplace=True)
#     data_test.drop(['Tipo de vendedor'], axis=1, inplace=True)
#     data_test.drop(['Título'], axis=1, inplace=True)
#     data_test.drop(['Color'], axis=1, inplace=True)

#     X_test = data_test.drop(['Precio'], axis=1)
#     Y_test = data_test['Precio']

#     #best_hparams = {'subsample': 0.9, 'n_estimators': 500, 'min_child_weight': 1, 'max_depth': 5, 'learning_rate': 0.15, 'gamma': 0.2, 'colsample_bytree': 0.7} 
#     #best_hparams = {'subsample': 0.6, 'n_estimators': 300, 'min_child_weight': 1, 'max_depth': 8, 'learning_rate': 0.18, 'gamma': 0.4, 'colsample_bytree': 0.7}
#     best_hparams = {'subsample': 0.6, 'n_estimators': 500, 'min_child_weight': 5, 'max_depth': 9, 'learning_rate': 0.12, 'gamma': 0.25, 'colsample_bytree': 0.9}
#     best_model = XGBRegressor(**best_hparams)

#     best_model.fit(X_test, Y_test)

#     y_pred = best_model.predict(X_test)

#     mse = mean_squared_error(Y_test, y_pred)
#     r2 = r2_score(Y_test, y_pred)
#     print(f'RMSE: {mse**(1/2)}')
#     print(f'R2: {r2}')




if __name__ == '__main__':
    main()
    #test()
    


    

    










    


