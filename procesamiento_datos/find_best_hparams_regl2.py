import pandas as pd
import numpy as np
from features import one_hot_encoding
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run.run_reg_l2 import add_regresion_features, drop_columns_regresion

def main():
    file_path = r'procesamiento_datos/data_dev.csv'
    data_dev = pd.read_csv(file_path)
    categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final', 'Tipo de vendedor']
    data_dev = add_regresion_features(data_dev, categorical_columns, mode='train')
    data_dev = drop_columns_regresion(data_dev)
    X_dev = data_dev.drop(['Precio'], axis=1)
    Y_dev = data_dev['Precio']
    param_grid_important = {'alpha': np.linspace(0, 1.4, 100)}
    model = Ridge(max_iter=10000, random_state=42)
    n_iters = 100
    grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid_important, cv=5, scoring='neg_mean_squared_error', verbose=3, n_iter=n_iters)
    grid_search.fit(X_dev, Y_dev)
    best_score = grid_search.best_score_
    best_hparams = grid_search.best_params_   
    print(f'best score: {best_score}')
    print(f'best hparams: {best_hparams}')


main()