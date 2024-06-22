
import pandas as pd
import numpy as np
# from xgboost_model import xgboost, xgboost_cv
from features import one_hot_encoding
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV



def main():
    file_path = r'data_dev.csv'
    data = pd.read_csv(file_path)
    categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final']
    data = one_hot_encoding(data, categorical_columns, mode='train')
    # print(data.columns)
    data.drop(['Versión'], axis=1, inplace=True)
    data.drop(['Motor'], axis=1, inplace=True)
    data.drop(['Tipo de vendedor'], axis=1, inplace=True)
    data.drop(['Título'], axis=1, inplace=True)
    data.drop(['Color'], axis=1, inplace=True)

    # divido el dev en train y valid
    X = data.drop(['Precio'], axis=1)
    y = data['Precio']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=31)

    # Define the parameter grid
    # param_grid = {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [3, 6, 9],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'subsample': [0.8, 1.0],
    #     'colsample_bytree': [0.8, 1.0]
    # }
    # param_grid = {
    #     'n_estimators': [50, 100, 200, 300, 500],
    #     'max_depth': [3, 5, 6, 8, 9, 12],
    #     'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    #     'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    #     'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    #     'min_child_weight': [1, 3, 5, 7],
    #     'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    #     'reg_alpha': [0, 0.01, 0.1, 1, 10],
    #     'reg_lambda': [0, 0.01, 0.1, 1, 10]
    # }

    # # Initialize the XGBRegressor
    # model = XGBRegressor()

    # # Perform grid search
    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
    # grid_search.fit(X_train, y_train)

    # # Get the best model
    # best_model = grid_search.best_estimator_

    # # Make predictions
    # y_pred = best_model.predict(X_valid)

    # # Evaluate the model
    # mse = mean_squared_error(y_valid, y_pred)
    # print(f'Best parameters found: {grid_search.best_params_}')
    # print(f'MSE: {mse}')

    # Best parameters found: {'colsample_bytree': 0.8, 'learning_rate': 0.2, 'max_depth': 9, 'n_estimators': 200, 'subsample': 0.8}
    # MSE: 22521609.699799165

    model = XGBRegressor(colsample_bytree=0.8, learning_rate=0.2, max_depth=9, n_estimators=200, subsample=0.8)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    print(f'Validation set')
    print(f'MSE: {mse}')
    print(f'R2: {r2}')
    print()

    #Ahora veo TEST usando el modelo entrenado
    file_path = r'data_test.csv'
    data = pd.read_csv(file_path)
    data = one_hot_encoding(data, categorical_columns, mode='test')
    data.drop(['Versión'], axis=1, inplace=True)
    data.drop(['Motor'], axis=1, inplace=True)
    data.drop(['Tipo de vendedor'], axis=1, inplace=True)
    data.drop(['Título'], axis=1, inplace=True)
    data.drop(['Color'], axis=1, inplace=True)
    X_test = data.drop(['Precio'], axis=1)
    y_test = data['Precio']
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Test set')
    print(f'MSE: {mse}')
    print(f'R2: {r2}')
    print()


if __name__ == '__main__':
    main()