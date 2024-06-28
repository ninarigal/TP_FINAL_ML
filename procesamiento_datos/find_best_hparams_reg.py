import pandas as pd
import numpy as np
from features import one_hot_encoding
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def main():
    file_path = r'data_dev.csv'
    data_dev = pd.read_csv(file_path)
    categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final']
    data_dev = one_hot_encoding(data_dev, categorical_columns, mode='train')
    data_dev.drop(['Versión'], axis=1, inplace=True)
    data_dev.drop(['Motor'], axis=1, inplace=True)
    data_dev.drop(['Tipo de vendedor'], axis=1, inplace=True)
    data_dev.drop(['Título'], axis=1, inplace=True)
    data_dev.drop(['Color'], axis=1, inplace=True)


    print(data_dev.columns)

    # busco los mejores hiperparámetros
    X_dev = data_dev.drop(['Precio'], axis=1)
    Y_dev = data_dev['Precio']

    # Define the parameter grid
    param_grid_important = {
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'n_jobs': [None, -1],
        'positive': [True, False]
    }

    # Initialize the XGBRegressor
    model = LinearRegression()

    # # Perform grid search
    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid_important, cv=5, scoring='neg_mean_squared_error', verbose=3)
    # grid_search.fit(X_dev, Y_dev)

    # # Get the best model
    # best_score = grid_search.best_score_
    # best_hparams = grid_search.best_params_

    #perform randomized search
    n_iter_search = 10
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid_important, n_iter=n_iter_search, cv=5, scoring='neg_mean_squared_error', verbose=3, random_state=42)
    random_search.fit(X_dev, Y_dev)

    best_score = random_search.best_score_
    best_hparams = random_search.best_params_

   
    print(f'best score: {best_score}')
    print(f'best hparams: {best_hparams}')

    #creo el modelo con los mejores hiperparámetros
    best_model = LinearRegression(**best_hparams)

    #divido el dev en train y valid
    X_train, X_valid, y_train, y_valid = train_test_split(X_dev, Y_dev, test_size=0.2, random_state=0)

    #entreno el modelo
    best_model.fit(X_train, y_train)

    #predigo
    y_pred = best_model.predict(X_valid)

    #evalúo
    mse = mean_squared_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    print(f'RMSE: {mse**(1/2)}')
    print(f'R2: {r2}')


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
    


    

    










    


