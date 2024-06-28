import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from xgboost_model import xgboost, xgboost_cv
from procesamiento_datos.features import one_hot_encoding
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


def main():

    file_path = r'procesamiento_datos/data_train.csv'
    data_train = pd.read_csv(file_path)

    file_path = r'procesamiento_datos/data_valid.csv'
    data_valid = pd.read_csv(file_path)

    categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final']
    data_train = one_hot_encoding(data_train, categorical_columns, mode='train')
    # print(data.columns)
    data_train.drop(['Versión'], axis=1, inplace=True)
    data_train.drop(['Motor'], axis=1, inplace=True)
    data_train.drop(['Tipo de vendedor'], axis=1, inplace=True)
    data_train.drop(['Título'], axis=1, inplace=True)
    data_train.drop(['Color'], axis=1, inplace=True)
    # data_train.drop(['Kilómetros2'], axis=1, inplace=True)
    # data_train.drop(['Edad2'], axis=1, inplace=True)
    # data_train.drop(['Km_Edad'], axis=1, inplace=True)
    # data_train.drop(['Log_Kilómetros'], axis=1, inplace=True)
    # data_train.drop(['Log_Edad'], axis=1, inplace=True)




    data_valid = one_hot_encoding(data_valid, categorical_columns, mode='test')
    data_valid.drop(['Versión'], axis=1, inplace=True)
    data_valid.drop(['Motor'], axis=1, inplace=True)
    data_valid.drop(['Tipo de vendedor'], axis=1, inplace=True)
    data_valid.drop(['Título'], axis=1, inplace=True)
    data_valid.drop(['Color'], axis=1, inplace=True)
    # data_valid.drop(['Kilómetros2'], axis=1, inplace=True)
    # data_valid.drop(['Edad2'], axis=1, inplace=True)
    # data_valid.drop(['Km_Edad'], axis=1, inplace=True)
    # data_valid.drop(['Log_Kilómetros'], axis=1, inplace=True)
    # data_valid.drop(['Log_Edad'], axis=1, inplace=True)



    X_train = data_train.drop(['Precio'], axis=1)
    y_train = data_train['Precio']
    X_valid = data_valid.drop(['Precio'], axis=1)
    y_valid = data_valid['Precio']


    model = LinearRegression()


    # #best_hparams = {'subsample': 0.6, 'n_estimators': 500, 'min_child_weight': 5, 'max_depth': 9, 'learning_rate': 0.12, 'gamma': 0.25, 'colsample_bytree': 0.9}
    # best_hparams = {'subsample': 0.5, 'n_estimators': 500, 'min_child_weight': 3, 'max_depth': 7, 'learning_rate': 0.18, 'gamma': 0.1, 'colsample_bytree': 0.7}


    # model = LinearRegression(**best_hparams)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    # for i in range(len(y_pred)):
    #     if np.abs(y_pred[i] - y_valid.iloc[i]) > 20000:
    #         print(f'Pred: {y_pred[i]} Real: {y_valid.iloc[i]}')
    #         #encontrar marca y modelo (estan en one hot encoding)
    #         for j in range(len(X_valid.columns)):
    #             if X_valid.iloc[i, j] == 1:
    #                 print(X_valid.columns[j])

            

    print(f'Validation set')
    print(f'MSE: {mse}')
    print(f'RMSE: {np.sqrt(mse)}')
    print(f'R2: {r2}')
    print(f'MAE: {np.mean(np.abs(y_pred - y_valid))}')
    print()



if __name__ == '__main__':
    main()
