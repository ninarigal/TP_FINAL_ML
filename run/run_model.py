import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from xgboost_model import xgboost, xgboost_cv
from procesamiento_datos.features import one_hot_encoding
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

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


    data_valid = one_hot_encoding(data_valid, categorical_columns, mode='train')
    data_valid.drop(['Versión'], axis=1, inplace=True)
    data_valid.drop(['Motor'], axis=1, inplace=True)
    data_valid.drop(['Tipo de vendedor'], axis=1, inplace=True)
    data_valid.drop(['Título'], axis=1, inplace=True)
    data_valid.drop(['Color'], axis=1, inplace=True)

    # Obtener todas las columnas únicas de ambos DataFrames
    all_columns = sorted(set(data_train.columns).union(set(data_valid.columns)))

    # Crear DataFrames vacíos con todas las columnas
    df_train_full = pd.DataFrame(columns=all_columns)
    df_valid_full = pd.DataFrame(columns=all_columns)

    # Combinar los DataFrames originales con los DataFrames completos
    df_train_full = pd.concat([df_train_full, data_train], ignore_index=True).fillna(0)
    df_valid_full = pd.concat([df_valid_full, data_valid], ignore_index=True).fillna(0)

    # Asegurarse de que las columnas estén en el mismo orden
    df_train_full = df_train_full[all_columns]
    df_valid_full = df_valid_full[all_columns]

    X_train = df_train_full.drop(['Precio'], axis=1)
    y_train = df_train_full['Precio']
    X_valid = df_valid_full.drop(['Precio'], axis=1)
    y_valid = df_valid_full['Precio']


    best_hparams = {'subsample': 0.6, 'n_estimators': 500, 'min_child_weight': 5, 'max_depth': 9, 'learning_rate': 0.12, 'gamma': 0.25, 'colsample_bytree': 0.9}


    model = XGBRegressor(**best_hparams)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    for i in range(len(y_pred)):
        if np.abs(y_pred[i] - y_valid.iloc[i]) > 100000:
            print(f'Pred: {y_pred[i]} Real: {y_valid.iloc[i]}')

    print(f'Validation set')
    print(f'MSE: {mse}')
    print(f'RMSE: {np.sqrt(mse)}')
    print(f'R2: {r2}')
    print(f'MAE: {np.mean(np.abs(y_pred - y_valid))}')
    print()



if __name__ == '__main__':
    main()

