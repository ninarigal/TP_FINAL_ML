import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from xgboost_model import xgboost, xgboost_cv
from procesamiento_datos.features import one_hot_encoding
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor



def main():

    file_path = r'procesamiento_datos/data_train.csv'
    data_train = pd.read_csv(file_path)

    file_path = r'procesamiento_datos/data_valid.csv'
    data_valid = pd.read_csv(file_path)

    categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final']
    data_train = one_hot_encoding(data_train, categorical_columns, mode='train')

    # Crear nuevas columnas de manera eficiente para evitar fragmentación
    new_columns_train = {}
    new_columns_valid = {}

    for col in data_train.columns:
        if 'Modelo_' in col:
            new_columns_train[col + 'Edad'] = data_train[col] * data_train['Edad']
            # new_columns_train[col + 'Km_Edad'] = data_train[col] * data_train['Km_Edad']
            # data_train[col + 'Edad2'] = data_train[col] * data_train['Edad2']
        if 'Marca_' in col:
            new_columns_train[col + 'Edad'] = data_train[col] * data_train['Edad']
            new_columns_train[col + 'Edad2'] = data_train[col] * data_train['Edad2']
            # new_columns_train[col + 'Kilómetros2'] = data_train[col] * data_train['Kilómetros2']
        if 'Gama_' in col:
            new_columns_train[col + 'Edad'] = data_train[col] * data_train['Edad']
            new_columns_train[col + 'Edad2'] = data_train[col] * data_train['Edad2']
            new_columns_train[col + 'Kilómetros2'] = data_train[col] * data_train['Kilómetros2']
            # new_columns_train[col + 'Log_Edad'] = data_train[col] * data_train['Log_Edad']
        if 'Motor final_' in col:
            new_columns_train[col + 'Edad'] = data_train[col] * data_train['Edad']
            new_columns_train[col + 'Edad2'] = data_train[col] * data_train['Edad2']
            # new_columns_train[col + 'Kilómetros2'] = data_train[col] * data_train['Kilómetros2']
        if 'Transmisión_' in col:
            new_columns_train[col + 'Edad'] = data_train[col] * data_train['Edad']
            new_columns_train[col + 'Edad2'] = data_train[col] * data_train['Edad2']
            new_columns_train[col + 'Kilómetros2'] = data_train[col] * data_train['Kilómetros2']


    # Agregar nuevas columnas al DataFrame de entrenamiento
    new_columns_train_df = pd.DataFrame(new_columns_train)
    data_train = pd.concat([data_train, new_columns_train_df], axis=1)
    
    data_train.drop(['Versión'], axis=1, inplace=True)
    data_train.drop(['Motor'], axis=1, inplace=True)
    data_train.drop(['Tipo de vendedor'], axis=1, inplace=True)
    data_train.drop(['Título'], axis=1, inplace=True)
    data_train.drop(['Color'], axis=1, inplace=True)


    data_valid = one_hot_encoding(data_valid, categorical_columns, mode='test')
    for col in data_valid.columns:
        if 'Modelo_' in col:
            new_columns_valid[col + 'Edad'] = data_valid[col] * data_valid['Edad']
            # new_columns_valid[col + 'Km_Edad'] = data_valid[col] * data_valid['Km_Edad']
            # data_valid[col + 'Edad2'] = data_valid[col] * data_valid['Edad2']
        if 'Marca_' in col:
            new_columns_valid[col + 'Edad'] = data_valid[col] * data_valid['Edad']
            new_columns_valid[col + 'Edad2'] = data_valid[col] * data_valid['Edad2']
            # new_columns_valid[col + 'Kilómetros2'] = data_valid[col] * data_valid['Kilómetros2']
        if 'Gama_' in col:
            new_columns_valid[col + 'Edad'] = data_valid[col] * data_valid['Edad']
            new_columns_valid[col + 'Edad2'] = data_valid[col] * data_valid['Edad2']
            new_columns_valid[col + 'Kilómetros2'] = data_valid[col] * data_valid['Kilómetros2']
            # new_columns_valid[col + 'Log_Edad'] = data_valid[col] * data_valid['Log_Edad']
        if 'Motor final_' in col:
            new_columns_valid[col + 'Edad'] = data_valid[col] * data_valid['Edad']
            new_columns_valid[col + 'Edad2'] = data_valid[col] * data_valid['Edad2']
            # new_columns_valid[col + 'Kilómetros2'] = data_valid[col] * data_valid['Kilómetros2']
        if 'Transmisión_' in col:
            new_columns_valid[col + 'Edad'] = data_valid[col] * data_valid['Edad']
            new_columns_valid[col + 'Edad2'] = data_valid[col] * data_valid['Edad2']
            new_columns_valid[col + 'Kilómetros2'] = data_valid[col] * data_valid['Kilómetros2']

    # Agregar nuevas columnas al DataFrame de validación
    new_columns_valid_df = pd.DataFrame(new_columns_valid)
    data_valid = pd.concat([data_valid, new_columns_valid_df], axis=1)
                                              

    data_valid.drop(['Versión'], axis=1, inplace=True)
    data_valid.drop(['Motor'], axis=1, inplace=True)
    data_valid.drop(['Tipo de vendedor'], axis=1, inplace=True)
    data_valid.drop(['Título'], axis=1, inplace=True)
    data_valid.drop(['Color'], axis=1, inplace=True)

    X_train = data_train.drop(['Precio'], axis=1)
    y_train = data_train['Precio']
    X_valid = data_valid.drop(['Precio'], axis=1)
    y_valid = data_valid['Precio']


    # model = LinearRegression()
    # model = Ridge(alpha=1.0, random_state=42)
    model = Lasso(alpha=0.1, max_iter=10000, random_state=42)


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



if __name__ == '__main__':
    main()
