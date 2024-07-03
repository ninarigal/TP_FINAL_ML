import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from procesamiento_datos.features import one_hot_encoding
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt
from procesamiento_datos.clean_data import clean_dataset
from procesamiento_datos.features import add_features

def drop_columns_regresion(df):
    df.drop(['Gama', 'Motor final', 'Tipo de vendedor', 'Versión final', 'Kilómetros_Edad', 'Kilómetros2',
       'Edad2', 'Km_Edad', 'Log_Kilómetros', 'Log_Edad', 'Color_extraño', 'Transmisión', 'Versión', 'Motor',
        'Título', 'Color'], axis=1, inplace=True)
    return df


def main():
    file_path = r'procesamiento_datos/data_train.csv'
    data_train = pd.read_csv(file_path)
    file_path = r'procesamiento_datos/data_valid.csv'
    data_valid = pd.read_csv(file_path)
    categorical_columns = ['Marca', 'Modelo']

    data_train = drop_columns_regresion(data_train)
    data_valid = drop_columns_regresion(data_valid)

    data_train = one_hot_encoding(data_train, categorical_columns, mode='train')
    data_valid = one_hot_encoding(data_valid, categorical_columns, mode='test')

    X_train = data_train.drop(['Precio'], axis=1)
    y_train = data_train['Precio']
    X_valid = data_valid.drop(['Precio'], axis=1)
    y_valid = data_valid['Precio']

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    mse = mean_squared_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    plt.scatter(y_valid, y_pred)
    plt.xlabel('Real')
    plt.ylabel('Predicción')
    plt.title('Predicción vs Real')
    plt.plot([0, 300000], [0, 300000], color='red')
    plt.show()

    print(f'Validation set')
    print(f'MSE: {mse}')
    print(f'RMSE: {np.sqrt(mse)}')
    print(f'R2: {r2}')
    print(f'MAE: {np.mean(np.abs(y_pred - y_valid))}')
    print()

    with open('metrics/metrics_baseline.txt', 'w') as file:
        file.write(f'MSE: {mse}\n')
        file.write(f'RMSE: {np.sqrt(mse)}\n')
        file.write(f'R2: {r2}\n')

    joblib.dump(model, 'models/model_baseline.pkl')


if __name__ == '__main__':
    main()
