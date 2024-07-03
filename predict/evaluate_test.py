import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from procesamiento_datos.features import one_hot_encoding
from sklearn.metrics import mean_squared_error, r2_score
from run.run_reg_l2 import add_regresion_features, drop_columns_regresion
from run.run_xgb import drop_columns_xgb
import joblib
import matplotlib.pyplot as plt


def run_regression_l2(model_path, cleaned_data):
    categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final', 'Tipo de vendedor']
    data_test = add_regresion_features(cleaned_data, categorical_columns, mode='test', encoder_file='procesamiento_datos/ohe.pkl')
    data_test = drop_columns_regresion(data_test)
    X_test = data_test.drop(['Precio'], axis=1)
    y_test = data_test['Precio']
    model = joblib.load(model_path)
    y_pred = np.expm1(model.predict(X_test))
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Test set')
    print(f'MSE: {mse}')
    print(f'RMSE: {np.sqrt(mse)}')
    print(f'R2: {r2}')

    plot_results(y_pred, y_test)

def run_regression_l1(model_path, cleaned_data):
    categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final', 'Tipo de vendedor']
    data_test = add_regresion_features(cleaned_data, categorical_columns, mode='test', encoder_file='procesamiento_datos/ohe.pkl')
    data_test = drop_columns_regresion(data_test)
    X_test = data_test.drop(['Precio'], axis=1)
    y_test = data_test['Precio']
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Test set')
    print(f'MSE: {mse}')
    print(f'RMSE: {np.sqrt(mse)}')
    print(f'R2: {r2}')


def run_xgb(model_path, cleaned_data):
    categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final', 'Tipo de vendedor']
    data_test = one_hot_encoding(cleaned_data, categorical_columns, mode='test', encoder_file='procesamiento_datos/ohe.pkl')
    data_test = drop_columns_xgb(data_test)
    X_test = data_test.drop(['Precio'], axis=1)
    y_test = data_test['Precio']
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Test set')
    print(f'MSE: {mse}')
    print(f'RMSE: {np.sqrt(mse)}')
    print(f'R2: {r2}')


def plot_results(pred, real):
    plt.scatter(real, pred, c='b', alpha=0.4)
    plt.xlabel('Real')
    plt.ylabel('Predicción')
    plt.plot([0, 300000], [0, 300000], color='red')
    plt.show()










if __name__ == '__main__':
    model_path_regl2 = 'models/model_regl2.pkl'
    cleaned_data = pd.read_csv('procesamiento_datos/data_test.csv')
    run_regression_l2(model_path_regl2, cleaned_data)

    model_path_regl1 = 'models/model_regl1.pkl'
    run_regression_l1(model_path_regl1, cleaned_data)

    model_path_xgb = 'models/model_xgb.pkl'
    run_xgb(model_path_xgb, cleaned_data)


