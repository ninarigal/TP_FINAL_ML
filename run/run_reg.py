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
import joblib
import matplotlib.pyplot as plt
from procesamiento_datos.clean_data import clean_dataset
from procesamiento_datos.features import add_features

def add_regresion_features(df, categorical_columns, mode='train'):
    df = one_hot_encoding(df, categorical_columns, mode=mode)
    new_columns = {}
    for col in df.columns:
        if 'Modelo_' in col:
            new_columns[col + 'Edad'] = df[col] * df['Edad']
        if 'Marca_' in col:
            new_columns[col + 'Edad'] = df[col] * df['Edad']
            new_columns[col + 'Edad2'] = df[col] * df['Edad2']
        if 'Gama_' in col:
            new_columns[col + 'Edad'] = df[col] * df['Edad']
            new_columns[col + 'Edad2'] = df[col] * df['Edad2']
            new_columns[col + 'Kilómetros2'] = df[col] * df['Kilómetros2']
        if 'Motor final_' in col:
            new_columns[col + 'Edad'] = df[col] * df['Edad']
            new_columns[col + 'Edad2'] = df[col] * df['Edad2']
        if 'Transmisión_' in col:
            new_columns[col + 'Edad'] = df[col] * df['Edad']
            new_columns[col + 'Edad2'] = df[col] * df['Edad2']
            new_columns[col + 'Kilómetros2'] = df[col] * df['Kilómetros2']
    new_columns_df = pd.DataFrame(new_columns)
    df = pd.concat([df, new_columns_df], axis=1)
    return df

def drop_columns_regresion(df):
    df.drop(['Versión'], axis=1, inplace=True)
    df.drop(['Motor'], axis=1, inplace=True)
    df.drop(['Título'], axis=1, inplace=True)
    df.drop(['Color'], axis=1, inplace=True)
    return df

def predict_precio_regresion(model_path, marca, modelo, version, color, transmision, motor, kilometros, titulo, tipo_vendedor, anio, tipo_combustible, tipo_carroceria, puertas, camara_retroceso):
    auto = {
        'Marca': marca,
        'Modelo': modelo,
        'Versión': version,
        'Color': color,
        'Transmisión': transmision,
        'Motor': motor,
        'Kilómetros': kilometros,
        'Título': titulo,
        'Tipo de vendedor': tipo_vendedor,
        'Año': anio,
        'Tipo de combustible': tipo_combustible,
        'Tipo de carrocería': tipo_carroceria,
        'Puertas': puertas,
        'Con cámara de retroceso': camara_retroceso
    }
    auto_df = pd.DataFrame(auto, index=[0])
    auto_df = clean_dataset(auto_df, mode='test')
    auto_df = add_features(auto_df, mode='test')
    categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final', 'Tipo de vendedor']
    auto_df = add_regresion_features(auto_df, categorical_columns, mode='test')
    auto_df = drop_columns_regresion(auto_df)
    model = joblib.load(model_path)
    precio = model.predict(auto_df)
    return precio[0]   

def main():
    file_path = r'procesamiento_datos/data_train.csv'
    data_train = pd.read_csv(file_path)
    file_path = r'procesamiento_datos/data_valid.csv'
    data_valid = pd.read_csv(file_path)
    categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final', 'Tipo de vendedor']
    data_train = add_regresion_features(data_train, categorical_columns, mode='train')
    data_train = drop_columns_regresion(data_train)

    data_valid = add_regresion_features(data_valid, categorical_columns, mode='test')                                       
    data_valid = drop_columns_regresion(data_valid)
    
    X_train = data_train.drop(['Precio'], axis=1)
    y_train = data_train['Precio']
    X_valid = data_valid.drop(['Precio'], axis=1)
    y_valid = data_valid['Precio']

    model = Lasso(alpha=1, max_iter=30000, random_state=42)
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
    # save model as pkl
    joblib.dump(model, 'models/model_reg.pkl')


if __name__ == '__main__':
    main()
    # model_path = 'models/model_reg.pkl'
    # print("Precio estimado: ", predict_precio_regresion(model_path, 'Volkswagen', 'Golf', '1.0 Highline', 'Gris', 'Automática', 1.0, 58000, 'Golf Highline 1.0 2020', 'Particular', 2020, 'Nafta', 'Hatchback', 'Media', 'Sí'))


