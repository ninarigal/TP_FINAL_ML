import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from xgboost_model import xgboost, xgboost_cv
from procesamiento_datos.features import one_hot_encoding
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
import joblib
import matplotlib.pyplot as plt
from procesamiento_datos.clean_data import clean_dataset
from procesamiento_datos.features import add_features

def add_regresion_features(df, categorical_columns, mode='train', encoder_file='procesamiento_datos/ohe.pkl'):
    df = one_hot_encoding(df, categorical_columns, mode=mode, encoder_file=encoder_file)
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

def calculate_rmse_by_brand(X_valid, y_valid, y_pred):
    rmses = []
    marcas = ['peugeot', 'toyota', 'fiat', 'ds', 'ssangyong', 'chevrolet',
       'citroen', 'isuzu', 'ford', 'renault', 'porsche', 'jeep',
       'mercedesbenz', 'mini', 'honda', 'hyundai', 'volkswagen',
       'land rover', 'audi', 'geely', 'jaguar', 'daihatsu', 'subaru',
       'suzuki', 'haval', 'dodge', 'nissan', 'lexus', 'kia', 'mitsubishi',
       'lifan', 'jac', 'bmw', 'alfa romeo', 'chery', 'baic', 'jetour',
       'volvo']
    for col in X_valid.columns:
        if col.startswith('Marca_'):
            marca = col.split('_')[1]
            if marca in marcas:
                mask = X_valid[col] == 1
                y_mask = y_valid[mask]
                y_pred_mask = y_pred[mask]
                rmse = np.sqrt(mean_squared_error(y_mask, y_pred_mask))
                rmses.append(rmse)
    return marcas, rmses

def plot_rmse_by_brand(marcas, rmses):
    plt.figure(figsize=(10, 6))
    plt.bar(marcas, rmses)
    plt.xlabel('Marca')
    plt.ylabel('RMSE')
    plt.title('RMSE por marca')
    plt.xticks(rotation=90)
    plt.show()

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

    model = Lasso(alpha=1, max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    marcas, rmses = calculate_rmse_by_brand(X_valid, y_valid, y_pred)
    print(marcas)
    print(len(marcas))
    print(rmses)
    print(len(rmses))
    plot_rmse_by_brand(marcas, rmses)

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
    # # print("Precio estimado: ", predict_precio_regresion(model_path, 'Volkswagen', 'Golf', '1.0 Highline', 'Gris', 'Automática', 1.0, '58000 km', 'Golf Highline 1.0 2020', 'Particular', 2020, 'Nafta', 'Hatchback', 'Media', 'Sí'))
    # data_test = pd.read_csv('procesamiento_datos/data_test.csv')
    # categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final', 'Tipo de vendedor']
    # data_test = add_regresion_features(data_test, categorical_columns, mode='test', encoder_file='procesamiento_datos/ohe.pkl')
    # data_test = drop_columns_regresion(data_test)
    # X_test = data_test.drop(['Precio'], axis=1)
    # y_test = data_test['Precio']
    # model = joblib.load(model_path)
    # y_pred = model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # print(f'Test set')
    # print(f'MSE: {mse}')
    # print(f'RMSE: {np.sqrt(mse)}')
    # print(f'R2: {r2}')
    # print(f'MAE: {np.mean(np.abs(y_pred - y_test))}')
    # print()
    # plt.scatter(y_test, y_pred)
    # plt.xlabel('Real')
    # plt.ylabel('Predicción')
    # plt.title('Predicción vs Real')
    # plt.plot([0, 300000], [0, 300000], color='red')
    # plt.show()
    # for i in range(len(y_test)):
    #     if abs(y_test.iloc[i] - y_pred[i]) > 50000:
    #         for j in range(len(data_test.columns)):
    #             if data_test.iloc[i, j] == 1:
    #                 print(data_test.columns[j])
    #         print(f'Pred: {y_pred[i]} Real: {y_test.iloc[i]}')
    #         print()


