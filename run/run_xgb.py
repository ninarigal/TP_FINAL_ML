import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from xgboost_model import xgboost, xgboost_cv
from procesamiento_datos.features import one_hot_encoding
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
from procesamiento_datos.clean_data import clean_dataset
from procesamiento_datos.features import add_features

def add_features_xgb(df, categorical_columns, mode='train'):
    df = one_hot_encoding(df, categorical_columns, mode=mode)
    new_columns = {}
    for col in df.columns:
        if 'Marca_' in col:
            new_columns[col + 'Edad'] = df[col] * df['Edad']
    new_columns_df = pd.DataFrame(new_columns)
    df = pd.concat([df, new_columns_df], axis=1)
    return df

def drop_columns_xgb(df):
    df.drop(['Versión'], axis=1, inplace=True)
    df.drop(['Motor'], axis=1, inplace=True)
    # df.drop(['Tipo de vendedor'], axis=1, inplace=True)
    df.drop(['Título'], axis=1, inplace=True)
    df.drop(['Color'], axis=1, inplace=True)
    df.drop(['Edad2'], axis=1, inplace=True)
    df.drop(['Km_Edad'], axis=1, inplace=True)
    df.drop(['Log_Kilómetros'], axis=1, inplace=True)
    df.drop(['Log_Edad'], axis=1, inplace=True)
    df.drop(['Color_extraño'], axis=1, inplace=True)
    return df

def predict_precio_xgb(model_path, marca, modelo, version, color, transmision, motor, kilometros, titulo, tipo_vendedor, anio, tipo_combustible, tipo_carroceria, puertas, camara_retroceso):
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
    auto_df = add_features_xgb(auto_df, categorical_columns, mode='test')
    auto_df = drop_columns_xgb(auto_df)
    model = joblib.load(model_path)
    precio = model.predict(auto_df)
    return precio[0] 


def main():
    file_path = r'procesamiento_datos/data_train.csv'
    data_train = pd.read_csv(file_path)

    file_path = r'procesamiento_datos/data_valid.csv'
    data_valid = pd.read_csv(file_path)

    categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final', 'Tipo de vendedor']
    
    data_train = add_features_xgb(data_train, categorical_columns, mode='train')
    data_train = drop_columns_xgb(data_train)

    data_valid = add_features_xgb(data_valid, categorical_columns, mode='test')
    data_valid = drop_columns_xgb(data_valid)

    X_train = data_train.drop(['Precio'], axis=1)
    y_train = data_train['Precio']
    X_valid = data_valid.drop(['Precio'], axis=1)
    y_valid = data_valid['Precio']

    best_hparams = {'subsample': 0.5, 'n_estimators': 500, 'min_child_weight': 3, 'max_depth': 7, 'learning_rate': 0.18, 'gamma': 0.1, 'colsample_bytree': 0.7}
    # best_hparams = {'subsample': 0.6, 'n_estimators': 300, 'min_child_weight': 1, 'max_depth': 8, 'learning_rate': 0.18, 'gamma': 0.4, 'colsample_bytree': 0.7}
    # best_hparams: {'subsample': 0.9, 'n_estimators': 500, 'min_child_weight': 5, 'max_depth': 8, 'learning_rate': 0.14, 'gamma': 0.3, 'colsample_bytree': 0.9}
    # best_hparams = {'subsample': 0.9, 'n_estimators': 500, 'min_child_weight': 5, 'max_depth': 8, 'learning_rate': 0.14, 'gamma': 0.3, 'colsample_bytree': 0.9}
    

    model = XGBRegressor(**best_hparams)
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

    joblib.dump(model, 'models/xgb_model.pkl')



if __name__ == '__main__':
    main()
    file_path = r'procesamiento_datos/data_test.csv'
    data_test = pd.read_csv(file_path)
    categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final', 'Tipo de vendedor']
    data_test = add_features_xgb(data_test, categorical_columns, mode='test')
    data_test = drop_columns_xgb(data_test)
    X_test = data_test.drop(['Precio'], axis=1)
    y_test = data_test['Precio']
    model = joblib.load('models/xgb_model.pkl')
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Test set')
    print(f'MSE: {mse}')
    print(f'RMSE: {np.sqrt(mse)}')
    print(f'R2: {r2}')
    print(f'MAE: {np.mean(np.abs(y_pred - y_test))}')
    print()
    
    plt.scatter(y_test, y_pred)
    plt.xlabel('Real')
    plt.ylabel('Predicción')
    plt.title('Predicción vs Real')
    plt.plot([0, 300000], [0, 300000], color='red')
    plt.show()

    # ver que autos son los que quedan mal
    for i in range(len(y_test)):
        if abs(y_test.iloc[i] - y_pred[i]) > 50000:
            for j in range(len(data_test.columns)):
                if data_test.iloc[i, j] == 1:
                    print(data_test.columns[j])
            print(f'Pred: {y_pred[i]} Real: {y_test.iloc[i]}')
            print()



