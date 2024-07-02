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


if __name__ == '__main__':
    main()

