import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from procesamiento_datos.clean_data import clean_dataset
from procesamiento_datos.features import add_features
from procesamiento_datos.features import one_hot_encoding
import joblib

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

def get_depreciation(model_path: str, model_name: str, SUV_model: dict, years: int, km_per_year: int):
    """	 Predict the depreciation of a car using a regression model.
    Args:
        model_path (str): Path to the model.
        model_name (str): Type of model.
        SUV_model (dict): Dictionary with the car data.
        years (int): Number of years to predict.
        km_per_year (int): Number of kilometers per year.
       
    Returns:
        np.array: Array with the predicted prices.
    """
    prices = np.zeros(years)
    SUV_model = pd.DataFrame(SUV_model)

    for i in range(years):
        if model_name == 'reg':
            km = SUV_model['Kilómetros'].replace(' km','', regex=True).astype(int) 
            km += i*km_per_year
            SUV_model['Kilómetros'] = km
            anio = SUV_model['Año'].astype(float) - i
            SUV_model['Año'] = anio

    
            prices[i] = predict_precio_regresion(model_path, marca=SUV_model['Marca'], modelo=SUV_model['Modelo'], version=SUV_model['Versión'], color=SUV_model['Color'], transmision=SUV_model['Transmisión'], motor=SUV_model['Motor'], kilometros=(SUV_model['Kilómetros']), titulo=SUV_model['Título'], tipo_vendedor=SUV_model['Tipo de vendedor'], anio= (SUV_model['Año']), tipo_combustible=SUV_model['Tipo de combustible'], tipo_carroceria=SUV_model['Tipo de carrocería'], puertas=SUV_model['Puertas'], camara_retroceso=SUV_model['Con cámara de retroceso'])
        #elif model_name == 'xgb':
        #    prices[i] = predict_precio_xgb(model_path, marca=SUV_model['Marca'], modelo=SUV_model['Modelo'], version=SUV_model['Versión'], color=SUV_model['Color'], transmision=SUV_model['Transmisión'], motor=SUV_model['Motor'], kilometros=(SUV_model['Kilómetros'] + i*km_per_year), titulo=SUV_model['Título'], tipo_vendedor=SUV_model['Tipo de vendedor'], anio= (SUV_model['Año'] - i), tipo_combustible=SUV_model['Tipo de combustible'], tipo_carroceria=SUV_model['Tipo de carrocería'], puertas=SUV_model['Puertas'], camara_retroceso=SUV_model['Con cámara de retroceso'])
    print(prices)
    
    return prices


def plot_depreciation(prices1, prices2, model1, model2):
    """	 Plot the depreciation of two cars.
    Args:
        prices1 (np.array): Array with the prices of the first car.
        prices2 (np.array): Array with the prices of the second car.
        model1 (str): Name of the first car.
        model2 (str): Name of the second car.
    """

    years = range(len(prices1))
    plt.plot(years, prices1, color='b')
    plt.scatter(years, prices1, label=model1, color='b')
    plt.scatter(years, prices2, label=model2, color='r')
    plt.plot(years, prices2, color='r')
    plt.legend()
    plt.show()

def estudio_toyota():
    years = 5
    km_per_year = 10000
    model_path = r'models/model_reg.pkl'
    model_name = 'reg'
    corolla_cross = {'Marca': ['Toyota'],	'Modelo': ['Corolla Cross'],	'Año': ['2024.0'],	'Versión': ['Seg'],	'Color': ['Blanco'], 'Tipo de combustible': ['Nafta'],	'Puertas': ['5.0'], 'Tipo de carrocera': ['SUV'],	'Transmisión': ['Automática'],	'Motor': ['2.0'],	'Tipo de carrocería': ['SUV'], 'Kilómetros': ['0 km'],	'Título': ['Corolla Cross'],	'Tipo de vendedor': ['particular'],	'Con cámara de retroceso': ['1']}
    rav4 = {'Marca': ['Toyota'],	'Modelo': ['Rav4'],	'Año': ['2024.0'],	'Versión': ['Vx'],	'Color': ['Blanco'], 'Tipo de combustible': ['Nafta'],	'Puertas': ['5.0'],	'Transmisión': ['Automática'],	'Motor': ['2.5'],	'Tipo de carrocería': ['SUV'], 'Kilómetros': ['0 km'],	'Título': ['Rav4'],	'Tipo de vendedor': ['particular'],	'Con cámara de retroceso': ['1']}
    sw4 = {'Marca': ['Toyota'],	'Modelo': ['Sw4'],	'Año': ['2024.0'],	'Versión': ['diamond'],	'Color': ['Blanco'], 'Tipo de combustible': ['Nafta'],	'Puertas': ['5.0'],	'Transmisión': ['Automática'],	'Motor': ['2.8'],	'Tipo de carrocería': ['SUV'], 'Kilómetros': ['0 km'],	'Título': ['Sw4'],	'Tipo de vendedor': ['particular'],	'Con cámara de retroceso': ['1']}

    prices_ccross = get_depreciation(model_path, model_name, corolla_cross, years, km_per_year)
    prices_rav4 = get_depreciation(model_path, model_name, rav4, years, km_per_year)
    prices_sw4 = get_depreciation(model_path, model_name, sw4, years, km_per_year)

    depre_ccross = (prices_ccross - prices_ccross[0]) / prices_ccross[0] * 100
    depre_rav4 = (prices_rav4 - prices_rav4[0]) / prices_rav4[0] * 100
    depre_sw4 = (prices_sw4 - prices_sw4[0]) / prices_sw4[0] * 100

    plt.plot(range(years), prices_ccross, label='Corolla Cross')
    plt.plot(range(years), prices_rav4, label='Rav4')
    plt.plot(range(years), prices_sw4, label='Sw4')
    plt.legend()
    plt.show()

    plt.plot(range(years), depre_ccross, label='Corolla Cross')
    plt.plot(range(years), depre_rav4, label='Rav4')
    plt.plot(range(years), depre_sw4, label='Sw4')
    plt.legend()
    plt.show()



if __name__ == '__main__':

    years = 4
    km_per_year = 10000
    model_path = r'models/model_reg.pkl'
    model_name = 'reg'
    tiguan = {'Marca': ['Volkswagen'],	'Modelo': ['Tiguan'],	'Año': ['2024.0'],	'Versión': ['Premium'],	'Color': ['Blanco'],	'Tipo de combustible': ['Nafta'],	'Puertas': ['5.0'], 'Tipo de carrocería': ['SUV'],	'Transmisión': ['Automática'],	'Motor': ['2.0'],	'Tipo de carrocería': ['SUV'], 'Kilómetros': ['0 km'],	'Título': ['Tiguan'],	'Tipo de vendedor': ['particular'],	'Con cámara de retroceso': ['1']}
    x5 = {'Marca': ['bmw'],	'Modelo': ['X5'],	'Año': ['2014.0'],	'Versión': ['Premium'],	'Color': ['Blanco'],	'Tipo de combustible': ['Nafta'],	'Puertas': ['5.0'], 'Tipo de carrocería': ['SUV'],	'Transmisión': ['Automática'],	'Motor': ['3.0'],	'Tipo de carrocería': ['SUV'], 'Kilómetros': ['100000 km'],	'Título': ['bmw'],	'Tipo de vendedor': ['particular'],	'Con cámara de retroceso': ['1']}

    prices_tiguan = get_depreciation(model_path, model_name, tiguan, years, km_per_year)
    prices_x5 = get_depreciation(model_path, model_name, x5, years, km_per_year)

    depre_tiguan = (prices_tiguan - prices_tiguan[0]) / prices_tiguan[0] * 100
    depre_x5 = (prices_x5 - prices_x5[0]) / prices_x5[0] * 100

    plot_depreciation(prices_tiguan, prices_x5, 'Tiguan', 'X5')
    plot_depreciation(depre_tiguan, depre_x5, 'Tiguan', 'X5')


    q5 = {'Marca': ['Audi'],	'Modelo': ['Q5'],	'Año': ['2020.0'],	'Versión': ['Quattro'],	'Color': ['Blanco'], 'Tipo de combustible': ['Nafta'],	'Puertas': ['5.0'], 'Tipo de carrocería': ['SUV'],	'Transmisión': ['Automática'],	'Motor': ['2.0'],	'Tipo de carrocería': ['SUV'], 'Kilómetros': ['30000 km'],	'Título': ['Audi'],	'Tipo de vendedor': ['particular'],	'Con cámara de retroceso': ['1']}
    rav4 = {'Marca': ['Toyota'],	'Modelo': ['Rav4'],	'Año': ['2024.0'],	'Versión': ['Vx'],	'Color': ['Blanco'], 'Tipo de combustible': ['Nafta'],	'Puertas': ['5.0'], 'Tipo de carrocería': ['SUV'],	'Transmisión': ['Automática'],	'Motor': ['2.5'],	'Tipo de carrocería': ['SUV'], 'Kilómetros': ['0 km'],	'Título': ['Rav4'],	'Tipo de vendedor': ['particular'],	'Con cámara de retroceso': ['1']}
  
    prices_rav4 = get_depreciation(model_path, model_name, rav4, years, km_per_year)
    prices_q5 = get_depreciation(model_path, model_name, q5, years, km_per_year)

    depre_q5 = (prices_q5 - prices_q5[0]) / prices_q5[0] * 100
    depre_rav4 = (prices_rav4 - prices_rav4[0]) / prices_rav4[0] * 100
    plot_depreciation(prices_rav4, prices_q5, 'Rav4', 'Q5')
    plot_depreciation(depre_rav4, depre_q5, 'Rav4', 'Q5')

    














