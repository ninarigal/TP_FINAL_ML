import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_depreciation(model_path: str, , model_name: str, SUV_model: dict, years: int, km_per_year: int):
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
    for i in range(years):
        if model_name == 'reg':
            prices[i] = predict_precio_regresion(model_path, marca=SUV_model['Marca'], modelo=SUV_model['Modelo'], version=SUV_model['Versión'], color=SUV_model['Color'], transmision=SUV_model['Transmisión'], motor=SUV_model['Motor'], kilometros=(SUV_model['Kilómetros'] + i*km_per_year), titulo=SUV_model['Título'], tipo_vendedor=SUV_model['Tipo de vendedor'], anio= (SUV_model['Año'] - i), tipo_combustible=SUV_model['Tipo de combustible'], tipo_carroceria=SUV_model['Tipo de carrocería'], puertas=SUV_model['Puertas'], camara_retroceso=SUV_model['Con cámara de retroceso'])
        elif model_name == 'xgb':
            prices[i] = predict_precio_xgb(model_path, marca=SUV_model['Marca'], modelo=SUV_model['Modelo'], version=SUV_model['Versión'], color=SUV_model['Color'], transmision=SUV_model['Transmisión'], motor=SUV_model['Motor'], kilometros=(SUV_model['Kilómetros'] + i*km_per_year), titulo=SUV_model['Título'], tipo_vendedor=SUV_model['Tipo de vendedor'], anio= (SUV_model['Año'] - i), tipo_combustible=SUV_model['Tipo de combustible'], tipo_carroceria=SUV_model['Tipo de carrocería'], puertas=SUV_model['Puertas'], camara_retroceso=SUV_model['Con cámara de retroceso'])
    return prices


def plot_depreciation(prices1, prices2, model1, model2):
    """	 Plot the depreciation of two cars.
    Args:
        prices1 (np.array): Array with the prices of the first car.
        prices2 (np.array): Array with the prices of the second car.
        model1 (str): Name of the first car.
        model2 (str): Name of the second car.
    """
    years = len(prices1)
    plt.scatter(years, prices1, label=model1)
    plt.scatter(years, prices2, label=model2)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    years = 5
    km_per_year = 10000
    model_path = r'models/model_reg.plk.csv'
    model_name = 'reg'
    tiguan = {'Marca': 'Volkswagen',	'Modelo': 'Tiguan',	'Año': 2024.0,	'Versión': 'Premium',	'Color': 'Blanco',	'Tipo de combustible': 'Nafta',	'Puertas': 5.0, 'Tipo de carrocería': 'SUV',	'Transmisión': 'Automática',	'Motor': 2.0,	'Tipo de carrocería': 'SUV', 'Kilómetros': '0 km',	'Título': 'Tiguan',	'Tipo de vendedor': 'tienda',	'Con cámara de retroceso': 1}
    x5 = {'Marca': 'bmw',	'Modelo': 'X5',	'Año': 2014.0,	'Versión': 'Premium',	'Color': 'Blanco',	'Tipo de combustible': 'Nafta',	'Puertas': 5.0, 'Tipo de carrocería': 'SUV',	'Transmisión': 'Automática',	'Motor': 3.0,	'Tipo de carrocería': 'SUV', 'Kilómetros': '100000 km',	'Título': 'bmw',	'Tipo de vendedor': 'tienda',	'Con cámara de retroceso': 1}
    prices1 = get_depreciation(model_path, model_name, tiguan, years, km_per_year)
    prices2 = get_depreciation(model_path, model_name, x5, years, km_per_year)
    plot_depreciation(prices1, prices2, 'Tiguan', 'X5')











