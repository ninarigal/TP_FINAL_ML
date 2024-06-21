import clean_dataset
import pandas as pd

from reg_lineal import regresion_lineal
from xgb import XGBoost
from xgb import xgboost
from create_features import new_features
from random_forest import random_forest
from xgb_linear_reg import xgboost_reg
from xgb import xgboost_cv
import numpy as np


def main():
    file_path = r'pf_suvs_i302_1s2024.csv'
    data = pd.read_csv(file_path)
    data = clean_dataset.clean_dataset(data)
    data = new_features(data)
    print(data.columns)

    #eliminar los datos que en version final sean other
    #data = data[data['Versión final'] != 'other']
    #data = data[data['Motor final'] != 'other']

        
    #CONVERTIR VARIABLES CATEGÓRICAS A ONE HOT ENCODING
    data = pd.get_dummies(data, columns=['Marca'])
    data = pd.get_dummies(data, columns=['Modelo'])
    data = pd.get_dummies(data, columns=['Transmisión'])
    data = pd.get_dummies(data, columns=['Versión final']) 
    data = pd.get_dummies(data, columns=['Gama'])
    data = pd.get_dummies(data, columns=['Motor final'])

    #me quedo con el logaritmo de los km
    data['log_km'] = data['Kilómetros'].apply(lambda x: np.log(x+1))
    data.drop(['Kilómetros'], axis=1, inplace=True)
 
    # Ejemplo: Crear una característica de ratio entre 'Kilómetros' y 'Edad' del auto
    data['km_per_year'] = data['log_km'] / (data['Edad'] + 1e-9)  # Añadir una pequeña constante para evitar división por cero

        # Obtener las columnas dummy de cada tipo ('Modelo', 'Marca', 'Versión final')
    dummies_modelo = data.filter(like='Modelo').columns
    dummies_marca = data.filter(like='Marca').columns
    dummies_version = data.filter(like='Versión final').columns
    dummies_gama = data.filter(like='Gama').columns
    dummies_motor = data.filter(like='Motor final').columns
    dummies_transmision = data.filter(like='Transmisión').columns
    
    # Crear una lista de DataFrames que contienen las nuevas columnas
    new_columns = []

    # Multiplicar 'Kilómetros' por cada columna dummy y agregar a la lista
    for modelo in dummies_modelo:
        new_columns.append(data['km_per_year'] * data[modelo])

    for marca in dummies_marca:
        new_columns.append(data['km_per_year'] * data[marca])
     


    for version in dummies_version:
        new_columns.append(data['km_per_year'] * data[version])
      
    
    for gama in dummies_gama:
        new_columns.append(data['km_per_year'] * data[gama])
    

    for motor in dummies_motor:
        new_columns.append(data['km_per_year'] * data[motor])


    for transmision in dummies_transmision:
        new_columns.append(data['km_per_year'] * data[transmision])

    for marca in dummies_marca:
        for modelo in dummies_modelo:
            new_columns.append(data[marca] * data[modelo])
    
    for modelo in dummies_modelo:
        for version in dummies_version:
            new_columns.append(data[modelo] * data[version])
    


    


    # Concatenar todos los DataFrames en uno solo a lo largo del eje de columnas (axis=1)
    new_columns_df = pd.concat(new_columns, axis=1)

    # Agregar las nuevas columnas al DataFrame original
    data = pd.concat([data, new_columns_df], axis=1)
    


    


    data.drop(['Versión'], axis=1, inplace=True)
    data.drop(['Motor'], axis=1, inplace=True)
    data.drop(['Tipo de vendedor'], axis=1, inplace=True)
    data.drop(['Título'], axis=1, inplace=True)
    data.drop(['Color'], axis=1, inplace=True)
    #data.drop(['Versión final'], axis=1, inplace=True)
    #data.drop(['Motor final'], axis=1, inplace=True)

  

    print(data.columns)

    target = 'Precio'
    #regresion_lineal(data, target)
    xgboost(data, target)
    #xgboost_cv(data, target)
    #random_forest(data, target)
    #xgboost_reg(data, target)


if __name__ == '__main__':
    main()
    
