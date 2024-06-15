import clean_dataset
import pandas as pd

from reg_lineal import RegresionLineal
from reg_lineal import regresion_lineal
from xgb import XGBoost
from xgb import xgboost

def main():
    file_path = r'pf_suvs_i302_1s2024.csv'
    data = pd.read_csv(file_path)
    data = clean_dataset.clean_dataset(data)
    #CONVERTIR VARIABLES CATEGÓRICAS A ONE HOT ENCODING
    data = pd.get_dummies(data, columns=['Marca'])
    data = pd.get_dummies(data, columns=['Modelo'])
    data = pd.get_dummies(data, columns=['Transmisión'])
    data.drop(['Versión'], axis=1, inplace=True)
    data.drop(['Motor'], axis=1, inplace=True)
    data.drop(['Tipo de vendedor'], axis=1, inplace=True)
    data.drop(['Título'], axis=1, inplace=True)
    data.drop(['Color'], axis=1, inplace=True)

    print(data.columns)

    target = 'Precio'
    regresion_lineal(data, target)
    xgboost(data, target)


    

if __name__ == '__main__':
    main()
    
