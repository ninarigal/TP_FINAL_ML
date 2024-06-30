import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from features import one_hot_encoding
from clean_data import clean_dataset
from features import add_features
from xgboost import XGBRegressor


#MARCA: TOYOTA
#MODELOS: RAV4, COROLLA CROSS, SW4

#Analizar el valor despues de 5 años de uso, si la uso 20 mil km por año

#Armar el dataset con los datos de los modelos de toyota

data = pd.read_csv(r'predecir.csv')


#limpiar datos
data = clean_dataset(data, mode='test')

#agregar columnas
data = add_features(data, mode='test')




#modelo
file_path = r'data_train.csv'
data_train = pd.read_csv(file_path)

categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final']
data_train = one_hot_encoding(data_train, categorical_columns, mode='train')

data_train.drop(['Versión'], axis=1, inplace=True)
data_train.drop(['Motor'], axis=1, inplace=True)
data_train.drop(['Tipo de vendedor'], axis=1, inplace=True)
data_train.drop(['Título'], axis=1, inplace=True)
data_train.drop(['Color'], axis=1, inplace=True)

#one hot encoding
categorical_columns = ['Marca', 'Modelo', 'Transmisión', 'Versión final', 'Gama', 'Motor final']
data = one_hot_encoding(data, categorical_columns, mode='test')

data.drop(['Versión'], axis=1, inplace=True)
data.drop(['Motor'], axis=1, inplace=True)
data.drop(['Tipo de vendedor'], axis=1, inplace=True)
data.drop(['Título'], axis=1, inplace=True)
data.drop(['Color'], axis=1, inplace=True)
data.drop(['Precio'], axis=1, inplace=True)

model = XGBRegressor()
model.fit(data_train.drop(['Precio'], axis=1), data_train['Precio'])

#predicción
y_pred = model.predict(data)
print(y_pred)
print('La SW4 0km despues de 4 años y 80 mil km se deprecio: ',np.abs(y_pred[0] - y_pred[1]))
print('La SW4 15000km despues de 4 años y 80 mil km se deprecio: ',np.abs(y_pred[2] - y_pred[3]))
















