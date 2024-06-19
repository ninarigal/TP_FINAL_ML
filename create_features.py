import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Crear gamas de auto

def new_features(df):
    #Quiero ver que modelos no tienen 0km

   # for marca in df['Marca'].unique():
        #modelos = df[df['Marca'] == marca]['Modelo'].unique()
        #for modelo in modelos:
        #    if df[(df['Marca'] == marca) & (df['Modelo'] == modelo) & (df['Kilómetros'] == 0)].shape[0] == 0:
        #        print(f'{marca} {modelo} no tiene 0km')
        #if df[(df['Marca'] == marca) & (df['Kilómetros'] == 0)].shape[0] == 0:
        #    print(f'{marca} no tiene 0km')

    #df = create_gamas(df)
    df = assign_gamas(df)
    df = clean_precios_gama(df)
    df.to_csv('cleaned_data.csv')
    return df



def create_gamas(df):

    #Crear gamas de autos
    df['Gama'] = np.nan
    #Promedio ponderado de los precios de cada marca
    marcas = df['Marca'].unique()
    print(marcas)
   # precios_promedio = []
    for marca in marcas:
        df_marca = df[df['Marca'] == marca]
        y = df_marca['Precio']
        X = df_marca[['Edad']]

        #precios = y[X['Edad'] < 3]
        precios = y

        # Calcular Q1 y Q3
        Q1 = np.percentile(precios, 25)
        Q3 = np.percentile(precios, 75)

        # Calcular el rango intercuartil (IQR)
        IQR = Q3 - Q1

        # Determinar los límites
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        # Filtrar los outliers
        datos_filtrados = [x for x in precios if limite_inferior <= x <= limite_superior]

        # Calcular el promedio sin outliers
        precios = np.mean(datos_filtrados)
        if precios < 20000:
            df['Gama'] = 'Económico'
        elif precios < 40000:
            df['Gama'] = 'Medio'
        elif precios < 60000:
            df['Gama'] = 'Alto'
        else:
            df['Gama'] = 'Premium'
    
    return df

    
   # precios_promedio = np.array(precios_promedio)
   # print(sorted(precios_promedio))

def assign_gamas(df):
    gama_lujo = ['ds', 'porsche', 'mercedesbenz', 'mini', 'land rover', 'audi', 'jaguar', 
             'lexus', 'bmw', 'alfa romeo', 'volvo']

    gama_media = ['peugeot', 'toyota', 'citroen', 'ford', 'renault', 'jeep', 'honda', 'hyundai', 
              'volkswagen', 'subaru', 'nissan', 'kia', 'mitsubishi']

    gama_baja = ['fiat', 'ssangyong', 'chevrolet', 'isuzu', 'suzuki', 'haval', 'dodge', 'lifan', 
             'jac', 'chery', 'baic', 'jetour', 'geely', 'daihatsu']
    
    df['Gama'] = np.nan
    df.loc[df['Marca'].isin(gama_lujo), 'Gama'] = 'Lujo'
    df.loc[df['Marca'].isin(gama_media), 'Gama'] = 'Media'
    df.loc[df['Marca'].isin(gama_baja), 'Gama'] = 'Baja'

    return df


precios_max_gama = {
    'Baja': 50000,
    'Media': 80000,
    'Lujo': 700000
}

def clean_precios_gama(df):
    for gama, precio_max in precios_max_gama.items():
        #eliminar los datos
        df = df.drop(df[(df['Gama'] == gama) & (df['Precio'] > precio_max)].index)
        df.reset_index(drop=True, inplace=True)
    return df

