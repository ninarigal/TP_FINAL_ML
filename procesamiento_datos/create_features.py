import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def new_features(df, mode):
    df = assign_gamas(df)
    df = clean_precios_gama(df, mode)
    df = create_km_per_year(df)
    return df


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

def clean_precios_gama(df, mode):
    if mode == 'train':
        for gama, precio_max in precios_max_gama.items():
            df = df.drop(df[(df['Gama'] == gama) & (df['Precio'] > precio_max)].index)
            df.reset_index(drop=True, inplace=True)
    return df

def create_km_per_year(df):
    for i in range(len(df)):
        if df.loc[i, 'Edad'] == 0:
            df.loc[i, 'Kilómetros_Edad'] = 0
        else:
            df.loc[i, 'Kilómetros_Edad'] = df.loc[i, 'Kilómetros'] / df.loc[i, 'Edad']
    return df
