import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import joblib

def add_features(df, mode):
    df = assign_gamas(df)
    df = clean_precios_gama(df, mode)
    df = create_km_per_year(df)
    df = create_km2(df)
    df = create_edad2(df)
    df = create_km_edad(df)
    df = create_log_km(df)
    df = create_log_edad(df)


    print(df.head())

    # if mode == 'train':
    #     name = 'data_dev.csv'
    # else:
    #     name = 'data_test.csv'
    
    # df.to_csv(name, index=False)
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


def encode_and_concat(data, encoder, features):
    encoded_features = encoder.transform(data[features])
    encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(features))
    return pd.concat([data.drop(columns=features), encoded_df], axis=1)

def one_hot_encoding(data, features, mode='train', encoder_file='ohe.pkl'):
    if mode == 'train':
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(data[features])
        data = encode_and_concat(data, encoder, features)
        joblib.dump(encoder, encoder_file)
    else:
        encoder = joblib.load(encoder_file)
        data = encode_and_concat(data, encoder, features)
    return data

def create_km2(df):
    df['Kilómetros2'] = df['Kilómetros'] ** 2
    return df

def create_edad2(df):
    df['Edad2'] = df['Edad'] ** 2
    return df

def create_km_edad(df):
    df['Km_Edad'] = df['Kilómetros'] * df['Edad']
    return df

def create_log_km(df):
    df['Log_Kilómetros'] = np.log(df['Kilómetros'] + 1)
    return df

def create_log_edad(df):
    df['Log_Edad'] = np.log(df['Edad'] + 1)
    return df
