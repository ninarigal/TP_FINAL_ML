
import pandas as pd
import numpy as np

def clean_dataset(df):
    df = to_lower_case(df)
    df = remove_special_characters(df)
    df = create_year_column(df, 2024)
    df = clean_transmission(df)
    df = convert_to_dollars(df)
    df = clean_marcas(df)
    df = clean_modelos(df)
    df = clean_precios(df)
    df = clean_km(df)
    df = clean_año(df)
    df = df.drop(columns=[ 'Tipo de combustible', 'Tipo de carrocería', 'Con cámara de retroceso', 'Puertas', 'Año', 'Moneda'])
        
    return df 

def to_lower_case(df):
    return df.apply(lambda x: x.astype(str).str.lower())

def remove_special_characters(df):
    df = df.replace('-', '', regex=True)	
    df = df.replace('á', 'a', regex=True)
    df = df.replace('é', 'e', regex=True)
    df = df.replace('í', 'i', regex=True)
    df = df.replace('ó', 'o', regex=True)
    df = df.replace('ú', 'u', regex=True)
    return df


def create_year_column(df, current_year):
    df['Año'] = pd.to_numeric(df['Año'], errors='coerce')
    df.dropna(subset=['Año'], inplace=True)
    df['Edad'] = current_year - df['Año']
    return df

def clean_transmission(df):
    df['Transmisión'] = df['Transmisión'].replace('semiautomatica', 'automatica secuencial')
    #remplazar nan por automática
    df = df.fillna({'Transmisión': 'automatica'})

    return df

def convert_to_dollars(df):
    df['Precio'] = pd.to_numeric(df['Precio'], errors='coerce')
    df.dropna(subset=['Precio'], inplace=True)
    for i in range(len(df)):
        if df['Moneda'][i] == '$':
            df.loc[i, 'Precio'] = df.loc[i, 'Precio'] / 1045 # 1 peso = 1045 dolars (9th may 2024)
    return df

def clean_marcas(df):
    ds = ['ds7', 'ds automobiles']
    renault = ['sandero']
    jetour = ['jetur']
    hyundai = ['hiunday']
    fiat = ['abarth']
    for i in range(len(df)):
        if df['Marca'][i] in ds:
            df.loc[i, 'Marca'] = 'ds' 
        elif df['Marca'][i] in renault:
            df.loc[i, 'Marca'] = 'renault'
        elif df['Marca'][i] in jetour:
            df.loc[i, 'Marca'] = 'jetour'
        elif df['Marca'][i] in hyundai:
            df.loc[i, 'Marca'] = 'hyundai'
        elif df['Marca'][i] in fiat:
            df.loc[i, 'Marca'] = 'fiat'
    return df

def clean_modelos(df):
    df['Modelo'] = df['Modelo'].replace('7', 'ds7')
    df['Modelo'] = df['Modelo'].replace('ml', 'clase ml')

    return df

def clean_precios(df):
    df = df.drop(df[df['Precio'] < 1000 ].index) # Eliminar precios menores a 1000

    #esto no es correcto para preprocesar en valid/test
    etron = df[df['Modelo'] == 'etron']
    for e in etron.index:
        if df['Edad'][e] == 2:
            df = df.drop(e)
    df.reset_index(drop=True, inplace=True)
    return df

def clean_km(df):
    df['Kilómetros'] = df['Kilómetros'].replace(' km', '', regex=True).astype(int)
    for i in range(len(df)):
        if df['Kilómetros'][i] in [111, 1111, 11111, 111111, 1111111, 99999, 999999, 9999999]:
            df.loc[i, 'Kilómetros'] = df.loc[i, 'Edad'] * 10000
        if df.loc[i, 'Edad'] < 0:
            df.loc[i, 'Edad'] = df.loc[i, 'Kilómetros'] /10000
    return df

def clean_año(df):
    #esto no es correcto para preprocesar en valid/test
    for i in range(len(df)):
        if df['Año'][i] > 2024:	
            df.loc[i, 'Año'] = 2012 
    #borrar filas con años incorrectos
    for i in range(len(df)):
        if df['Año'][i] > 2024 or df['Año'][i] < 0:
            df = df.drop(i)
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    df = pd.read_csv("pf_suvs_i302_1s2024.csv")
    df = clean_dataset(df)
    df.to_csv("cleaned_data.csv", index=False)


main()
