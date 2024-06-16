
import pandas as pd
import numpy as np
import re

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
    df = clean_versiones(df)
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
    df = df.replace('ë', 'e', regex=True)
    return df

def create_year_column(df, current_year):
    df['Año'] = pd.to_numeric(df['Año'], errors='coerce')
    df.dropna(subset=['Año'], inplace=True)
    df['Edad'] = current_year - df['Año']
    return df

def clean_transmission(df):
    df['Transmisión'] = df['Transmisión'].replace('semiautomatica', 'automatica secuencial')
    #remplazar nan por automática
    for i in range(len(df)):
        if df['Transmisión'][i] != 'manual' and df['Transmisión'][i] != 'automatica secuencial' and df['Transmisión'][i] != 'automatica':
            df.loc[i, 'Transmisión'] = 'automatica'
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


def map_version(version, keywords):
    for keyword in keywords:
        if re.search(rf'\b{keyword}\b', version):
            return keyword
    return 'other'

def replace(version, keywords, keyword):
    for k in keywords: 
        if re.search(rf'\b{k}\b', version):
            return keyword
    return version

def clean_versiones(df):
    df['Título versión'] = df['Título'].str.cat(df['Versión'], sep=' ')
    df['Versión final'] = df['Título versión']
    for marca in df['Marca'].unique():
        if marca == 'peugeot':
            df_marca = df[df['Marca'] == 'peugeot'].copy()
            keywords = ['active', 'allure', 'feline', 'thp', 'concert', 'crossway', 'gt', 'hybrid'] 
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'toyota':
            df_marca = df[df['Marca'] == 'toyota'].copy()
            keywords = ['d', 'limited', 'hev', 'seg', 'xli', 'xei', 'gr', 'srx', 'sr', 'srv', 'runner', 'diamond', 'vx', 'dl', 'tx', 'xle', 'wide body', '4x4', '4x2'] # ver bien la clasificacion porque nos da dudas !!!
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['grsport', 'grs', 'gazoo racing'], 'gr')).apply(lambda x: replace(x, ['diamons', 'diamante'], 'diamond')).apply(lambda x: replace(x, ['prado'], 'vx'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'ds':
            df_marca = df[df['Marca'] == 'ds'].copy()
            keywords = ['bastille', 'rivoli', 'chic', 'etense']
            df_marca['Título versión'].apply(lambda x: replace(x, ['rivolli'], 'rivoli'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'fiat':
            df_marca = df[df['Marca'] == 'fiat'].copy()
            keywords = ['pop', 'star', 'drive', 'cross', 'abarth', 'impetus', 'audace']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['audance'], 'audace'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'ssangyong':
            df_marca = df[df['Marca'] == 'ssangyong'].copy()
            keywords = ['601', '602', '2']
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'chevrolet':
            df_marca = df[df['Marca'] == 'chevrolet'].copy()
            keywords = ['turbo mt', 'turbo at', 'ltz']
            df_chevrolet_tracker = df_marca[df_marca['Modelo'] == 'tracker'].copy().apply(lambda x: map_version(x, keywords))
            df.update(df_chevrolet_tracker)
            keywords = ['lt', 'dlx', 'hp', 'ls', 'ltz', 'premier', 'rs', 'activ']
            df_marca = df_marca[df_marca['Modelo'] != 'tracker'].copy()
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'citroen':
            df_marca = df[df['Marca'] == 'citroen'].copy()
            keywords = ['shine', 'thp', 'first', 'feel', 'tendance', 'feel', 'live', 'noir', 'puretech', 'series', 'origins']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['pure tech'], 'puretech')).apply(lambda x: replace(x, ['firts'], 'first')).apply(lambda x: replace(x, ['tendence'], 'tendance')).apply(lambda x: replace(x, ['fell'], 'feel'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'isuzu':
            df_marca = df[df['Marca'] == 'isuzu'].copy()
            keywords = ['xs', 'ls', 'wagon'] 
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'baic':
            df_marca = df[df['Marca'] == 'baic'].copy()
            keywords = ['luxury', 'elite', 'confort turbo', 'luxury turbo', 'fahion turbo', 'fashion', 'turbo']
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'ford':
            df_marca = df[df['Marca'] == 'ford'].copy()
            keywords = ['wildtrack', 'big bend', 'titanium', 'xls', 'se', 's', 'xlt', 'xl', 'freestyle', 'storm', 'sel', 'platinum', 'e', 'trend', 'wildtrak', 'awd']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['beng'], 'big bend')).apply()


    return df         

        
def main():
    df = pd.read_csv("pf_suvs_i302_1s2024.csv")
    df = clean_dataset(df)
    df.to_csv("cleaned_data.csv", index=False)


main()
