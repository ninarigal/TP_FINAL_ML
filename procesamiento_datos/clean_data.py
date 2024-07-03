import pandas as pd
import numpy as np
import re
from fuzzywuzzy import process
import difflib

def clean_dataset(df, mode): 
    df = to_lower_case(df)
    df = remove_special_characters(df) 

    df = clean_marcas(df)
    df = clean_modelos(df)

    df = clean_año(df, mode)
    df = create_year_column(df, mode)

    df = clean_transmission(df)
    if 'Precio' in df.columns:
        df = convert_to_dollars(df)
        df = clean_precios(df, mode)
        df = df.drop(columns=['Moneda'])
    df = clean_km(df)
    df = clean_versiones(df)
    df = clean_motores(df)
    df = df.drop(columns=[ 'Tipo de combustible', 'Tipo de carrocería', 'Con cámara de retroceso', 'Puertas', 'Año', 'Título versión', 'TV', 'Título versión motor'])
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

def create_year_column(df, mode, current_year=2024):
    #df['Año'] = pd.to_numeric(df['Año'], errors='coerce')
    #df.dropna(subset=['Año'], inplace=True)
    df['Año'] = df['Año'].astype(float)

    df['Edad'] = current_year - df['Año']
    #df['Edad'] = pd.to_numeric(df['Edad'], errors='coerce')
    #df.dropna(subset=['Edad'], inplace=True)


    if mode == 'train':
        etron = df[df['Modelo'] == 'etron']
        for e in etron.index:
            if df['Edad'][e] == 2:
                df = df.drop(e)
    df.reset_index(drop=True, inplace=True)
    return df

def clean_transmission(df):
    # VER DE LOS QUE NO ESTAN ASIGNADOS SI POR EJEMPLO EN EL TITULO TIENE UN AT O MT
    df['Transmisión'] = df['Transmisión'].replace('semiautomatica', 'automatica secuencial')
    df['Transmisión'] = df['Transmisión'].replace('automatica secuencial', 'automatica')
    for i in range(len(df)):
        if df['Transmisión'][i] != 'manual' and df['Transmisión'][i] != 'automatica secuencial' and df['Transmisión'][i] != 'automatica':
            if 'at' in df['Título'][i] or '5at' in df['Título'][i] or 'aut' in df['Título'][i]:
                df.loc[i, 'Transmisión'] = 'automatica'
            elif 'mt' in df['Título'][i]:
                df.loc[i, 'Transmisión'] = 'manual'
            else: 
                df.loc[i, 'Transmisión'] = 'automatica'
    return df

def convert_to_dollars(df, dollar=1045):
    #df['Precio'] = pd.to_numeric(df['Precio'], errors='coerce')
    #df.dropna(subset=['Precio'], inplace=True)
    df['Precio'] = df['Precio'].astype(float)
    for i in range(len(df)):
        if df['Moneda'][i] == '$':
            df.loc[i, 'Precio'] = df.loc[i, 'Precio'] / dollar # 1 peso = 1045 dolars (9th may 2024)
    df.reset_index(drop=True, inplace=True)                
    return df

def clean_marcas(df):
    marcas = ['peugeot', 'toyota', 'fiat', 'ds', 'ssangyong', 'chevrolet',
       'citroen', 'isuzu', 'ford', 'renault', 'porsche', 'jeep',
       'mercedesbenz', 'mini', 'honda', 'hyundai', 'volkswagen',
       'land rover', 'audi', 'geely', 'jaguar', 'daihatsu', 'subaru',
       'suzuki', 'haval', 'dodge', 'nissan', 'lexus', 'kia', 'mitsubishi',
       'lifan', 'jac', 'bmw', 'alfa romeo', 'chery', 'baic', 'jetour',
       'volvo']
    for i in range(len(df)):
        if df['Marca'][i] in ['abarth']:
            df.loc[i, 'Marca'] = 'fiat'
    def get_closest_brand(brand):
        closest_match, score = process.extractOne(brand, marcas)
        if score > 70: 
            return closest_match
        return brand
    marcas_para_revisar = df[~df['Marca'].isin(marcas)]['Marca'].copy()
    marcas_actualizadas = marcas_para_revisar.apply(lambda x: get_closest_brand(x))
    df.loc[~df['Marca'].isin(marcas), 'Marca'] = marcas_actualizadas
    return df

def clean_modelos(df):
    df['Modelo'] = df['Modelo'].replace('7', 'ds7')
    df['Modelo'] = df['Modelo'].replace('ml', 'clase ml')
    df['Modelo'] = df['Modelo'].replace('clase e', 'clase gle')
    df['Modelo'] = df['Modelo'].replace('coupe', 'clase glc')
    df['Modelo'] = df['Modelo'].replace('bronco sport', 'bronco')
    df['Modelo'] = df['Modelo'].replace('c4', 'c4 cactus')
    df['Modelo'] = df['Modelo'].replace('c3', 'c3 aircross')
    # modelos = ['2008', '208', '3008', '4008', '4runner', '5008', '500x', 'ds7',
    #    'actyon', 'agile', 'aircross', 'amigo', 'blazer', 'bronco', 'chr', 'c3 aircross', 'c4 aircross',
    #    'c4 cactus', 'c5 aircross', 'captiva', 'captur', 'cayenne',
    #    'cherokee', 'clase gl', 'clase gla', 'clase glb',
    #    'clase glc', 'clase gle', 'clase glk', 'clase ml', 'commander',
    #    'compass', 'cooper countryman', 'corolla cross', 'countryman',
    #    'crv', 'creta', 'crossfox', 'defender', 'discovery',
    #    'ds3', 'duster', 'duster oroch', 'etron',
    #    'ecosport', 'emgrand x7 sport', 'equinox', 'evoque', 'explorer',
    #    'fpace', 'feroza', 'forester', 'freelander', 'galloper',
    #    'grand blazer', 'grand cherokee', 'grand santa fe', 'grand vitara',
    #    'h1', 'h6', 'hilux', 'hilux sw4', 'hrv', 'jimny', 'jolion',
    #    'journey', 'kangoo', 'kicks', 'koleos', 'kona', 'kuga',
    #    'land cruiser', 'lx', 'macan', 'mohave', 'montero', 'murano',
    #    'musso', 'mustang', 'myway', 'nativa', 'nivus', 'nx', 'oroch',
    #    'outback', 'outlander', 'panamera', 'pathfinder', 'patriot',
    #    'pilot', 'pulse', 'q2', 'q3', 'q3 sportback', 'q5', 'q5 sportback',
    #    'q7', 'q8', 'range rover', 'range rover sport', 'rav4', 'renegade',
    #    'rodeo', 's2', 's5', 'samurai', 'sandero', 'sandero stepway',
    #    'santa fe', 'seltos', 'serie 4', 'sorento', 'soul', 'spin',
    #    'sportage', 'sq5', 'stelvio', 'suran', 'sw4', 'tcross', 'taos',
    #    'terios', 'terrano ii', 'territory', 'tiggo', 'tiggo 2', 'tiggo 3',
    #    'tiggo 4', 'tiggo 4 pro', 'tiggo 5', 'tiggo 8 pro', 'tiguan',
    #    'tiguan allspace', 'touareg', 'tracker', 'trailblazer', 'trooper',
    #    'tucson', 'ux', 'veracruz', 'vitara', 'wrangler', 'xterra',
    #    'xtrail', 'x1', 'x2', 'x25', 'x3', 'x35', 'x4', 'x5', 'x55', 'x6',
    #    'x70', 'xc40', 'xc60']
    # def get_closest_model(model):
    #     closest_match, score = process.extractOne(model, modelos)
    #     if score > 70: 
    #         return closest_match
    #     return model
    # modelos_para_revisar = df[~df['Marca'].isin(modelos)]['Marca'].copy()
    # modelos_actualizados = modelos_para_revisar.apply(lambda x: get_closest_model(x))
    # df.loc[~df['Marca'].isin(modelos), 'Marca'] = modelos_actualizados
    return df

def clean_precios(df, mode):
    if mode == 'train':
        df = df.drop(df[df['Precio'] < 1000 ].index) # Eliminar precios menores a 1000
        df = df.drop(df[(df['Edad'] == 0) & (df['Precio'] > 600000)].index)
        df = df.drop(df[(df['Edad'] == 0) & (df['Precio'] < 15000)].index)
        df = df.drop(df[(df['Edad'] == 1) & (df['Precio'] < 12000)].index)
        df = df.drop(df[(df['Edad'] < 5) & (df['Precio'] < 8000)].index)
        df.reset_index(drop=True, inplace=True)
    return df

def clean_km(df, mode='train'):
    df['Kilómetros'] = df['Kilómetros'].replace(' km', '', regex=True).astype(int)
    for i in range(len(df)):
        if df['Kilómetros'][i] in [111, 1111, 11111, 111111, 1111111, 99999, 999999, 9999999]:
            df.loc[i, 'Kilómetros'] = df.loc[i, 'Edad'] * 10000
    
        if df.loc[i, 'Edad'] < 0 or df.loc[i, 'Edad'] > 100:
                df.loc[i, 'Edad'] = int(df.loc[i, 'Kilómetros'] /10000)

        if df['Kilómetros'][i] > 100000 and df['Edad'][i] < 3:
            df.loc[i, 'Kilómetros'] = df.loc[i, 'Edad'] * 10000

    return df

def clean_año(df, mode, current_year=2024):
    df['Año'] = df['Año'].astype(float)
    if mode == 'train':
        for i in range(len(df)):
            if df['Año'][i] > current_year:	
                df.loc[i, 'Año'] = 2012 
        for i in range(len(df)):
            if df['Año'][i] > current_year or df['Año'][i] < 0:
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

def clean_vendedor(df):
    df['Tipo de vendedor'] = df['Tipo de vendedor'].replace('tienda', 'concesionaria')
    df['Tipo de vendedor'] = df['Tipo de vendedor'].replace(np.nan, 'particular')
    return df


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
            df_chevrolet_tracker= df_marca[df_marca['Modelo'] == 'tracker'].copy()
            df_chevrolet_tracker['Versión final'] = df_chevrolet_tracker['Título versión'].apply(lambda x: map_version(x, keywords))
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
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['beng'], 'big bend')).apply(lambda x: replace(x, ['titanuim', 'titaniun'], 'titanium')).apply(lambda x: replace(x, ['freest'], 'freestyle'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'volvo':
            df_marca = df[df['Marca'] == 'volvo'].copy()
            keywords = ['momentum', 'r design', 'inscription',  'plus', 'ultimate', 'luxury', 'comfort', 'high']
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'alfa romeo':
            df_marca = df[df['Marca'] == 'alfa romeo'].copy()
            keywords = ['distinctive', 'super', 'veloce'] 
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'renault':
            df_marca = df[df['Marca'] == 'renault'].copy()
            keywords = ['intens', 'zen', 'life', 'bose', 'confort', 'iconic', 'privilege', 'dynamique', 'expression', 'luxe', 'dakar', 'outsider', 'tech road', 'los pumas', 'emotion', '4wd', 'volcom']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['intense'], 'intens'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'porsche':
            df_marca = df[df['Marca'] == 'porsche'].copy()
            keywords = ['300cv', '400cv', '245cv', '252cv', '340cv', '500cv'] #NO ENTIENDO COMO SON !!!
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'jeep':
            df_marca = df[df['Marca'] == 'jeep'].copy()
            keywords = ['limited', 'sport', 'classic', 'trailhawk', 'longitude', 's', 'overland', 'laredo', 'srt', 'rubicon', 'unlimited', 'aniversario', 'se', 'opening', 'at6']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['limite'], 'limited')).apply(lambda x: replace(x, ['longitud'], 'longitude')).apply(lambda x: replace(x, ['srt8'], 'srt')).apply(lambda x: replace(x, ['anniversary'], 'aniversario')).apply(lambda x: replace(x, ['s'], 'sport'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'mercedesbenz':
            df_marca = df[df['Marca'] == 'mercedesbenz'].copy()
            keywords = ['amg', 'sport', 'progressive', 'urban', 'advance', 'luxury', 'cdi', 'coupe', 'nature', '4matic']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['4 matic'], '4matic'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'mini':
            df_marca = df[df['Marca'] == 'mini'].copy()
            keywords = ['s', 'jcw', 'peppers']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['classic'], 's')).apply(lambda x: replace(x, ['jonh cooper works', 'john cooper works'], 'jcw'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'volkswagen':
            df_marca = df[df['Marca'] == 'volkswagen'].copy()
            keywords = ['comfortline', 'highline', 'trendline', 'life', 'elegance', 'premium', 'hero', 'sport', 'exclusive', 'track', 'motion', 'style', 'allspace', 'dsg']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['confortline', 'b170', 'confort', ''], 'comfortline')).apply(lambda x: replace(x, ['trendlinde', 'trendlin'], 'trendline')).apply(lambda x: replace(x, ['high'], 'highline')).apply(lambda x: replace(x, ['4motion'], 'motion'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'honda':
            df_marca = df[df['Marca'] == 'honda'].copy()
            keywords = ['lx', 'ex', 'si', 'ext', 'i', 'exl', 'active']
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)

        elif marca == 'bmw':
            df_marca = df[df['Marca'] == 'bmw'].copy()
            keywords = ['sdrive', 'xdrive', 'premium', 'm', 'selective', 'mpa']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['s drive', 'sdrive20i', 'sport', 'sportline', 'msport', 'msportx', '18i', '25i'], 'sdrive')).apply(lambda x: replace(x, ['x drive', 'x5xdrive40i', 'executive',  '30e', 'xdrive25i', 'x25i', '30i', '20i',  '40i' ], 'xdrive')).apply(lambda x: replace(x, ['m35i'], 'mpa'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'lexus':
            df_marca = df[df['Marca'] == 'lexus'].copy()
            keywords = ['luxury', 'f sport']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['ux250h', '250h'], ' luxury'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'nissan':
            df_marca = df[df['Marca'] == 'nissan'].copy()
            keywords = ['sense', 'exclusive', 'advance', 'uefa', 'se', 'd', 'i', 'acenta', 'e power', 'tekna', 'visia', 'nft', '4x4']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['sence'], 'sense')).apply(lambda x: replace(x, ['excluisive', '3.5'], 'exclusive')).apply(lambda x: replace(x, ['special edition'], 'se')).apply(lambda x: replace(x, ['uefe'], 'uefa')).apply(lambda x: replace(x, ['teckna'], 'tekna')).apply(lambda x: replace(x, ['epower'], 'e power'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'subaru':
            df_marca = df[df['Marca'] == 'subaru'].copy()
            keywords = ['awd', 'dynamic', 'xs', 'r', 'x', 'a']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['sawd'], 'awd'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'haval':
            df_marca = df[df['Marca'] == 'haval'].copy()
            keywords = ['luxury', 'elite', 'great wall', 'coupe']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['4wd'], 'great wall')).apply(lambda x: replace(x, ['2wd'], 'coupe'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'dodge':
            df_marca = df[df['Marca'] == 'dodge'].copy()
            keywords =['se', 'sxt', 'rt']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['r/t'], 'rt'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'suzuki':
            df_marca = df[df['Marca'] == 'suzuki'].copy()
            keywords = ['jlx', 'xl', 'jiii', 'glx', 'jx', 'mt']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['gl plus', 'gl'], 'glx')).apply(lambda x: replace(x, ['jx', '1.3'], 'jlx')).apply(lambda x: replace(x, ['2.0', '2.0td'], 'mt'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'daihatsu':
            df_marca = df[df['Marca'] == 'daihatsu'].copy()
            keywords = ['1.6', '1.3']
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'lifan':
            df_marca = df[df['Marca'] == 'lifan'].copy()
            keywords=['1.8', '2.0']
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'mitsubishi':
            df_marca = df[df['Marca'] == 'mitsubishi'].copy()
            keywords = ['gls', 'glx', 'gl', 'gt', 'ls', 'semi high']
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'kia':
            df_marca = df[df['Marca'] == 'kia'].copy()
            keywords = ['ex', 'sx', 'x line', 'lx', 'rock', 'pop', 'classic']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['xline'], 'x line')).apply(lambda x: replace(x, ['exclusive'], 'ex'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'jaguar':
            df_marca = df[df['Marca'] == 'jaguar'].copy()
            keywords = ['sc prestige', 'prestige']
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'geely':
            df_marca = df[df['Marca'] == 'geely'].copy()
            keywords = ['gs', 'gt', 'gl']
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'jac':
            df_marca = df[df['Marca'] == 'jac'].copy()
            keywords = ['intelligent']
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'chery':
            df_marca = df[df['Marca'] == 'chery'].copy()
            keywords = ['comfort', 'luxury']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['confort'], 'comfort'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'jetour':
            df_marca = df[df['Marca'] == 'jetour'].copy()
            keywords  = ['lt']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['turbo', 'premium', '7'], 'lt'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'audi':
            df_marca = df[df['Marca'] == 'audi'].copy()
            keywords = ['quattro', 'advanced', 'sport', 'stronic', 'offroad']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['quatro'], 'quattro'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'land rover':
            df_marca = df[df['Marca'] == 'land rover'].copy()
            keywords = ['sw', 'x', 'se', 'dynamic','sport', 'prestige', 'i', 'coupe', 'xedi', 'hse']
            df_marca['Título versión'] = df_marca['Título versión'].apply(lambda x: replace(x, ['plus'], 'prestige'))
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
        elif marca == 'hyundai':
            df_marca = df[df['Marca'] == 'hyundai'].copy()
            keywords = ['safety', 'style', 'gl', 'exd', 'xl', 'crdi', 'gls', 'tgdi', 'premium']
            df_marca['Versión final'] = df_marca['Título versión'].apply(lambda x: map_version(x, keywords))
            df.update(df_marca)
    
    return df         

def clean_motores(df):
    df['TV'] = df['Título'].str.cat(df['Versión'], sep=' ')
    df['Título versión motor'] = df['TV'].str.cat(df['Motor'], sep=' ')
    df['Motor final'] = df['Título versión motor']
    keywords = ['1.0', '1.2', '1.3', '1.4', '1.5', '1.6', '1.8', '1.9', '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7', '2.8', '2.9', '3.0', '3.1', '3.2', '3.3', '3.5', '3.6', '3.7', '3.8', '4.0', '4.2', '4.3', '4.4', '4.5', '4.7', '4.8', '5.0', '5.2', '5.5', '5.7', '6.1', '6.4', '55 s']

    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['1.0t', 'fiat pulse audace', 'aircross t200 shine', 'aircros shine t200', 'aircross t200 at7 shine', '1,0', 'nivus', '1.0l', 'volkswagen tcross 170 tsi 170 tsi 170 tsi', '1.0tsi', 'volkswagen tcross 170tsi'], '1.0'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['chevrolet tracker tracker lt', '1.2t'], '1.2'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['fiat pulse abarth', '1.3t', '1.3l', 'jeep compass s s nan', 'jeep compass serie s serie s nan'], '1.3'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['1,4', 'taos', '1.4t', '1.4l', 'volkswagen tiguan allspace 250 250 nan', 'audi q3 sportback 35 tfsi okm 2024 35 tfsi 35 tfsi 150 cv'], '1.4'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['1.5t', 'mini countryman cooper classic genco mini classic 136 cv', 'honda hrv lx nan', 'hyundai new creta new creta nan', 'bmw x1 18i sdrive 18i sdrive 1500 cc', 'bmw x1 18i okm 2024', 'bmw x1 18i', 'chery tiggo 2 luxury', 'xc40 t5 262hp', 'xc40 t5 awd'], '1.5'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['active pack at', '1.6thp', 'gt pack hybrid 2022 smart garage', 'peugeot 5008 allure', 'peugeot 5008 gt pack', 'peugeot 3008 gt pack', '1.6t', 'ds7 bastille', '1,6', 'c4 cactus feel', 'c4 cactus shine', 'c4 cactus vti at feel', '1.6i', 'c4 cactus noir', 'renault captur bose', 'renault sandero stepway expression', 'mercedez benz gla 250 4matic 2022 gla 250 4matic nan', 'mercedes benz clase gla 2016 250', '1.6l', 'nissan kicks', 'kia sportage lx'], '1.6'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['toyota chr hev cvt', 'toyota corolla cross seg hybrid', 'seg hybrida ecvt' '1.8l', 'chevrolet tracker awd premier', 'chevrolet tracker 2024 automatica lt', 'premier 1800', '1.8t', 'jeep renegade sport sport nan', 'jeep renegade sport sport', 'jeep renegade 80 aniversario edic 80 aniversario edic nan', 'jeep renegade 1.3t longitude t270 1.3t longitude t270', 'jeep renegade sport mt sport mt nan', '1.8i'], '1.8'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['corolla cross xli', '2,0', '2.0l', 'ford ecosport storm', 'mercedesbenz clase glc glc 300 amgline glc 300 amgline nan', 'mercedesbenz glc 300 coupe glc 300 coupe nan', 'glc 300 coupe glc 300 coupe nan', 'mercedes benz glc 300 coupe amg line', 'mini countryman cooper s classic confort genco mini countryman cooper s classic confort 192 hp', 'honda crv exl exl naftero', 'tiguan life 350 tsi 4m life 350 tsi 4m 2.0 l 230 cv 350 tsi', '2.0t', 'audi q5 45 tfsi advanced 0km 2024 45 tfsi advanced 45 tfsi 245 cv', '2.0tfsi', 'bmw x1 20i xdrive okm año 2024', 'bmw x1 25i año 2018 4x4 automatica', 'bmw x1 xdrive 20 i', 'bmw x1 xdrive 25i', 'bmw x1 25i xdrive', 'bmw x2 35i', 'bmw x2 m35i', 'volvo xc40 b4', 'volvo xc40 t4', 'volvo xc60 t8 462hp'], '2.0'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['honda crv lx 2016 lx nan', 'suzuki grand vitara jlx td mazda diesel jlx td mazda diesel mazda', '"2.4', '"""2.4', 'mitsubishi outlander at 4x4 gls'], '2.4'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['toyota rav4 limited', 'toyota rav4 4wd', 'toyota rav limited', '2.5l', '2.5i', 'nissan xtrail exclusive', 'xtrail epower', 'xtrail teckna', '2.5t'], '2.5'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['corolla cross xli', 'range rover sport tdv6 hse 4x4 at'], '2.7'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['toyota hilux sw4 srx', 'toyota sw4 diamond', 'toyota sw4 srx', 'toyota sw4 gr', '2.8tdi', 'chevrolet trailblazer premier'], '2.8'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['mercedes benz gle 53 amg 53 amg 6 cilindros', 'mercedesbenz gle 53 gle 53s 3.0 6 cilindros', 'luxury 3', 'bmw x5 40i xdrive', 'bmw x5 40i', 'bmw x3e hibrida enchufafle', 'bmw x3 xdrive 30i', 'bmw x3 xdrive 30e', 'bmw x3 m40i sport 2018', 'bmw x3 30e', 'bmw x3 30i', 'bmw x3 hibrida enchufable okm 2024', 'volvo xc60 inscription t6', 'volvo xc60 t6'], '3.0'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['mercedes benz ml 350 350 sport 4matic nan', 'mercedesbenz clase ml 350 blue efficiency blue efficiency nan', 'mercedes benz ml 350 4 matic 2012 ml 350 4 matic luxury 3498 cm3', 'honda pilot ex 2021 v6 ex v6 v'], '3.5'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['porsche macan turbo electric', 'porsche macan electric (g4)', 'jeep grand cherokee limited 4x4 2024 0km atx6 limited v6', 'jeep grand cherokee limited 4x4 2024 0km limited v6', 'jeep grand cherokee limited grand cherokee limited nan', '3.6l', 'jeep cherokee nc cherokee v6'], '3.6'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['jeep wrangler wrangler unlimited sport v6 wrangler unlimited sport v6 v6'], '3.8'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['mercedes benz ml 63 amg 63 amg nan'], '4.0'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['4.4l', 'bmw x6 m'], '4.4'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['mercedes benz gle 53 amg | blindaje rb3 gle 53 amg nan'], '5.5'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['lexus lx 570 año 2020'], '5.7'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['etron sportback s line 55 quattro', 'nuevo audi etron 55 quattro'], '55 s'))
    df['Título versión motor'] = df['Título versión motor'].apply(lambda x: replace(x, ['1.8l'], '1.8'))
    df['Motor final'] = df['Título versión motor'].apply(lambda x: map_version(x, keywords))
   
    return df
        

def clean_data_test(df):
    # Filtrar las filas donde el precio es menor a 1000
    df = df[df['Precio'] >= 1000]

    # Crear una máscara booleana combinando todas las condiciones
    mask = ~(
        ((df['Modelo'] == 'bronco') & (df['Versión final'] == 'wildtrak') & (df['Precio'] > 80000)) |
        ((df['Modelo'] == '3008') & (df['Versión final'] == 'other') & (df['Precio'] > 80000)) |
        ((df['Modelo'] == 'renegade') & (df['Versión final'] == 'sport') & (df['Precio'] > 80000))
    )

    # Aplicar la máscara para filtrar el DataFrame
    df = df[mask]

    df.reset_index(drop=True, inplace=True)
    return df



    

# def main():
#     df = pd.read_csv("pf_suvs_i302_1s2024.csv")
#     df = clean_dataset(df, 'test')
#     df.to_csv("cleaned_data.csv", index=False)
#     print("Data cleaned and saved")


# main()


