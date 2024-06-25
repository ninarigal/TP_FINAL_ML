import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split


file_path = r'data_dev.csv'
data = pd.read_csv(file_path)
# Definir las características y la variable objetivo

groups = data['Modelo']
modelos_uno = []
for modelo in groups.unique():
    if data[groups == modelo].shape[0] < 2:
        print(f"El modelo {modelo} tiene menos de 2 autos")
        modelos_uno.append(modelo)

#Eliminar modelos_uno de groups


# Eliminar los autos de los modelos con menos de 2 autos
#data_sin = data[~groups.isin(modelos_uno)]
data_sin = data[~data['Modelo'].isin(modelos_uno)]
data_sin.reset_index(drop=True, inplace=True)

groups = data_sin['Modelo']


X = data_sin.drop('Precio', axis=1)
y = data_sin['Precio']

# Crear el objeto GroupShuffleSplit
#gss = GroupShuffleSplit(valid_size=0.2, n_splits=1, random_state=0)


        
success = False
i = 0
while not success:
    i += 1
    print(f"Intento {i}")
    """
    for train_idx, valid_idx in gss.split(X, y, groups=groups):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
    """
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True, stratify=groups)

    # Verificar que cada conjunto contenga al menos un auto de cada modelo
    train_models = set(X_train['Modelo'].unique())
    valid_models = set(X_valid['Modelo'].unique())

    # Asegurarse de que cada conjunto tenga al menos un auto de cada modelo
    all_models = set(groups)

    #imprimir los que quedaron distintos
    print(all_models.difference(valid_models))
    if all_models.issubset(train_models) and all_models.issubset(valid_models):
        success = True
    else:

        # Identificar modelos faltantes en X_valid y agregar un ejemplo de cada modelo faltante
        models_to_add = all_models - valid_models
        for modelo in models_to_add:
            print(f"Modelo faltante: {modelo}")
            # Encontrar un ejemplo del modelo en X_train
            example_idx = X_train[X_train['Modelo'] == modelo].index[0]
            
            # Agregar el ejemplo a X_valid y su etiqueta correspondiente a y_valid
            X_valid = pd.concat([X_valid, X_train.loc[[example_idx]]], ignore_index=True)
            y_valid = pd.concat([y_valid, y_train.loc[[example_idx]]], ignore_index=True)
            
            # Eliminar el ejemplo de X_train e y_train
            X_train.drop(example_idx, inplace=True)
            y_train.drop(example_idx, inplace=True)

        valid_models = set(X_valid['Modelo'].unique())
        train_models = set(X_train['Modelo'].unique())


        if all_models.issubset(train_models) and all_models.issubset(valid_models):
            success = True

        #agrego los modelos que faltan (que tienen un solo auto) a X_train y y_train

        idx = data[data['Modelo'].isin(modelos_uno)].index
        print("Modelos con un solo auto:", data.loc[idx, 'Modelo'].unique)
        X_train = pd.concat([X_train, data.loc[idx].drop('Precio', axis=1)], ignore_index=True)
        y_train = pd.concat([y_train, data.loc[idx, 'Precio']], ignore_index=True)

        valid_models = set(X_valid['Modelo'].unique())
        train_models = set(X_train['Modelo'].unique())


print("Modelos en el conjunto de entrenamiento:", train_models)
print("Modelos en el conjunto de prueba:", valid_models)
print("Cada conjunto contiene al menos un auto de cada modelo")

#quiero guardar x e y
print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
X_train['Precio'] = y_train
X_valid['Precio'] = y_valid
print(X_train.shape)
print(X_valid.shape)

# Guardar los conjuntos de entrenamiento y prueba
X_train.to_csv('data_train.csv', index=False)
X_valid.to_csv('data_valid.csv', index=False)

