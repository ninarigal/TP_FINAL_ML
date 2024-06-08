import pandas as pd
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Cargar los datos
data = pd.read_csv('data.csv')

# A la columna de Kilómetros le sacamos el km y lo convertimos a número
data['Kilómetros'] = data['Kilómetros'].str.replace(' km', '').str.replace(',', '').astype(int)

# Visualizar los primeros registros para asegurarnos de que se han cargado correctamente
print(data.head())

# Definir las características (features) y el objetivo (target)
X = data[['Kilómetros', 'Edad', 'Modelo']]
y = data['Precio']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_max = y_train.max()
y_test = y_test / y_train.max()
y_train = y_train / y_train.max()

# Codificar la columna de modelos utilizando target encoding
encoder = TargetEncoder(cols=['Modelo'])
X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)

X_train = X_train.astype(float)
X_test = X_test.astype(float)

X_test = X_test - X_train.min() / X_train.max() - X_train.min()
X_train = X_train - X_train.min() / X_train.max() - X_train.min()


# Crear el modelo
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test) * y_train_max
y_test = y_test * y_train_max

# Calcular el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calcular el coeficiente de determinación (R^2)
r2 = r2_score(y_test, y_pred)

print(f'Error Cuadrático Medio (MSE): {mse}')
print(f'Coeficiente de Determinación (R^2): {r2}')

# Guardar el modelo en un archivo
joblib.dump(model, 'modelo_precio_auto.pkl')

# Plotear el precio predecido de los autos contra el real
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('Precio Real')
plt.ylabel('Precio Predicho')
plt.title('Precio Real vs Precio Predicho')
plt.show()

