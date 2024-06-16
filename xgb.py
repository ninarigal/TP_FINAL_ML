import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#XGBOOST
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class XGBoost:
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.model = XGBRegressor()

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, y_test, y_preds):
        mse = mean_squared_error(y_test, y_preds)
        r2 = r2_score(y_test, y_preds)
        return mse, r2
    
    def train_test_split(self, data):
        X = data[self.features]
        Y = data[self.target]
        return train_test_split(X, Y, test_size=0.3, random_state=0)
    
    def plot(self, y_test, y_preds):
        plt.scatter(y_test, y_preds, color='black', alpha=0.5, s=10, )
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='y=x', linestyle='--')
        plt.xlabel('Valores reales')
        plt.ylabel('Predicciones')
        plt.title('Valores reales vs Predicciones')
        plt.show()
                
               

def xgboost(data, target):
    X = data.drop([target], axis=1)
    xgb = XGBoost(X.columns, 'Precio')
    X_train, X_test, Y_train, Y_test = xgb.train_test_split(data)
    xgb.fit(X_train, Y_train)
    Y_pred = xgb.predict(X_test)
    
    #for i in range(len(Y_test)):
    #    print(f'Real: {y_test.iloc[i]}, Predicción: {y_preds[i]}')
       # if (Y_test.iloc[i] - Y_pred[i] > 10000):
        #    print(f'Real: {Y_test.iloc[i]}, Predicción: {Y_pred[i]}')
        #    print(f'Diferencia: {Y_test.iloc[i] - Y_pred[i]}')
            
           
    mse, r2 = xgb.evaluate(Y_test, Y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R2 Score: {r2}')
    xgb.plot(Y_test, Y_pred)
