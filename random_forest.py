#random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class RandomForest:
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.model = RandomForestRegressor(n_estimators=100, random_state=0)

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
        return train_test_split(X, Y, test_size=0.2, random_state=0)
    
    def plot(self, y_test, y_preds):
        plt.scatter(y_test, y_preds, color='black', alpha=0.5, s=10, )
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='y=x', linestyle='--')
        plt.xlabel('Valores reales')
        plt.ylabel('Predicciones')
        plt.title('Valores reales vs Predicciones')
        plt.show()

def random_forest(data, target):
    X = data.drop([target], axis=1)
    rf = RandomForest(X.columns, 'Precio')
    X_train, X_test, Y_train, Y_test = rf.train_test_split(data)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)
    mse, r2 = rf.evaluate(Y_test, Y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R2 Score: {r2}')
    rf.plot(Y_test, Y_pred)