import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def divide_dev_test(file_path):
    data = pd.read_csv(file_path)
    data_dev, data_test = train_test_split(data, test_size=0.1, random_state=0)

    data_dev.to_csv('data_dev.csv', index=False)
    data_test.to_csv('data_test.csv', index=False)

    return data_dev, data_test




    