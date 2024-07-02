import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def divide_dev_test(file_path):
    data = pd.read_csv(file_path)
    data_dev, data_test = train_test_split(data, test_size=0.1, random_state=42)

    data_dev.reset_index(drop=True, inplace=True)
    data_test.reset_index(drop=True, inplace=True)

    return data_dev, data_test
    






    