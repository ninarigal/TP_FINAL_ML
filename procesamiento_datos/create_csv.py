import pandas as pd
import numpy as np
from clean_data import clean_dataset, clean_data_test
from divide_data import divide_dev_test
from features import add_features


if __name__ == '__main__':
    file_path = 'procesamiento_datos/pf_suvs_i302_1s2024.csv'
    data_dev, data_test = divide_dev_test(file_path)
    print(data_dev.shape)
    print(data_test.shape)
 
    data_dev = clean_dataset(data_dev, mode='train')
    data_dev = add_features(data_dev, mode='train')
    data_test = clean_dataset(data_test, mode='test')
    data_test = add_features(data_test, mode='test')
    data_dev.reset_index(drop=True, inplace=True)
    data_test.reset_index(drop=True, inplace=True)
    data_test = clean_data_test(data_test)
    data_dev.to_csv('procesamiento_datos/data_dev.csv', index=False)
    data_test.to_csv('procesamiento_datos/data_test.csv', index=False)



