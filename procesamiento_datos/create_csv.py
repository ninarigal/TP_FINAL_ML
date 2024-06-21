import pandas as pd
import numpy as np
from clean_data import clean_dataset
from divide_data import divide_dev_test
from create_features import new_features


if __name__ == '__main__':
    file_path = 'pf_suvs_i302_1s2024.csv'
    data_dev, data_test = divide_dev_test(file_path)
    print(data_dev.shape)
    print(data_test.shape)
 
    data_dev = clean_dataset(data_dev, mode='train')
    data_dev = new_features(data_dev, mode='train')
    data_test = clean_dataset(data_test, mode='test')
    data_test = new_features(data_test, mode='test')
    data_dev.to_csv('data_dev.csv', index=False)
    data_test.to_csv('data_test.csv', index=False)




