import pandas as pd
import numpy as np
from clean_data import clean_dataset
from divide_data import divide_dev_test
from features import add_features


if __name__ == '__main__':
    file_path = 'pf_suvs_i302_1s2024.csv'
    data_dev, data_test = divide_dev_test(file_path)
    print(data_dev.shape)
    print(data_test.shape)
 
    data_dev = clean_dataset(data_dev, mode='train')
    data_dev = add_features(data_dev, mode='train')
    data_test = clean_dataset(data_test, mode='test')
    data_test = add_features(data_test, mode='test')
    data_dev.to_csv('data_dev.csv', index=False)
    data_test.to_csv('data_test.csv', index=False)

    # # separo en train y valid
    # data_train = data_dev.sample(frac=0.8, random_state=0)
    # data_valid = data_dev.drop(data_train.index)
    # data_train.to_csv('data_train.csv', index=False)
    # data_valid.to_csv('data_valid.csv', index=False)




