import pandas as pd
import glob
import numpy as np


def read_data(path: str):
    """
    Read the data from the given path
    :param path: path of the data
    :return: data
    """
    data = pd.read_csv(path)
    return data


def null_separator(data):
    """
    Separate the null values from non-null values
    :param data: data to be separated
    :return: non-null values
    """
    # null_data = data[data.isnull().any(axis=1)]
    non_null_data = data.dropna()
    return non_null_data


def null_replacer(data):
    """
    Replace the null values with mean of the column
    :param data: data to be replaced
    :return: data with replaced null values
    """
    data = data.fillna(data.mean())
    return data


def outlier_remover(data):
    """
    Remove the outliers from the data
    :param data: data to be cleaned
    :return: cleaned data
    """
    data = data[(np.abs(data - data.mean()) <= (3 * data.std())).all(axis=1)]
    return data


def split_data(data, ratio: float):
    """
    Split the data into train and test sets
    :param data: data to be split
    :param ratio: ratio of the split
    :return: train and test sets
    """
    train = data.sample(frac=ratio)
    test = data.drop(train.index)
    return train, test


def save_data(data, path: str):
    """
    Save the data to the given path
    :param data: data to be saved
    :param path: path to save the data
    :return: None
    """
    data.to_csv(path, index=False)



if __name__ == '__main__':

    path = 'Classification Datasets/'
    # data = read_data(path)
    # data = null_separator(path)
    # ...
    # train, test = split_data(data)
    # save_data(train, 'train'+ "..." + '.csv')
    # save_data(test, 'test' + "..." + '.csv') 


