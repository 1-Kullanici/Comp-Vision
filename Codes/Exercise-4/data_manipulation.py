import pandas as pd
import glob
import numpy as np



def parse_files(path: str):
    """
    Parse the files in the given path
    :param path: path of the files
    :return: list of files
    """
    files = glob.glob(path + '*.csv')
    return files


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


def change_column(data, column, target_column):
    """
    Move the column to the target column index
    :param data: data to be changed
    :param column: column to be moved
    :param target_column: target column index
    :return: changed data
    """
    temp_col = data[column]
    data.pop(column)
    data.insert(target_column, column, temp_col)
    return data


def string_to_int(data, column):
    """
    Convert the string values to int values
    :param data: data to be converted
    :param column: column to be converted
    :param map: mapping of the values
    :return: converted data
    """
    mapDict = {}
    for i, value in enumerate(data[column].unique()):
        mapDict[value] = i
    data[column] = data[column].map(mapDict)
    return data, mapDict


def data_divider(data, target):
    """
    Divide the data into x and y
    :param data: data to be divided
    :param target: target column
    :return: x and y
    """
    y = data.iloc[:,[target]]      # (shape is (n, 1), n is the number of rows)
    x = data.iloc[:,0:target-1]    # Someting wrong (shape should be (n, m), n is the number of rows, m is the number of columns)
    return x, y


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
    files = parse_files(path)
    for file in files:
        data = read_data(file)
        data = null_separator(data)
        train, test = split_data(data, 0.8)

        filename = file.split('/')[-1].removesuffix('.csv')
        save_data(train, path + filename + '_train')
        save_data(test,  path + filename + '_test')
        print('Saved', path + filename + '_train')
        print('Saved', path + filename + '_test')
        
    # data = read_data(path)
    # data = null_separator(path)
    # ...
    # train, test = split_data(data)
    # save_data(train, 'train'+ "..." + '.csv')
    # save_data(test, 'test' + "..." + '.csv') 


