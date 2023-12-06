import numpy as np
import pandas as pd
from sklearn import preprocessing
import data_manipulation as dm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(train, test, target, model: function):
    """
    Train the model using Logistic Regression
    :param train: training data
    :param test: testing data
    :param target: target variable
    :return: trained model
    """
    # model = LogisticRegression()
    model.fit(train, target)
    prediction = model.predict(test)
    return model, prediction


if __name__ == '__main__':

    path = 'Classification Datasets/'
    files = dm.parse_files(path)
    for file in files:
        data = dm.read_data(file)
        data = dm.null_separator(data)
        train, test = dm.split_data(data, 0.8)

        filename = file.split('/')[-1].removesuffix('.csv')
        dm.save_data(train, path + filename + '_train')
        dm.save_data(test,  path + filename + '_test')
        print('Saved', path + filename + '_train')
        print('Saved', path + filename + '_test')