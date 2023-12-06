import numpy as np
import pandas as pd
from sklearn import preprocessing
import data_separation as ds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    path = 'Classification Datasets/'
    files = ds.parse_files(path)
    for file in files:
        data = ds.read_data(file)
        data = ds.null_separator(data)
        train, test = ds.split_data(data, 0.8)

        filename = file.split('/')[-1].removesuffix('.csv')
        ds.save_data(train, path + filename + '_train')
        ds.save_data(test,  path + filename + '_test')
        print('Saved', path + filename + '_train')
        print('Saved', path + filename + '_test')