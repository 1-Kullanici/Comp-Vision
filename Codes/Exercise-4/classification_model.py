import numpy as np
import pandas as pd
from sklearn import preprocessing
import data_manipulation as dm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, mean_absolute_error, r2_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    path = 'Classification Datasets/'
    files = dm.parse_files(path)
    results = {}

    for file in files:
        data             = dm.read_data(file)
        data             = dm.null_separator(data)
        target           = len(data.columns) - 1
        print(target)
         
        train,   test    = dm.split_data(data,    0.8)
        x_train, y_train = dm.data_divider(train, target)
        print(x_train.shape)
        print(y_train.shape)
        x_test,  y_test  = dm.data_divider(test,  target)
        filename         = file.split('/')[-1].removesuffix('.csv')
    
        model_1            = LogisticRegression().fit(x_train, y_train)
        prediction_1       = LogisticRegression.predict(model_1, x_test)

        acc_1              = accuracy_score(test['class'],        prediction_1)
        mae_1              = mean_absolute_error(test['class'],   prediction_1)
        r2s_1              = r2_score(test['class'],              prediction_1)
        conf_matrix_1      = confusion_matrix(test['class'],      prediction_1)

        results[str(model_1) + filename] = (acc_1, mae_1, r2s_1, conf_matrix_1)


        model_2            = SVC().fit(x_train, y_train)
        prediction_2       = SVC.predict(model_2, x_test)

        acc_2              = accuracy_score(test['class'],        prediction_2)
        mae_2              = mean_absolute_error(test['class'],   prediction_2)
        r2s_2              = r2_score(test['class'],              prediction_2)
        conf_matrix_2      = confusion_matrix(test['class'],      prediction_2)

        results[str(model_2) + filename] = (acc_2, mae_2, r2s_2, conf_matrix_2)


        model_3            = DecisionTreeClassifier().fit(x_train, y_train)
        prediction_3       = DecisionTreeClassifier.predict(model_3, x_test)

        acc_3              = accuracy_score(test['class'],        prediction_3)
        mae_3              = mean_absolute_error(test['class'],   prediction_3)
        r2s_3              = r2_score(test['class'],              prediction_3)
        conf_matrix_3      = confusion_matrix(test['class'],      prediction_3)
        
        results[str(model_3) + filename] = (acc_3, mae_3, r2s_3, conf_matrix_3)

        print(results)

        fpr, tpr, thresholds = roc_curve(test['class'], prediction_1)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

        # dm.save_data(train, path + filename + '_train')
        # dm.save_data(test,  path + filename + '_test')
        # print('Saved', path + filename + '_train')
        # print('Saved', path + filename + '_test')