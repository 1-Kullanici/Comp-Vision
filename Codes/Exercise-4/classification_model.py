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


# def train_model(train, test, target, models = [LogisticRegression, SVC, DecisionTreeClassifier]):
#     """
#     Train the model using Logistic Regression
#     :param train: training data
#     :param test: testing data
#     :param target: target variable
#     :return: trained model
#     """
#     for m in models:
#         model = m.fit(train, target)
#         prediction = m.predict(test)

#         acc = accuracy_score(test['class'], prediction)
#         print('Accuracy of', m, 'is', acc)

#         prec = precision_score(test['class'], prediction)
#         mae = mean_absolute_error(test['class'], prediction)
#         conf_matrix = confusion_matrix(test['class'], prediction)
#         r2s = r2_score(test['class'], prediction)
#         roc = roc_auc_score(test['class'], prediction)

#         results[str(m) + filename] = (acc, prec, mae, conf_matrix, r2s, roc)

#     return model, prediction


if __name__ == '__main__':

    path = 'Classification Datasets/'
    files = dm.parse_files(path)
    results = {}
    target = 4

    for file in files:
        data = dm.read_data(file)
        data = dm.null_separator(data)
        train, test = dm.split_data(data, 0.8)
        filename = file.split('/')[-1].removesuffix('.csv')
    
        model = LogisticRegression.fit(file, file[target])
        prediction = LogisticRegression.predict(model, test[target])

        acc = accuracy_score(test['class'], prediction)
        mae = mean_absolute_error(test['class'], prediction)
        conf_matrix = confusion_matrix(test['class'], prediction)
        r2s = r2_score(test['class'], prediction)
        # prec = precision_score(test['class'], prediction)
        # roc = roc_auc_score(test['class'], prediction)

        results[str(model) + filename] = (acc, mae, conf_matrix, r2s)


        # model = SVC.fit(file)
        # prediction = SVC.predict(model, test)

        # acc = accuracy_score(test['class'], prediction)
        # prec = precision_score(test['class'], prediction)
        # mae = mean_absolute_error(test['class'], prediction)
        # conf_matrix = confusion_matrix(test['class'], prediction)
        # r2s = r2_score(test['class'], prediction)
        # roc = roc_auc_score(test['class'], prediction)

        # results[str(models[i] + filename)] = (acc, prec, mae, conf_matrix, r2s, roc)

        # model = DecisionTreeClassifier.fit(file)
        # prediction = DecisionTreeClassifier.predict(model, test)
        
        # acc = accuracy_score(test['class'], prediction)
        # prec = precision_score(test['class'], prediction)
        # mae = mean_absolute_error(test['class'], prediction)
        # conf_matrix = confusion_matrix(test['class'], prediction)
        # r2s = r2_score(test['class'], prediction)
        # roc = roc_auc_score(test['class'], prediction)

        # results[str(models[i] + filename)] = (acc, prec, mae, conf_matrix, r2s, roc)


        # print('Accuracy of', model, 'is', acc)
        # print('Precision of', model, 'is', prec)
        # print('Mean Absolute Error of', model, 'is', mae)
        # print('R2 Score of', model, 'is', r2s)
        # print('Confusion Matrix of', model, 'is', conf_matrix)
        # print('ROC AUC Score of', model, 'is', roc)

        fpr, tpr, thresholds = roc_curve(test['class'], prediction)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

        dm.save_data(train, path + filename + '_train')
        dm.save_data(test,  path + filename + '_test')
        print('Saved', path + filename + '_train')
        print('Saved', path + filename + '_test')