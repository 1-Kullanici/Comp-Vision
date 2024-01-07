
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# from sklearn import preprocessing
# from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score, mean_squared_error
# from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt


class model_grader:

    def __init__(self, y_test, prediction):
        self.y_test = y_test
        self.prediction = prediction

    def accuracy_score(self):
        return accuracy_score(self.y_test, self.prediction)
    
    def mean_absolute_error(self):
        return mean_absolute_error(self.y_test, self.prediction)
    
    def r2_score(self):
        return r2_score(self.y_test, self.prediction)
    
    def mean_squared_error(self):
        return mean_squared_error(self.y_test, self.prediction)
    
    def confusion_matrix(self):
        return confusion_matrix(self.y_test, self.prediction)
    
    def regression(self):
        return self.mean_absolute_error(), self.r2_score(), self.mean_squared_error()
    
    def classification(self):
        return self.accuracy_score(), self.mean_absolute_error(), self.r2_score(), self.confusion_matrix()

    # def model_grader(y_test, prediction):
    #     """Returns the accuracy, mean absolute error, r2 score and confusion matrix of the model"""
    #     acc              = accuracy_score(y_test,        prediction)
    #     mae              = mean_absolute_error(y_test,   prediction)
    #     r2s              = r2_score(y_test,              prediction)
    #     mse              = mean_squared_error(y_test,    prediction)
    #     conf_matrix      = confusion_matrix(y_test,      prediction)
    #     return acc, mae, r2s, mse, conf_matrix


if __name__ == '__main__':

    """

    if __name__ == '__main__':

    #     path = 'Classification Datasets/'
    #     files = dm.parse_files(path)
    #     print('Files:', files)
    #     results = []
    #     filenames = []


    #     ##########################################################################################################

    #     # Regression (cars.csv)
    #     file = files[1]
            
    #     print('Reading', file)

    #     data             = dm.read_data(file)
    #     data             = dm.null_separator(data)

    #     target           = len(data.columns) - 1
    #     # data             = dm.change_column(data, 'Origin', target)
    #     # data, mapDict    = dm.string_to_int(data, 'Origin')
    #     train,   test    = dm.split_data(data,    0.8)

    #     x_train, y_train = dm.data_divider(train, target)
    #     x_test,  y_test  = dm.data_divider(test,  target)
    #     filename         = file.split('/')[-1].removesuffix('.csv')
    #     filenames.append(filename)


    #     classifier_1     = BayesianRidge()
    #     model_1          = classifier_1.fit(x_train, y_train) 
    #     prediction_1     = classifier_1.predict(x_test)

    #     classifier_2     = SVR()
    #     model_2          = classifier_2.fit(x_train, y_train)
    #     prediction_2     = classifier_2.predict(x_test)

    #     classifier_3     = LinearRegression()
    #     model_3          = classifier_3.fit(x_train, y_train)
    #     prediction_3     = classifier_3.predict(x_test)


    #     results.append((y_test, prediction_1, prediction_2, prediction_3))

    #     # Accuracy Measure
    #     fpr = []
    #     tpr = []
    #     thresholds = []
    #     scores = []
    #     conf_matrices = []
    #     for i in range(len(results)):
    #         y_test, prediction_1, prediction_2, prediction_3 = results[i]
    #         # fpr_1, tpr_1, thresholds_1 = roc_curve(y_test, prediction_1)
    #         # fpr_2, tpr_2, thresholds_2 = roc_curve(y_test, prediction_2)
    #         # fpr_3, tpr_3, thresholds_3 = roc_curve(y_test, prediction_3)

    #         # fpr.append([fpr_1, fpr_2, fpr_3])
    #         # tpr.append([tpr_1, tpr_2, tpr_3])
    #         # thresholds.append([thresholds_1, thresholds_2, thresholds_3])
            
    #         mg_1 = model_grader(y_test, prediction_1)
    #         mg_2 = model_grader(y_test, prediction_2)
    #         mg_3 = model_grader(y_test, prediction_3)

    #         mae_1, r2s_1, mse_1 = mg_1.regression()
    #         mae_2, r2s_2, mse_2 = mg_2.regression()
    #         mae_3, r2s_3, mse_3 = mg_3.regression()

    #         print('Mean Absolute Error of Bayesian Ridge Regression: ', mae_1)
    #         print('Mean Absolute Error of SVR: ', mae_2)
    #         print('Mean Absolute Error of Linear Regression: ', mae_3)

    #         print('R^2 Score of Bayesian Ridge Regression: ', r2s_1)
    #         print('R^2 Score of SVR: ', r2s_2)
    #         print('R^2 Score of Linear Regression: ', r2s_3)

    #         print('Mean Square Error of Bayesian Ridge Regression: ', mse_1)
    #         print('Mean Square Error of SVR: ', mse_2)
    #         print('Mean Square Error of Linear Regression: ', mse_3)

    #         scores.append(([mae_1, r2s_1, mse_1], [mae_2, r2s_2, mse_2], [mae_3, r2s_3, mse_3]))

    #     ##########################################################################################################

    #     # Classification (heart.csv)
    #     file = files[0]

    #     print('Reading', file)

    #     data             = dm.read_data(file)
    #     data             = dm.null_separator(data)

    #     target           = len(data.columns) - 1
    #     train,   test    = dm.split_data(data,    0.8)

    #     x_train, y_train = dm.data_divider(train, target)
    #     x_test,  y_test  = dm.data_divider(test,  target)
    #     filename         = file.split('/')[-1].removesuffix('.csv')
    #     filenames.append(filename)


    #     classifier_1     = GradientBoostingClassifier()
    #     model_1          = classifier_1.fit(x_train, y_train) 
    #     prediction_1     = classifier_1.predict(x_test) 

    #     classifier_2     = SVC()
    #     model_2          = classifier_2.fit(x_train, y_train)
    #     prediction_2     = classifier_2.predict(x_test)

    #     classifier_3     = DecisionTreeClassifier()
    #     model_3          = classifier_3.fit(x_train, y_train)
    #     prediction_3     = classifier_3.predict(x_test)

    #     results = []                                                            # Clear previous results
    #     results.append((y_test, prediction_1, prediction_2, prediction_3))

    #     # Accuracy Measure
    #     fpr = []
    #     tpr = []
    #     thresholds = []
    #     scores = []
    #     conf_matrices = []
    #     for i in range(len(results)):
    #         y_test, prediction_1, prediction_2, prediction_3 = results[i]
    #         # fpr_1, tpr_1, thresholds_1 = roc_curve(y_test, prediction_1)
    #         # fpr_2, tpr_2, thresholds_2 = roc_curve(y_test, prediction_2)
    #         # fpr_3, tpr_3, thresholds_3 = roc_curve(y_test, prediction_3)

    #         # fpr.append([fpr_1, fpr_2, fpr_3])
    #         # tpr.append([tpr_1, tpr_2, tpr_3])
    #         # thresholds.append([thresholds_1, thresholds_2, thresholds_3])

    #         mG_1 = model_grader(y_test, prediction_1)
    #         mG_2 = model_grader(y_test, prediction_2)
    #         mG_3 = model_grader(y_test, prediction_3)

    #         acc_1, mae_1, r2s_1, conf_matrix_1 = mG_1.classification()
    #         acc_2, mae_2, r2s_2, conf_matrix_2 = mG_2.classification()
    #         acc_3, mae_3, r2s_3, conf_matrix_3 = mG_3.classification()

    #         print('Accuracy of Gradient Boosting Classifier: ', acc_1)
    #         print('Accuracy of SVC: ', acc_2)
    #         print('Accuracy of Decision Tree Classifier: ', acc_3)

    #         print('Mean Absolute Error of Gradient Boosting Classifier: ', mae_1)
    #         print('Mean Absolute Error of SVC: ', mae_2)
    #         print('Mean Absolute Error of Decision Tree Classifier: ', mae_3)

    #         print('R^2 Score of Gradient Boosting Classifier: ', r2s_1)
    #         print('R^2 Score of SVC: ', r2s_2)
    #         print('R^2 Score of Decision Tree Classifier: ', r2s_3)

    #         scores.append(([acc_1, mae_1, r2s_1], [acc_2, mae_2, r2s_2], [acc_3, mae_3, r2s_3]))
    #         conf_matrices.append([conf_matrix_1, conf_matrix_2, conf_matrix_3])



    #     # fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(figsize=(3, 3), 
    #     #                                                         facecolor='lightskyblue',
    #     #                                                         layout='constrained',
    #     #                                                         nrows=2,
    #     #                                                         ncols=3)


    #     # ax1.plot(conf_matrices[0,0], label=str(filenames[0]) + ' - Logistic Regression')
    #     # ax1.xlabel('False Positive Rate')
    #     # ax1.ylabel('True Positive Rate')
    #     # ax1.title(str(filenames[0]) + ' - Logistic Regression')

    #     # ax2.plot(conf_matrices[0,1], label=str(filenames[0]) + ' - SVC')
    #     # ax2.xlabel('False Positive Rate')
    #     # ax2.ylabel('True Positive Rate')
    #     # ax2.title(str(filenames[0]) + ' - SVC')

    #     # ax3.plot(conf_matrices[0,2], label=str(filenames[0]) + ' - Decision tree')
    #     # ax3.xlabel('False Positive Rate')
    #     # ax3.ylabel('True Positive Rate')
    #     # ax3.title(str(filenames[0]) + ' - Decision tree')
        
    #     # ax4.plot(conf_matrices[1,0], label=str(filenames[1]) + ' - Logistic Regression')
    #     # ax4.xlabel('False Positive Rate')
    #     # ax4.ylabel('True Positive Rate')
    #     # ax4.title(str(filenames[1]) + ' - Logistic Regression')

    #     # ax5.plot(conf_matrices[1,1], label=str(filenames[1]) + ' - SVC')
    #     # ax5.xlabel('False Positive Rate')
    #     # ax5.ylabel('True Positive Rate')
    #     # ax5.title(str(filenames[1]) + ' - SVC')

    #     # ax6.plot(conf_matrices[1,2], label=str(filenames[1]) + ' - Decision tree')
    #     # ax6.xlabel('False Positive Rate')
    #     # ax6.ylabel('True Positive Rate')
    #     # ax6.title(str(filenames[1]) + ' - Decision tree')

    #     # # ax7.plot(fpr[0,0], tpr[0,0], label=str(filenames[0]) + ' - Logistic Regression')
    #     # # ax7.plot(fpr[0,1], tpr[0,1], label=str(filenames[0]) + ' - SVC')
    #     # # ax7.plot(fpr[0,2], tpr[0,2], label=str(filenames[0]) + ' - Decision Tree')
    #     # # ax7.title('ROC Curve')
    #     # # ax7.legend()

    #     # fig.show()

    #     # fig.savefig('Codes/Exercise-4/Results.png')
    """