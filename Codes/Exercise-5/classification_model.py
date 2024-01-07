import sample_manipulation as sm
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier





if __name__ == '__main__':
    main_folder = 'Classification Datasets/'
    sub_folder = 'apple-dataset-test/'
    ratio = 0.8
    sm.dividor(main_folder, sub_folder, ratio)

