import sample_manipulation as sm
import cv2
import numpy as np
import os

from skimage.transform import resize 
from skimage.io import imread

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics



def pre_process(Categories, datadir): # Example: Categories=['apple','banana','orange'] ; datadir='IMAGES/'
    flat_data_arr=[] #input array 
    target_arr=[]    #output array 
    #path which contains all the categories of images 
    for i in Categories: 
        print(f'loading... category : {i}') 
        path=os.path.join(datadir,i) 
        for img in os.listdir(path): 
            img_array=imread(os.path.join(path,img)) 
            img_resized=resize(img_array,(150,150,3)) 
            flat_data_arr.append(img_resized.flatten()) 
            target_arr.append(Categories.index(i)) 
        print(f'loaded category:{i} successfully') 
    flat_data=np.array(flat_data_arr) 
    target=np.array(target_arr)
    return flat_data, target




if __name__ == '__main__':
    main_folder = 'Classification Datasets/'
    sub_folder = 'apple-dataset/'
    ratio = 0.8
    categories, train_path, test_path = sm.dividor(main_folder, sub_folder, ratio)
    print(categories)
    # pre_process()



