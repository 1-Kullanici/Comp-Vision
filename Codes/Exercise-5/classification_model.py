import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import sample_manipulation as sm

from skimage.io import imread
from skimage.transform import resize 

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report


def pre_process(Categories, datadir): # Example: Categories=['apple','banana','orange'] ; datadir='IMAGES/'
    flat_data_arr = [] #input array  (x)
    target_arr    = [] #output array (y)
    #path which contains all the categories of images 
    for i in Categories: 
        print(f'loading... category : {i}') 
        path = os.path.join(datadir,i) 
        for img in os.listdir(path): 
            img_array   = imread(os.path.join(path,img)) 
            img_resized = resize(img_array,(150,150,3)) 
            flat_data_arr.append(img_resized.flatten()) 
            target_arr.append(Categories.index(i)) 
        print(f'loaded category:{i} successfully') 
    flat_data = np.array(flat_data_arr) 
    target    = np.array(target_arr)
    return flat_data, target




if __name__ == '__main__':
    # Define path to the dataset and the ratio to split the dataset. Ratio is the percentage of train data.
    main_folder = 'Classification Datasets/' # Path except the last portion
    sub_folder = 'apple-dataset/'            # Last portion of the path
    ratio = 0.8                              # The ratio of train data among all data

    # Seperate the dataset into train and test. Get the categories and paths to seperated data.
    categories, train_path, test_path = sm.dividor(main_folder, sub_folder, ratio)
    flat_data, target = pre_process(categories, train_path)

    # Make dataframe 
    df = pd.DataFrame(flat_data)  
    df['Target'] = target 
    print('Dataframe shape is: ', df.shape, '. (# of images, # of features + 1 "which is the target").')

    #input data  
    x = df.iloc[:,:-1]  
    #output data 
    y = df.iloc[:,-1]

    # Splitting the data into training and testing sets 
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20, 
                                                     random_state=77, 
                                                     stratify=y) 

    # Defining the parameters grid for GridSearchCV 
    param_grid = {'C':[0.1,1,10,100], 
                  'gamma':[0.0001,0.001,0.1,1], 
                  'kernel':['rbf','poly']} 

    # Creating a support vector classifier 
    svc = SVC(probability=True) 

    # Creating a model using GridSearchCV with the parameters grid 
    model = GridSearchCV(svc,param_grid)

    # Training the model using the training data 
    model.fit(x_train,y_train)

    # Testing the model using the testing data 
    y_pred = model.predict(x_test) 

    # Calculating the accuracy of the model 
    accuracy = accuracy_score(y_pred, y_test) 

    # Print the accuracy and the classification report of the model 
    print(f"The model is {accuracy*100}% accurate")
    print(classification_report(y_test, y_pred, target_names=categories))

    # Predicting the category of an image from test data
    path = test_path + categories[0] + '/'
    img=imread(path + os.listdir(path)[0]) 
    plt.imshow(img) 
    plt.show() 
    img_resize=resize(img,(150,150,3)) 
    l=[img_resize.flatten()] 
    probability=model.predict_proba(l) 
    for ind,val in enumerate(categories): 
        print(f'{val} = {probability[0][ind]*100}%') 
    print("The predicted image is : "+ categories[model.predict(l)[0]])








