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

    # Make dataframe 
    df = pd.DataFrame(flat_data)  
    df['Target'] = target 
    print('Dataframe shape is: ', df.shape, '. (# of images, # of features "or pixels in the image" + 1 "which is the target").')

    #input data  - the features (x)
    x = df.iloc[:,:-1]  
    #output data - the target   (y) 
    y = df.iloc[:,-1]

    # Splitting the data into training and testing sets 
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20, 
                                                     random_state=77, 
                                                     stratify=y)
    
    return x_train,x_test,y_train,y_test




if __name__ == '__main__':
    # Define path to the dataset and the ratio to split the dataset. Ratio is the percentage of train data.
    main_folder = 'Classification Datasets/' # Path except the last portion
    sub_folder = 'apple-dataset/'            # Last portion of the path
    ratio = 0.8                              # The ratio of train data among all data


    # Pre-processing the data
    categories, train_path, validate_path = sm.dividor(main_folder, sub_folder, ratio) # Seperate the dataset into train and validate.
    x_train,x_test,y_train,y_test         = pre_process(categories, train_path)        # Pre-process the train data


    # Preparing the model
    param_grid = {'C':[0.1,1,10,100], 'gamma':[0.0001,0.001,0.1,1], 
                  'kernel':['rbf','poly']}    # Defining the parameters grid for GridSearchCV 
    svc = SVC(probability=True)               # Creating a support vector classifier 
    model = GridSearchCV(svc,param_grid)      # Creating a model using GridSearchCV with the parameters grid 


    # Training and testing the model
    model.fit(x_train,y_train)     # Training the model using the pre-processed training data 
    y_pred = model.predict(x_test) # Testing the model using the pre-processed testing data 


    # Evaluating the model
    accuracy = accuracy_score(y_pred, y_test) 
    print(f"The model is {accuracy*100}% accurate")
    print(classification_report(y_test, y_pred, target_names=categories))


    # Predicting the category of an image from validation data
    path = validate_path + categories[0] + '/'
    img  = imread(path + os.listdir(path)[0]) 
    plt.imshow(img) 
    plt.show() 
    img_resize = resize(img,(150,150,3)) 
    l = [img_resize.flatten()] 
    probability = model.predict_proba(l) 
    for ind,val in enumerate(categories): 
        print(f'{val} = {probability[0][ind]*100}%') 
    print("The selected image is :" + categories[0] + ". The predicted image is : " + categories[model.predict(l)[0]] + ".")








