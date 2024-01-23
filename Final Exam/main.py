import time

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from keras import layers, models
import os

def load_split_nvgesture(file_with_split = './nvgesture_train_correct.lst',list_split = list()):
    params_dictionary = dict()
    with open(file_with_split,'r') as f:
          dict_name  = file_with_split[file_with_split.rfind('/')+1 :]
          dict_name  = dict_name[:dict_name.find('_')]

          for line in f:
            params = line.split(' ')
            params_dictionary = dict()

            params_dictionary['dataset'] = dict_name

            path = params[0].split(':')[1]
            for param in params[1:]:
                    parsed = param.split(':')
                    key = parsed[0]
                    if key == 'label':
                        # make label start from 0
                        label = int(parsed[1]) - 1 
                        params_dictionary['label'] = label
                    elif key in ('depth','color','duo_left'):
                        #othrwise only sensors format: <sensor name>:<folder>:<start frame>:<end frame>
                        sensor_name = key
                        #first store path
                        params_dictionary[key] = path + '/' + parsed[1]
                        #store start frame
                        params_dictionary[key+'_start'] = int(parsed[2])

                        params_dictionary[key+'_end'] = int(parsed[3])
        
            params_dictionary['duo_right'] = params_dictionary['duo_left'].replace('duo_left', 'duo_right')
            params_dictionary['duo_right_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_right_end'] = params_dictionary['duo_left_end']          

            params_dictionary['duo_disparity'] = params_dictionary['duo_left'].replace('duo_left', 'duo_disparity')
            params_dictionary['duo_disparity_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_disparity_end'] = params_dictionary['duo_left_end']                  

            list_split.append(params_dictionary)
 
    return list_split


def prepare_data(example_config):
    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    # mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hand = mp.solutions.hands

    landmark_list = []

    path = example_config["color"] + ".avi"
    start_frame = example_config["color"+'_start']
    end_frame = example_config["color"+'_end']
    label = example_config['label']

    cap = cv2.VideoCapture(path)
    frames_to_load = range(start_frame, end_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    with mp_hand.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hand:
        for indx in enumerate(frames_to_load):
            ret, image = cap.read()
            if ret:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_height, image_width, _ = image.shape
                print(image.shape)
                resultant = hand.process(image)
                if resultant.multi_hand_landmarks:
                    for landmark in resultant.multi_hand_landmarks:
                        print('hand_landmarks(x):', landmark.landmark[0].x * image_width)
                        print('hand_landmarks(y):', landmark.landmark[0].y * image_height)
                        landmark_list.append(landmark.landmark)
                        mp_drawing.draw_landmarks(
                            image,
                            landmark,
                            mp_hand.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
            cv2.imshow('frame', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    column_names = []
    dimensions = ["x", "y"]
    for i in range(21):
        for j in dimensions:
            column_names.append(f"{i}_{j}")

    column_names.append("label")

    row_of_rows = []
    for i in range(len(landmark_list)):
        rows = []
        for j in range(21):
            rows.append(landmark_list[i][j].x)
            rows.append(landmark_list[i][j].y)
        rows.append(label)
        row_of_rows.append(rows)

    cv2.destroyAllWindows()

    return row_of_rows, column_names

def data_to_csv(test_lst_path="./nvgesture_test_correct_cvpr2016_v2.lst", train_lst_path="./nvgesture_train_correct_cvpr2016_v2.lst"):
    file_lists = dict()
    file_lists["test"] = test_lst_path
    file_lists["train"] = train_lst_path
    train_list = list()
    test_list = list()
    general_train_list = list()
    general_test_list = list()
    
    load_split_nvgesture(file_with_split = file_lists["train"],list_split = train_list)
    load_split_nvgesture(file_with_split = file_lists["test"],list_split = test_list)

    for i, _ in enumerate(train_list):
        train_data, column_names = prepare_data(example_config = train_list[i])
        general_train_list.extend(train_data)
    
    general_train_list_np = np.array(general_train_list)

    for i, _ in enumerate(test_list):
        test_data, _= prepare_data(example_config = test_list[i])
        general_test_list.extend(test_data)

    general_test_list_np = np.array(general_test_list)

    df_train = pd.DataFrame(general_train_list_np, columns=column_names)
    df_train.to_csv("train.csv", encoding="utf-8")

    df_test = pd.DataFrame(general_test_list_np, columns=column_names)
    df_test.to_csv("test.csv", encoding="utf-8")


if __name__ == "__main__":

    from os.path import exists
    
    if not exists("./train.csv"):
        path_test = "./nvGesture_v1/nvgesture_test_correct_cvpr2016_v2_mini.lst"
        path_train = "./nvGesture_v1/nvgesture_train_correct_cvpr2016_v2_mini.lst"
        data_to_csv(test_lst_path=path_test, train_lst_path=path_train)

    train_df = pd.read_csv("all_train.csv")
    test_df = pd.read_csv("all_test.csv")

    train_df.drop(columns=["Unnamed: 0"], inplace=True)
    test_df.drop(columns=["Unnamed: 0"], inplace=True)

    y_train = train_df.loc[:, ["label"]]
    x_train = train_df.iloc[:, :-1]

    y_test = test_df.loc[:, ["label"]]
    x_test = test_df.iloc[:, :-1]

    knn_model_2 = KNeighborsClassifier(n_neighbors=5, leaf_size=1, p=1)
    knn_model_2.fit(x_train, y_train)

    y_pred = knn_model_2.predict(x_test.values)

    # print(roc_auc_score(y_test, y_pred, multi_class='ovr'))
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))

    mp_hand = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    frame_count = 0
    possible_state = [0] * 25
    max_frame_4_evidence = 5
    predicted_state = None

    cap = cv2.VideoCapture(0)
    landmark_list = []
    label_dict = { 0:'Move hand left', 
                   1:'Move hand right',
                   2:'Move hand up',
                   3:'Move hand down', 
                   
                   4:'Move two fingers left', 
                   5:'Move two fingers right', 
                   6:'Move two fingers up', 
                   7:'Move two fingers down', 
                   
                   8:'Click index finger', 
                   9:'Call someone',

                   10:'Open hand', 
                   11:'Shaking hand', 
                   
                   12:'Show index finger', 
                   13:'Show two fingers', 
                   14:'Show three fingers',

                   15:'Push hand up', 
                   16:'Push hand down', 
                   17:'Push hand out', 
                   18:'Pull hand in', 
                   
                   19:'Rotate fingers CW',
                   20:'Rotate fingers CCW', 
                   
                   21:'Push two fingers away', 
                   22:'Close hand two times', 

                   23:'Thumb up', 
                   24:'Okay gesture',
                }

    with mp_hand.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.9
    ) as hand:
        while cap.isOpened():
            ret, image = cap.read()
            if ret:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resultant = hand.process(image)
                if resultant.multi_hand_landmarks:
                    for landmark in resultant.multi_hand_landmarks:
                        landmark_list.append(landmark.landmark)
                        mp_drawing.draw_landmarks(
                                image,
                                landmark,
                                mp_hand.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                    row_of_rows = []
                    for i in range(len(landmark_list)):
                        rows = []
                        for j in range(21):
                            rows.append(landmark_list[i][j].x)
                            rows.append(landmark_list[i][j].y)
                        row_of_rows.append(rows)

                    arr = np.array(row_of_rows)

                    if frame_count < max_frame_4_evidence:
                        y_pred = knn_model_2.predict(arr)
                        frame_count += 1
                        possible_state[int(y_pred[-1])] += 1
                    elif frame_count == max_frame_4_evidence:
                        predicted_state = possible_state.index(max(possible_state))
                        frame_count = 0
                        possible_state = [0] * 25
                    
                    if predicted_state is not None:
                        # image = cv2.putText(image, f"Predicted class: {label_dict[predicted_state]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)                        
                        if predicted_state == 23:
                            image = cv2.putText(image, "Open the light.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)                        
                        elif predicted_state == 13:
                            image = cv2.putText(image, "Close curtains.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)  
                    else:
                        image = cv2.putText(image, f"Predicted class: {predicted_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)
                    # if y_pred[-1] == 11:
                    #     image = cv2.putText(image, "Predicted class: Waving", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)
                    # elif y_pred[-1] == 24:
                    #     image = cv2.putText(image, "Predicted class: Okey", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)
                    # else:
                    #     image = cv2.putText(image, "Predicted class: Nothing", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)
                    #print(y_pred[-1])
                cv2.imshow('frame', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

    cv2.destroyAllWindows()

