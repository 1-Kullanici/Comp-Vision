import os
import glob
import shutil
import numpy as np


def make_directory(path: str):
    """
    Make a directory in the given path
    :param path: path of the directory
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        print('Directory already exists!')
        return True


def rename_file(path_to_file: str, new_name: str):
    """
    Rename a file in the given path
    :param path_to_file: path of the file
    :param new_name: new name of the file
    :return: None
    """
    if not os.path.exists(new_name):
        os.rename(path_to_file, new_name)
    else:
        print('File already exists!')


def move_file(path_to_file: str, path_to_move: str):
    """
    Move a file to the given path
    :param path_to_file: path of the file
    :param path_to_move: path to move
    :return: None
    """
    if not os.path.exists(path_to_move):
        shutil.move(path_to_file, path_to_move)
    else:
        print('Directory already exists!')


def parse_folders(path: str):
    """
    Parse the files in the given path
    :param path: path of the files
    :return: list of files
    """
    all_entries = os.listdir(path)
    for entry in all_entries:
        if os.path.isfile(entry):
            all_entries.remove(entry)
            # print('Removed', entry)        
    # print(all_entries)
    return all_entries


def parse_files(path: str):
    """
    Parse the files in the given path
    :param path: path of the files
    :return: list of files
    """
    files = glob.glob(path + '*.JPG')
    return files


def split_files(files, ratio: float, path_to_file:str, path_to_move: str):
    """
    Split the data into train and test sets
    :param data: data to be split
    :param ratio: ratio of the split
    :return: train and test sets
    """

    max_range = int(np.round(len(files) * ratio))
    flag = make_directory(path_to_move)
    if flag:
        print('Files already moved!')
        return
    else:
        for i in range(max_range):
            shutil.move(files[i], path_to_move)

    
def dividor(main_folder, sub_folder, ratio):
    path = main_folder + sub_folder
    folders = parse_folders(path)
    for folder in folders:
        files = parse_files(path + folder + '/')
        split_files(files, ratio, path + folder + '/', main_folder + sub_folder.split('/')[0] + '-train' + '/' + folder + '/') 
    rename_file(path, main_folder + sub_folder.split('/')[0] + '-test')
    print('Done!')



if __name__ == '__main__':

    main_folder = 'Classification Datasets/'
    sub_folder = 'apple-dataset-test/'
    ratio = 0.8
    dividor(main_folder, sub_folder, ratio)