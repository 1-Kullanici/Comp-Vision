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
        # print('Directory already exists!')
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
    try:
        shutil.move(path_to_file, path_to_move)
        return False
    except:
        print('Could not move file!')
        return True


def parse_folders(path: str):
    """
    Parse the files in the given path
    :param path: path of the files
    :return: list of files
    """
    try:
        all_entries = os.listdir(path)
        for entry in all_entries:
            if os.path.isfile(entry):
                all_entries.remove(entry)
                # print('Removed', entry)        
        # print(all_entries)
        return all_entries
    except:
        print('No such directory exists!')
        return None


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
        return True
    else:
        for i in range(max_range):
            flag = move_file(files[i], path_to_move)
            if flag:
                return True
        return False

    
def dividor(main_folder, sub_folder, ratio):
    path = main_folder + sub_folder
    folders = parse_folders(path)

    if folders is None:
        print('Invalid path!')
        return None, None, None

    path_to_train = main_folder + sub_folder + 'train' + '/' 
    path_to_test = main_folder + sub_folder + 'test' + '/'

    # Check if train and test folders exist
    flag = make_directory(path_to_train)
    if flag:
        print('The dataset is separated already!')
        folders = parse_folders(path_to_train)
        return folders, path_to_train, path_to_test

    # Seperate train data and store in train folder
    for folder in folders:
        files = parse_files(path + folder + '/')
        flag = split_files(files, ratio, path + folder + '/', path_to_train + folder + '/')
        if flag:
            print(f'Could not move files to {path_to_train + folder}/')
            break
    
    # Move the rest of the data to test folder
    ratio = 1
    for folder in folders:
        files = parse_files(path + folder + '/')
        flag = split_files(files, ratio, path + folder + '/', path_to_test + folder + '/')
        if flag:
            print(f'Could not move files to {path_to_test + folder}/')
            break
    
    # Check any errors
    if not flag:
        # Remove excess folders
        for folder in folders:
            shutil.rmtree(path+ folder + '/')

        # Finished!
        print('Done!')
        return folders, path_to_train, path_to_test
    else:
        print('Done with errors!')
        return folders, path_to_train, path_to_test


if __name__ == '__main__':

    main_folder = 'Classification Datasets/'
    sub_folder = 'apple-dataset/'
    ratio = 0.8
    categories, train_path, test_path = dividor(main_folder, sub_folder, ratio)