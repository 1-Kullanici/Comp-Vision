import os
import numpy as np
import pandas as pd
import multiprocessing

import torch
import torch.nn as nn
import torchvision.models as models

from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.io import read_image
from torchvision.transforms import v2
from sklearn.model_selection import KFold

import sample_manipulation as sm
import classification_model as cm

class CustomImageDataset(Dataset):
    def __init__(self, labels:dict, tensor_arr, target_transform=None):
        self.img_labels = pd.DataFrame.from_dict(labels)
        self.tensor_arr = tensor_arr
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.tensor_arr[idx]
        label = self.img_labels.iloc[idx, 0]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# Note: The function may not stack images to tensors as I intended. 
# Check the tensor structure and debug tihs part. Check if the "data" in this function matches the tensor structure. 
def pre_process(Categories, datadir, transform=None):
    data_arr      = [] #input array  (x)
    target_arr    = [] #output array (y)
    #path which contains all the categories of images 
    for i in Categories: 
        print(f'loading... category : {i}') 
        path = os.path.join(datadir,i) 
        for img in os.listdir(path): 
            img_array  = Image.open(os.path.join(path,img)) # Open image as PIL image
            if transform:
                img_array = transform(img_array) # Convert PIL image to tensor
            data_arr.append(img_array)
            target_arr.append(Categories.index(i)) 
        print(f'loaded category:{i} successfully') 
    data   = np.array(data_arr)                  # Store the tensors in a numpy array
    target = np.array(target_arr)
    print('\n',data.shape,'\n',target.shape,'\n')

    # Make target a dataframe
    df_target = pd.DataFrame()
    df_target['Label'] = target

    return data, df_target


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Define path to the dataset and the ratio to split the dataset. Ratio is the percentage of train data.
    main_folder = 'Classification Datasets/' # Path except the last portion
    sub_folder = 'apple-dataset/'            # Last portion of the path
    ratio = 0.8                              # The ratio of train data among all data

    # Load and preprocess the apple dataset
    categories, train_path, validate_path = sm.dividor(main_folder, sub_folder, ratio)      # Seperate the dataset into train and validate.
    transform = v2.Compose([v2.RandomResizedCrop(size=(150, 150), antialias=True),
                            v2.ToImageTensor(),
                            v2.ConvertImageDtype(),
                            # v2.RandomHorizontalFlip(p=0.5),
                            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])
    train_data, train_target  = pre_process(categories, train_path, transform=transform)            # Pre-process the train data
    train_dataset = CustomImageDataset(labels=train_target, tensor_arr=train_data)
    # validate_data, validate_target  = pre_process(categories, validate_path, transform=transform)   # Pre-process the validate data
    # validate_dataset  = CustomImageDataset(labels=validate_target, tensor_arr=validate_data)

    ##############################################################################################################################
    # Load the pre-trained ResNet model
    resnet = models.resnet50(pretrained=True)

    # Freeze the parameters of the pre-trained model
    for param in resnet.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer with a new one for our classification task
    num_classes = len(categories)  # Number of apple types
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)

    ##############################################################################################################################
    # Train the model
    num_epochs = 10
    batch_size = 32
    num_folds = 5

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    train_features, train_labels = next(iter(train_dataloader))      # iter function to load batches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet.to(device)

    kf = KFold(n_splits=num_folds, shuffle=True)

    for fold, (train_indices, val_indices) in enumerate(kf.split(train_features)):
        print(f"Fold {fold+1}/{num_folds}")
        train_fold_dataset = TensorDataset(train_features[train_indices], train_labels[train_indices])
        val_fold_dataset   = TensorDataset(train_features[val_indices], train_labels[val_indices])

        train_fold_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)
        val_fold_loader = DataLoader(val_fold_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_fold_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = resnet(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_fold_loader):.4f}")

        print("Validation")
        with torch.no_grad():
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            for inputs, labels in val_fold_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = resnet(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

            print(f"Validation Loss: {val_loss/len(val_fold_loader):.4f}")
            print(f"Validation Accuracy: {100 * val_correct/val_total:.2f}%")

    print("Training finished.")

    ##############################################################################################################################
    # Evaluate the model
    print("Evaluating the model...")
    for label in categories:
        images = sm.parse_files(validate_path + label + '/')
        print(f'---------------{label}---------------')
        for image in images:
            image = Image.open(image)
            image = transform(image)
            image = image.unsqueeze(0)

            with torch.no_grad():
                output = resnet(image)
                _, predicted = torch.max(output, 1)

            print(f"Image: {label}, Predicted Class: {categories[predicted.item()]}")

    print("Evaluation finished.")

    ##############################################################################################################################
    # Save the trained model
    torch.save(resnet.state_dict(), 'apple_classifier.pth')
    print("Model saved.")