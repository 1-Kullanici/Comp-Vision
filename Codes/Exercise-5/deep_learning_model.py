import os
import torch
import torch.nn as nn
import torchvision.models as models

from PIL import Image
from torchvision.transforms import v2
from sklearn.model_selection import KFold

import sample_manipulation as sm
import classification_model as cm


# Define path to the dataset and the ratio to split the dataset. Ratio is the percentage of train data.
main_folder = 'Classification Datasets/' # Path except the last portion
sub_folder = 'apple-dataset/'            # Last portion of the path
ratio = 0.8                              # The ratio of train data among all data

# Load the pre-trained ResNet model
resnet = models.resnet50(pretrained=True)

# Freeze the parameters of the pre-trained model
for param in resnet.parameters():
    param.requires_grad = False

# Replace the last fully connected layer with a new one for our classification task
num_classes = 6  # Number of apple types
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)

# Load and preprocess the apple dataset
categories, train_path, validate_path = sm.dividor(main_folder, sub_folder, ratio)    # Seperate the dataset into train and validate.
x_train,x_test,y_train,y_test         = cm.pre_process(categories, train_path)        # Pre-process the train data

x_train = torch.tensor(x_train.values)
x_test  = torch.tensor(x_test.values)
y_train = torch.tensor(y_train.values)
y_test  = torch.tensor(y_test.values)


# Train the model
num_epochs = 10
batch_size = 32
num_folds = 5

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

kf = KFold(n_splits=num_folds, shuffle=True)

for fold, (train_indices, val_indices) in enumerate(kf.split(x_train)):
    print(f"Fold {fold+1}/{num_folds}")
    train_fold_dataset = torch.utils.data.TensorDataset(x_train[train_indices], y_train[train_indices])
    val_fold_dataset = torch.utils.data.TensorDataset(x_train[val_indices], y_train[val_indices])

    train_fold_loader = torch.utils.data.DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)
    val_fold_loader = torch.utils.data.DataLoader(val_fold_dataset, batch_size=batch_size, shuffle=False)

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

# Evaluate the model
transforms = v2.Compose([v2.RandomResizedCrop(size=(150, 150), antialias=True),
                         v2.ToDtype(torch.float32, scale=True),
                         # v2.RandomHorizontalFlip(p=0.5),
                         # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])

for folder in categories:
    image_path = validate_path + folder + '/'
    image = Image.open(image_path + os.listdir(image_path)[0])
    image = transforms(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = resnet(image)
        _, predicted = torch.max(output, 1)

    print(f"Image: {image_path}, Predicted Class: {categories[predicted.item()]}")

# Save the trained model
torch.save(resnet.state_dict(), 'apple_classifier.pth')
