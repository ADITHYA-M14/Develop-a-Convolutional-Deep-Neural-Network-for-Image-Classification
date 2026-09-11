# Develop a Convolutional Deep Neural Network for Image Classification
# NAME: ADITHYA M
# REG NO: 212224230008
## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

## STEP 1:
Load the image dataset and divide it into training and testing datasets. Apply suitable transformations such as resizing and normalization to the images.

## STEP 2:
Create DataLoader objects for the training and testing datasets to load images in batches.

## STEP 3:
Design a CNN model consisting of convolution, ReLU activation, max-pooling, flattening, and fully connected layers.

## STEP 4:
Initialize the CNN model, cross-entropy loss function, and Adam optimizer. Move the model to the available CPU/GPU device.

## STEP 5:
Train the CNN for the specified number of epochs by performing forward propagation, calculating loss, backpropagation, and updating the model parameters.

## STEP 6:
Evaluate the trained model using test images. Calculate accuracy, generate a confusion matrix and classification report, and verify the prediction using a new sample image.




## PROGRAM

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_set = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# Check dataset
im, lbl = train_set[0]

print("Image shape:", im.shape)
print("Training images:", len(train_set))
print("Testing images:", len(test_set))

trl = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True
)

tstl = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False
)

class CNNclassifier1(nn.Module):

    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.c2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.c3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.l1 = nn.Linear(
            128 * 3 * 3,
            64
        )

        self.l2 = nn.Linear(
            64,
            32
        )

        self.l3 = nn.Linear(
            32,
            10
        )

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x = self.c1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.c3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.l1(x))
        x = self.dropout(x)

        x = torch.relu(self.l2(x))

        x = self.l3(x)

        return x

model = CNNclassifier1()
model = model.to(device)

print(model)
criterion = nn.CrossEntropyLoss()

op = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
epochs = 5

for i in range(epochs):

    model.train()

    running_loss = 0.0

    for a, b in trl:

        # Move data to CPU/GPU
        a = a.to(device)
        b = b.to(device)

        # Clear previous gradients
        op.zero_grad()

        # Forward pass
        pred = model(a)

        # Calculate loss
        loss = criterion(pred, b)

        # Backward pass
        loss.backward()

        op.step()
        running_loss += loss.item()
    average_loss = running_loss / len(trl)
    print(
        f"Epoch [{i + 1}/{epochs}] "
        f"Loss: {average_loss:.4f}"
    )
t = 0
c = 0
act = []
pre = []
model.eval()
with torch.no_grad():
    for img, labels in tstl:
        img = img.to(device)
        labels = labels.to(device)
        output = model(img)
        _, predicted = torch.max(output, 1)
        t += labels.size(0)
        c += (predicted == labels).sum().item()
        pre.extend(
            predicted.cpu().numpy()
        )
        act.extend(
            labels.cpu().numpy()
        )
accuracy = c / t * 100
print("\n--------------------------------")
print("Accuracy Score:", accuracy, "%")
print("--------------------------------")
conf_matrix = confusion_matrix(
    act,
    pre
)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(
    classification_report(
        act,
        pre,
        target_names=test_set.classes
    )
)
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=test_set.classes,
    yticklabels=test_set.classes
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Fashion-MNIST Confusion Matrix")
plt.tight_layout()
plt.show()
with torch.no_grad():
    img1, label = test_set[0]
    input_image = img1.unsqueeze(0).to(device)
    output = model(input_image)
    _, pred = torch.max(output, 1)
    classes = test_set.classes
    display_img = img1 * 0.3530 + 0.2860
    plt.figure(figsize=(4, 4))
    plt.imshow(
        display_img.squeeze(),
        cmap="gray"
    )
    plt.title(
        f"Predicted: {classes[pred.item()]}"
    )
    plt.axis("off")
    plt.show()
    print(
        f"Actual: {classes[label]}"
    )
    print(
        f"Predicted: {classes[pred.item()]}"
    )
original_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=None
)
image, label = original_dataset[0]
plt.figure(figsize=(4, 4))
plt.imshow(
    image,
    cmap="gray"
)
plt.title(
    original_dataset.classes[label]
)
plt.axis("off")
plt.show()
```


### OUTPUT

## Training Loss per Epoch

<img width="260" height="125" alt="image" src="https://github.com/user-attachments/assets/5463ff79-86af-4735-9f55-319feecd14d4" />


## Confusion Matrix

<img width="988" height="871" alt="image" src="https://github.com/user-attachments/assets/a3f7b2d7-6162-4ebb-b5db-74c1213d9171" />

<img width="449" height="280" alt="image" src="https://github.com/user-attachments/assets/17cb6183-c4d1-41e1-9391-94a7d71cf734" />

## Classification Report
<img width="582" height="414" alt="image" src="https://github.com/user-attachments/assets/11577365-2836-4c55-b149-e164fffd0773" />


### New Sample Data Prediction
<img width="390" height="468" alt="image" src="https://github.com/user-attachments/assets/5942943b-bf3d-4521-a2ef-36628f0c5fd6" />


## RESULT
The Convolutional Neural Network was successfully developed and trained for image classification using PyTorch.
