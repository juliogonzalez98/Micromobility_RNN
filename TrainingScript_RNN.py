import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import datasets

import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.models import resnet50


if not torch.cuda.is_available():
    raise Exception("You should enable GPU in the runtime menu.")
device = torch.device("cuda:0")


# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir = 'ModelTrain/train'

# The directory where we will
# store our smaller dataset
base_dir = 'ModelTrain/processed_datalab'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# Directories for our training,
# validation and test splits
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'val')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# Directory with our training BikeBi pictures
train_BikeBi_dir = os.path.join(train_dir, 'BikeBi')
if not os.path.exists(train_BikeBi_dir):
    os.mkdir(train_BikeBi_dir)

# Directory with our training BikeU pictures
train_BikeU_dir = os.path.join(train_dir, 'BikeU')
if not os.path.exists(train_BikeU_dir):
    os.mkdir(train_BikeU_dir)

# Directory with our training Road pictures
train_Road_dir = os.path.join(train_dir, 'Road')
if not os.path.exists(train_Road_dir):
    os.mkdir(train_Road_dir)

# Directory with our training Sidewalk pictures
train_Sidewalk_dir = os.path.join(train_dir, 'Sidewalk')
if not os.path.exists(train_Sidewalk_dir):
    os.mkdir(train_Sidewalk_dir)

# Directory with our training Sidewalk pictures
train_Crosswalk_dir = os.path.join(train_dir, 'Crosswalk')
if not os.path.exists(train_Crosswalk_dir):
    os.mkdir(train_Crosswalk_dir)

# Directory with our validationing BikeBi pictures
validation_BikeBi_dir = os.path.join(validation_dir, 'BikeBi')
if not os.path.exists(validation_BikeBi_dir):
    os.mkdir(validation_BikeBi_dir)

# Directory with our validationing BikeU pictures
validation_BikeU_dir = os.path.join(validation_dir, 'BikeU')
if not os.path.exists(validation_BikeU_dir):
    os.mkdir(validation_BikeU_dir)

# Directory with our validationing Road pictures
validation_Road_dir = os.path.join(validation_dir, 'Road')
if not os.path.exists(validation_Road_dir):
    os.mkdir(validation_Road_dir)

# Directory with our validationing Sidewalk pictures
validation_Sidewalk_dir = os.path.join(validation_dir, 'Sidewalk')
if not os.path.exists(validation_Sidewalk_dir):
    os.mkdir(validation_Sidewalk_dir)

# Directory with our validationing Sidewalk pictures
validation_Crosswalk_dir = os.path.join(validation_dir, 'Crosswalk')
if not os.path.exists(validation_Crosswalk_dir):
    os.mkdir(validation_Crosswalk_dir)

# Directory with our testing BikeBi pictures
test_BikeBi_dir = os.path.join(test_dir, 'BikeBi')
if not os.path.exists(test_BikeBi_dir):
    os.mkdir(test_BikeBi_dir)

# Directory with our testing BikeU pictures
test_BikeU_dir = os.path.join(test_dir, 'BikeU')
if not os.path.exists(test_BikeU_dir):
    os.mkdir(test_BikeU_dir)

# Directory with our testing Road pictures
test_Road_dir = os.path.join(test_dir, 'Road')
if not os.path.exists(test_Road_dir):
    os.mkdir(test_Road_dir)

# Directory with our testing Sidewalk pictures
test_Sidewalk_dir = os.path.join(test_dir, 'Sidewalk')
if not os.path.exists(test_Sidewalk_dir):
    os.mkdir(test_Sidewalk_dir)

# Directory with our testing Sidewalk pictures
test_Crosswalk_dir = os.path.join(test_dir, 'Crosswalk')
if not os.path.exists(test_Crosswalk_dir):
    os.mkdir(test_Crosswalk_dir)
'''
# Copy first 500 BikeBi images to train_BikeBis_dir
fnames = ['BikeBi ({}).jpg'.format(i) for i in range(500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_BikeBi_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 BikeBi images to validation_BikeBi_dir
fnames = ['BikeBi ({}).jpg'.format(i) for i in range(500, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_BikeBi_dir, fname)
    shutil.copyfile(src, dst)

# Copy first 500 BikeU images to train_BikeU_dir
fnames = ['BikeU ({}).jpg'.format(i) for i in range(500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_BikeU_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 BikeU images to validation_BikeU_dir
fnames = ['BikeU ({}).jpg'.format(i) for i in range(500, 1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_BikeU_dir, fname)
    shutil.copyfile(src, dst)

# Copy first 500 Road images to train_Road_dir
fnames = ['Road ({}).jpg'.format(i) for i in range(330)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_Road_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 Road images to validation_Road_dir
fnames = ['Road ({}).jpg'.format(i) for i in range(330, 660)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_Road_dir, fname)
    shutil.copyfile(src, dst)

# Copy first 500 Sidewalk images to train_Sidewalk_dir
fnames = ['Sidewalk ({}).jpg'.format(i) for i in range(187)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_Sidewalk_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 Sidewalk images to validation_Sidewalk_dir
fnames = ['Sidewalk ({}).jpg'.format(i) for i in range(187, 373)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_Sidewalk_dir, fname)
    shutil.copyfile(src, dst)
'''

print('total training BikeBi images:', len(os.listdir(train_BikeBi_dir)))
print('total training BikeU images:', len(os.listdir(train_BikeU_dir)))
print('total training Road images:', len(os.listdir(train_Road_dir)))
print('total training Sidewalk images:', len(os.listdir(train_Sidewalk_dir)))
print('total training Crosswalk images:', len(os.listdir(train_Crosswalk_dir)))

print('total validation BikeBi images:', len(os.listdir(validation_BikeBi_dir)))
print('total validation BikeU images:', len(os.listdir(validation_BikeU_dir)))
print('total validation Road images:', len(os.listdir(validation_Road_dir)))
print('total validation Sidewalk images:', len(os.listdir(validation_Sidewalk_dir)))
print('total validation Crosswalk images:', len(os.listdir(validation_Crosswalk_dir)))
print(' ')

imgsize = 150

# needed transforms
transform = transforms.Compose(
    [transforms.Resize(int(imgsize)), # Resize the short side of the image to 150 keeping aspect ratio
     transforms.CenterCrop(int(imgsize)), # Crop a square in the center of the image
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

batch_size1=20

# Directory with our training pictures
trainset = ImageFolder(train_dir, transform=transform)
testset = ImageFolder(validation_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(batch_size1),
                                          shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=int(batch_size1),
                                         shuffle=False, num_workers=8)


batch_size=20

# needed transformations
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose(
    [transforms.RandomHorizontalFlip(), # Resize the short side of the image to 150 keeping aspect ratio
     transforms.RandomRotation(10), # Crop a square in the center of the image
     transforms.RandomResizedCrop(imgsize),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'val': transforms.Compose(
    [transforms.Resize(int(imgsize)), # Resize the short side of the image to 150 keeping aspect ratio
     transforms.CenterCrop(int(imgsize)), # Crop a square in the center of the image
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(base_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=8) for x in ['train', 'val']}


classes = ('BikeBi', 'BikeU', 'Crosswalk', 'Road', 'Sidewalk')


class Model_LSTM(nn.Module):
    def __init__(self, num_classes=5):
        super(Model_LSTM, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x

# ResNet-50 pretrained model load and freezing of the first layers
pretrained_model = resnet50(pretrained=True)
pretrained_model.eval()
pretrained_model.to(device)

ct = 0
for child in pretrained_model.children():
    ct += 1
    if ct <= 5:
        for params in child.parameters():
            params.requires_grad = False

# model
model = nn.Sequential(
        pretrained_model,
        nn.Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(1000, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 5),
        nn.LogSoftmax(dim=1)
)

# in case there's fine tunning
'''
PATH1 = './final_model.pth'
model.load_state_dict(torch.load(PATH1))
'''

model = Model_LSTM
model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

from torchsummary import summary
summary(model, (3, imgsize, imgsize))
print(model)

t = time.time()
epochs = 60
val_acc_history = []
val_loss_history = []
train_acc_history = []
train_loss_history = []

epoch_loss_Train = 0
epoch_loss_Val = 0
epoch_acc_Train = 0
epoch_acc_Val = 0

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
for epoch in range(epochs):  # loop over the dataset multiple times

    for phase in ['train', 'val']:

        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        i = 0
        for inputs, labels in dataloaders[phase]:
            # get the inputs to gpu; data is a list of [inputs, labels]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                #forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)


            # print statistics
            if i % 125 == 0:  # print every 8 mini-batches
                    print(f"{phase} --> Epoch {epoch + 1}/{epochs} [{i}/{len(dataloaders[phase])}] loss: {loss.item():.2f}")
            i+=1


        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        print('Epoch {}/{}:'.format(epoch + 1, epochs))
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print("")
            print("Best Model Updated!")
            print("")
        if phase == 'val':
            val_acc_history.append(epoch_acc)
            val_loss_history.append(epoch_loss)
        if phase == 'train':
            train_acc_history.append(epoch_acc)
            train_loss_history.append(epoch_loss)

    print()


time_elapsed = time.time() - t
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
model.load_state_dict(best_model_wts)

print(" ")

# load best model weights
model.load_state_dict(best_model_wts)

# Disable gradients computation
torch.set_grad_enabled(False);


PATH = './final_model_ResNet3.pth'
torch.save(model.state_dict(), PATH)


acc_hist = [h.cpu().numpy() for h in val_acc_history]

train_acc_hist = [h.cpu().numpy() for h in train_acc_history]


#model.load_state_dict(torch.load(PATH))
testset = ImageFolder(validation_dir, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)


print(' ')



plt.title("Validation and Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("[%]")
plt.plot(range(1, epochs+1), val_acc_history, label="Accuracy")
plt.plot(range(1, epochs+1), train_acc_history, label="Training Accuracy")
plt.ylim((0,1.))
plt.xticks(np.arange(1, epochs+1, 1.0))
plt.legend()
plt.show()


plt.title("Validation and Training Loss")
plt.xlabel("Epochs")
plt.ylabel("[%]")
plt.plot(range(1, epochs+1), val_loss_history, label="Loss")
plt.plot(range(1, epochs+1), train_loss_history, label="Training Loss")
plt.ylim((0,1.))
plt.xticks(np.arange(1, epochs+1, 1.0))
plt.legend()
plt.show()