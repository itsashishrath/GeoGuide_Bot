'''
* Team Id: 1194
* Author List: Keshav Joshi, Ashish Rathore, Disha Chhabra.
* Filename: task_2b_model_training.py
* Theme: eYRC Geo Guide(GG)
* Functions: None
* Global Variables: custom_image_path, train_transforms, Model, dataset,
                    class_names, train_size, val_size, train_dataset, val_dataset, 
                    batch_size, train_loader, val_loader, model, criterion,
                    optimizer, scheduler, num_epochs, device
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import InterpolationMode


# Setting up our folder path where all our images are stored
custom_image_path = r'training'


# Setting up transformation according to the model requirements and model that we our using is shufflenetV2
train_transforms = transforms.Compose([
    transforms.Resize([232], interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Setting up our model arciteture and initiating it
class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        
        # Load shufflenet_v2_x2_0 with specified arguments
        self.base = models.shufflenet_v2_x2_0(pretrained=True, progress=True)
        
        # Modify the classifier part
        self.base.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 5)
        )
        
    def forward(self, x):
        x = self.base(x)
        return x
    

# Loding the dataset using built in imagefolder function this function uses label as our subfolder name
dataset = ImageFolder(root=custom_image_path, transform=train_transforms)
class_names = dataset.classes

# Spliting traing testing data set 
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Making our data ready for training
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# Initialize the model
model = Model()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

# Defining all the hyper parameters and using cuda for faster traning
num_epochs = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = torch.nn.DataParallel(model)


# Our traditional Traning and Tesing loop in one go.
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)

    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = val_corrects.double() / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    scheduler.step()



# Saving our model to using it in our main file.
torch.save(model.state_dict(), 'custom_cnn_model.pth')