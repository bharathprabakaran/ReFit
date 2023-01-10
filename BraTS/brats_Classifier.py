import torchvision
import torch
import torch.nn.functional as F
import torch.nn as nn


from torch.utils.data import random_split
from torchvision import transforms
from torch.utils.data import DataLoader
from bratsloader import USDataset
import numpy as np

SAMPLE1_PATH =  /path/to/BRATS/SLICES/
SAMPLE1_GT_PATH = /path/to/BRATS/GT/

SAMPLE2_PATH =  /path/to/BRATS/SLICES/
SAMPLE2_GT_PATH = /path/to/BRATS/GT/


SAMPLE3_PATH =  /path/to/BRATS/SLICES/
SAMPLE3_GT_PATH = /path/to/BRATS/GT/


RES_PATH = './MODEL.pt'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torchvision.models.resnet34(pretrained = False)


model.fc = nn.Sequential(

    nn.Linear(512, 2, bias = False))
model.to(device)


def train_epoch(model, device, train_loader, optimizer, epoch, scheduler):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    model.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data.to(device)
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_epoch(model, device, test_loader):
    model.to(device)
    model.cuda()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data.to(device)
            output = model(data.cuda())
            pred = output.argmax(dim=1, keepdim=True)
            pred = pred.cpu()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), accuracy))

    return accuracy











def evaluate_model(model):
 


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)


    
    
    # Generate Dataset from paths
    dataset1 = USDataset(SAMPLE1_PATH, SAMPLE1_GT_PATH, transform=None)
    dataset2 = USDataset(SAMPLE2_PATH, SAMPLE2_GT_PATH, transform=None)
    dataset3 = USDataset(SAMPLE3_PATH, SAMPLE3_GT_PATH, transform=None)
    

    ## DECIDE WHICH SAMPLE TO TRAIN AND VALIDATE ON:
    dataset = dataset1 + dataset2
    val_dataset = dataset3


    
    


    batch_size = 32


    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,  shuffle=True) 
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    
    epoch = 10
    


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epoch):  # loop over the dataset multiple times

     running_loss = 0.0
     for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        
        inputs, labels = data
        inputs.to(device), labels.to(device)
        inputs.cuda()
        labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.cuda())
        loss = criterion(outputs.cuda(), labels.cuda())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches

            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
     test_epoch(model, device, test_loader)

evaluate_model(model)




# Generate Dataset from paths
dataset1 = USDataset(SAMPLE1_PATH, SAMPLE1_GT_PATH, transform=None)
dataset2 = USDataset(SAMPLE2_PATH, SAMPLE2_GT_PATH, transform=None)
dataset3 = USDataset(SAMPLE3_PATH, SAMPLE3_GT_PATH, transform=None)
    

## DECIDE WHICH SAMPLE TO TRAIN AND VALIDATE ON:
val_dataset = dataset3




batch_size = 32



test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    
test_epoch(model, device, test_loader)

torch.save(model.state_dict(), RES_PATH)

