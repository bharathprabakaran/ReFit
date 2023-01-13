import torchvision
import torch
import torch.nn.functional as F
import torch.nn as nn


from torch.utils.data import random_split
from torchvision import transforms
from torch.utils.data import DataLoader
from decaloader import USDataset
import numpy as np

TRAIN_PATH =  /path/to/DECATHLON/ADC/
TRAIN_GT_PATH = /path/to/DECATHLON/GT/

VAL_PATH =  /path/to/DECATHLON/ADC/VAL/
VAL_GT_PATH = /path/to/DECATHLON/GT/VAL/

RES_PATH = './MODEL.pt'



model = torchvision.models.resnet34(pretrained = True)

model.fc = nn.Sequential(

    nn.Linear(512, 2, bias = False),
)



def train_epoch(model, device, train_loader, optimizer, epoch, scheduler):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    model.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
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
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), accuracy))

    return accuracy











def evaluate_model(model):
    

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    
    image_path = IMAGE_PATH 
    gt_path =  GT_PATH

    val_image_path = VAL_PATH
    val_gt_pat = VAL_GT_PATH 
    
    
    # Generate Dataset from paths
    dataset = USDataset(image_path, gt_path, transform=None)
    val_dataset = USDataset(val_image_path, val_gt_path, transform=None)


    batch_size = 4


    classes = ('benign', 'malignant')

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,  shuffle=True) #  sampler=sampler) #
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    epoch = 5
    


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epoch):  # loop over the dataset multiple times

     running_loss = 0.0
     for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data
        inputs.to(device), labels.to(device)
        inputs
        labels
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)#.cuda())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches

            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
     test_epoch(model, device, test_loader)

evaluate_model(model)


dataset = USDataset(TRAIN_PATH, TRAIN_GT_PATH, transform=None)
val_dataset = USDataset(VAL_PATH, VAL_GT_PATH, transform=None)

    

    


batch_size = 32


classes = ('benign', 'malignant')

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,  shuffle=True) #  sampler=sampler) #
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    
test_epoch(model, device, test_loader)


torch.save(model.state_dict(), RES_PATH)


