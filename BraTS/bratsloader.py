from __future__ import print_function, division
from PIL import Image
import xml.etree.ElementTree as ET
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random
#random.seed(0)
import cv2
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader

class USDataset(Dataset):


    def __init__(self, img_path,gt_path, transform=transforms.ToTensor()):#, flip= True):
 
        self.img_path = img_path
        self.gt_path = gt_path
        self.transform = transform

        self.X = []
        self.Y = []

        for obj in os.listdir(gt_path):
                  imgs = img_path  + 'image' + obj[4:]

                  mask = np.load( gt_path  + obj)
                  if np.any(mask):
                      label = 1

                  else:
                      label = 0

                  self.X.append(imgs)
                  self.Y.append(label)
                 





    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # read images and transform them to tensors
        img_name = self.X[idx]

        image = np.load(img_name)
        tensor_transform = transforms.ToTensor()
       
        image = tensor_transform(image)

        label = self.Y[idx]
        label = np.array(label)

        
        image = torch.as_tensor(image, dtype=torch.float32)

        label = torch.as_tensor(label, dtype=torch.int64)


        sample = (image, label)

        return sample

