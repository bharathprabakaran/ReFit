import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import nibabel
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import MinMaxScaler

import nibabel as nib
scaler = MinMaxScaler()


TRAIN_DATASET_PATH = '/PATH/TO/MICCAI_BraTS2020_TrainingData/'


t2_list = sorted(glob.glob('/PATH/TO/MICCAI_BraTS2020_TrainingData/*/*t2.nii.gz'))
t1ce_list = sorted(glob.glob('/PATH/TO/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii.gz'))
flair_list = sorted(glob.glob('/PATH/TO/MICCAI_BraTS2020_TrainingData/*/*flair.nii.gz'))
mask_list = sorted(glob.glob('/PATH/TO/MICCAI_BraTS2020_TrainingData/*/*seg.nii.gz'))

# Each volume generates 18 64x64x64x4 sub-volumes.
# Total 369 volumes = 6642 sub volumes

for img in range(len(t2_list)):  # Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", img)

    temp_image_t2 = nib.load(t2_list[img]).get_fdata()
    temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(
        temp_image_t2.shape)

    temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(
        temp_image_t1ce.shape)

    temp_image_flair = nib.load(flair_list[img]).get_fdata()
    temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(
        temp_image_flair.shape)

    temp_mask = nib.load(mask_list[img]).get_fdata()
    temp_mask = temp_mask.astype(np.uint8)
    temp_mask[temp_mask == 4] = 1  # Reassign mask values 4 to 1
    temp_mask[temp_mask == 2] = 1  # Reassign mask values 2 to 1

    # print(np.unique(temp_mask))

    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)

    # Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
    # cropping x, y, and z
    temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]

    val, counts = np.unique(temp_mask, return_counts=True)

    for n_slice in range(temp_combined_images.shape[2]):
        np.save('./brats/img/image_' + str(img) + '_' + str(n_slice) + '.npy', temp_combined_images[:,:,n_slice])
        np.save('./brats/gt/mask_' + str(img)   + '_' + str(n_slice) + '.npy', temp_mask[:,:,n_slice])


