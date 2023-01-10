import os
import time

import cv2
import numpy as np
from skimage import segmentation
from PIL import Image
import torch
import torch.nn as nn
import skimage
from skimage.color import rgb2gray
import torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

outer = './OUTPUT_PATH/'
path = './PATH/TO/BRATS/'


class Args(object):
    train_epoch = 2 ** 6
    mod_dim1 = 64  #
    mod_dim2 = 32
    gpu_id = 0

    min_label_num = 40
    max_label_num = 256


class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
        )

    def forward(self, x):
        return self.seq(x)



def run():
    args = Args()
    i = 1
    for obj in os.listdir(path):
            out = outer + obj

            start_time0 = time.time()
            n = obj 

            rat = path + n 
            
            torch.cuda.manual_seed_all(43)
            torch.manual_seed(43)
            np.random.seed(43)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # choose GPU:0
            full_path = rat
            
            img1 = np.load(full_path)
            img = np.zeros([128,128,3])
            
            img[:,:,0] = img1[:,:,0]
            
            
            
            img[:,:,0] = (img1[:,:,0])
            img[:,:,1] = img[:,:,0]
            img[:,:,2] = img[:,:,0]
            img[img<0.8]=0
            
            
            image = img*100
            
            
            '''segmentation ML'''
            
            
            
            
            seg_map = segmentation.slic(image, n_segments=700, compactness=20, sigma=1, start_label=1) 
            
            seg_map = seg_map.flatten()
            seg_lab = [np.where(seg_map == u_label)[0]
                       for u_label in np.unique(seg_map)]

            '''train init'''
            device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

            tensor = image.transpose((2, 0, 1))
            tensor = tensor.astype(np.float32) / 255.0
            tensor = tensor[np.newaxis, :, :, :]
            tensor = torch.from_numpy(tensor).to(device)

            model = MyNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
            

            image_flatten = image.reshape((-1, 3))
            color_avg = np.random.randint(255, size=(args.max_label_num, 3))
            show = image

            '''train loop'''
            start_time1 = time.time()
            model.train()

            for batch_idx in range(args.train_epoch):
                '''forward'''
                optimizer.zero_grad()
                output1 = model(tensor)[0]
                
                output = output1.permute(1, 2, 0).view(-1, args.mod_dim2)
                target = torch.argmax(output, 1)
                im_target = target.data.cpu().numpy()

                '''refine'''
                for inds in seg_lab:
                    u_labels, hist = np.unique(im_target[inds], return_counts=True)
                    im_target[inds] = u_labels[np.argmax(hist)]

                '''backward'''
                target = torch.from_numpy(im_target)
                target = target.to(device)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                '''show image'''
                un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
                if un_label.shape[0] < args.max_label_num:  # update show
                    img_flatten = image_flatten.copy()
                    if len(color_avg) != un_label.shape[0]:
                        color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=int) for label in un_label]
                    for lab_id, color in enumerate(color_avg):
                        img_flatten[lab_inverse == lab_id] = color
                    show = img_flatten.reshape(image.shape)
                
                if len(un_label) < args.min_label_num:
                    break

            '''save'''
            time0 = time.time() - start_time0
            time1 = time.time() - start_time1
            
            out = outer + n[:-4]+'.png' 
            show = np.float32(show)
            show = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(out, show)
            
            print(f'Image: {i}')
            i +=1
            #break
                   

if __name__ == '__main__':
    run()
