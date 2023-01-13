from PIL import Image, ImageDraw
import numpy as np
import os

import cv2



predict_path = './out_path/' 

segs = './CAM_RESULTS/'

image_path = './USS_RESULTS/'


# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

bild=0

for TH in range(1):
    th = 0
   

    for obj in os.listdir(segs):
        
        name = obj
        
        
        img1 = cv2.imread(image_path + obj[:-3] + 'png', 1)# /255
        
        img = np.float32(img1)# / 255
        
        
        
        
        # NORMALIZE IMAGES
        
        m = np.amax(img)
        #print(img.shape)
        img = (img / m)*255
        #print(np.amax(img))
        img = img.astype(int)
        #print(img)

        cam = np.load(segs + obj[:-3] + 'npy')
        
        for er in range(128):
                  for err in range(128):
                    if cam[er,err]<0.7:
                      cam[er,err]=0
                    else:
                      cam[er,err]=255

       

        



        
        for cla in range(1):

         
            add = img
            
            add = Image.fromarray((add).astype(np.uint8))
            
            rep_value = (0, 0, 0)

            
            h, w = cam.shape
            # print(f' H : {h}, W : {w}')
            for k in range(h - 1):
                        for j in range(w - 1):
                            if cam[(k), (j)] == 0:
                                for b in range(1):
                                    ImageDraw.floodfill(add, (j, k), rep_value, thresh=1) #10)
                                    
            out = predict_path + obj[:-4] + '.npy'
            add = np.array(add)
            
            add[add > 0] = 255  # 255

            
            add = Image.fromarray((add).astype(np.uint8))
            #add.show()
  
            #break
        #break
        add.save(predict_path + obj[:-3] + 'png')
        
        bild +=1
        print(bild)

# restore np.load for future normal usage
np.load = np_load_old
