from PIL import Image, ImageDraw
import numpy as np
import os
from PIL import Image, ImageOps
# Opening the image and
# converting its type to RGBA


cam_path = './Dataset/CAM_busi/'
out_path = './Dataset/BoundaryFit_busi/'

segs = './Dataset/USS_busi_slic/'
# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
bild=0



for obj in os.listdir(cam_path):
        name = obj[:-4]
  
        img = Image.open(segs + name + '.png')
        newsize = (512, 512)
        img = img.resize(newsize)


        img = np.array(img)
        m = np.amax(img)

        img = (img / m)*255

        img = img.astype(int)



        cams2 = Image.open(cam_path + name + '.png')
        newsize = (512, 512)

        cams2 = cams2.resize(newsize)


        cams2 = np.array(cams2)
 

        cams2[cams2 >0] = 1
        cams2[cams2 < 1] = 0


        rough_cam = cams2



        end = np.zeros(cams2.shape)
        
        
        for cla in range(1):


            add = img

            add = Image.fromarray((add).astype(np.uint8))

            rep_value = (0, 0, 0)


            h, w = rough_cam.shape
            for k in range(h - 1):
                        for j in range(w - 1):
                            if rough_cam[ (k), (j)] == 0:
                                for b in range(3):
                                    ImageDraw.floodfill(add, (j, k), rep_value, thresh=10)#10)
            out = out_path + obj[:-4] + '.npy'
            add = np.array(add)
            add = np.sum(add, axis = 2)

            add[add > 0] = 255  # 255


            add[add > 0] = 255


            #add = Image.fromarray((add).astype(np.uint8))
            #add.show()
            #add.save(out)

            #break
        #break
        np.save(out,add)
        bild +=1
        print(bild)

# restore np.load for future normal usage
np.load = np_load_old
