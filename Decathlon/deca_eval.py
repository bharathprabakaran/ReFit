import os
import cv2
import math
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing


predict_folder = '/Path/to/results/'


gt_folder = '/Path/to/Ground-truth/'


categories = ['1', '0']
num_cls = len(categories)


def compare(start,step,TP,P,T, name_list):

    for obj in os.listdir(gt_folder): 

        name = obj[:-4]

        full_path = predict_folder + name + '.png'


        if os.path.isfile(predict_folder + name + '.png'):

                img = Image.open(full_path)

                img = np.array(img)

                img[img > 0] = 255

                predict = img
                gt_file = os.path.join(gt_folder,'%s.png'%name)

                
                gt = Image.open(gt_file)
                newsize = (320, 320)
                gt = gt.resize(newsize)
                gt = np.array(gt)


                predict[predict>0]=1
                

                gt[gt>0]= 1


                cal = gt<255

                mask = (predict==gt) * cal

                for i in range(num_cls):
                    P[i].acquire()
                    P[i].value += np.sum((predict==i)*cal)

                    P[i].release()
                    T[i].acquire()
                    T[i].value += np.sum((gt==i)*cal)
                    T[i].release()
                    TP[i].acquire()
                    TP[i].value += np.sum((gt==i)*mask)
                    TP[i].release()
            
def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))
    
    p_list = []
    for i in range(1):
        p = multiprocessing.Process(target=compare, args=(i,1,TP,P,T, name_list))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = []
    DICE = [] 
    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        
    loglist = {}
    for i in range(num_cls):

        loglist[categories[i]] = IoU[i] * 100
    
    miou = np.mean(np.array(IoU))
    t_tp = np.mean(np.array(T_TP)[1:])
    p_tp = np.mean(np.array(P_TP)[1:])
    fp_all = np.mean(np.array(FP_ALL)[1:])
    fn_all = np.mean(np.array(FN_ALL)[1:])
    miou_foreground = np.mean(np.array(IoU)[1:])
    dic = np.mean(np.array(DICE)[1:])

    loglist['mIoU'] = miou * 100
    loglist['t_tp'] = t_tp
    loglist['p_tp'] = p_tp
    loglist['fp_all'] = fp_all
    loglist['fn_all'] = fn_all
    loglist['miou_foreground'] = miou_foreground 
    return loglist

if __name__ == '__main__':
    name_list = categories

    if True:#args.mode == 'png':
        loglist = do_python_eval(predict_folder, gt_folder, categories, 2)
        DSC = (2 * loglist['mIoU']/100)/(loglist['mIoU']/100 +1)*100
        print('DSC={:.3f}%, mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(DSC, loglist['mIoU'], loglist['fp_all'], loglist['fn_all']))


