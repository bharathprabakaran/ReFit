import os
import cv2
import math
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse

# Define all paths
parser = argparse.ArgumentParser()
parser.add_argument("--domain", default='one_CT', type=str)
parser.add_argument("--threshold", default=0.5, type=float)

parser.add_argument('--gt_dir', default='/home/erik/PycharmProjects/Test/data/Dataset_BUSI_with_GT/GT/benign/', type=str) # './Dataset/Dataset_BUSI_with_GT/GT/benign/', type=str) 


parser.add_argument('--mode', default='npy', type=str)

args = parser.parse_args()
predict_folder = './Dataset/BoundaryFit_busi/'  



gt_folder = args.gt_dir
args.list = './data/' + args.domain + '.txt'
args.predict_dir = predict_folder

categories = ['benign', 'malignant']
num_cls = len(categories)



def compare(start,step,TP,P,T, name_list):
    
    # load all masks
    for obj in os.listdir(gt_folder): 

        name = obj[:-4]
        
        full_masks2 = predict_folder + name + '.npy'

        if os.path.isfile(predict_folder + name + '.npy'):


                

                full_masks = np.load(full_masks2)
                full_masks[full_masks > 0] = 255
                
                
                predict = full_masks
                
                #load gt files and pre-process
                gt_file = os.path.join(gt_folder,'%s.png'%name)
                gt = Image.open(gt_file)
                newsize = (512, 512)
                gt = gt.resize(newsize)
                gt = np.array(gt)

                predict[predict>0]=1
                
                gt[gt>0]= 1
                

                # actuall True positive calculations
                cal = gt<255
                
                mask = (predict==gt) * cal
                
                for i in range(num_cls):
                    P[i].acquire()
                    P[i].value += np.sum((predict==i)*cal)
                    #print(P[i])
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
    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))

    loglist = {}
    for i in range(num_cls):
        # if i%2 != 1:
        #     print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
        # else:
        #     print('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
        loglist[categories[i]] = IoU[i] * 100
    
    miou = np.mean(np.array(IoU))
    t_tp = np.mean(np.array(T_TP)[1:])
    p_tp = np.mean(np.array(P_TP)[1:])
    fp_all = np.mean(np.array(FP_ALL)[1:])
    fn_all = np.mean(np.array(FN_ALL)[1:])
    miou_foreground = np.mean(np.array(IoU)[1:])
    # print('\n======================================================')
    # print('%11s:%7.3f%%'%('mIoU',miou*100))
    # print('%11s:%7.3f'%('T/TP',t_tp))
    # print('%11s:%7.3f'%('P/TP',p_tp))
    # print('%11s:%7.3f'%('FP/ALL',fp_all))
    # print('%11s:%7.3f'%('FN/ALL',fn_all))
    # print('%11s:%7.3f'%('miou_foreground',miou_foreground))
    loglist['mIoU'] = miou * 100
    loglist['t_tp'] = t_tp
    loglist['p_tp'] = p_tp
    loglist['fp_all'] = fp_all
    loglist['fn_all'] = fn_all
    loglist['miou_foreground'] = miou_foreground 
    return loglist

if __name__ == '__main__':
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values

    if args.mode == 'png':
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 3)
        print('mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(loglist['mIoU'], loglist['fp_all'], loglist['fn_all']))
    elif args.mode == 'rw':
        th_list = np.arange(0.05, args.max_th, 0.05).tolist()

        over_activation = 1.60
        under_activation = 0.60
        
        mIoU_list = []
        FP_list = []

        for th in th_list:
            args.threshold = th
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21)

            mIoU, FP = loglist['mIoU'], loglist['fp_all']

            print('Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(th, mIoU, FP))

            FP_list.append(FP)
            mIoU_list.append(mIoU)
        
        best_index = np.argmax(mIoU_list)
        best_th = th_list[best_index]
        best_mIoU = mIoU_list[best_index]
        best_FP = FP_list[best_index]

        over_FP = best_FP * over_activation
        under_FP = best_FP * under_activation

        print('Over FP : {:.4f}, Under FP : {:.4f}'.format(over_FP, under_FP))

        over_loss_list = [np.abs(FP - over_FP) for FP in FP_list]
        under_loss_list = [np.abs(FP - under_FP) for FP in FP_list]

        over_index = np.argmin(over_loss_list)
        over_th = th_list[over_index]
        over_mIoU = mIoU_list[over_index]
        over_FP = FP_list[over_index]

        under_index = np.argmin(under_loss_list)
        under_th = th_list[under_index]
        under_mIoU = mIoU_list[under_index]
        under_FP = FP_list[under_index]
        
        print('Best Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(best_th, best_mIoU, best_FP))
        print('Over Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(over_th, over_mIoU, over_FP))
        print('Under Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(under_th, under_mIoU, under_FP))
    else:
        
        
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 2)
            print('mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(loglist['mIoU'], loglist['fp_all'], loglist['fn_all']))

