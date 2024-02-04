from mmdet.apis import init_detector, inference_detector, show_result_pyplot

import random
import ignite

from collections import OrderedDict
import cv2
import numpy as np
from scipy import signal
import os
import torch
from torch import nn, optim

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *



import cv2 
def draw_gt(img_path, gt_path):
    img = cv2.imread(img_path)
    
    anno = open(gt_path,'r')
    for j in anno:
        # print(j)
        anno = j.split(' ')
        start_point = (int(float(anno[0])),int(float(anno[1])))
        # print(start_point)
        end_point = (int(float(anno[2])),int(float(anno[3])))
        thicknesss = 2
        color = (0,255,0)
        img = cv2.rectangle(img,start_point,end_point,color,thicknesss)

    return img


# demo function
def demo(model, img_path):  

    # config_file = '/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/configs_nwdrka/nwd_rka/RS_cl_two_stage.py'
    # checkpoint_file = '/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/pretrain/base_24.pth'
    # checkpoint_file = '/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/pretrain/CL_26.5.pth'


    # img = '/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/tools/feature_map/psnr/P2349__1.0__600___0.png'

    result = inference_detector(model, img_path)
    return 



def eval_step(engine, batch):
    return batch


def cal_psnr(target_img, gt_img):

    default_evaluator = Engine(eval_step)

    psnr = ignite.metrics.PSNR(data_range=255)
    psnr.attach(default_evaluator, 'psnr')

    # target_img = cv2.imread('/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/tools/feature_map/psnr/base_21.png')
    # gt_img = cv2.imread('/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/tools/feature_map/psnr/kernel2.png')

    try:
        target_img = torch.from_numpy(target_img)
        gt_img = torch.from_numpy(gt_img)   
        state = default_evaluator.run([[target_img, gt_img]])
        p = state.metrics['psnr']
        # print(type(p))
        return p
    except:
        return 0

    # state = default_evaluator.run([[target_img, gt_img]])


    # return state.metrics['psnr']




def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d /= 1.8
    # gkern2d *= (1/gkern2d.max())
    return gkern2d



def gau(gt_file_path, image_path):


    img_name = image_path.split('/')[:-1]
    anno_data = open(gt_file_path, 'r')
    image = cv2.imread(image_path) #read image shape

    image_filter = np.zeros((800, 800), dtype=np.float64)
    # print(image_filter.shape)

    for j in anno_data:
        anno = j.split(' ')
        # print(j)
        x = int(int(float(anno[0])) + int(float(anno[2]))/2)
        y = int(int(float(anno[1])) + int(float(anno[3]))/2)
        x_tl, y_tl = int(float(anno[0])), int(float(anno[1]))
        x_t2, y_t2 = int(float(anno[2])), int(float(anno[3]))

    
        size = (x_t2-x_tl)*(y_t2-y_tl)
        # print(size)
        
        kernel = gkern(7,4)
        # print(kernel)
        # print(kernel.shape)
        # for x1 in range(kernel.shape[0]):
        #     for y1 in range(kernel.shape[1]):
        #         # print(int(kernel.shape[0]/2))
        #         if  (x - int(kernel.shape[0]/2) + x1) < 0 or (x - int(kernel.shape[0]/2) + x1) >= image.shape[1]:
        #             continue
        #         elif (y - int(kernel.shape[1]/2) + y1) < 0 or (y - int(kernel.shape[1]/2) + y1) >= image.shape[0]:
        #             continue
        #         else :
        #             image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1] = max(image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1], kernel[x1][y1])
    # print(image_filter.max())
        cnt=kernel.shape[0] # kernel shape need to be odd and can be a square root

        A= image_filter[y_tl:y_tl+cnt, x_tl:x_tl+cnt]
        # print("A:",A.shape) 
        # print(A.any())
        if ((x_tl<cnt or x_tl>(800-cnt)) or (y_tl<cnt or y_tl>(800-cnt))):
            continue
        if A.any()==0:
            
            # print("kernel.shape:",kernel.shape)
            # print(image_filter[y_tl:y_tl+cnt, x_tl:x_tl+cnt].shape)
            # print(x_tl)
            # print(x_tl,y_tl,x_br,y_br,cnt)
            # print(A.shape)
            # print('here')

            image_filter[y_tl:y_tl+cnt, x_tl:x_tl+cnt] = kernel
        else:
            # print('there')
            for l in range(A.shape[0]):
                for k in range(A.shape[1]):
                    pass
                    # print(max(A[l][k], kernel[l][k]))
                    # A[l][k]=max(A[l][k], kernel[l][k])
                    # A[l][k] = (0.3*A[l][k]+0.7*kernel[l][k])
            image_filter[y_tl:y_tl+cnt, x_tl:x_tl+cnt]= A[l][k]  
        # print(A)
        # print(kernel.max())
        
        
        # for l in range(A.shape[0]):
        #     for k in range(A.shape[1]):
        #         if A[l][k] == 0:
        #             A[l][k] = kernel[l][k] 
        #         else:
        #             print('here')

        #             A[l][k]=max(A[l][k], kernel[l][k])
        #             # A[l][k] = (0.3*A[l][k]+0.7*kernel[l][k])
        # image_filter[y_tl:y_tl+cnt, x_tl:x_tl+cnt]= A[l][k]  
        # print(image_filter.max())


    # cv2.imwrite(FLAGS.output+'/test_blur/'+i[:-4]+'.jpg', image_filter*255)
    # print(image_filter.shape)
    image_filter = image_filter * 255
    cv2.imwrite('/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr_2/g.png', image_filter)
    img = cv2.imread('/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr_2/g.png')
    # print(img.shape)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    # cv2.imwrite('/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/gau/'+image_path.split('/')[-1],img)

    # cv2.imwrite('/mnt/data0/Garmin/kaimmdet/nwd/mmdet-nwdrka/tools/visdrone/'+img_name, image_filter)
    # img = cv2.imread('/mnt/data0/Garmin/kaimmdet/nwd/mmdet-nwdrka/tools/visdrone/'+img_name)
    # # print(img.shape)
    # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    # cv2.imwrite('/mnt/data0/Garmin/kaimmdet/nwd/mmdet-nwdrka/tools/visdrone/'+img_name,img)
    return img



def main():
    # args = parse_args()
    intro_files = [
    # '/mnt/data0/Garmin/datasets/ai-tod/test/images/79__2400_0.png',
    # '/mnt/data0/Garmin/datasets/ai-tod/test/images/129__0_1800.png',
    '/mnt/data0/Garmin/datasets/ai-tod/test/images/322__1200_1200.png',
    # '/mnt/data0/Garmin/datasets/ai-tod/test/images/1037__600_2416.png',
    ]
    # gt_files = ['/mnt/data0/Garmin/datasets/ai-tod/test/labels/79__2400_0.txt',
    # '/mnt/data0/Garmin/datasets/ai-tod/test/labels/129__0_1800.txt',
    # '/mnt/data0/Garmin/datasets/ai-tod/test/labels/322__1200_1200.txt',
    # '/mnt/data0/Garmin/datasets/ai-tod/test/labels/1037__600_2416.txt',
    # ]
    config_file = '/mnt/data0/Garmin/DNTR/mmdet-dntr/configs/aitod-dntr/aitod_RS_baseline.py'
    # checkpoint_file = '/mnt/data0/Garmin/pinjyun/nwd/mmdet-nwdrka/work_dirs/visdrone_RS_cascade_t2t_crop_4x/latest.pth'
    # checkpoint_file = '/mnt/data0/Garmin/pinjyun/nwd/mmdet-nwdrka/work_dirs/visdrone_RS_cascade_t2t_nms100_4x/epoch_48.pth'
    checkpoint_file_cl = '/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/pretrain/CL_26.5.pth'
    checkpoint_file_base = '/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/pretrain/base_24.pth'

    # visdrone
    # image_folder = '/mnt/data0/Garmin/datasets/visdrone/VisDrone2019-DET-val/images/'
    # gt_folder = '/mnt/data0/Garmin/datasets/visdrone/VisDrone2019-DET-val/annotations/'

    # aitod
    image_folder = '/mnt/data0/Garmin/datasets/ai-tod/test/images/'
    # '/mnt/data0/Garmin/datasets/ai-tod/val/images'
    gt_folder = '/mnt/data0/Garmin/datasets/ai-tod/test/labels/'

    # uavdt
    # image_folder = '/mnt/data0/Garmin/datasets/uavdt/val/'
    # gt_folder = '/mnt/data0/Garmin/datasets/uavdt/val_txt/'

    imgs = os.listdir(image_folder)
    gts = os.listdir(gt_folder)

    # print(imgs)
    imgs.sort()
    gts.sort()
    image_num = len(imgs)
    print('img num:', len(imgs))
    print('img num:', len(gts))
    total_psnr_cl = 0
    total_psnr_base = 0

    # # define model
    device = 'cuda:0'
    model_cl = init_detector(config_file, checkpoint_file_cl, device=device)
    model_base = init_detector(config_file, checkpoint_file_base, device=device)
    for img in in
    for img in imgs:
        print('img:',image_folder + img)
        i_name = img.split('.')[:-1]
        i_name = ".".join(i_name)
        g_name = gt_folder + i_name + '.txt'
        print('gt:',g_name)


        demo (model_cl, (image_folder + img))
        feature_model_cl = cv2.imread('/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr_2/c1.png')
        
            
        demo (model_base, (image_folder + img))
        feature_model_base = cv2.imread('/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr_2/c1.png')
        

        feature_gt = gau(g_name, image_folder+img)
        psnr_cl = cal_psnr(feature_model_cl, feature_gt)
        psnr_base = cal_psnr(feature_model_base, feature_gt)
        print('psnr_cl:', psnr_cl)
        print('psnr_base:', psnr_base)

        total_psnr_cl += psnr_cl
        total_psnr_base += psnr_base

    print('psnr cal done...')
    print('total number of images : ', image_num)
    print('average psnr_cl :', total_psnr_cl/image_num)
    print('average psnr_base :', total_psnr_base/image_num)

    # for path, gt_path in zip(intro_files, gt_files):
    #     print(type(path))
    #     feature_gt = gau(gt_path, path)
    #     img_gt = draw_gt(path, gt_path)
    #     cv2.imwrite('/mnt/data0/Garmin/kaimmdet/nwd/mmdet-nwdrka/tools/psnr_exp/gt/'+path.split('/')[-1],img_gt)

    #     demo (model_cl, (path))
    #     feature_model_cl = cv2.imread('/mnt/data0/Garmin/kaimmdet/nwd/mmdet-nwdrka/tools/uavdt_psnr/c2.png')
    #     cv2.imwrite('/mnt/data0/Garmin/kaimmdet/nwd/mmdet-nwdrka/tools/psnr_exp/cl/'+path.split('/')[-1], feature_model_cl)

    #     demo (model_base, (path))
    #     feature_model_base = cv2.imread('/mnt/data0/Garmin/kaimmdet/nwd/mmdet-nwdrka/tools/uavdt_psnr/c2.png')
    #     cv2.imwrite('/mnt/data0/Garmin/kaimmdet/nwd/mmdet-nwdrka/tools/psnr_exp/base/'+path.split('/')[-1], feature_model_base)
    #     # demo (model_base, (path))
    #     # feature_model_base = cv2.imread('/mnt/data0/Garmin/kaimmdet/nwd/mmdet-nwdrka/tools/uavdt_psnr/c1.png')

    #     psnr_cl = cal_psnr(feature_model_cl, feature_gt)
    #     psnr_base = cal_psnr(feature_model_base, feature_gt)
    #     print('path:', path.split('/')[-1])
    #     print('psnr_cl:', psnr_cl)
    #     print('psnr_base:', psnr_base)

if __name__ == '__main__':
    main()