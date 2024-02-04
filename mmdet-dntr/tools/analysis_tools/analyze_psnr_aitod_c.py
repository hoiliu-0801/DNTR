from mmdet.apis import init_detector, inference_detector

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

    target_img = torch.from_numpy(target_img)
    gt_img = torch.from_numpy(gt_img)

    state = default_evaluator.run([[target_img, gt_img]])


    return state.metrics['psnr']




def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d /= 1.8
    # gkern2d *= (1/gkern2d.max())
    return gkern2d



def gau(gt_file_path, image_path):


    img_name = image_path.split('/')[-1]
    anno_data = open(gt_file_path, 'r')
    image = cv2.imread(image_path) #read image shape

    image_filter = np.zeros((800, 800), dtype=np.float64)
    # print(image_filter.shape)

    for j in anno_data:
        anno = j.split(' ')
        # print(j)
        x = int(int(float(anno[0])) + int(float(anno[2]))/2)
        y = int(int(float(anno[1])) + int(float(anno[3]))/2)
        x_tl, y_tl = int(float(anno[1])*800), int(float(anno[2])*800)
        x_t2, y_t2 = int(float(anno[3])*800), int(float(anno[4])*800)
        # print(x_tl,x_t2,y_tl,y_t2)

    
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
    cv2.imwrite('/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/g.png', image_filter)
    img = cv2.imread('/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/g.png')
    # print(img.shape)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    # print('/mnt/data0/Garmin/pinjyun/mmdetection/tools/psnr_exp/gau/'+image_path.split('/')[-1])
    

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
    # '/mnt/data0/Garmin/datasets/ai-tod/test/images/285__1800_1200.png',
    # '/mnt/data0/Garmin/datasets/ai-tod/test/images/322__1200_1200.png',
    # '/mnt/data0/Garmin/datasets/ai-tod/test/images/1037__600_2416.png',
    # '322__1200_1200.png'
    '9999977_00000_d_0000056__0_278.png'
    # '/mnt/data0/Garmin/datasets/ai-tod/test/images/12460.png',
    ]
    # gt_files = [
    # '/mnt/data0/Garmin/datasets/ai-tod/test/labels/79__2400_0.txt',
    # '/mnt/data0/Garmin/datasets/ai-tod/test/labels/129__0_1800.txt',
    # '/mnt/data0/Garmin/datasets/ai-tod/test/labels/285__1800_1200.txt',
    # '/mnt/data0/Garmin/datasets/ai-tod/test/labels/322__1200_1200.txt',
    # '/mnt/data0/Garmin/datasets/ai-tod/test/labels/1037__600_2416.txt',
    # '/mnt/data0/Garmin/datasets/ai-tod/test/labels/12460.txt',
    # ]
    
    # checkpoint_file = '/mnt/data0/Garmin/pinjyun/nwd/mmdet-nwdrka/work_dirs/visdrone_RS_cascade_t2t_crop_4x/latest.pth'
    # checkpoint_file = '/mnt/data0/Garmin/pinjyun/nwd/mmdet-nwdrka/work_dirs/visdrone_RS_cascade_t2t_nms100_4x/epoch_48.pth'
    # checkpoint_file_cl = '/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/pretrain/CL_26.5.pth'

    # config_file = '/mnt/data0/Garmin/pinjyun/mmdetection/configs/aitod/deformable-detr-refine-twostage_r50_1xb8-50e_aitod.py'
    # checkpoint_file_base = "/mnt/data0/Garmin/pinjyun/mmdetection/work_dirs/deformable-detr-refine-twostage_r50_1xb8-50e_aitod/epoch_50.pth"
    # config_file = "/mnt/data0/Garmin/DNTR/mmdet-dntr/configs/aitod-dntr/aitod_DNTR_mask.py"
    # checkpoint_file_base = "/mnt/data0/Garmin/DNTR/mmdet-dntr/work_dirs/aitod_DNTR_mask/epoch_48.pth"

    # config_file = "/mnt/data0/Garmin/pinjyun/NWD/work_dirs/detectors_cascade_rcnn_r50_aitod_baseline/detectors_cascade_rcnn_r50_aitod_baseline.py"
    # checkpoint_file_base = "/mnt/data0/Garmin/pinjyun/NWD/work_dirs/detectors_cascade_rcnn_r50_aitod_baseline/epoch_12.pth"

    config_file = "/mnt/data0/Garmin/DNTR/mmdet-dntr/configs/aitod-dntr/aitod_RS_baseline.py"
    checkpoint_file_cl = "/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/pretrain/CL_26.5.pth"
    checkpoint_file_base = "/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/pretrain/base_24.pth"

    # visdrone
    # image_folder = '/mnt/data0/Garmin/datasets/visdrone/VisDrone2019-DET-val/images/'
    # gt_folder = '/mnt/data0/Garmin/datasets/visdrone/VisDrone2019-DET-val/annotations/'

    # aitod
    image_folder = '/mnt/data0/Garmin/datasets/ai-tod/test/images/'
    gt_folder = '/mnt/data0/Garmin/datasets/ai-tod/test/labels/'

    # uavdt
    # image_folder = '/mnt/data0/Garmin/datasets/uavdt/val/'
    # gt_folder = '/mnt/data0/Garmin/datasets/uavdt/val_txt/'

    imgs = os.listdir(image_folder)
    gts = os.listdir(gt_folder)

    
    imgs.sort()
    gts.sort()
    image_num = len(imgs)
    gt_num = len(gts)
    print("total img:", image_num , "total gt:", gt_num)
    total_psnr_cl = 0
    total_psnr_base_c0 = 0
    total_psnr_base_c1 = 0
    total_psnr_base_c2 = 0
    total_psnr_base_c3 = 0
    total_psnr_cl_c1 = 0

    # # define model
    device = 'cuda:0'
    model_cl = init_detector(config_file, checkpoint_file_cl, device=device)
    model_base = init_detector(config_file, checkpoint_file_base, device=device)

    for img in intro_files:
        print('img:',image_folder + img)
        print('gt:',gt_folder + img.replace("png","txt"))
        feature_gt = gau(gt_folder + img.replace("png","txt"), image_folder+img)
        cv2.imwrite('/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/gau/'+(image_folder+img).split('/')[-1],feature_gt)

    # for (img, gt) in zip(imgs, gts):
    #     print('img:',image_folder + img)
    #     print('gt:',gt_folder + img.replace("png","txt"))


    #     # demo (model_cl, (image_folder + img))
    #     # feature_model_cl = cv2.imread('/mnt/data0/Garmin/kaimmdet/nwd/mmdet-nwdrka/tools/uavdt_psnr/c1.png')
        
            
    #     demo (model_base, (image_folder + img))
    #     feature_model_base_c0 = cv2.imread("/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/c0.png")
    #     feature_model_base_c1 = cv2.imread("/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/c1.png")
    #     feature_model_base_c2 = cv2.imread("/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/c2.png")
    #     feature_model_base_c3 = cv2.imread("/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/c3.png")

    #     demo (model_cl, (image_folder + img))

    #     feature_model_cl_c0 = cv2.imread("/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/c0.png")
    #     feature_model_cl_c1 = cv2.imread("/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/c1.png")
    #     feature_model_cl_c2 = cv2.imread("/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/c2.png")
    #     feature_model_cl_c3 = cv2.imread("/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/c3.png")

    #     feature_gt = gau(gt_folder + img.replace("png","txt"), image_folder+img)
    #     # cv2.imshow("a",feature_gt)
    #     # cv2.waitKey(0)
    #     psnr_base_c0 = cal_psnr(feature_model_base_c0, feature_gt)
    #     psnr_base_c1 = cal_psnr(feature_model_base_c1, feature_gt)
    #     psnr_base_c2 = cal_psnr(feature_model_base_c2, feature_gt)
    #     psnr_base_c3 = cal_psnr(feature_model_base_c3, feature_gt)

    #     psnr_cl_c0 = cal_psnr(feature_model_cl_c0, feature_gt)
    #     psnr_cl_c1 = cal_psnr(feature_model_cl_c1, feature_gt)
    #     psnr_cl_c2 = cal_psnr(feature_model_cl_c2, feature_gt)
    #     psnr_cl_c3 = cal_psnr(feature_model_cl_c3, feature_gt)
    #     print("psnr_base:",psnr_base_c1)
    #     print("psnr_cl:",psnr_cl_c1)
    #     cv2.imwrite('/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/gau/'+(image_folder+img).split('/')[-1],feature_gt)
    #     cv2.imwrite('/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/base/'+str(int(psnr_base_c1))+'_'+(image_folder+img).split('/')[-1],feature_model_base_c1)
    #     cv2.imwrite('/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/cl/'+str(int(psnr_cl_c1))+'_'+(image_folder+img).split('/')[-1],feature_model_cl_c1)


    #     # total_psnr_cl += psnr_cl
    #     total_psnr_base_c1 += psnr_base_c1
    #     total_psnr_cl_c1 += psnr_cl_c1

    # print('psnr cal done...')
    # print('total number of images : ', image_num)
    # # print('average psnr_cl :', total_psnr_cl/image_num)
    # print('average psnr_base_c1 :', total_psnr_base_c1/image_num)
    # print('average psnr_cl_c1 :', total_psnr_cl_c1/image_num)



if __name__ == '__main__':
    main()