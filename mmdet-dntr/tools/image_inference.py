from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import argparse
import sys

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

def parse_args():
    parser = argparse.ArgumentParser(description='in and out imgs')
    parser.add_argument('--img_path', type=str, help = "image name", default="/mnt/data0/Garmin/datasets/coco/test2017/000000485208.jpg")
    parser.add_argument('--config', type=str,default='/mnt/data0/Garmin/DNTR/mmdet-dntr/configs/aitod-dntr/coco_RS_CL_base.py')
    parser.add_argument('--ckpt', type=str, default="/mnt/data0/Garmin/DNTR/mmdet-dntr/work_dirs/coco_RS_CL_base/epoch_24.pth")
    parser.add_argument('--out', type=str, default="outputs/base")

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_iou(detection_1, detection_2):
    # dec1 (gt[x1_tl,y1_tl,x1_br,y1_br]) dec2 (result[x2_tl,y2_tl,x2_br,y2_br])
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[2]
    x2_br = detection_2[2]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[3]
    y2_br = detection_2[3]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = (detection_1[2]-detection_1[0])*(detection_1[3]-detection_1[1])
    area_2 = (detection_2[2]-detection_2[0])*(detection_2[3]-detection_2[1])
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)

def count_TP_FP_FN(result, result_classes, gt, gt_classes):

    ######### TP
    tp = []
    tp_idx=[]
    gt_idx=[]
    for i_idx,i in enumerate(result):
        bbox = None
        Iou=0.5
        for s,j in enumerate(gt):
            if s in gt_idx:
                continue
            elif get_iou(i,j)>Iou and result_classes[i_idx] == gt_classes[s]:
                Iou=get_iou(i,j)
                bbox=i
                tp.append(bbox)
                gt_idx.append(s)
                tp_idx.append(i_idx)

    ######### FN
    fn=[]
    for q1,q in enumerate(gt):
        if q1 not in gt_idx:
            fn.append(q)

    ######### FP
    fp=[]
    for l1,l in enumerate(result):
        if l1 not in tp_idx:
            fp.append(l)

    return tp,fp,fn

def draw_bbox(image, bboxes, is_gt=True, color=None):
    image_h, image_w, _ = image.shape
    if is_gt: out_boxes,  num_boxes = bboxes
    else: out_boxes, out_scores, num_boxes = bboxes
    for i in range(num_boxes):
        coor = out_boxes[i]
        if is_gt == False: score = out_scores[i]
        bbox_color = color
        bbox_thick = int(0.8 * (image_h + image_w) / 600)
        c1, c2 = (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3]))
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
    return image

def eval_step(engine, batch):
    return batch

def cal_psnr(target_img, gt_img):

    default_evaluator = ignite.engine.Engine(eval_step)

    psnr = ignite.metrics.PSNR(data_range=255)
    psnr.attach(default_evaluator, 'psnr')

    try:
        target_img = torch.from_numpy(target_img)
        gt_img = torch.from_numpy(gt_img)
        state = default_evaluator.run([[target_img, gt_img]])
        p = state.metrics['psnr']
        # print(type(p))
        return p
    except:
        return 0


def main():
    args = parse_args()
    # img = args.img

    # config_file = '/mnt/data0/Garmin/DNTR/mmdet-dntr/configs/aitod-dntr/coco_RS_CL.py'
    # checkpoint_file = '/mnt/data0/Garmin/DNTR/mmdet-dntr/work_dirs/coco_RS_CL/epoch_12.pth'
    # img_path = "/mnt/data0/Garmin/datasets/coco/test2017/000000485208.jpg"
    # config_file = "/mnt/data0/Garmin/DNTR/mmdet-dntr/configs/aitod-dntr/coco_RS_CL_base.py"
    # checkpoint_file = "/mnt/data0/Garmin/DNTR/mmdet-dntr/work_dirs/coco_RS_CL_base/epoch_24.pth"
    device = 'cuda:0'

    model_cl = init_detector(args.config, args.ckpt, device=device)
    print(inference_detector(model_cl, args.img_path))

if __name__ == '__main__':
    main()