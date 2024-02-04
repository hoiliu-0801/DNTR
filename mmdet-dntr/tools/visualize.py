import mmcv
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, get_feature
import os
import argparse
import cv2
import sys
import torch
from mmcv import Config, DictAction
import json
import numpy as np
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from tools.feature_visualization import featuremap_2_heatmap
import ignite


def parse_args():
    parser = argparse.ArgumentParser(description='in and out imgs')
    parser.add_argument('--config', dest='config',help='config_file',default=None, type=str)
    parser.add_argument('--ckpt_ours', dest='ckpt_ours',help='checkpoint_file',default=None, type=str)
    parser.add_argument('--ckpt_base', dest='ckpt_base',help='checkpoint_file',default=None, type=str)
    parser.add_argument('--out', dest='out_path',help='output path for the visualization',default=None, type=str)

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
    ################################################################
    ############### Initialization
    ################################################################
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model_ours = init_detector(args.config, args.ckpt_ours)
    catorgy = model_ours.CLASSES
    print('Classes :' ,len(catorgy))

    annotations = json.load(open(cfg.data.test.ann_file, 'r'))
    # print(annotations.keys())
    # dict_keys(['categories', 'annotations', 'images'])
    anno_id_transfer = {}
    for i,cls in enumerate(annotations['categories']):
        anno_id_transfer.update({cls['id']:i})

    os.makedirs(args.out_path, exist_ok = True)
    merge_path = os.path.join(args.out_path, "merge")
    os.makedirs(merge_path, exist_ok = True)
    ################################################################
    ############### Start of visualization
    ################################################################
    for image_meta in annotations["images"]:
        # print(image_meta.keys())
        # dict_keys(['file_name', 'id', 'width', 'height'])
        # file_target = os.listdir("/mnt/data0/Garmin/datasets/coco/test_coco")
        # file_n= image_meta["file_name"]
        # print("file_target:",file_target)
        # print("file_n:",file_n)
        # if file_n in file_target:
        image_id = image_meta["id"]
        ################################################################
        ############### Get Predicted Boxes
        ################################################################
        score_threshold = 0.5
        result = inference_detector(model_ours, os.path.join(cfg.data.test.img_prefix, image_meta["file_name"]))
        # len(result)=8 → 8 classes
        # each class has [n, 5], n for # boxes, 5 for [points and score]
        box_list, cls_list,  score_list, cnt = [], [], [], 0
        for cls,boxes in enumerate(result):
            for box in boxes:
                if box[-1]<score_threshold: continue
                box_list.append(box[:4])
                cls_list.append(cls)
                score_list.append(box[-1])
                cnt += 1
        pred_bbox = [np.array(box_list), np.array(score_list), np.array(cls_list), np.array(cnt)]

        ################################################################
        ############### Get Ground Truth Boxes
        ################################################################
        box_list, cls_list = [], []
        for anno in annotations['annotations']:
            if anno['image_id']!=image_id: continue
            x, y, w, h = anno["bbox"]
            cls = anno["category_id"]
            box_list.append([x, y, (x+w), (y+h)])
            cls_list.append(anno_id_transfer[cls])
            cnt += 1
        ground_truth = [np.array(box_list), np.array(cls_list), np.array(cnt)]

        # print(np.unique(pred_bbox[2]), np.unique(ground_truth[1]))
        ################################################################
        ############### Draw TP_FP_FN boxes
        ################################################################
        tp,fp,fn=count_TP_FP_FN(pred_bbox[0], pred_bbox[2],ground_truth[0], ground_truth[1])
        ori_image = cv2.imread(os.path.join(cfg.data.test.img_prefix, image_meta["file_name"]))
        red, blue, green =(0,0,255), (255,255,0), (0,255,0)
        false_negative = [np.array(fn), np.array(len(fn))]
        false_postive = [np.array(fp), np.array(len(fp))]
        true_postive = [np.array(tp), np.array(len(tp))]
        image = draw_bbox(ori_image, false_negative, is_gt=True, color=blue) # FN
        image = draw_bbox(image, false_postive, is_gt=True, color=red) # FP
        image = draw_bbox(image, true_postive, is_gt=True, color=green) # TP

        # ################################################################
        # ############### Visualize the feature
        # ################################################################
        # w,h = image.shape[:2]
        # feats = get_feature(model_ours, os.path.join(cfg.data.test.img_prefix, image_meta["file_name"]))
        # feat = feats[1] # choose level
        # heatmap = featuremap_2_heatmap(feat)[0]
        # heatmap = cv2.resize(heatmap, (h, w))
        # heatmap = np.uint8(255 * heatmap)
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # 这行将热力图转换为RGB格式 ，如果注释掉就是灰度图

        # from tools.feature_visualization import draw_feature_map
        # draw_feature_map(feat)
        # ################################################################
        # ############### Calculating the psnr value
        # ################################################################
        # gaussian_image = cv2.imread(os.path.join('/mnt/data0/Garmin/DNTR/mmdet-dntr/tools/aitod_psnr/gau', image_meta['file_name']))
        # psnr = cal_psnr(heatmap, gaussian_image)
        # print(image_meta['file_name']+" PSNR :%.2f" % psnr)
        # # ################################################################
        # # ############### Save Image
        # # ################################################################
        # merged_image = np.hstack((image, heatmap))
        # cv2.imwrite(os.path.join(merge_path, image_meta["file_name"]), ori_image)
        cv2.imwrite(os.path.join(merge_path, image_meta["file_name"]), image)


if __name__ == '__main__':
    main()

# ours
# CUDA_VISIBLE_DEVICES=1 python tools/visualize.py --config configs/aitod-dntr/aitod_RS_baseline.py --ckpt_ours /mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/pretrain/CL_26.5.pth --out outputs/aitod/

# baseline
# CUDA_VISIBLE_DEVICES=1 python tools/visualize.py --config configs/aitod-dntr/aitod_RS_baseline.py --ckpt_ours /mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/pretrain/base_24.pth --out outputs/base