'''
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.279 | bridge        | 0.095 | storage-tank | 0.215 |
| ship     | 0.214 | swimming-pool | 0.130 | vehicle      | 0.167 |
| person   | 0.059 | wind-mill     | 0.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-11-02 08:06:48,919 - mmdet - INFO -
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.753 | bridge        | 0.911 | storage-tank | 0.801 |
| ship     | 0.813 | swimming-pool | 0.862 | vehicle      | 0.844 |
| person   | 0.941 | wind-mill     | 1.000 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-11-02 08:06:49,320 - mmdet - INFO - Exp name: v001.01.17_aitodv2_faster_hrnet_w32.py
2021-11-02 08:06:49,321 - mmdet - INFO - Epoch(val) [12][7004]	bbox_mAP: 0.1450, bbox_mAP_50: 0.3280, bbox_mAP_75: 0.2970, bbox_mAP_vt: 0.2580, bbox_mAP_t: 0.2130, bbox_mAP_s: 0.1640, bbox_mAP_m: 0.1060, bbox_oLRP: 0.2740, bbox_oLRP_Localisation: 0.3780, bbox_oLRP_false_positive: 0.2180, bbox_oLRP_false_negative: 0.2240, bbox_mAP_copypaste: 0.145 -1.000 0.328 0.297 0.258 0.213
Loading and preparing results...
DONE (t=2.67s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=682.71s).
Accumulating evaluation results...
DONE (t=12.13s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.145
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.328
Average Precision  (AP) @[ IoU=0.55      | area=   all | maxDets=1500 ] = 0.297
Average Precision  (AP) @[ IoU=0.60      | area=   all | maxDets=1500 ] = 0.258
Average Precision  (AP) @[ IoU=0.65      | area=   all | maxDets=1500 ] = 0.213
Average Precision  (AP) @[ IoU=0.70      | area=   all | maxDets=1500 ] = 0.164
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.106
Average Precision  (AP) @[ IoU=0.80      | area=   all | maxDets=1500 ] = 0.057
Average Precision  (AP) @[ IoU=0.85      | area=   all | maxDets=1500 ] = 0.021
Average Precision  (AP) @[ IoU=0.90      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.95      | area=   all | maxDets=1500 ] = 0.001
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.001
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.111
Average Precision  (AP) @[ IoU=0.50      | area=  tiny | maxDets=1500 ] = 0.294
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.274
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.378
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.218
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.224
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.224
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.166
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.416
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.489
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.866
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.267
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.378
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.652
# Class-specific LRP-Optimal Thresholds #
 [0.666 0.619 0.443 0.428 0.861 0.313 0.426   nan]
'''
_base_ = ['../_base_/models/faster_rcnn_r50_fpn_aitod.py',
    '../_base_/datasets/aitodv2_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py']
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        _delete_=True,
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)))),
    neck=dict(
        _delete_=True,
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256))