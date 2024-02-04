'''
------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.262 | bridge        | 0.096 | storage-tank | 0.240 |
| ship     | 0.243 | swimming-pool | 0.132 | vehicle      | 0.175 |
| person   | 0.058 | wind-mill     | 0.001 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-10-20 01:43:55,490 - mmdet - INFO -
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.766 | bridge        | 0.913 | storage-tank | 0.783 |
| ship     | 0.791 | swimming-pool | 0.866 | vehicle      | 0.839 |
| person   | 0.941 | wind-mill     | 0.999 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-10-20 01:43:55,943 - mmdet - INFO - Exp name: v001.01.02_aitodv2_cascade_rcnn_r50_baseline_1x.py
2021-10-20 01:43:55,944 - mmdet - INFO - Epoch(val) [12][7004]	bbox_mAP: 0.1510, bbox_mAP_50: 0.3420, bbox_mAP_75: 0.3050, bbox_mAP_vt: 0.2630, bbox_mAP_t: 0.2190, bbox_mAP_s: 0.1680, bbox_mAP_m: 0.1130, bbox_oLRP: 0.2670, bbox_oLRP_Localisation: 0.3850, bbox_oLRP_false_positive: 0.2180, bbox_oLRP_false_negative: 0.2240, bbox_mAP_copypaste: 0.151 -1.000 0.342 0.305 0.263 0.219
Loading and preparing results...
DONE (t=2.93s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=738.38s).
Accumulating evaluation results...
DONE (t=14.77s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.151
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.342
Average Precision  (AP) @[ IoU=0.55      | area=   all | maxDets=1500 ] = 0.305
Average Precision  (AP) @[ IoU=0.60      | area=   all | maxDets=1500 ] = 0.263
Average Precision  (AP) @[ IoU=0.65      | area=   all | maxDets=1500 ] = 0.219
Average Precision  (AP) @[ IoU=0.70      | area=   all | maxDets=1500 ] = 0.168
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.113
Average Precision  (AP) @[ IoU=0.80      | area=   all | maxDets=1500 ] = 0.064
Average Precision  (AP) @[ IoU=0.85      | area=   all | maxDets=1500 ] = 0.027
Average Precision  (AP) @[ IoU=0.90      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.95      | area=   all | maxDets=1500 ] = 0.002
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.001
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.115
Average Precision  (AP) @[ IoU=0.50      | area=  tiny | maxDets=1500 ] = 0.297
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.267
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.385
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.218
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.224
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.224
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.175
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.394
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.482
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.862
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.291
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.388
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.658
# Class-specific LRP-Optimal Thresholds #
 [0.529 0.308 0.419 0.31  0.833 0.351 0.373 0.05 ]
'''
_base_ = [
     '../_base_/models/cascade_rcnn_r50_fpn_aitod.py',
    '../_base_/datasets/aitodv2_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# model settings


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)