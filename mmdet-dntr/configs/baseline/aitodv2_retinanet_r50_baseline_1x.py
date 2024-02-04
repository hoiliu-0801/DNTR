'''
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.013 | bridge        | 0.118 | storage-tank | 0.143 |
| ship     | 0.236 | swimming-pool | 0.058 | vehicle      | 0.114 |
| person   | 0.023 | wind-mill     | 0.005 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-10-20 20:43:50,539 - mmdet - INFO -
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.971 | bridge        | 0.894 | storage-tank | 0.872 |
| ship     | 0.782 | swimming-pool | 0.937 | vehicle      | 0.885 |
| person   | 0.968 | wind-mill     | 0.987 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-10-20 20:43:53,765 - mmdet - INFO - Exp name: v001.01.05_aitodv2_retinanet_r50_baseline_1x.py
2021-10-20 20:43:53,766 - mmdet - INFO - Epoch(val) [12][7004]	bbox_mAP: 0.0890, bbox_mAP_50: 0.2420, bbox_mAP_75: 0.2040, bbox_mAP_vt: 0.1610, bbox_mAP_t: 0.1180, bbox_mAP_s: 0.0790, bbox_mAP_m: 0.0460, bbox_oLRP: 0.1310, bbox_oLRP_Localisation: 0.2040, bbox_oLRP_false_positive: 0.2210, bbox_oLRP_false_negative: 0.2440, bbox_mAP_copypaste: 0.089 -1.000 0.242 0.204 0.161 0.118
Loading and preparing results...
DONE (t=31.87s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=4899.80s).
Accumulating evaluation results...
DONE (t=110.65s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.089
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.242
Average Precision  (AP) @[ IoU=0.55      | area=   all | maxDets=1500 ] = 0.204
Average Precision  (AP) @[ IoU=0.60      | area=   all | maxDets=1500 ] = 0.161
Average Precision  (AP) @[ IoU=0.65      | area=   all | maxDets=1500 ] = 0.118
Average Precision  (AP) @[ IoU=0.70      | area=   all | maxDets=1500 ] = 0.079
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.046
Average Precision  (AP) @[ IoU=0.80      | area=   all | maxDets=1500 ] = 0.022
Average Precision  (AP) @[ IoU=0.85      | area=   all | maxDets=1500 ] = 0.010
Average Precision  (AP) @[ IoU=0.90      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.95      | area=   all | maxDets=1500 ] = 0.001
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.027
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.084
Average Precision  (AP) @[ IoU=0.50      | area=  tiny | maxDets=1500 ] = 0.248
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.131
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.204
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.221
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.244
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.258
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.089
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.262
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.276
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.379
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.912
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.309
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.609
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.685
# Class-specific LRP-Optimal Thresholds #
 [0.175 0.322 0.348 0.388 0.295 0.404 0.326 0.153]
'''
_base_ = [
    '../_base_/models/retinanet_r50_fpn_aitod.py',
    '../_base_/datasets/aitodv2_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]


# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001) 
# Note: We find that the gradient of one-stage detector is quite unstable and very easy to explode on Tiny Object Detection tasks.
# we recommend the following solutions:
## lower the learning rate and increase the warm up iterations. 
## change the regression loss to DIoU loss, which is much more stable and easier to converge.
## use P2-P6 of FPN instead of P3-P7, the improvement is significant but the problem is that it will bring great computation burden. 

# learning policy
checkpoint_config = dict(interval=4)