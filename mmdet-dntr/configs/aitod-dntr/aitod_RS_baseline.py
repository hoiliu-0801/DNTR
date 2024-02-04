'''
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.320 | bridge        | 0.202 | storage-tank | 0.378 |
| ship     | 0.434 | swimming-pool | 0.166 | vehicle      | 0.273 |
| person   | 0.116 | wind-mill     | 0.091 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-11-03 21:30:33,140 - mmdet - INFO -
+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.721 | bridge        | 0.822 | storage-tank | 0.672 |
| ship     | 0.614 | swimming-pool | 0.861 | vehicle      | 0.766 |
| person   | 0.904 | wind-mill     | 0.917 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2021-11-03 21:30:33,767 - mmdet - INFO - Exp name: v001.06.07_aitodv2_detectors_r50_rpn_NWD_rk.py
2021-11-03 21:30:33,768 - mmdet - INFO - Epoch(val) [12][7004]	bbox_mAP: 0.2470, bbox_mAP_50: 0.5740, bbox_mAP_75: 0.5130, bbox_mAP_vt: 0.4420, bbox_mAP_t: 0.3590, bbox_mAP_s: 0.2650, bbox_mAP_m: 0.1710, bbox_oLRP: 0.2980, bbox_oLRP_Localisation: 0.3930, bbox_oLRP_false_positive: 0.3450, bbox_oLRP_false_negative: 0.3670, bbox_mAP_copypaste: 0.247 -1.000 0.574 0.513 0.442 0.359
Loading and preparing results...
DONE (t=6.38s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1963.37s).
Accumulating evaluation results...
DONE (t=24.17s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.247
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.574
Average Precision  (AP) @[ IoU=0.55      | area=   all | maxDets=1500 ] = 0.513
Average Precision  (AP) @[ IoU=0.60      | area=   all | maxDets=1500 ] = 0.442
Average Precision  (AP) @[ IoU=0.65      | area=   all | maxDets=1500 ] = 0.359
Average Precision  (AP) @[ IoU=0.70      | area=   all | maxDets=1500 ] = 0.265
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.171
Average Precision  (AP) @[ IoU=0.80      | area=   all | maxDets=1500 ] = 0.095
Average Precision  (AP) @[ IoU=0.85      | area=   all | maxDets=1500 ] = 0.041
Average Precision  (AP) @[ IoU=0.90      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.95      | area=   all | maxDets=1500 ] = 0.002
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.097
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.242
Average Precision  (AP) @[ IoU=0.50      | area=  tiny | maxDets=1500 ] = 0.582
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.298
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.393
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.367
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.370
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.158
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.378
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.413
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.484
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.785
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.278
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.319
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.424
# Class-specific LRP-Optimal Thresholds #
 [0.436 0.651 0.699 0.682 0.721 0.504 0.54  0.32 ]
'''
_base_ = [
    # '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/aitod_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
# '../_base_/datasets/aitodv2_detection.py',

# model settings
model = dict(
    type='CascadeRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='DetectoRS_ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        conv_cfg=dict(type='ConvAWS'),
        norm_eval=True,
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True,
        style='pytorch'),
    neck=dict(
        type='RFP',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='torchvision://resnet50',
            style='pytorch')),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='RankingAssigner',
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='nwd',
                topk=3),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    gpu_assign_thr=256),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    gpu_assign_thr=256),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    gpu_assign_thr=256),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=3000)))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[8, 11])
checkpoint_config = dict(interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=12)

evaluation = dict(interval=12, metric='bbox')

# load_from = None
# load_from = "/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/pretrain/base_24.pth"
load_from = "/mnt/data0/Garmin/pinjyun/nwd/mmdet-nwdrka/work_dirs/pretrain/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth"
# load_from = "/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/RS_cl_two_stage/e12_mAP251.pth"

# resume_from = '/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/aitodv2_detectors_r50_nwdrka_1x/latest.pth'
resume_from = None
# resume_from = "/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/RS_cl_two_stage/e12_mAP251.pth"
