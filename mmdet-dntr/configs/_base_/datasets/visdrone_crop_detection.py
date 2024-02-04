dataset_type = 'VisdroneDataset'
data_root = '/mnt/data0/Garmin/datasets/visdrone/'

classes = ('pedestrian', 'people', 'bicycle', 'car', 'van',
             'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PhotoMetricDistortion',
#         brightness_delta=32,
#         contrast_range=(0.5, 1.5),
#         saturation_range=(0.5, 1.5),
#         hue_delta=18),
#     dict(
#         type='RandomCenterCropPad',
#         crop_size=(512, 512),
#         ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
#         mean=[0, 0, 0],
#         std=[1, 1, 1],
#         to_rgb=True,
#         test_pad_mode=None),
#     dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile', to_float32=True),
#     dict(
#         type='MultiScaleFlipAug',
#         scale_factor=1.0,
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(
#                 type='RandomCenterCropPad',
#                 ratios=None,
#                 border=None,
#                 mean=[0, 0, 0],
#                 std=[1, 1, 1],
#                 to_rgb=True,
#                 test_mode=True,
#                 test_pad_mode=['logical_or', 31],
#                 test_pad_add_pix=1),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='DefaultFormatBundle'),
#             dict(
#                 type='Collect',
#                 meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
#                            'scale_factor', 'flip', 'flip_direction',
#                            'img_norm_cfg', 'border'),
#                 keys=['img'])
#         ])
# ]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'anno_crop_10cls/crop_train.json',
        img_prefix=data_root + 'VisDrone2019-DET-train/crop_images/',
        pipeline=train_pipeline,
        ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'anno_crop_10cls/crop_test.json',
        img_prefix=data_root + 'VisDrone2019-DET-test-dev/crop_images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'anno_crop_10cls/crop_test.json',
        img_prefix=data_root + 'VisDrone2019-DET-test-dev/crop_images/',
        pipeline=test_pipeline))
    # train=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/small_trainval_v1_1.0.json',
    #     img_prefix=data_root + 'trainval/',
    #     pipeline=train_pipeline),
    # val=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/small_test_v1_1.0.json',
    #     img_prefix=data_root + 'test/',
    #     pipeline=test_pipeline),
    # test=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/small_test_v1_1.0.json',
    #     img_prefix=data_root + 'test/',
    #     pipeline=test_pipeline))
# gmap_image='/mnt/data0/Garmin/datasets/ai-tod/train/image_blur/train_blur/'

