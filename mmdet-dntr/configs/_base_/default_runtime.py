checkpoint_config = dict(interval=6)
# yapf:disable
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "/home/hoiliu/Desktop/DNTR/mmdet-dntr/work_dirs/aitod_CL_mask/epoch_36.pth"
# load_from = None
resume_from = None
# resume_from ='/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/aitodv2_detectors_r50_nwdrka_1x_copy/epoch_6.pth'
# resume_from = '/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/pretrain/base_24.pth'
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=2)

### Publish model
# python tools/model_converters/publish_model.py work_dirs/RS_heatmap_test1/24.2.pth  work_dirs/pretrain/SOTA_24.2.pth
# Multi GPU
# CUDA_VISIBLE_DEVICES=0,1,2,5 ./tools/dist_train.sh configs/aitod/RS_crcnn_wasserstein_assigner.py 4
