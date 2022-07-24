# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardImageLoggerHook', by_epoch=True),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/work_dirs/depthformer_swinl_22k_w7_kitti_baseline/best_abs_rel_iter_25600.pth"
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
