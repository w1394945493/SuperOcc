_base_ = [
    # '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    # '../../../mmdetection3d/configs/_base_/default_runtime.py'
    "/home/lianghao/wangyushen/Projects/SuperOcc/projects/configs/_base_/datasets/nus-3d.py",
    "/home/lianghao/wangyushen/Projects/SuperOcc/projects/configs/_base_/default_runtime.py",

]
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
voxel_size = [0.5, 0.5, 0.5]
scale_range = [0.01, 3.2]
u_range = [0.1, 2]
v_range = [0.1, 2]

# arch config
embed_dims = 256
num_layers = 6
num_query = 3600
memory_len = 3000
topk_proposals = 3000
num_propagated = 3000

prop_query = True
temp_fusion = True
with_ego_pos = True
num_frames = 8
num_levels = 4
num_points = 2
num_refines = [1, 1, 2, 2, 4, 4]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

object_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_names = [
     'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
     'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
     'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
     'vegetation'
]

# num_gpus = 4
# batch_size = 2
num_gpus = 1
batch_size = 1
workers_per_gpu=0

num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 24
num_epochs_single_frame = 2
seq_mode = True

collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']

use_ego = False
ignore_label = 0
manual_class_weight = [
    1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
    1.26960524, 1.06258364, 1.189019, 1.06217292, 1.00595144, 0.85706115,
    1.03923299, 0.90867526, 0.8936431, 0.85486129, 0.8527829, 0.5]

model = dict(
    type='SuperOCC',
    seq_mode=seq_mode,
    data_aug=dict(
        img_color_aug=False,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)
    ),
    stop_prev_grad=0,
    img_backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint="ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth",
            prefix='backbone.'),
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        with_cp=True,
        style='pytorch',
        # pretrained='torchvision://resnet50'
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=embed_dims,
        num_outs=num_levels),
    pts_bbox_head=dict(
        type='StreamOccHead',
        num_classes=len(occ_names),
        in_channels=embed_dims,
        num_query=num_query,
        memory_len=memory_len,
        topk_proposals=topk_proposals,
        num_propagated=num_propagated,
        prop_query=prop_query,
        temp_fusion=temp_fusion,
        with_ego_pos=with_ego_pos,
        scale_range=scale_range,
        u_range=u_range,
        v_range=v_range,
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        manual_class_weight=manual_class_weight,
        ignore_label=ignore_label,
        transformer=dict(
            type='StreamOccTransformer',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=len(occ_names),
            num_refines=num_refines,
            pc_range=point_cloud_range,
            use_ego=use_ego
        ),
        loss_occ=dict(
            type='CELoss',
            activated=True,
            loss_weight=10.0
        ),
        loss_pts=dict(type='SmoothL1Loss', beta=0.2, loss_weight=0.5),
    )
)


dataset_type = 'NuScenesDatasetSurroundOcc'
data_root = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-trainval/'
occ_gt = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data'


file_client_args = dict(backend='disk')

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

ida_aug_conf = {
    "resize_lim": (0.38, 0.55),
    "final_dim": (256, 704),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1),
    dict(type='LoadOccupancySurroundOcc'),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='CustomFormatBundle3D', class_names=object_names, collect_keys=collect_keys),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'mask_camera'] + collect_keys,
         meta_keys=('filename', 'occ_gt_path', 'ori_shape', 'img_shape',  'scale_factor', 'flip', 'scene_token'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, test_mode=True),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='CustomFormatBundle3D',
                collect_keys=collect_keys,
                class_names=object_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'] + collect_keys,
            meta_keys=('filename', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'scene_token'))
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file=data_root + 'nuscenes_infos_train_sweep.pkl',
        ann_file = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nusc_annos/superocc/nuscenes_infos_train_sweep.pkl',
        seq_split_num=1, # streaming video training
        seq_mode=seq_mode, # streaming video training
        pipeline=train_pipeline,
        classes=object_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline,
            #  ann_file=data_root + 'nuscenes_infos_val_sweep.pkl',
             data_root=data_root,
             occ_gt=occ_gt,
             ann_file='/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nusc_annos/superocc/nuscenes_infos_val_sweep.pkl',
             classes=object_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline,
            #   ann_file=data_root + 'nuscenes_infos_val_sweep.pkl',
              data_root=data_root,
              occ_gt=occ_gt,
              ann_file='/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nusc_annos/superocc/nuscenes_infos_val_sweep.pkl',
              classes=object_names, modality=input_modality),
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
        seq_split_num=2,
        num_iters_to_seq=num_epochs_single_frame*num_iters_per_epoch,
        random_drop=0.0
    ),
    nonshuffler_sampler=dict(type='DistributedSampler')
)


optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'sampling_offset': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2)
)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

evaluation = dict(interval=num_iters_per_epoch*num_epochs, pipeline=test_pipeline)
find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from=None
resume_from=None


# ===> per class IoU of 6019 samples:
# ===> noise - IoU = nan
# ===> barrier - IoU = 23.92
# ===> bicycle - IoU = 12.43
# ===> bus - IoU = 29.77
# ===> car - IoU = 33.62
# ===> construction_vehicle - IoU = 17.36
# ===> motorcycle - IoU = 17.74
# ===> pedestrian - IoU = 14.95
# ===> traffic_cone - IoU = 14.79
# ===> trailer - IoU = 13.42
# ===> truck - IoU = 26.25
# ===> driveable_surface - IoU = 48.22
# ===> other_flat - IoU = 29.22
# ===> sidewalk - IoU = 32.16
# ===> terrain - IoU = 30.2
# ===> manmade - IoU = 18.67
# ===> vegetation - IoU = 30.04
# ===> mIoU of 6019 samples: 24.55
#
# Starting Evaluation...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6019/6019 [01:29<00:00, 67.23it/s]
# ===> per class IoU of 6019 samples:
# ===> non-free - IoU = 38.13
# ===> mIoU of 6019 samples: 38.13
# {'mIoU': 24.55, 'binary_mIoU': 38.13}
