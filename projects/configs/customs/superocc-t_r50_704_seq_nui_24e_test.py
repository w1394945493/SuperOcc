_base_ = [
    # '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    # '../../../mmdetection3d/configs/_base_/default_runtime.py'
    "/vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/_base_/datasets/nus-3d.py",
    "/vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/_base_/default_runtime.py",

]
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
voxel_size = [0.5, 0.5, 0.5]
scale_range = [0.01, 3.2]
u_range = [0.1, 2]
v_range = [0.1, 2]

# arch config
embed_dims = 256   # 特征编码维度
num_layers = 6     # 特征层
num_query = 600
memory_len = 500   # 内存队列长度 
topk_proposals = 500
num_propagated = 500

#===============================#
# prop_query = True    # temp_fusion、prop_query、with_ego_pos和num_frames 应该是一起的
# temp_fusion = True   # 用于控制是否做时序融合
# with_ego_pos = True
# num_frames = 8       # 控制视频帧
prop_query = False    # temp_fusion、prop_query、with_ego_pos和num_frames 应该是一起的
temp_fusion = False   # 用于控制是否做时序融合
with_ego_pos = False
num_frames = 1       # 控制视频帧
#===============================#

num_levels = 4
num_points = 4
num_refines = [2, 2, 4, 4, 8, 8]  # 各层预测超二次曲面数量

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

#======================= train params =============================#
# num_gpus = 4
# batch_size = 2
num_gpus = 2
batch_size = 1
workers_per_gpu=4

log_interval=5 # 日志打印间隔
# num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_iters_per_epoch = 10 // (num_gpus * batch_size)   # 每个epoch迭代步数：num_train_dataset // (num_gpus * batch_size)
num_epochs = 24                                       # 最大训练epoch数
max_iters = num_epochs * num_iters_per_epoch          # 最大迭代次数
max_keep_ckpts=2                                      # 最多保留model数量
# val_interval=num_iters_per_epoch*num_epochs           # 评估间隔
val_interval=num_iters_per_epoch           # 评估间隔

# num_epochs_single_frame = 2
# seq_mode = True # 视频流训练
seq_mode = False

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
            checkpoint="/c20250502/wangyushen/Weights/pretrained/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth",
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
    #=========================================#
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


dataset_type = 'NuScenesDatasetSurroundOcc' #! SurroundOcc
data_root = '/c20250502/wangyushen/Datasets/NuScenes/v1.0-trainval/v1.0-trainval/'
ann_root = '/c20250502/wangyushen/Datasets/NuScenes/method/superocc/'
occ_gt = '/c20250502/wangyushen/Datasets/'

train_ann = ann_root + 'nuscenes_infos_train_sweep.pkl'
val_ann = ann_root + 'nuscenes_infos_val_sweep.pkl'


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
    # dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1),
    dict(type='LoadOccupancySurroundOcc'),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='CustomFormatBundle3D', class_names=object_names, collect_keys=collect_keys),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'mask_camera'] + collect_keys,
         meta_keys=('filename', 'occ_gt_path', 'ori_shape', 'img_shape',  'scale_factor', 'flip', 'scene_token'))
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=False, color_type="color"),
    # dict(
    #     type="LoadMultiViewImageFromMultiSweeps",
    #     sweeps_num=num_frames - 1,
    #     test_mode=True,
    # ),
    dict(type='LoadOccupancySurroundOcc'), # test 推理没有加这一步
    dict(type="RandomTransformImage", ida_aug_conf=ida_aug_conf, training=False),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="CustomFormatBundle3D",
                collect_keys=collect_keys,
                class_names=object_names,
                with_label=False,
            ),
            dict(
                type="Collect3D",
                # keys=["img"] + collect_keys, # test:未保存occ标注
                keys=["img", "voxel_semantics", "mask_camera"] + collect_keys,
                meta_keys=(
                    "filename",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                    "flip",
                    "scene_token",
                ),
            ),
        ],
    ),
]

data = dict(
    samples_per_gpu=batch_size,
    # workers_per_gpu=4,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file=data_root + 'nuscenes_infos_train_sweep.pkl',
        ann_file = train_ann,
        occ_gt=occ_gt,
        #=================================#
        seq_split_num=1, # streaming video training
        seq_mode=seq_mode, # streaming video training
        
        
        pipeline=train_pipeline,
        classes=object_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type, 
        pipeline=test_pipeline, 
        # ann_file=data_root + 'nuscenes_infos_val_sweep.pkl', 
        ann_file=val_ann,
        occ_gt=occ_gt,
        classes=object_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, 
        # ann_file=data_root + 'nuscenes_infos_val_sweep.pkl', 
        ann_file=val_ann,
        occ_gt=occ_gt,
        classes=object_names, modality=input_modality),
    
    # shuffler_sampler=dict(
    #     type='InfiniteGroupEachSampleInBatchSampler',
    #     seq_split_num=2,
    #     num_iters_to_seq=num_epochs_single_frame*num_iters_per_epoch,
    #     random_drop=0.0
    # ),
    # shuffler_sampler = dict(type='DistributedSampler'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
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

evaluation = dict(interval=val_interval, pipeline=test_pipeline)
find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=max_keep_ckpts)

log_config = dict(
    interval=log_interval,   # 每10个iter打印一次
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')  # 如果你用tensorboard
    ])

runner = dict(
    type='IterBasedRunner', max_iters=max_iters)
load_from=None
resume_from=None
