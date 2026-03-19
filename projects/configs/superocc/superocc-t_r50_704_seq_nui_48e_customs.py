_base_ = [
    # '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    # '../../../mmdetection3d/configs/_base_/default_runtime.py'
    "/vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/_base_/datasets/nus-3d.py",
    "/vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/_base_/default_runtime.py",
]
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
voxel_size = [0.4, 0.4, 0.4]
scale_range = [0.01, 3.2]
u_range = [0.1, 2]
v_range = [0.1, 2]

# arch config
embed_dims = 256
num_layers = 6
num_query = 600
memory_len = 500
topk_proposals = 500
num_propagated = 500

prop_query = True
temp_fusion = True
with_ego_pos = True
num_frames = 8
num_levels = 4
num_points = 4
num_refines = [2, 2, 4, 4, 8, 8]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

object_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]

num_gpus = 4
batch_size = 2
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 48
num_epochs_single_frame = 2
seq_mode = True

collect_keys = ['ego2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']

nusc_class_frequencies = [
    2082349,
    3012970,
    234046,
    5385402,
    34146494,
    2044124,
    325765,
    3330253,
    543815,
    5785079,
    13521112,
    198278651,
    4895895,
    56540471,
    66504617,
    227803562,
    252374615,
    17126390780
]

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
        nusc_class_frequencies=nusc_class_frequencies,
        score_thres=0.25,
        transformer=dict(
            type='StreamOccTransformer',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=len(occ_names),
            num_refines=num_refines,
            pc_range=point_cloud_range
        ),
        loss_occ=dict(
            type='CELoss',
            activated=True,
            loss_weight=10.0
        ),
        loss_pts=dict(type='SmoothL1Loss', beta=0.2, loss_weight=0.5),
    )
)


dataset_type = 'NuScenesDatasetOcc3D' #! Occ3D
# data_root = './data/nuscenes/'
data_root = "/c20250502/wangyushen/Datasets/NuScenes/v1.0-trainval/v1.0-trainval/"
ann_root = "/c20250502/wangyushen/Datasets/NuScenes/method/superocc/"
occ_gt = "/c20250502/wangyushen/Datasets/occ3d_nuscenes/"

train_ann = ann_root + "nuscenes_infos_train_sweep.pkl"
val_ann = ann_root + "nuscenes_infos_val_sweep.pkl"

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
    dict(type='LoadOccGTFromFile'),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='CustomFormatBundle3D', class_names=object_names, collect_keys=collect_keys),
    dict(type='Collect3D', 
         keys=['img', 'voxel_semantics', 'mask_camera'] + collect_keys,
         meta_keys=('filename', 'occ_gt_path', 'ori_shape', 'img_shape',  'scale_factor', 'flip', 'scene_token'))
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=False, color_type="color"),
    dict(
        type="LoadMultiViewImageFromMultiSweeps",
        sweeps_num=num_frames - 1,
        test_mode=True,
    ),
    dict(type="LoadOccGTFromFile"),  # test: 没有加载occ标注
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
                # keys=["img"] + collect_keys,
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
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file=data_root + 'nuscenes_infos_train_sweep.pkl',
        ann_file=train_ann,
        occ_gt=occ_gt,
        seq_split_num=1,  # streaming video training
        seq_mode=seq_mode,  # streaming video training
        pipeline=train_pipeline,
        classes=object_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d="LiDAR",
    ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        # ann_file=data_root + "nuscenes_infos_val_sweep.pkl",
        ann_file=val_ann,
        occ_gt=occ_gt,
        classes=object_names,
        modality=input_modality,
    ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        # ann_file=data_root + "nuscenes_infos_val_sweep.pkl",
        ann_file=val_ann,
        occ_gt=occ_gt,
        classes=object_names,
        modality=input_modality,
    ),
    shuffler_sampler=dict(
        type="InfiniteGroupEachSampleInBatchSampler",
        seq_split_num=2,
        num_iters_to_seq=num_epochs_single_frame * num_iters_per_epoch,
        random_drop=0.0,
    ),
    nonshuffler_sampler=dict(type="DistributedSampler"),
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
# ===> others - IoU = 12.23
# ===> barrier - IoU = 47.25
# ===> bicycle - IoU = 25.71
# ===> bus - IoU = 41.25
# ===> car - IoU = 49.67
# ===> construction_vehicle - IoU = 24.33
# ===> motorcycle - IoU = 28.46
# ===> pedestrian - IoU = 25.65
# ===> traffic_cone - IoU = 32.64
# ===> trailer - IoU = 25.41
# ===> truck - IoU = 36.41
# ===> driveable_surface - IoU = 68.96
# ===> other_flat - IoU = 42.45
# ===> sidewalk - IoU = 45.63
# ===> terrain - IoU = 42.88
# ===> manmade - IoU = 33.1
# ===> vegetation - IoU = 31.52
# ===> mIoU of 6019 samples: 36.09
#
# Starting Evaluation...
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6019/6019 [00:46<00:00, 130.49it/s]
# ===> per class IoU of 6019 samples:
# ===> non-free - IoU = 52.73
# ===> mIoU of 6019 samples: 52.73
#
# Starting Evaluation...
# Using /home/zichen/.cache/torch_extensions/py39_cu117 as PyTorch extensions root...
# Detected CUDA files, patching ldflags
# Emitting ninja build file /home/zichen/.cache/torch_extensions/py39_cu117/dvr/build.ninja...
# Building extension module dvr...
# Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
# ninja: no work to do.
# Loading extension module dvr...
# 6019it [05:50, 17.19it/s]
# +----------------------+----------+----------+----------+
# |     Class Names      | RayIoU@1 | RayIoU@2 | RayIoU@4 |
# +----------------------+----------+----------+----------+
# |        others        |  0.107   |  0.116   |  0.118   |
# |       barrier        |  0.454   |  0.499   |  0.515   |
# |       bicycle        |  0.268   |  0.318   |  0.327   |
# |         bus          |  0.542   |  0.647   |  0.708   |
# |         car          |  0.550   |  0.628   |  0.657   |
# | construction_vehicle |  0.214   |  0.310   |  0.352   |
# |      motorcycle      |  0.297   |  0.339   |  0.355   |
# |      pedestrian      |  0.324   |  0.385   |  0.408   |
# |     traffic_cone     |  0.361   |  0.382   |  0.389   |
# |       trailer        |  0.247   |  0.317   |  0.403   |
# |        truck         |  0.461   |  0.567   |  0.614   |
# |  driveable_surface   |  0.645   |  0.710   |  0.774   |
# |      other_flat      |  0.379   |  0.421   |  0.455   |
# |       sidewalk       |  0.327   |  0.383   |  0.428   |
# |       terrain        |  0.324   |  0.398   |  0.457   |
# |       manmade        |  0.414   |  0.516   |  0.582   |
# |      vegetation      |  0.311   |  0.438   |  0.542   |
# +----------------------+----------+----------+----------+
# |         MEAN         |  0.366   |  0.434   |  0.476   |
# +----------------------+----------+----------+----------+
# {'mIoU': 36.09, 'binary_mIoU': 52.73, 'RayIoU': 0.4251917232434056, 'RayIoU@1': 0.3661318238264223, 'RayIoU@2': 0.4337494921860217, 'RayIoU@4': 0.4756938537177729}
