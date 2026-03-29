import os
import mmcv
import torch
import numpy as np
import os.path as osp
from mmdet3d.datasets.builder import PIPELINES
from numpy.linalg import inv
from mmcv.runner import get_dist_info
from mmdet3d.core.points import BasePoints
from torchvision.transforms.functional import rotate
from mmdet.datasets.pipelines import to_tensor
from mmcv.parallel import DataContainer as DC


def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def compose_ego2img(cur_ego2global, prev_sensor2global_translation, prev_sensor2global_rotation, prev_cam_intrinsic):
    prev_cam2global_r = prev_sensor2global_rotation
    prev_cam2global_t = prev_sensor2global_translation
    prev_cam2global_rt = convert_egopose_to_matrix_numpy(prev_cam2global_r, prev_cam2global_t)
    prev_global2cam_rt = invert_matrix_egopose_numpy(prev_cam2global_rt)

    viewpad = np.eye(4)
    viewpad[:prev_cam_intrinsic.shape[0], :prev_cam_intrinsic.shape[1]] = prev_cam_intrinsic
    prev_global2img_rt = (viewpad @ prev_global2cam_rt)

    ego2img = prev_global2img_rt @ cur_ego2global
    return ego2img


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweeps:
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False,
                 train_interval=[4, 8],
                 test_interval=6):
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        self.train_interval = train_interval
        self.test_interval = test_interval

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def load_offline(self, results):
        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['cam_sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            if self.test_mode: # 情况1：测试阶段(固定采样)
                interval = self.test_interval # 6
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)] # 每个interval取一帧：[5 11 17 23 29 35 41]
            elif len(results['cam_sweeps']['prev']) <= self.sweeps_num:  # 情况2：历史帧不足
                pad_len = self.sweeps_num - len(results['cam_sweeps']['prev'])
                choices = list(range(len(results['cam_sweeps']['prev']))) + \
                    [len(results['cam_sweeps']['prev']) - 1] * pad_len # 原有索引 + 重复最后一帧
            else:  # 情况3: 训练阶段(随机采样)
                max_interval = len(results['cam_sweeps']['prev']) // self.sweeps_num # 最大间隔
                max_interval = min(max_interval, self.train_interval[1]) # 限制采样间隔不要太大
                min_interval = min(max_interval, self.train_interval[0]) # 限制采样间隔不要太小
                interval = np.random.randint(min_interval, max_interval + 1) # 随机取一个interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)): 
                sweep_idx = min(idx, len(results['cam_sweeps']['prev']) - 1)
                sweep = results['cam_sweeps']['prev'][sweep_idx] # 当前帧六路相机数据

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['cam_sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_ego2img(
                        results['lidar2global'],                      # (4 4)
                        sweep[sensor]['sensor2global_translation'],   # (3,)
                        sweep[sensor]['sensor2global_rotation'],      # (3,3)
                        sweep[sensor]['cam_intrinsic'],               # (3,3)
                    ))

        # results: dict{
        #     'img': List[(H, W, 3), (H, W, 3), ...]
        #     'img_timestamp': List[float, float, ...]
        #     'filename': List[str, str, ...]
        #     'lidar2img': List[(4, 4), (4, 4), ...]
        # }

        return results

    def load_online(self, results):
        # only used when measuring FPS
        assert self.test_mode
        assert self.test_interval % 6 == 0

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['cam_sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            interval = self.test_interval
            choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['cam_sweeps']['prev']) - 1)
                sweep = results['cam_sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['cam_sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    # skip loading history frames # 在线加载：跳过加载历史帧图片
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_ego2img(
                        results['lidar2global'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results

    def __call__(self, results):
        if self.sweeps_num == 0: # self.sweeps_num:7  8-1=7
            return results

        # world_size = get_dist_info()[1]
        # if world_size == 1 and self.test_mode: # 当gpus
        #     #===================================#
        #     # online模式：使用当前帧作为过去帧
        #     return self.load_online(results)
        # else:
        #     return self.load_offline(results) # 其余状态，一律调用offline模式
        
        # 只加载单帧图像
        return self.load_offline(results)


@PIPELINES.register_module()
class LoadOccGTFromFile(object): # Occ3d
    def __init__(self, num_classes=18):
        self.num_classes = num_classes

    def __call__(self, results):
        occ_labels = np.load(results['occ_gt_path'])
        semantics = torch.from_numpy(occ_labels['semantics'])  # (200, 200, 16)
        mask_camera = torch.from_numpy(occ_labels['mask_camera']).to(torch.bool) # (200, 200, 16)

        if results.get('flip_dx', False):
            semantics = torch.flip(semantics, [0])
            mask_camera = torch.flip(mask_camera, [0])

        if results.get('flip_dy', False):
            semantics = torch.flip(semantics, [1])
            mask_camera = torch.flip(mask_camera, [1])

        results['voxel_semantics'] = semantics  # (200, 200, 16)
        results['mask_camera'] = mask_camera  # (200, 200, 16)

        return results


@PIPELINES.register_module()
class LoadOccupancySurroundOcc(object): # SurroundOcc
    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_gt_path = os.path.join(occ_gt_path, results['pts_filename'].split('/')[-1]+'.npy') # surroundocc gt

        label = np.load(occ_gt_path)
        new_label = np.ones((200, 200, 16), dtype=np.int64) * 17
        new_label[label[:, 0], label[:, 1], label[:, 2]] = label[:, 3]

        new_label = torch.from_numpy(new_label)
        mask_camera = new_label != 0

        if results.get('flip_dx', False):
            new_label = torch.flip(new_label, [0])
            mask_camera = torch.flip(mask_camera, [0])

        if results.get('flip_dy', False):
            new_label = torch.flip(new_label, [1])
            mask_camera = torch.flip(mask_camera, [1])

        results['voxel_semantics'] = new_label # voxel_semantics
        results['mask_camera'] = mask_camera

        return results