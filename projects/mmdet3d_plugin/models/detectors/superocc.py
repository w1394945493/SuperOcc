import torch
import queue
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner import get_dist_info
from mmcv.runner.fp16_utils import cast_tensor_type
from mmdet.models import DETECTORS, builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.aug import GpuPhotoMetricDistortion, pad_multiple


@DETECTORS.register_module()
class SuperOCC(MVXTwoStageDetector):
    def __init__(self,
                 data_aug=None,
                 stop_prev_grad=0,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 seq_mode=False,
                 pretrained=None,
                 **kwargs):
        super(SuperOCC, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.color_aug = GpuPhotoMetricDistortion()

        self.prev_scene_token = None
        self.seq_mode = seq_mode
        self.memory = {}
        self.queue = queue.Queue()

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        """
        Args:
            img: (B*N, 3, img_H, img_W)
        Returns:
            img_feats: Tuple((B*N, C=256, H2, W2), (B*N, C=256, H3, W3), (B*N, C=256, H4, W4), (B*N, C=256, H5, W5))
        """
        # Tuple((B*N, C2, H2, W2), (B*N, C3, H3, W3), (B*N, C4, H4, W4), (B*N, C5, H5, W5))
        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            # Tuple((B*N, C=256, H2, W2), (B*N, C=256, H3, W3), (B*N, C=256, H4, W4), (B*N, C=256, H5, W5))
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_feat(self, img, img_metas):
        """
        Args:
            img: (B, N=N_sweep*N_views, 3, H, W)
        Returns:
            img_feats_reshaped: List[(B, N, C=256, H2, W2), ..., (B, N, C=256, H5, W5)]
        """
        if isinstance(img, list):
            img = torch.stack(img, dim=0)   # (B, N, 3, H, W)

        assert img.dim() == 5

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)      # (B*N, C=3, H, W)
        img = img.float()

        # move some augmentations to GPU
        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]['img_shape'] = [img_shape for _ in range(N)]
                img_metas[b]['ori_shape'] = [img_shape for _ in range(N)]

            # if 'img_pad_cfg' in self.data_aug:
            #     img_pad_cfg = self.data_aug['img_pad_cfg']
            #     img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'])

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if self.training and self.stop_prev_grad > 0:
            H, W = input_shape
            img = img.reshape(B, -1, 6, C, H, W)

            img_grad = img[:, :self.stop_prev_grad]
            img_nograd = img[:, self.stop_prev_grad:]

            all_img_feats = [self.extract_img_feat(img_grad.reshape(-1, C, H, W))]

            with torch.no_grad():
                self.eval()
                for k in range(img_nograd.shape[1]):
                    all_img_feats.append(self.extract_img_feat(img_nograd[:, k].reshape(-1, C, H, W)))
                self.train()

            img_feats = []
            for lvl in range(len(all_img_feats[0])):
                C, H, W = all_img_feats[0][lvl].shape[1:]
                img_feat = torch.cat([feat[lvl].reshape(B, -1, 6, C, H, W) for feat in all_img_feats], dim=1)
                img_feat = img_feat.reshape(-1, C, H, W)
                img_feats.append(img_feat)
        else:
            # img_feats: Tuple((B*N, C=256, H2, W2), (B*N, C=256, H3, W3), (B*N, C=256, H4, W4), (B*N, C=256, H5, W5))
            img_feats = self.extract_img_feat(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped # todo (1 6 256 67 174) (1 6 256 32 88) (1 6 256 16 44) (1 6 256 8 22)

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:  # data['img'][0].shape: (1 6 3 256 704)
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      **data):
        img = data['img']
        B, N = data['img'].shape[:2]
        # img_feats: List[(B, N, C=256, H2, W2), ..., (B, N, C=256, H5, W5)]
        img_feats = self.extract_feat(img, img_metas)
        data['img_feats'] = img_feats

        if self.seq_mode:
            if self.prev_scene_token is None:
                prev_exists = [0] * B
            else:
                prev_exists = [
                    int(img_metas[i]['scene_token'] == self.prev_scene_token[i])
                    for i in range(B)
                ]
            # 更新为当前帧的 token 列表
            self.prev_scene_token = [meta['scene_token'] for meta in img_metas]
            data['prev_exists'] = data['img'].new_tensor(prev_exists)

        outs = self.pts_bbox_head(img_metas, **data)
        loss_inputs = [voxel_semantics, mask_camera, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses

    def forward_test(self, img_metas, rescale, **data):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            data[key] = data[key][0] # 把batch维度去掉
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        outs = self.pts_bbox_head(img_metas, **data)
        occ_list = self.pts_bbox_head.get_occ(
            outs, img_metas)
        return occ_list

    def simple_test(self, img_metas, **data):
        """Test function of point cloud branch."""
        if img_metas[0]['scene_token'] != self.prev_scene_token: # 判断当前帧是不是同一个scene的连续帧，决定是否使用上一帧的memory，是否重置模型的时序状态
            self.prev_scene_token = img_metas[0]['scene_token'] # scene_token:判断是否进入新场景
            data['prev_exists'] = data['img'].new_zeros(1) # 更新一下记录
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = data['img'].new_ones(1)

        # todo ----------------------------------------- #
        # todo offline 和 online
        world_size = get_dist_info()[1]
        if world_size == 1:  # 
            return self.simple_test_online(img_metas, **data) # online：顺序处理(有时间关系)
        else:  # offline
            return self.simple_test_offline(img_metas, **data) # offine：并行处理(无时序) 帧之间没有时序

    def simple_test_offline(self, img_metas, **data):
        img = data['img']
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        data['img_feats'] = img_feats
        return self.simple_test_pts(img_metas, **data)

    def simple_test_online(self, img_metas, **data):
        self.fp16_enabled = False
        assert len(img_metas) == 1  # batch_size = 1
        img = data['img'] # todo (1 6 3 256 704)

        B, N, C, H, W = img.shape
        img = img.reshape(B, N // 6, 6, C, H, W) # todo (1 1 6 3 256 704) (bs 时间帧 相机 图像3HW)

        img_filenames = img_metas[0]['filename']
        num_frames = len(img_filenames) // 6

        img_shape = (H, W, C)
        img_metas[0]['img_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['ori_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['pad_shape'] = [img_shape for _ in range(len(img_filenames))]

        img_feats_list, img_metas_list = [], []
        # todo 将多帧图像按时间顺序拆解，逐帧提取特征，使用队列和字典来管理特征：逐帧提取特征；若已经提取过，直接复用，同时清楚太早的缓存视频帧特征
        # extract feature frame by frame
        for i in range(num_frames): # todo 逐帧处理元数据 
            img_indices = list(np.arange(i * 6, (i + 1) * 6)) # todo 当前帧对应的索引如第0帧：0-5 第1帧6-11 ...

            img_metas_curr = [{}]
            for k in img_metas[0].keys(): # todo len(img_metas[0]['filename]):48
                item = img_metas[0][k]
                if isinstance(item, list) and (len(item) == 6 * num_frames):
                    img_metas_curr[0][k] = [item[j] for j in img_indices]
                else:
                    img_metas_curr[0][k] = item
            # todo 特征缓存
            if img_filenames[img_indices[0]] in self.memory: # len(img_filenames): 48 6x8 8帧图像，每帧6路环视图像的路径(按顺序命名)
                # found in memory
                img_feats_curr = self.memory[img_filenames[img_indices[0]]] # 使用当前帧的第一路图像路径作为标记，所以这里取第一张图像对应的索引编号img_indices[0]
            else:
                # extract feature and put into memory
                img_feats_curr = self.extract_feat(img[:, i], img_metas_curr) # todo 重新计算 i:视频帧号(0-7) 输出：4: (1 6 256 64 176) (1 6 256 32 88) (1 6 256 16 44) (1 6 256 8 22)
                self.memory[img_filenames[img_indices[0]]] = img_feats_curr # key(图片路径)：value(多尺度图像特征): 使用该帧第一张图像的路径作为索引
                self.queue.put(img_filenames[img_indices[0]]) 
                while self.queue.qsize() > 16:  # avoid OOM # todo 缓存上限：16帧
                    pop_key = self.queue.get()
                    self.memory.pop(pop_key)

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)
        # 把逐帧提取出来的特征，重新拼回多帧输入的格式
        # reorganize
        feat_levels = len(img_feats_list[0]) # todo 4
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0) # (8 6 256 64 176)
            feat_l = feat_l.flatten(0, 1)[None, ...] # 将T和V两个维度展平，并增加bs维度
            img_feats_reorganized.append(feat_l) # 4:(1 48 256 64 176) (1 48 256 32 88) (1 48 256 16 44) (1 48 256 8 22)

        img_metas_reorganized = img_metas_list[0] # img_metas重组：取第一帧的meta作为基础
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v) # len(img_metas_reorganized[0]['filename']): 48

        img_feats = img_feats_reorganized
        img_metas = img_metas_reorganized
        img_feats = cast_tensor_type(img_feats, torch.half, torch.float32) # todo 从float16转回float32
        data['img_feats'] = img_feats # todo img_feats: list:4 (1 48 256 64 176) (1 48 256 32 88) (1 48 256 16 44) (1 48 256 8 22) 48=6x

        # run occupancy predictor
        return self.simple_test_pts(img_metas, **data)
