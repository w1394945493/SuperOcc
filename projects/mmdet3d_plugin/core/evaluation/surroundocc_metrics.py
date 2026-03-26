import numpy as np
import torch.distributed as dist
import torch


# class MeanIoU:
#
#     def __init__(self,
#                  class_indices,
#                  empty_label,
#                  use_mask=False):
#         self.class_indices = class_indices
#         self.num_classes = len(class_indices)
#         self.empty_label = empty_label
#         self.class_names = [
#             'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
#             'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
#             'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
#             'vegetation']
#         self.use_mask = use_mask
#         self.reset()
#
#     def reset(self) -> None:
#         self.total_seen = torch.zeros(self.num_classes+1).cuda()
#         self.total_correct = torch.zeros(self.num_classes+1).cuda()
#         self.total_positive = torch.zeros(self.num_classes+1).cuda()
#         self.cnt = 0
#
#     def add_batch(self, outputs, targets, mask=None):
#         self.cnt += 1
#
#         if mask is not None:
#             outputs = outputs[mask]
#             targets = targets[mask]
#
#         for i, c in enumerate(self.class_indices):
#             self.total_seen[i] += torch.sum(targets == c).item()
#             self.total_correct[i] += torch.sum((targets == c)
#                                                & (outputs == c)).item()
#             self.total_positive[i] += torch.sum(outputs == c).item()
#
#         self.total_seen[-1] += torch.sum(targets != self.empty_label).item()
#         self.total_correct[-1] += torch.sum((targets != self.empty_label)
#                                             & (outputs != self.empty_label)).item()
#         self.total_positive[-1] += torch.sum(outputs != self.empty_label).item()
#
#     def count_miou(self):
#         ious = []
#         precs = []
#         recas = []
#
#         for i in range(self.num_classes):
#             if self.total_positive[i] == 0:
#                 precs.append(0.)
#             else:
#                 cur_prec = self.total_correct[i] / self.total_positive[i]
#                 precs.append(cur_prec.item())
#
#             if self.total_seen[i] == 0:
#                 ious.append(1)
#                 recas.append(1)
#             else:
#                 cur_iou = self.total_correct[i] / (self.total_seen[i]
#                                                    + self.total_positive[i]
#                                                    - self.total_correct[i])
#                 cur_reca = self.total_correct[i] / self.total_seen[i]
#                 ious.append(cur_iou.item())
#                 recas.append(cur_reca)
#
#         miou = np.mean(ious)
#         # logger = get_root_logger()
#         print(f'===> per class IoU of {self.cnt} samples:')
#         for iou, prec, reca, cls_name in zip(ious, precs, recas, self.class_names):
#             print('%s : %.2f%%, %.2f, %.2f' % (cls_name, iou * 100, prec, reca))
#
#
#         occ_iou = self.total_correct[-1] / (self.total_seen[-1]
#                                             + self.total_positive[-1]
#                                             - self.total_correct[-1])
#         return miou * 100, occ_iou * 100


class Metric_mIoU():
    def __init__(self,
                 num_classes=18,
                 empty_label=17,
                 use_image_mask=False
                 ):
        if num_classes == 18:
            self.class_names = [
                'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                'vegetation', 'free']
        elif num_classes == 2:
            self.class_names = ['non-free', 'free']

        self.use_image_mask = use_image_mask
        self.num_classes = num_classes
        self.empty_label = empty_label
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape  # (num,)
        k = (gt >= 0) & (gt < n_cl)    # n_cl:18
        labeled = np.sum(k)            # 统计有效点数
        correct = np.sum((pred[k] == gt[k]))  # 统计预测正确数
        #==================================================#
        # 构建混淆矩阵：
        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl), # (n_cl, n_cl)
            correct,
            labeled,
        )
    # 从混淆矩阵里计算每个类别的IoU
    def per_class_iu(self, hist):
        result = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  # TP TP+FN TP+FP TP/(TP+FN+FP)
        result[hist.sum(1) == 0] = float('nan') # 某一类在真实里完全没有出现，该类IoU无意义，设为nan
        return result

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten()) # new_hist:混淆矩阵 (18 ,18)
        hist += new_hist
        mIoUs = self.per_class_iu(hist) # 各类别mIoU
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def add_batch(self, semantics_pred, semantics_gt, mask_camera):
        self.cnt += 1
        if self.use_image_mask: # True
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]

        if self.num_classes == 2: # self.num_classes:18
            masked_semantics_pred = np.copy(masked_semantics_pred)
            masked_semantics_gt = np.copy(masked_semantics_gt)
            masked_semantics_pred[masked_semantics_pred != self.empty_label] = 0
            masked_semantics_pred[masked_semantics_pred == self.empty_label] = 1
            masked_semantics_gt[masked_semantics_gt != self.empty_label] = 0
            masked_semantics_gt[masked_semantics_gt == self.empty_label] = 1

        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {self.cnt} samples:') # 给类别IoU
        for ind_class in range(self.num_classes - 1): # 17
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))

        print(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(mIoU[:self.num_classes - 1]) * 100, 2))) # mIoU
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))
        IoU=compute_occ_iou(self.hist, self.num_classes - 1)
        return round(np.nanmean(mIoU[:self.num_classes - 1]) * 100, 2)


def compute_occ_iou(hist, free_index):
    tp = (
        hist[:free_index, :free_index].sum() +
        hist[free_index + 1:, free_index + 1:].sum())
    return tp / (hist.sum() - hist[free_index, free_index])
