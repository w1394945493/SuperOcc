# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
import math
import itertools
import copy
import torch.distributed as dist
import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from .sampler import SAMPLER
import random
from IPython import embed


@SAMPLER.register_module()
class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                indice = indice[list(
                    torch.randperm(int(size), generator=g).numpy())].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def sync_random_seed(seed=None, device='cuda'):
    """Make sure different ranks share the same seed.
    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.
    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, num_replicas = get_dist_info()

    if num_replicas == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


@SAMPLER.register_module()
class InfiniteGroupEachSampleInBatchSampler(Sampler):
    """
    Pardon this horrendous name. Basically, we want every sample to be from its own group.
    If batch size is 4 and # of GPUs is 8, each sample of these 32 should be operating on
    its own group.
    Shuffling is only done for group order, not done within groups.
    Arguments:
        dataset: Dataset used for sampling.
        min_len: Minimum sequence sampling length
        max_len: Maximum sequence sampling length
        num_iters_to_seq: After `num_iters_to_seq` iterations,
            start sequential sampling. Default: 0
        samples_per_gpu (optional): Per gpu batchsize. Default: 1
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 seq_split_num=-1,
                 num_iters_to_seq=0,
                 random_drop=0,
                 samples_per_gpu=1, # batch大小
                 num_replicas=None,
                 rank=None,
                 seed=0):
      
        _rank, _num_replicas = get_dist_info() # 来自from mmcv.runner import get_dist_info _rank: 当前进程编号；_num_replicas = 总进程数(world_size)
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.batch_size = samples_per_gpu # batch_size 全局batch=samples_per_gpu x GPU数量
        self.num_replicas = num_replicas  # num_replicas: 总进程数(world_size)
        self.rank = rank
        self.seq_split_num = seq_split_num
        self.sub_seq_generator = torch.Generator() # torch.Generator(): 创建一个“独立的随机数生成器”：sampler随机性独立；可控(可手动设置seed)；多卡一致 
        self.sub_seq_generator.manual_seed(self.rank + seed) # 每个gpu使用不同的随机种子：
        self.seed = sync_random_seed(seed)
        self.random_drop = random_drop

        self.size = len(self.dataset)
        self._iters = 0
        self.num_iters_to_seq = num_iters_to_seq

        assert hasattr(self.dataset, 'flag') # dataset必须有flag这个属性！
        #==============================================#
        self.flag = self.dataset.flag  # self.flag = self.dataset.flag 是一个数组：
        self.group_sizes = np.bincount(self.flag) # 输出数组：第i个位置 = flag中值为i的元素个数
        self.groups_num = len(self.group_sizes)   # group数量
        self.global_batch_size = samples_per_gpu * num_replicas # 全局batch_size: samples_per_gpu * GPUs
        assert self.groups_num >= self.global_batch_size

        #===============================================#
        # group划分：每个group里有多少样本
        # Now, for efficiency, make a dict {group_idx: List[dataset sample_idxs]}
        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist() # np.where: 找出满足条件为元素位置(索引)
            for group_idx in range(self.groups_num)}  # 把整个batch_size按group分桶：每个group对应一堆样本index

        #==============================================#
        #：这是每个group分配一个数据生成器
        self.group_idx_to_sample_idxs_generator = {         
            group_idx: self._sample_sub_sequence(group_idx) # 给每个group分配一个专属的数据生成器(不断输出该group的样本) 注：生成器，有yield的函数，只有next()时才会调用
            for group_idx in range(self.groups_num)  # {0: generator_0,1: generator_1,...} 每个generator负责不断生成该group的样本序列
        }

        #===============================================#
        # self.group_indices_per_global_sample_idx: 关键！
        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator
        self.group_indices_per_global_sample_idx = [ # 给当前gpu里的每一个样本位置，分配一个专属的group流
            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx) # local_sample_idx=当前gpu里的第结果sample位置：范围：0~batch_size-1 self.rank:当前是第几个gpu 计算得到：该sample在全局batch的位置
            for local_sample_idx in range(self.batch_size)] # 把group流按global_batch_size划分：确保每个GPU用的是不同位置的group √

        # Keep track of a buffer of dataset sample idxs for each local sample idx
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]

    def _infinite_group_indices(self):
        g = torch.Generator()    # 创建随机数生成器：不用全局随机，每个sampler自己控制随机性
        g.manual_seed(self.seed) # 固定随机数种子
        while True:              # while True：无限循环
            yield from torch.randperm(self.groups_num, generator=g).tolist() # 生成随机排列：生成[0,1,....,group_num-1]的随机排列

    # 把“全局group随机序列”拆成“global_batch_size条互不重叠的数据流”：每个sample取其中一条流
    def _group_indices_per_global_sample_idx(self, global_sample_idx): # itertools.islice: “按规则从一个序列里抽子序列” 基本形式：itertools.islice(iterable,start,stop,step)
        # yield from: 把整个iterable的内容展开吐出来
        yield from itertools.islice(self._infinite_group_indices(),  # iterable:无限group序列
                                    global_sample_idx,               # start: 从第global_sample_idx开始
                                    None,                            # stop:  None(无限)
                                    self.global_batch_size)          # step:  每隔global_batch_size取一个

    
    def _sample_sub_sequence(self, group_idx): # 不是函数返回值，是一个生成器：函数里出现yield：不再是普通函数，变成生成器函数；或者说调用它不会执行函数，而是返回一个generator对象。只有调用next()时才会调用
        '''randomly split sub-sequences in a whole sequence'''

        sample_ids = self.group_idx_to_sample_idxs[group_idx] # group_idx_to_sample_idxs：每个group对应一堆样本索引(index) sample_ids: 样本索引
        while True:
            if self._iters < self.num_iters_to_seq or self.seq_split_num == -1:
                shuffled = torch.randperm(len(sample_ids), generator=self.sub_seq_generator).tolist() # 打乱group内的样本顺序
                yield from [[sample_ids[i]] for i in shuffled] 

            else: # 分段模式：将序列分段
                # split the sequence into parts
                # 随机决定把一个group的样本序列切成几段：
                idx = torch.randperm(len(sample_ids), generator=self.sub_seq_generator).tolist() # 随机打乱所有位置：idx：随机顺序：例如：sample_ids = [10, 11, 12, 13, 14] -> idx: [3, 0, 4, 1, 2]（随机顺序）  
                idx.remove(0) # 去掉0：避免产生空序列
                idx = sorted(idx[:self.seq_split_num - 1])  # choose n-1 split position 选前k个+排序，得到切分点：seq_split_num: 想切成3段：则需要seq_split_num-1个切点 排序目的：保证切点从小到大
                split_idx = [0] + idx + [len(sample_ids)]   # 构造完整切分边界：将切点补全为：起点+中间切点+终点
                sub_seq_idx = [sample_ids[split_idx[i]: split_idx[i + 1]]
                               for i in range(len(split_idx) - 1)] # 按区间切片，得到子序列：本质：按相邻边界，两两切一段  # [[1,2,3], [4,5], ...]
                # 对子序列继续做：打乱顺序+随机丢弃一部分元素
                shuffled = torch.randperm(len(sub_seq_idx), generator=self.sub_seq_generator).tolist() # 打乱子序列的顺序
                for i in shuffled: # 逐个处理每个子序列
                    sub_seq = sub_seq_idx[i]
                    length = len(sub_seq)
                    drop_num = math.floor(length * self.random_drop) # 计算要丢弃的元素数量
                    drop_idxs = torch.randperm(length, generator=self.sub_seq_generator).tolist()[:drop_num] # 随机选取要丢弃的index
                    new_sub_seq = [sub_seq[j] for j in range(length) if j not in drop_idxs]
                    yield new_sub_seq 
                # yield from [sub_seq_idx[i] for i in shuffled]
    
    #======================================#
    # 取出一段index(长度=batch_size)
    def __iter__(self): 
        last_group_idx_batch = [-1 for i in range(self.batch_size)]
        while True: # 一直产生batch # 动态生成，无限流
            curr_batch = []
            for local_sample_idx in range(self.batch_size): # batch_size 
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    new_group_idx = next(self.group_indices_per_global_sample_idx[local_sample_idx])

                    # 保证不会连续两段相同的序列
                    # 如果不加的话，在epoch轮换时会有概率连续两段相同序列
                    if new_group_idx == last_group_idx_batch[local_sample_idx]:
                        new_group_idx = next(self.group_indices_per_global_sample_idx[local_sample_idx])
                    last_group_idx_batch[local_sample_idx] = new_group_idx

                    self.buffer_per_local_sample[local_sample_idx] = \
                        copy.deepcopy(next(self.group_idx_to_sample_idxs_generator[new_group_idx]))

                curr_batch.append(self.buffer_per_local_sample[local_sample_idx].pop(0))

            self._iters += 1
            #===================================#
            # 返回一个索引列表
            yield curr_batch # 输出已经是一个batch

    def __len__(self):
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch):
        self.epoch = epoch