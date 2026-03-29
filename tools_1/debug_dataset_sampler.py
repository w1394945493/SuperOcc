import os
os.environ["TMPDIR"] = "/vepfs-mlp2/c20250502/haoce/wangyushen/tmp"
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from mmcv import Config
from mmdet3d.datasets import build_dataset

from projects.mmdet3d_plugin.datasets.builder import build_dataloader

#===============================================#
# debug dataset & InfiniteDistGroupBatchSampler
config = '/vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/customs/superocc-t_r50_704_seq_nui_24e_test.py'
cfg = Config.fromfile(config)

dataset = build_dataset(cfg.data.train)

data_loader = build_dataloader(
    dataset,
    cfg.data.samples_per_gpu,
    cfg.data.workers_per_gpu,    
    num_gpus=1,
    dist=False,
    seed=0,
    shuffler_sampler=cfg.data.shuffler_sampler,
    nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    runner_type=cfg.runner,
)
for data in tqdm(data_loader):
    a=1

