export PYTHONPATH=$PYTHONPATH:/home/lianghao/anaconda3/envs/pgocc_wys/lib/python3.8/site-packages/mmdet3d/.mim
# todo 生成nuscenes_infos_train.pkl和nuscenes_info_val.pkl
mim run mmdet3d create_data nuscenes \
    --root-path /home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-trainval \
    --out-dir /home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-trainval --extra-tag nuscenes


# todo 根据nuscenes_infos_train.pkl和nuscenes_info_val.pkl，再生成superocc使用的pkl文件
python /home/lianghao/wangyushen/Projects/SuperOcc/tools_1/create_data_nusc.py \
    --data-root /home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-trainval \
    --out-dir /home/lianghao/wangyushen/data/wangyushen/Datasets/data/nusc_annos/superocc \
    --version v1.0-trainval

python /home/lianghao/wangyushen/Projects/SuperOcc/tools_1/test.py \
    /home/lianghao/wangyushen/Projects/SuperOcc/projects/configs/superocc_surroundocc/superocc-l_r50_704_seq_nui_24e_custom.py \
    /home/lianghao/wangyushen/data/wangyushen/Weights/superocc/superocc-l_r50_704_seq_nui_24e.pth \
    --eval=bbox 

# ========================================================================== #
# todo 在A100上
conda config --append envs_dirs /vepfs-mlp2/c20250502/haoce/conda_env/wys_temp_2
conda activate /vepfs-mlp2/c20250502/haoce/conda_env/wys_temp_2

# 使用sparseocc提供的标注，应该也可
python /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/tools_1/create_data_nusc.py \
    --data-root /c20250502/wangyushen/Datasets/NuScenes/v1.0-trainval \
    --out-dir /c20250502/wangyushen/Datasets/NuScenes/method/superocc \
    --version v1.0-trainval

cd /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/mmdet3d_plugin/ops/localagg_prob_sq
pip install .  --no-build-isolation
cd /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/mmdet3d_plugin/ops/msmv_sampling
pip install .  --no-build-isolation
cd /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/mmdet3d_plugin/ops/tile_localagg_prob_sq
pip install .  --no-build-isolation 
python setup.py build_ext --inplace # 需要就地编译

# ========================================================================== #
# 在surroundocc数据集上 推理/可视化
# 推理
python /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/tools_1/test.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/superocc_surroundocc/superocc-t_r50_704_seq_nui_24e_experiment.py \
    /c20250502/wangyushen/Weights/superocc/surroundocc/superocc-t_r50_704_seq_nui_24e.pth \
    --eval=bbox

# 仅单帧预测：不做时序融合
python /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/tools_1/test.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/customs/superocc-t_r50_704_seq_nui_24e_test.py \
    /c20250502/wangyushen/Weights/superocc/surroundocc/superocc-t_r50_704_seq_nui_24e.pth \
    --eval=bbox

# 训练
bash tools/dist_train.sh projects/configs/superocc/superocc-t_r50_704_seq_nui_48e.py 4 --work-dir work_dirs/superocc-t/default

# 单卡
python /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/tools_1/train.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/customs/superocc-t_r50_704_seq_nui_24e_test.py \
    --work-dir /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/superocc/train

# 多卡
PYTHONPATH=. torchrun --nproc_per_node=2 \
    /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/tools_1/train.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/customs/superocc-t_r50_704_seq_nui_24e_test.py \
    --seed 0 \
    --launcher pytorch \
    --work-dir /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/superocc/train

# 可视化
python /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/tools_1/viz_prediction.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/superocc_surroundocc/superocc-t_r50_704_seq_nui_24e_experiment.py \
    /c20250502/wangyushen/Weights/superocc/surroundocc/superocc-t_r50_704_seq_nui_24e.pth \
    --eval=bbox

# ========================================================================== #
# 在occ3d数据集上 推理/可视化
python /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/tools_1/test.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/superocc/superocc-t_r50_704_seq_nui_48e_customs.py \
    /c20250502/wangyushen/Weights/superocc/occ3d/superocc-t_r50_704_seq_nui_48e.pth \
    --eval=bbox

python /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/tools_1/viz_prediction.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/superocc/superocc-t_r50_704_seq_nui_48e_customs.py \
    /c20250502/wangyushen/Weights/superocc/occ3d/superocc-t_r50_704_seq_nui_48e.pth \
    --eval=bbox

#===========================================#
# 环境配置
# 命令整理
conda clean -a -y
# conda下载缓存路径
conda config --show pkgs_dirs

pip cache purge # 清理缓存
pip cache dir # 查看pip缓存路径

# 指定pip缓存路径
export PIP_CACHE_DIR=/vepfs-mlp2/c20250502/haoce/pip_cache
mkdir -p /vepfs-mlp2/c20250502/haoce/pip_cache

echo 'export PIP_CACHE_DIR=/vepfs-mlp2/c20250502/haoce/pip_cache' >> ~/.bashrc
source ~/.bashrc

# 临时编译目录
echo $TMPDIR

export TMPDIR=/vepfs-mlp2/c20250502/haoce/tmp
mkdir -p /vepfs-mlp2/c20250502/haoce/tmp

# 指定临时编译路径
echo 'export TMPDIR=/vepfs-mlp2/c20250502/haoce/tmp' >> ~/.bashrc
source ~/.bashrc

conda create -n wys_superocc python=3.10 -y
conda create --prefix /vepfs-mlp2/c20250502/haoce/wangyushen/conda_env/wys_superocc python=3.10 pip openssl -y
conda env remove -n wys_superocc

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=12.1 -c pytorch -c nvidia