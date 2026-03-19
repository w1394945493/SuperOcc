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

# todo -------------------------------------------------- #
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
python setup.py build_ext --inplace # 就地编译

# todo -------------------------------- #
# surroundocc
python /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/tools_1/test.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/superocc_surroundocc/superocc-t_r50_704_seq_nui_24e_experiment.py \
    /c20250502/wangyushen/Weights/superocc/surroundocc/superocc-t_r50_704_seq_nui_24e.pth \
    --eval=bbox

python /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/tools_1/viz_prediction.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/superocc_surroundocc/superocc-t_r50_704_seq_nui_24e_experiment.py \
    /c20250502/wangyushen/Weights/superocc/surroundocc/superocc-t_r50_704_seq_nui_24e.pth \
    --eval=bbox

# todo -------------------------------- #
# occ3d
python /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/tools_1/test.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/superocc/superocc-t_r50_704_seq_nui_48e_customs.py \
    /c20250502/wangyushen/Weights/superocc/occ3d/superocc-t_r50_704_seq_nui_48e.pth \
    --eval=bbox

python /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/tools_1/viz_prediction.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/SuperOcc/projects/configs/superocc/superocc-t_r50_704_seq_nui_48e_customs.py \
    /c20250502/wangyushen/Weights/superocc/occ3d/superocc-t_r50_704_seq_nui_48e.pth \
    --eval=bbox