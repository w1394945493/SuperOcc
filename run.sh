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