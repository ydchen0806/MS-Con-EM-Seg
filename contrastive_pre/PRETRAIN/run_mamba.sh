export NCCL_P2P_DISABLE=1 &&
pip3 install segmentation_models_pytorch -i https://pypi.tuna.tsinghua.edu.cn/simple &&
python3 -m torch.distributed.launch --nproc_per_node=8 /data/ydchen/VLP/EM_Mamba/contrastive_pre/PRETRAIN/main3d.py 

