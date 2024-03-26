export NCCL_SOCKET_IFNAME=eth0 &&
export NCCL_SOCKET_NTHREADS=8 &&
export NCCL_P2P_DISABLE=1 &&
pip3 install segmentation_models_pytorch -i https://pypi.tuna.tsinghua.edu.cn/simple &&
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 \
--node_rank=${PAI_TASK_INDEX} --master_addr=${PAI_CONTAINER_LIST_0_0} --master_port=12365 \
--use_env /data/ydchen/VLP/EM_Mamba/contrastive_pre/PRETRAIN/main3d.py 

