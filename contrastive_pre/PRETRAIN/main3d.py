import random
import tempfile
import os
import pandas as pd
import numpy as np
import yaml
import sys
sys.path.append('/data/ydchen/VLP/EM_Mamba/contrastive_pre/PRETRAIN/utils')
sys.path.append('/data/ydchen/VLP/EM_Mamba/contrastive_pre/PRETRAIN/models')
from utils.trainer import Trainer
from utils.dataset import VLP_dataset
from tensorboardX import SummaryWriter
import torch
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from omegaconf import OmegaConf
from models.mamba3d import MambaAE
import argparse
import time
import datetime
# def parse_args():
#     parser = argparse.ArgumentParser(description="Mamba 3D")
#     parser.add_argument(
#         "--save_dir", type=str, default="/h3cstore_ns/EM_pretrain/mamba_pretrain")
#     return parser.parse_args()


def ddp_main():

    # dist.init_process_group("nccl")
    dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400)) 
    torch.cuda.empty_cache()
    rank = dist.get_rank()

    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()
    current_rank = rank

    # set up
    # config = yaml.load(open("config3d.yaml", "r"), Loader=yaml.FullLoader)
    config = OmegaConf.load("/data/ydchen/VLP/EM_Mamba/contrastive_pre/PRETRAIN/config3d.yaml")
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)

    # loading data path
    meta_data = config['csv_path']
    save_dir = os.path.join(config['csv_path']['save_path'], config['save_name'])
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)
    # if rank == 0:
    #     current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # args = parse_args()

    # define image-text dataset
    train_dataset = VLP_dataset(
        csv_path = meta_data['meta_path'], ct_dir = meta_data['ct_dir'])
    train_dataset = train_dataset.get_dataset(train_test='train')

    # building model part
    # --------------------
    model = MambaAE(config['network'], device_id=device_id)

    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    # --------------------

    # choose optimizer (no LARS, AdamW with small batch)
    # --------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **config['optimizer']['params'],
        betas=(0.9, 0.999)
    )

    # ---------xw-----------
    trainer = Trainer(save_dir, model=model,
                            optimizer=optimizer,
                            device=device_id,
                            model_name=config['wandb_name'],
                            **config['trainer'])
    # --------------------
    
    # --------------------
    # I_T_P_trainer
    trainer.train_process(train_dataset,writer)


ddp_main()
