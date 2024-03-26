# package import
import os
from typing import Type
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torchvision as tv
import pandas as pd
from torch.utils.data import DataLoader as TorchDataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from pprint import pprint
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import distributed as torch_dist
from torch.utils.data.distributed import DistributedSampler
try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('/data/ydchen/VLP/EM_Mamba/contrastive_pre/PRETRAIN/utils')
from loss import (clip_loss,
                   reconst_loss,
                     local_loss,
                       bceDiceLoss,
                         focal_loss
)


class Trainer:
    def __init__(self, save_dir, model,
                 optimizer, device, model_name, **args):
        self.model = model
        self.save_dir = save_dir
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.train_batch_size = args['batch_size']
        self.max_epochs = args['max_epochs']
        self.lr_max = args['lr']
        self.num_workers = args['num_workers']
        self.step_limit = int(args['step_limit'])
        self.log_interval = args['log_interval']
        self.epoch_num = 0
        self.global_step = 0
        self.decoder = 'no'
        # self.crop_img_size = args['crop_img_size']
        # self.merge_threshold = args['merge_threshold']
        # self.num_pseudo_map = args['num_pseudo_map']
        # self.quantil = args['quantil']
        # self.resize = tv.transforms.Resize((self.crop_img_size, self.crop_img_size), interpolation=Image.NEAREST)
        print(f'trainer init successful!')
        print('---------------------------')
        
    def create_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=200, eta_min=1e-8)

# modfiy here for wo encoder and w encoder training
    def train_epoch(self, train_loader, scheduler, scaler):
        epoch_loss = 0
        epoch_acc1 = []
        epoch_acc5 = []
        epoch_global, epoch_patch, epoch_word, epoch_seg, epoch_reconst = [], [], [], [], []

        for data in tqdm(train_loader):
            if self.global_step > self.step_limit:
                break

            img, raw_img = self.prepare_data(data)
            
            loss, acc1, acc5, metric = self.train_batch(img, raw_img, scaler, scheduler)
            current_rank = torch_dist.get_rank()
        
            self.log_and_save_model(train_loader, 
                                    epoch_loss, 
                                    epoch_acc1, 
                                    epoch_acc5, 
                                    metric, 
                                    self.model_checkpoints_folder)

            self.global_step += 1

            epoch_loss += loss.item()
            epoch_acc1.append(acc1.item())
            epoch_acc5.append(acc5.item())
            epoch_global.append(metric['clip_loss'])
            epoch_patch.append(metric['patch_loss'])
            epoch_word.append(metric['word_loss'])
            epoch_seg.append(metric['seg_loss'])
            epoch_reconst.append(metric['reconst_loss'])

            
        epoch_acc1 = np.array(epoch_acc1).mean()
        epoch_acc5 = np.array(epoch_acc5).mean()
        epoch_global = np.array(epoch_global).mean()
        epoch_patch = np.array(epoch_patch).mean()
        epoch_word = np.array(epoch_word).mean()
        epoch_seg = np.array(epoch_seg).mean()
        epoch_reconst = np.array(epoch_reconst).mean()

        metric = {'global_loss': epoch_global,
                    'patch_loss': epoch_patch,
                    'word_loss': epoch_word,
                    'seg_loss': epoch_seg,
                    'reconst_loss': epoch_reconst
                    }
        
            
        return epoch_loss, epoch_acc1, epoch_acc5, metric


    def prepare_data(self, data):
        # text = data['raw_text']
        img = data['ct_patch'].to(torch.float32).to(self.device).contiguous()
        raw_img = data['ori_ct_patch'].to(torch.float32).to(self.device).contiguous()
        return img, raw_img


# modfiy here for wo encoder and w encoder training
    def train_batch(self, img, raw_img, scaler, scheduler):
        loss_patch = None
        loss_word = None
        loss_seg = None
        loss_reconst = None
        self.optimizer.zero_grad()
        with autocast():
            output_dict = self.model(img)
            if self.decoder == 'no':
                view1_proj_img_emb = output_dict['view1_proj_img_emb']
                view2_proj_img_emb = output_dict['view2_proj_img_emb']

                world_size = torch_dist.get_world_size()
                with torch.no_grad():
                    agg_v1_proj_img_emb = [torch.zeros_like(view1_proj_img_emb) for _ in range(world_size)]
                    agg_v2_proj_img_emb = [torch.zeros_like(view2_proj_img_emb) for _ in range(world_size)]
                    # print('start gather')
                    dist.all_gather(agg_v1_proj_img_emb, view1_proj_img_emb)
                    dist.all_gather(agg_v2_proj_img_emb, view2_proj_img_emb)
                    # get current rank
                    rank = torch_dist.get_rank()
                agg_v1_proj_img_emb[rank] = view1_proj_img_emb
                agg_v2_proj_img_emb[rank] = view2_proj_img_emb

                agg_v1_proj_img_emb = torch.cat(agg_v1_proj_img_emb, dim=0)
                agg_v2_proj_img_emb = torch.cat(agg_v2_proj_img_emb, dim=0)

                global_loss, acc1, acc5 = clip_loss(agg_v1_proj_img_emb, agg_v2_proj_img_emb, device=self.device)

                loss = global_loss
                if self.device == 0:
                    print(f'global step {self.global_step}, global loss {global_loss.item()} acc1 {acc1.item()}, acc5 {acc5.item()}\n')
                    print('current gpu memory:', torch.cuda.memory_allocated(self.device)/1024/1024/1024)

            scaler.scale(loss).backward()
            if self.device == 0:
                print('back current gpu memory:', torch.cuda.memory_allocated(self.device)/1024/1024/1024)

            scaler.step(self.optimizer)
            scaler.update()
            scheduler.step()
            
            if loss_patch is None:
                loss_patch = torch.tensor(0)
            if loss_patch is None:
                loss_patch = torch.tensor(0)
            if loss_word is None:
                loss_word = torch.tensor(0)
            if loss_seg is None:
                loss_seg = torch.tensor(0)
            if loss_reconst is None:
                loss_reconst = torch.tensor(0)

            metric = {'clip_loss': global_loss.item(),
                      'patch_loss': loss_patch,
                      'word_loss': loss_word,
                      'seg_loss': loss_seg.item(),
                      'reconst_loss': loss_reconst.item()}

        return loss, acc1, acc5, metric
    
    # traing process
    def train_process(self, train_dataset,writer):

        train_loader = self.create_data_loader(train_dataset)
        model_checkpoints_folder = self.prepare_checkpoint_directory()
        self.model_checkpoints_folder = model_checkpoints_folder
        start_epoch, is_continued = self.load_checkpoint_if_exists(model_checkpoints_folder)

        self.epoch_num = start_epoch
        self.global_step = start_epoch * (len(train_dataset)//self.train_batch_size//8)

        total_metric = {'global_loss': [],
                        'patch_loss': [],
                        'word_loss': [],
                        'seg_loss': [],
                        'reconst_loss': [],
                        'acc1': [],
                        'acc5': []
                        }
        
        print('training start!')
        scheduler = self.create_scheduler()
        scaler = GradScaler()

        for epoch_counter in tqdm(range(start_epoch, self.max_epochs+1)):
            epoch_loss, epoch_acc1, epoch_acc5, metric = self.train_epoch(train_loader, scheduler, scaler)
            # self.log_and_save_model(train_dataset, epoch_counter, epoch_loss, epoch_acc1, epoch_acc5, metric, model_checkpoints_folder)
            
            metric.update({'acc1': epoch_acc1, 'acc5': epoch_acc5})
            
            total_metric['global_loss'].append(metric['global_loss'])
            total_metric['patch_loss'].append(metric['patch_loss'])
            total_metric['word_loss'].append(metric['word_loss'])
            total_metric['seg_loss'].append(metric['seg_loss'])
            total_metric['reconst_loss'].append(metric['reconst_loss'])
            total_metric['acc1'].append(epoch_acc1)
            total_metric['acc5'].append(epoch_acc5)
            writer.add_scalar('global_loss', metric['global_loss'], epoch_counter)
            writer.add_scalar('patch_loss', metric['patch_loss'], epoch_counter)
            writer.add_scalar('word_loss', metric['word_loss'], epoch_counter)
            writer.add_scalar('seg_loss', metric['seg_loss'], epoch_counter)
            writer.add_scalar('reconst_loss', metric['reconst_loss'], epoch_counter)
            writer.add_scalar('acc1', epoch_acc1, epoch_counter)
            writer.add_scalar('acc5', epoch_acc5, epoch_counter)


            if self.device == 0:
                if self.epoch_num == start_epoch:
                    csv = pd.DataFrame(metric, index=[epoch_counter])
                    csv.to_csv(model_checkpoints_folder + self.model_name+'_epoch_metric.csv',
                                header=True, index=False)
                else:
                    csv = pd.DataFrame(metric, index=[epoch_counter])
                    csv.to_csv(model_checkpoints_folder + self.model_name+'_epoch_metric.csv',
                                mode='a', header=False, index=False)
                    
            self.epoch_num += 1
            
            if self.global_step > self.step_limit:
                print('#########################################')
                print(f'global step {self.global_step} > step limit {self.step_limit}!\n'
                        'training finished!')
                break

        self.save_final_model(model_checkpoints_folder, total_metric)


    def create_data_loader(self, train_dataset):
        return TorchDataLoader(train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers,
                        drop_last=True, shuffle=False, sampler=DistributedSampler(train_dataset), pin_memory=True)


    def prepare_checkpoint_directory(self):
        
        model_checkpoints_folder = os.path.join(f'{self.save_dir}/{self.model_name}/')
        if self.device == 0:
     
            if not os.path.exists(model_checkpoints_folder):
                print('create directory "{}" for save checkpoint!'.format(model_checkpoints_folder))
                print('---------------------------')
                os.makedirs(model_checkpoints_folder, exist_ok=True)
            else:
                print('directory "{}" existing for save checkpoint!'.format(model_checkpoints_folder))
        return model_checkpoints_folder


    def load_checkpoint_if_exists(self, model_checkpoints_folder):
        print('#########################################')
        print('Be patient..., checking checkpoint now...')
        
        if os.path.exists(model_checkpoints_folder + self.model_name+'_0_0_checkpoint.pth'):
            ckpt = torch.load(model_checkpoints_folder + self.model_name+'_0_0_checkpoint.pth',
                            map_location='cpu')
            start_epoch = ckpt['epoch']
            # add module before all state keys
            new_state_dict = {}
            for k, v in ckpt['model_state_dict'].items():
                name = 'module.' + k
                new_state_dict[name] = v
            ckpt['model_state_dict'] = new_state_dict

            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print('continue training successful!')
            return start_epoch, True
        else:
            print('Start training from 0 epoch')
            return 0, False


    def log_and_save_model(self, train_loader, epoch_loss, epoch_acc1, epoch_acc5, metric, model_checkpoints_folder):
        if self.device == 0:
            epoch_iter = (len(train_loader))
            # print(f'{self.epoch_num} epoch loss is {epoch_loss/epoch_iter}, acc1 is {epoch_acc1}, acc5 is {epoch_acc5}')
            # pprint(metric)
            print(f'global step is {self.global_step}')
            if self.global_step % self.log_interval == 0:
                self.save_checkpoints(self.epoch_num, self.global_step, model_checkpoints_folder)


    def save_checkpoints(self, epoch, step, model_checkpoints_folder):
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            model_checkpoints_folder + self.model_name+f'_{epoch}_{step}'+'_checkpoint.pth')
        if self.decoder == 'no':
            torch.save(self.model.module.img_model.state_dict(),
                    model_checkpoints_folder + self.model_name+f'_{epoch}_{step}'+'_encoder.pth')
        else:
            torch.save(self.model.module.img_model.state_dict(),
                    model_checkpoints_folder + self.model_name+f'_{epoch}_{step}'+'_encoder.pth')
            # torch.save(self.model.module.img_model.state_dict(),
            #         model_checkpoints_folder + self.model_name+f'_{epoch}_{step}'+'_encoder_decoder.pth')
            
    def save_final_model(self, model_checkpoints_folder, total_metric):
        if self.decoder == 'no':
            torch.save(self.model.module.img_model.state_dict(),
                    model_checkpoints_folder + self.model_name+'_encoder.pth')
        else:
            torch.save(self.model.module.img_model.state_dict(),
                    model_checkpoints_folder + self.model_name +'_encoder.pth')
            # torch.save(self.model.module.img_model.state_dict(),
            #         model_checkpoints_folder + self.model_name +'_encoder_decoder.pth')
            

        # save metric as csv
        metric_df = pd.DataFrame(total_metric)
        metric_df.to_csv(model_checkpoints_folder + self.model_name+'_metric.csv')

        plt.figure(figsize=(12, 10))
        plt.subplot(2, 3, 1)
        plt.plot(total_metric['global_loss'])
        plt.title('global loss')
        plt.subplot(2, 3, 2)
        plt.plot(total_metric['patch_loss'])
        plt.title('patch loss')
        plt.subplot(2, 3, 3)
        plt.plot(total_metric['word_loss'])
        plt.title('word loss')
        plt.subplot(2, 3, 4)
        plt.plot(total_metric['seg_loss'])
        plt.title('seg loss')
        plt.subplot(2, 3, 5)
        plt.plot(total_metric['reconst_loss'])
        plt.title('reconst loss')
        plt.subplot(2, 3, 6)
        plt.plot(total_metric['acc1'])
        plt.plot(total_metric['acc5'])
        plt.title('acc1 and acc5')

        plt.savefig(model_checkpoints_folder + self.model_name+'_loss.png')
        print('training finished!')