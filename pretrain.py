from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import logging
import argparse
import numpy as np
from attrdict import AttrDict
from tensorboardX import SummaryWriter
import random
import torch
import torch.nn as nn
from apex import amp

from dataloader.data_provider_pretraining import Provider
from dataloader.provider_valid_pretraining import Provider_valid
from utils.show import show_affs, show_affs_whole,save_rec
from model.unet3d_mala import UNet3D_MALA, UNet3D_MALA_ns
from model.model_superhuman import UNet_PNI, UNet_PNI_Noskip
from model.unetr import UNETR
from model.PretrainModel import pretrainModel
from utils.utils import setup_seed


def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
                level    = logging.INFO,
                format   = '%(message)s',
                datefmt  = '%m-%d %H:%M',
                filename = path,
                filemode = 'w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = cfg.TRAIN.model_name
    else:
        model_name = prefix + '_' + cfg.NAME
    today = '_230313_batch_size_' + str(cfg.TRAIN.batch_size) + '_random_choice_' + str(cfg.TRAIN.random_choice) + \
        '_multi_scale_mse_' + str(cfg.TRAIN.multi_scale_mse)
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, cfg.MODEL.model_type + today, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, cfg.MODEL.model_type + today, model_name)
    cfg.record_path = os.path.join(cfg.TRAIN.record_path, cfg.MODEL.model_type + today, model_name)
  
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
     
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer

def load_dataset(cfg):
    print('Caching datasets ... ', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider

def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    model = pretrainModel(cfg)
    model = model.to(device)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model

def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model-%06d.pt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0

def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr

def loop(cfg, train_provider, model, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    sum_mse_loss = 0
    sum_cos_loss = 0
    sum_loss_mha_1 = 0
    sum_loss_mha_2 = 0
    device = torch.device('cuda:0')
    
    if cfg.TRAIN.loss_func == 'MSE':
        print('L2 loss...')
        criterion = nn.MSELoss()
    elif cfg.TRAIN.loss_func == 'L1':
        print('L1 loss...')
        criterion = nn.L1Loss()
    else:
        raise AttributeError("NO this criterion")
    cos = torch.nn.CosineSimilarity(dim= -1, eps=1e-6)
    while iters <= cfg.TRAIN.total_iters:
        # train
        model.train()
        iters += 1
        t1 = time.time()
        inputs1,inputs2, gt = train_provider.next()
        
        # decay learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        optimizer.zero_grad()

        gt, out1, out2, x_concat, y_concat, x_mha, y_mha = model(inputs1,inputs2,gt)
            
        mse_loss = criterion(out1, gt) / 2 + criterion(out2, gt) / 2 
        if cfg.TRAIN.multi_scale_mse:
            mse_loss += criterion(x_concat,y_concat)

        cos_sim = (-torch.log((torch.exp(cos(x_concat,y_concat)/0.1)/torch.exp(cos(x_concat,y_concat)/0.1).sum()))).mean()
        
        loss_mha_1 = (-torch.log((torch.exp(cos(x_concat,x_mha)/0.1)/torch.exp(cos(x_concat,x_mha)/0.1).sum()))).mean()
        loss_mha_2 = (-torch.log((torch.exp(cos(y_concat,y_mha)/0.1)/torch.exp(cos(y_concat,y_mha)/0.1).sum()))).mean()
        
        cos_sim = cos_sim * 0.1
        loss_mha_1 = loss_mha_1 * 0.5
        loss_mha_2 = loss_mha_2 * 0.5
        loss = mse_loss + cos_sim + loss_mha_1 + loss_mha_2
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        ##############################

        # if cfg.TRAIN.weight_decay is not None:
        #     for group in optimizer.param_groups:
        #         for param in group['params']:
        #             param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()
        
        sum_loss += loss.item()
        sum_mse_loss += mse_loss.item()
        sum_cos_loss += cos_sim.item()
        sum_loss_mha_1 += loss_mha_1.item()
        sum_loss_mha_2 += loss_mha_2.item()
        sum_time += time.time() - t1
        
        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info('step %d, loss = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss * 1, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss * 1, iters)
                writer.add_scalar('cosine_similarity', sum_cos_loss * 1, iters)
                writer.add_scalar('mse_loss', sum_mse_loss * 1, iters)
                writer.add_scalar('loss_mha_1', sum_loss_mha_1 * 1, iters)
                writer.add_scalar('loss_mha_2', sum_loss_mha_2 * 1, iters)
                f_loss_txt.write('step = ' + str(iters) + ', loss = ' + str(sum_loss * 1) + ',MSE loss = ' + \
                                str(sum_mse_loss * 1) + ',cosine similarity = ' + \
                                str(sum_cos_loss * 1) + ',sum_loss_mha_1 = ' + \
                                str(sum_loss_mha_1 * 1) + ',sum_loss_mha_2 = ' + \
                                str(sum_loss_mha_2 * 1) + ',lr = ' + str(current_lr) + ',time = ' + str(sum_time))
            else:
                logging.info('step %d, loss = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss / cfg.TRAIN.display_freq * 1, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)
                writer.add_scalar('cosine_similarity', sum_cos_loss / cfg.TRAIN.display_freq * 1, iters)
                writer.add_scalar('mse_loss', sum_mse_loss / cfg.TRAIN.display_freq * 1, iters)
                writer.add_scalar('loss_mha_1', sum_loss_mha_1 / cfg.TRAIN.display_freq * 1, iters)
                writer.add_scalar('loss_mha_2', sum_loss_mha_2 / cfg.TRAIN.display_freq * 1, iters)
                f_loss_txt.write('step = ' + str(iters) + ', loss = ' + str(sum_loss / cfg.TRAIN.display_freq * 1) + ',MSE loss = ' + \
                                str(sum_mse_loss / cfg.TRAIN.display_freq * 1) + ',cosine similarity = ' + \
                                str(sum_cos_loss / cfg.TRAIN.display_freq * 1) + ',sum_loss_mha_1 = ' + \
                                str(sum_loss_mha_1 / cfg.TRAIN.display_freq * 1) + ',sum_loss_mha_2 = ' + \
                                str(sum_loss_mha_2 / cfg.TRAIN.display_freq * 1) + ', lr = ' + str(current_lr) + \
                                    ', time = ' + str(sum_time))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0
            sum_cos_loss = 0
            sum_mse_loss = 0
            sum_loss_mha_1 = 0
            sum_loss_mha_2 = 0
        
        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            save_rec(iters, gt, inputs1, inputs2, out1, out2, cfg.cache_path, model_type=cfg.MODEL.model_type)
            torch.cuda.empty_cache()

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                    'model_weights': model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.pt' % iters))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='pretraining_all', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('/braindat/lab/chenyd/code/Miccai23/config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider = load_dataset(cfg)
        model = build_model(cfg, writer)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                 eps=0.01, weight_decay=1e-6, amsgrad=True)
        # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        cuda_count = torch.cuda.device_count()

        if cuda_count > 1:
            if cfg.TRAIN.batch_size % cuda_count == 0:
                print('%d GPUs ... ' % cuda_count, end='', flush=True)
                model = nn.DataParallel(model)
            else:
                raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
        else:
            print('a single GPU ... ', end='', flush=True)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, model, optimizer, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')