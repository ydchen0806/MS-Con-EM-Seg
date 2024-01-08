from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import cv2
import h5py
import random
import logging
import argparse
import numpy as np
from PIL import Image
from attrdict import AttrDict
try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
import multiprocessing as mp
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader.data_provider_labeled import Provider
from dataloader.provider_valid import Provider_valid
from model.resunet import ResUNet
from model.attresUnet import ResUNetWithAttention
from loss.loss import WeightedMSE, WeightedBCE
from loss.loss import MSELoss, BCELoss
from utils.show import show_affs, show_affs_whole
from model.unet3d_mala import UNet3D_MALA
from model.model_superhuman import UNet_PNI
from utils.utils import setup_seed, execute
from utils.shift_channels import shift_func
import sys
sys.path.append('Miccai23')
from model.unetr import UNETR
import waterz
from utils.lmc import mc_baseline
from utils.fragment import watershed, randomlabel
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings
warnings.filterwarnings("ignore")
os.getcwd()
def init_project(cfg, args):
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
    basename = os.path.basename(args.pretrained_path).split('.')[0]
    model_name = cfg.time + '_' + cfg.MODEL.model_type + '_pretrain_' + str(args.pretrained) + '_'  + basename + '_'\
        + cfg.DATA.dataset_name + '_' + str(cfg.DATA.train_split) + '_' + \
         str(cfg.DATA.valid_dataset) + '_' + str(cfg.DATA.test_split) + '_frozenLR_' + str(cfg.TRAIN.frozen_lr) + \
            '_batchsize_' + str(cfg.TRAIN.batch_size)
    # if cfg.TRAIN.resume:
    #     model_name = cfg.TRAIN.model_name
    # else:
    #     model_name = prefix + '_' + cfg.NAME

    cfg.pretrain_path = cfg.TRAIN.pretrain_path
    path_list = cfg.pretrain_path.split('/')[:-2]
    # 合并成新的路径
    path = '/'.join(path_list)
    path = '/braindat/lab/chenyd/MODEL/NeuriPIS0521_EMseg_swin'
    cfg.finetune_path = os.path.join(path,model_name)
    if not os.path.exists(cfg.finetune_path):
        os.makedirs(cfg.finetune_path)
    print('finetune_path:', cfg.finetune_path)
    cfg.cache_path = os.path.join(cfg.finetune_path, 'cache')
    cfg.save_path = os.path.join(cfg.finetune_path, 'model')
    # cfg.record_path = os.path.join(cfg.TRAIN.record_path, 'log')
    cfg.record_path = os.path.join(cfg.finetune_path, 'logs')
    cfg.valid_path = os.path.join(cfg.finetune_path, 'valid')
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
        if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    if not os.path.exists('/output/logs'):
        os.makedirs('/output/logs')
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer

def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    if cfg.TRAIN.if_valid:
        valid_provider = Provider_valid(cfg)
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider

def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    if cfg.MODEL.model_type == 'mala':
            print('load mala model!')
            model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc, if_sigmoid=cfg.MODEL.if_sigmoid, init_mode=cfg.MODEL.init_mode_mala, show_feature=False)
    elif cfg.MODEL.model_type == 'superhuman':
        print('load superhuman model!')
        model = UNet_PNI(in_planes=cfg.MODEL.input_nc,
                                out_planes=cfg.MODEL.output_nc,
                                filters=cfg.MODEL.filters,
                                upsample_mode=cfg.MODEL.upsample_mode,
                                decode_ratio=cfg.MODEL.decode_ratio,
                                merge_mode='add',
                                pad_mode=cfg.MODEL.pad_mode,
                                bn_mode=cfg.MODEL.bn_mode,
                                relu_mode=cfg.MODEL.relu_mode,
                                init_mode=cfg.MODEL.init_mode,
                                if_sigmoid=cfg.MODEL.if_sigmoid,
                                show_feature=False)
    elif cfg.MODEL.model_type == 'UNETR':
        print('load UNETR model!')
        model = UNETR(
                        in_channels=cfg.MODEL.input_nc,
                        out_channels=cfg.MODEL.output_nc,
                        img_size=cfg.MODEL.unetr_size,
                        patch_size=cfg.MODEL.patch_size,
                        feature_size=16,
                        hidden_size=768,
                        mlp_dim=2048,
                        num_heads=8,
                        pos_embed='perceptron',
                        norm_name='instance',
                        conv_block=True,
                        res_block=True,
                        kernel_size=cfg.MODEL.kernel_size,
                        skip_connection=False,
                        show_feature=False,
                        dropout_rate=0.1)
    elif cfg.MODEL.model_type == 'ResUNet':
        print('load ResUNet model!')
        model = ResUNet(args, input_channel=cfg.MODEL.input_nc, out_channels=cfg.MODEL.output_nc, input_size=(24,224,224))
    elif cfg.MODEL.model_type == 'ResUNetWithAttention':
        print('load ResUNetWithAttention model!')
        model = ResUNetWithAttention(args, input_channel=cfg.MODEL.input_nc, out_channels=cfg.MODEL.output_nc, input_size=(24,224,224))

    if cfg.TRAIN.pretrain:
        vit_state = torch.load(cfg.TRAIN.pretrain_path)
        vit_state_dict = vit_state['model_weights']
        vit_state_dict_new = {}
        for k,v in vit_state_dict.items():
            if k.startswith('module'):
                k = k.replace('module.model.','')
                vit_state_dict_new[k] = v
            else:
                vit_state_dict_new[k] = v
        model.load_state_dict(vit_state_dict_new,strict=False)
        # 删除vit_state_dict的缓存
        del vit_state
        del vit_state_dict
        del vit_state_dict_new
        
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model.to(device)

def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model-%06d.pth' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
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


def loop(cfg, train_provider, valid_provider, model, criterion, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    device = torch.device('cuda:0')
    
    if cfg.TRAIN.loss_func == 'MSELoss':
        criterion = MSELoss()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = BCELoss()
    elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
        criterion = WeightedBCE()
    elif cfg.TRAIN.loss_func == 'WeightedMSELoss':
        criterion = WeightedMSE()
    else:
        raise AttributeError("NO this criterion")

    while iters <= cfg.TRAIN.total_iters:
        # train
        model.train()
        iters += 1
        t1 = time.time()
        inputs, target, weightmap = train_provider.next()
        
        # decay learning rate
        if cfg.TRAIN.frozen_lr:
            if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
                current_lr = cfg.TRAIN.base_lr
            else:
                current_lr = calculate_lr(iters)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
        else:
            current_lr = cfg.TRAIN.base_lr
            # 对不同的group设置不同的学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr / 100
                current_lr *= 100
        
        optimizer.zero_grad()
        pred = model(inputs)

        ##############################
        # LOSS
        loss = criterion(pred, target, weightmap)
        loss.backward()
        ##############################

        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()
        
        sum_loss += loss.item()
        sum_time += time.time() - t1
        
        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info('step %d, loss = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss * 1, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('train/loss', sum_loss * 1, iters)
            else:
                logging.info('step %d, loss = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss / cfg.TRAIN.display_freq * 1, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('train/loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)
            f_loss_txt.write('step = ' + str(iters) + ', loss = ' + str(sum_loss / cfg.TRAIN.display_freq * 1))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0

        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            show_affs(iters, inputs, pred[:,:3], target[:,:3], cfg.cache_path, model_type=cfg.MODEL.model_type)

        # valid
        if cfg.TRAIN.if_valid:
            if iters % cfg.TRAIN.save_freq == 0:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model.eval()
                dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                                shuffle=False, drop_last=False, pin_memory=True)
                losses_valid = []
                #记录开始时间
                start = time.time()
                for k, batch in enumerate(dataloader, 0):
                    inputs, target, weightmap = batch
                    inputs = inputs.cuda()
                    target = target.cuda()
                    weightmap = weightmap.cuda()
                    with torch.no_grad():
                        pred = model(inputs)
                    tmp_loss = criterion(pred, target, weightmap)
                    losses_valid.append(tmp_loss.item())
                    valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
                #记录结束时间
                end = time.time()
                writer.add_scalar('train/training_time', end - start, iters)
                #logging记录训练时间
                logging.info('training time = %.2f sec' % (end - start))
                epoch_loss = sum(losses_valid) / len(losses_valid)
                out_affs = valid_provider.get_results()
                gt_affs = valid_provider.get_gt_affs().copy()
                gt_seg = valid_provider.get_gt_lb()
                valid_provider.reset_output()
                out_affs = out_affs[:3]
                # gt_affs = gt_affs[:, :3]
                show_affs_whole(iters, out_affs, gt_affs, cfg.valid_path)

                ##############
                # segmentation
                if cfg.TRAIN.if_seg:
                    if iters > 1:
                        fragments = watershed(out_affs, 'maxima_distance')
                        sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
                        seg_waterz = list(waterz.agglomerate(out_affs, [0.50],
                                                            fragments=fragments,
                                                            scoring_function=sf,
                                                            discretize_queue=256))[0]
                        arand_waterz = adapted_rand_ref(gt_seg, seg_waterz, ignore_labels=(0))[0]
                        voi_split, voi_merge = voi_ref(gt_seg, seg_waterz, ignore_labels=(0))
                        writer.add_scalar('valid/waterz_voi_split', voi_split, iters)
                        writer.add_scalar('valid/waterz_voi_merge', voi_merge, iters)
                        voi_sum_waterz = voi_split + voi_merge

                        seg_lmc = mc_baseline(out_affs)
                        arand_lmc = adapted_rand_ref(gt_seg, seg_lmc, ignore_labels=(0))[0]
                        voi_split, voi_merge = voi_ref(gt_seg, seg_lmc, ignore_labels=(0))
                        writer.add_scalar('valid/lmc_voi_split', voi_split, iters)
                        writer.add_scalar('valid/lmc_voi_merge', voi_merge, iters)
                        voi_sum_lmc = voi_split + voi_merge
                    else:
                        voi_sum_waterz = 0.0
                        arand_waterz = 0.0
                        voi_sum_lmc = 0.0
                        arand_lmc = 0.0
                        print('model-%d, segmentation failed!' % iters)
                else:
                    voi_sum_waterz = 0.0
                    arand_waterz = 0.0
                    voi_sum_lmc = 0.0
                    arand_lmc = 0.0
                ##############

                # MSE
                whole_mse = np.sum(np.square(out_affs - gt_affs)) / np.size(gt_affs)
                out_affs = np.clip(out_affs, 0.000001, 0.999999)
                bce = -(gt_affs * np.log(out_affs) + (1 - gt_affs) * np.log(1 - out_affs))
                whole_bce = np.sum(bce) / np.size(gt_affs)
                out_affs[out_affs <= 0.5] = 0
                out_affs[out_affs > 0.5] = 1
                # whole_f1 = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), out_affs.astype(np.uint8).flatten())
                whole_f1 = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - out_affs.astype(np.uint8).flatten())
                print('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f' % \
                    (iters, epoch_loss, whole_mse, whole_bce, whole_f1, voi_sum_waterz, arand_waterz, voi_sum_lmc, arand_lmc), flush=True)
                writer.add_scalar('valid/epoch_loss', epoch_loss, iters)
                writer.add_scalar('valid/mse_loss', whole_mse, iters)
                writer.add_scalar('valid/bce_loss', whole_bce, iters)
                writer.add_scalar('valid/f1_score', whole_f1, iters)
                writer.add_scalar('valid/voi_waterz', voi_sum_waterz, iters)
                writer.add_scalar('valid/arand_waterz', arand_waterz, iters)
                writer.add_scalar('valid/voi_lmc', voi_sum_lmc, iters)
                writer.add_scalar('valid/arand_lmc', arand_lmc, iters)
                f_valid_txt.write('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f' % \
                                (iters, epoch_loss, whole_mse, whole_bce, whole_f1, voi_sum_waterz, arand_waterz, voi_sum_lmc, arand_lmc))
                f_valid_txt.write('\n')
                f_valid_txt.flush()
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
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_3d', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    parser.add_argument('--input_size',default=(24,224,224),type=tuple)
    # parser.add_argument("--data_dir", type=str, default="/braindat/lab/chenyd/DATASET/MSD/")
    # parser.add_argument('--task_name', type=str, default='Task02_Heart')
    # parser.add_argument('--model_name', type=str, default='ResUNet')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--pretrained_path', type=str, default=
                        '/braindat/lab/chenyd/code_230508/Neurips23_imgSSL/res_encoder/testclip_bt_resnet_50001_iterations_encoder.pth')
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
    if cfg.DATA.shift_channels is None:
        # assert cfg.MODEL.output_nc == 3, "output_nc must be 3"
        cfg.shift = None
    else:
        assert cfg.MODEL.output_nc == cfg.DATA.shift_channels, "output_nc must be equal to shift_channels"
        cfg.shift = shift_func(cfg.DATA.shift_channels)

    if args.mode == 'train':
        writer = init_project(cfg, args)
        train_provider, valid_provider = load_dataset(cfg)
        model = build_model(cfg, writer)
        # if cfg.TRAIN.frozen_lr:
        #     if cfg.TRAIN.pretrain:
        #         vit_state = torch.load(cfg.TRAIN.pretrain_path)
        #         vit_state_dict = vit_state['model_weights']
        #         vit_state_dict_new = {}
        #         for key in vit_state_dict.keys():
        #             vit_state_dict_new['module.vit.'+key] = vit_state_dict[key]
        #         # 删除vit_state_dict的缓存
        #         del vit_state
        #         del vit_state_dict
        #         pretrained_params = []
        #         other_params = []
        #         for name,weight in model.named_parameters():
        #             if name not in vit_state_dict_new.keys():
        #                 pretrained_params.append(weight)
        #             else:
        #                 other_params.append(weight)
        #         del vit_state_dict_new

        #         optimizer = torch.optim.Adam([{'params': other_params},
        #                 {'params': pretrained_params, 'lr': cfg.TRAIN.base_lr / 100}], lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
        #                 eps=0.01, weight_decay=1e-6, amsgrad=True)
        #     else:
        #         raise ValueError('pretrain must be True')
        # else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                    eps=0.01, weight_decay=1e-6, amsgrad=True)
        # optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
        # optimizer = optim.Adamax(model.parameters(), lr=cfg.TRAIN.base_l, eps=1e-8)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, nn.L1Loss(), optimizer, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')