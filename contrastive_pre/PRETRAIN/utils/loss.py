import torch
import torch.nn.functional as F
import pandas as pd
import math
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from einops import rearrange
import segmentation_models_pytorch as smp


def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
    ''' Compute the accuracy over the k top predictions for the specified values of k'''
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def clip_loss(x, y, temperature=0.07, device='cuda'):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        sim = torch.einsum('i d, j d -> i j', x, y) * 1 / temperature

        labels = torch.arange(x.shape[0]).to(device)

        loss_t = F.cross_entropy(sim, labels) 
        loss_i = F.cross_entropy(sim.T, labels) 

        i2t_acc1, i2t_acc5 = precision_at_k(
            sim, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = precision_at_k(
            sim.T, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        return (loss_t + loss_i), acc1, acc5

def clip_loss_wPrior(x, y, prior=None, temperature=0.1, device='cuda', smooth='exp', prior_ratio=0.5):
        smooth = smooth

        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        sim = torch.einsum('i d, j d -> i j', x, y) * 1 / temperature

        labels = torch.arange(x.shape[0]).to(device)
        labels = torch.nn.functional.one_hot(labels, num_classes=-1).to(x.dtype)
        if prior is not None:
            prior = torch.corrcoef(prior)
            prior[prior<0] = 0
            prior.fill_diagonal_(0)
            if smooth == 'gau':
                prior = (1/torch.sqrt(torch.tensor(2*torch.pi))) * torch.exp(-0.5*(torch.square(prior)))
            elif smooth == 'lap':
                prior = 0.5 * torch.exp(-torch.abs(prior))
            elif smooth == 'sigmoid':
                prior = torch.sigmoid(prior)
            else:
                prior = 1 - torch.exp(-(prior_ratio) * prior)
            prior = prior.to(x.dtype)
            
            labels += prior

        loss_t = F.cross_entropy(sim, labels) 
        loss_i = F.cross_entropy(sim.T, labels) 

        return (loss_t + loss_i)

def orthogonal_loss(x1, x2, device='cuda'):
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    train_batch_size = x1.shape[0]

    logits = torch.mm(x1.T, x2).to(device)

    logits.div_(train_batch_size)
    on_diag = torch.diagonal(logits).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(logits).pow_(2).sum()
    loss = on_diag + 0.0051*off_diag
    return loss/2

def reconst_loss(x, y):
    loss = F.l1_loss(x, y)
    return loss.mean()

def align_loss(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    loss = 2 - 2 * (x * y).sum(dim=-1)
    loss += 2 - 2 * (y * x).sum(dim=-1)
    return loss.mean()

def local_loss(patch_emb, word_emb, temperature, device):
    bz = patch_emb.size(0)


    atten_sim = torch.bmm(word_emb, patch_emb.transpose(1, 2)) 
    word_num = word_emb.size(1)
    atten_scores = F.softmax(
        atten_sim / temperature, dim=-1) # [b, word_num, patch_num]
    word_atten_output = torch.bmm(atten_scores, patch_emb)

    with torch.no_grad():
        word_atten_output = F.normalize(word_atten_output, dim=-1)
        word_atten_weights = torch.ones(
            bz, word_num).type_as(word_emb) / word_num

    word_atten_weights /= word_atten_weights.sum(
        dim=1, keepdims=True)
    
    word_sim = torch.bmm(
        word_emb, word_atten_output.permute(0, 2, 1)) / temperature
    
    with torch.no_grad():
        word_sim = torch.bmm(word_emb, word_atten_output.permute(
            0, 2, 1)) / temperature
        word_num = word_sim.size(1)
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")
        targets = torch.arange(word_num).type_as(word_emb).long().repeat(bz).to(device)

    loss_word_1 = torch.sum(F.cross_entropy(
            word_sim_1, targets, reduction="mean") * word_atten_weights ) / bz

    word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
    loss_word_2 = torch.sum(F.cross_entropy(
        word_sim_2, targets, reduction="mean") * word_atten_weights) / bz

    loss_word = (loss_word_1 + loss_word_2) / 2.

    # visual local part
    atten_sim = torch.bmm(patch_emb, word_emb.permute(0, 2, 1))
    patch_num = patch_emb.size(1)

    atten_scores = F.softmax(
        atten_sim / temperature, dim=-1)  # bz, 196, 111
    patch_atten_output = torch.bmm(atten_scores, word_emb)
    with torch.no_grad():
        patch_atten_output = F.normalize(patch_atten_output, dim=-1)
        patch_num = patch_atten_output.size(1)
        patch_atten_weights = torch.ones(
            bz, patch_num).type_as(patch_emb) / patch_num

    patch_atten_weights /= patch_atten_weights.sum(
        dim=1, keepdims=True)
    
    patch_sim = torch.bmm(patch_emb, patch_atten_output.permute(
                0, 2, 1)) / temperature
    patch_num = patch_sim.size(1)
    patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
    targets = torch.arange(patch_num).type_as(
        patch_emb).long().repeat(bz).to(device)

    loss_patch_1 = torch.sum(F.cross_entropy(
        patch_sim_1, targets, reduction="mean") * patch_atten_weights.view(-1)) / bz

    patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
    loss_patch_2 = torch.sum(F.cross_entropy(
        patch_sim_2, targets, reduction="mean") * patch_atten_weights.view(-1)) / bz

    loss_patch = (loss_patch_1 + loss_patch_2) / 2.

    loss_local = loss_patch + loss_word
    return loss_local, loss_patch.item(), loss_word.item()


def bceDiceLoss(pred, target, train=True):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    smooth = 1e-5
    num = target.size(0)
    pred = pred.view(num, -1)
    target = target.view(num, -1)
    intersection = (pred * target)
    dice = (2. * intersection.sum(1) + smooth) / (pred.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    if train:
        return dice + 0.2 * bce
    return dice

def focal_loss(pred, target, mode, gamma):
    '''
    pred: [b, c, h, w]
    if mode == 'binary':
        target: [b, h, w]
    else:
        target: [b, c, h, w]
    mode: 'binary' or 'multiclass' or 'multilabel'
    gamma: default 2.0
    '''
    loss = smp.losses.FocalLoss(mode=mode, 
                                alpha=None, 
                                gamma=gamma, 
                                ignore_index=None, 
                                reduction='mean', 
                                normalized=False, 
                                reduced_threshold=None)
    loss = loss(pred, target)
    return loss