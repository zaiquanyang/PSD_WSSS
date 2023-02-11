import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable
import wandb

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import average_precision_score
from tqdm import tqdm

import utils
from func import *


def vanilla_train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True, args=None, writer=None, start_iter=0):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    # for samples in metric_logger.log_every(data_loader, print_freq, header):
    for i, (img, label) in enumerate(tqdm(data_loader)):
    
        start_iter += 1

        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
                
        with torch.cuda.amp.autocast():
            model.train()
            #-------------------------------------------------------------------------------------#
            # 计算主干分支的分类损失
            img_cls_logits, img_patch_logits, loss_dict, _, _, coefficient_matrix = model(img, label, retun_att_loss=True, n_layers=args.layer_index, attention_type=args.attention_type, epoch=epoch)
            gcl_loss, distill_loss, mKD_loss = loss_dict['gcl'], loss_dict['distill'], loss_dict['multi_kd']
            # adaptive coefficient_matrix
            if epoch >= args.start_ep:
                img_patch_logits = img_patch_logits * coefficient_matrix

            bce_loss = F.multilabel_soft_margin_loss(img_cls_logits, label)
            metric_logger.update(cls_loss=bce_loss.item())
            
            #-------------------------------------------------------------------------------------#
            # if  patch_outputs is not None:
            ploss = F.multilabel_soft_margin_loss(img_patch_logits, label)
            metric_logger.update(pat_loss=ploss.item())
            loss = bce_loss + ploss 
            
            #-------------------------------------------------------------------------------------#
            # if  gcl_loss is not None:
            metric_logger.update(gcl_loss=gcl_loss.item())
            if epoch >= args.start_ep:
                loss = loss + gcl_loss * args.gcl_lambda
            else:
                loss = loss + gcl_loss * 0.0
                
            #-------------------------------------------------------------------------------------#
            # if  distill_loss is not None:
            metric_logger.update(distill_loss=distill_loss.item())
            if epoch >= args.start_ep:
                loss = loss + distill_loss * args.distill_lambda
            else:
                loss = loss + distill_loss * 0.0
                
            #-------------------------------------------------------------------------------------#
            # if  mKD_loss is not None:
            metric_logger.update(mKD_loss=mKD_loss.item())
            if epoch >= args.start_ep:
                loss = loss + mKD_loss * args.multi_kd_lambda
            else:
                loss = loss + mKD_loss * 0.0

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        time.sleep(0.001)
        
        # tensorboard
        if utils.is_main_process():
            # writer.add_scalar('loss_iter/train_bce_loss', bce_loss.item(), start_iter)
            # writer.add_scalar('loss_iter/train_patch_loss', ploss.item(), start_iter)

            # writer.add_scalar('loss_iter/train_gcl_loss', gcl_loss.item(), start_iter)
            # writer.add_scalar('loss_iter/train_distill_loss', distill_loss.item(), start_iter)
            # writer.add_scalar('loss_iter/train_mKD_loss', mKD_loss.item(), start_iter)
            # writer.add_scalar('loss_iter/train_FixMatch_loss', FixMatch_loss.item(), start_iter)
            
            # writer.add_scalar('loss_iter/train_loss', loss.item(), start_iter)
            # writer.add_scalar('optimizer_iter/train_lr', optimizer.param_groups[0]["lr"], start_iter)

            wandb.log({
                'iter':start_iter, 
                'train_bce_loss': bce_loss.item(),
                'train_patch_loss': ploss.item(),
                'train_loss': loss.item(),
                'train_lr':optimizer.param_groups[0]["lr"],
                'coeff_max':coefficient_matrix.max().item(),
                'coeff_min':coefficient_matrix.min().item(),
            })


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, start_iter


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    epoch: int, 
    loss_scaler, 
    max_norm: float = 0,
    set_training_mode=True, 
    args=None, 
    writer=None, 
    start_iter=0,
    PPNet=None,
    EMA_CAM=None
    ):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    # for samples in metric_logger.log_every(data_loader, print_freq, header):
    for i, (img, label, 
            img_single, label_single, 
            img_multi_w_fg, label_multi_w_fg, single_fg_mask, fg_paste_position,
            img_multi_w_bg, label_multi_w_bg, single_bg_mask, bg_paste_position) in enumerate(tqdm(data_loader)):
    # for i, (img, label) in enumerate(tqdm(data_loader)):
        start_iter += 1

        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        img_single, label_single = img_single.cuda(), label_single.cuda()
        img_multi_w_fg, label_multi_w_fg = img_multi_w_fg.cuda(), label_multi_w_fg.cuda()
        img_multi_w_bg, label_multi_w_bg = img_multi_w_bg.cuda(), label_multi_w_bg.cuda()

        # bs x 224 x 224
        single_fg_mask = single_fg_mask.cuda()
        single_bg_mask = single_bg_mask.cuda()
        # paste_position = deepcopy(fg_paste_position)
        # paste_position[:, :-1] = paste_position[:, :-1] / 16     # 有关空间尺度信息进行缩放
        # paste_position_x1, paste_position_y1, paste_position_x2, paste_position_y2, \
        #     paste_w, paste_h, \
        #         scale_size = paste_position[:, 0].long().cuda(), paste_position[:, 1].long().cuda(), paste_position[:, 2].long().cuda(), paste_position[:, 3].long().cuda(), \
        #             paste_position[:, 4].long().cuda(), paste_position[:, 5].long().cuda(), paste_position[:, 6].float().cuda()
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                model.eval()
                # cls_attentions: bs x cls x 14 x 14, cls=20
                # patch_attn : 12 x bs x 196 x 196
                _, cls_attentions, patch_attn = model(img_single, return_att=True, n_layers=args.layer_index, \
                    attention_type=args.attention_type)
                # x_cls_logits, cls_attentions, patch_attn = x_cls_logits.detach(), cls_attentions.detach(), patch_attn.detach()
                patch_attn = torch.sum(patch_attn, dim=0)    # 平均 12 个 blk
                w_featmap, h_featmap = 14, 14
                
                # combine patch_attn(affinity) and cls_attentions
                # img_single_attn : bs x cls x 14 x 14
                img_single_attn = torch.matmul(patch_attn.unsqueeze(1), cls_attentions.view(cls_attentions.shape[0],cls_attentions.shape[1], -1, 1)).\
                    reshape(cls_attentions.shape[0],cls_attentions.shape[1], w_featmap, h_featmap)
                # img_single_cams : bs x cls x 14 x 14
                img_single_cams_cmm, img_single_cams_smm, img_single_cam = cal_cam(img_single_attn.detach(), label_single, channel_min_max=True)
        
                # 如果用于 copy 的图像只有单类别图像，则可以如下计算 label_id列表， 此时也就可以expand得到伪标签
                # label_single_idx_list = torch.Tensor([torch.nonzero(label_single[i]).reshape(-1) for i in range(label_single.shape[0])]).long()
                # BS x 14 x 14
                # mask_single_w = label_single_idx_list.reshape(-1, 1, 1).expand(label_single.shape[0], 14, 14)             

                # 如果用于copy的图像含有多类别图像，则不能直接expand得到伪标签, 可以考虑使用 cls_idx
        breakpoint()
                # EMA_CAM.update(cams=img_single_cams_smm, labels=label_single)

        with torch.cuda.amp.autocast():
            model.train()
            #---------------------------- 计算主干分支的分类损失 -----------------------------#
            img_cls_logits, img_patch_logits, _, _ = model(img, label, retun_att_loss=True, n_layers=args.layer_index, \
                attention_type=args.attention_type)
            
            # 可视化 cut_mix 后的图像
            # torchvision.utils.save_image(img_single, '/data/code/WSSS/MCTformer/images/img_single.png', normalize=True)
            # torchvision.utils.save_image(img_multi_w_fg, '/data/code/WSSS/MCTformer/images/img_multi_w_fg.png', normalize=True)
            # torchvision.utils.save_image(img_multi_w_bg, '/data/code/WSSS/MCTformer/images/img_multi_w_bg.png', normalize=True)
            # # # torchvision.utils.save_image(img_single*multi_cut_mask.unsqueeze(1), '/data/code/WSSS/MCTformer/images/img_single_mask.png', normalize=True)
            # breakpoint()

            # _________ 预测强 img_multi_w_fg 经过 cut_mix 后的 cam ________#
            _, cls_atten_multi_w_fg, patch_atten_multi_w_fg = model(img_multi_w_fg, label_multi_w_fg, return_att=True, n_layers=args.layer_index, \
                attention_type=args.attention_type)
            patch_atten_multi_w_fg = torch.sum(patch_atten_multi_w_fg, dim=0)    # 平均 12 个 head
            img_multi_w_fg_atten = torch.matmul(patch_atten_multi_w_fg.unsqueeze(1), \
                cls_atten_multi_w_fg.view(cls_atten_multi_w_fg.shape[0],cls_atten_multi_w_fg.shape[1], -1, 1))\
                .reshape(cls_atten_multi_w_fg.shape[0],cls_atten_multi_w_fg.shape[1], w_featmap, h_featmap)
            # img_multi_s_cams: BS x C x 14 x 14
            img_multi_w_fg_cams_cmm, img_multi_w_fg_cams_smm, _ = cal_cam(img_multi_w_fg_atten, label_multi_w_fg, channel_min_max=True)

            # ___________ 计算 img_multi_w_fg 的 fix_match_loss __________ #
            # FixMatch_FG_loss = F.cross_entropy(img_multi_s_cams * args.fixmatch_temp, mask_single_w.cuda(), reduction='none')
            logp_x = F.log_softmax(img_multi_w_fg_cams_cmm.permute(0, 2, 3, 1), dim=-1)
            p_y = F.softmax(img_single_cams_cmm.permute(0, 2, 3, 1) * args.fixmatch_temp, dim=-1)
            FixMatch_FG_loss = F.kl_div(logp_x, p_y, reduction='none')*(label_single.unsqueeze(dim=1).unsqueeze(dim=1))
            FixMatch_FG_loss = FixMatch_FG_loss.sum(dim=-1)
            # bs x 14 x 14
            fg_grid=(F.avg_pool2d(single_fg_mask.unsqueeze(1).expand(img_multi_w_fg.shape).float(), (16, 16)) >=1.0).float()[:, 0]
            # bs x 14 x 14
            confi_mask = torch.cat(img_single_cam, dim=0)  
            FixMatch_FG_loss = (FixMatch_FG_loss * fg_grid * (confi_mask > args.fixmatch_fg_thre))  

            # ------ 预测强 img_multi_w_bg 经过 cut_mix 后的 cam -------- #
            _, cls_atten_multi_w_bg, patch_atten_multi_w_bg = model(img_multi_w_bg, label_multi_w_bg, return_att=True, n_layers=args.layer_index, \
                attention_type=args.attention_type)
            patch_atten_multi_w_bg = torch.sum(patch_atten_multi_w_bg, dim=0)    # 平均 12 个 head
            img_multi_w_bg_atten = torch.matmul(patch_atten_multi_w_bg.unsqueeze(1), \
                cls_atten_multi_w_bg.view(cls_atten_multi_w_bg.shape[0],cls_atten_multi_w_bg.shape[1], -1, 1))\
                .reshape(cls_atten_multi_w_bg.shape[0],cls_atten_multi_w_bg.shape[1], w_featmap, h_featmap)
            # img_multi_w_bg_cams : BS x C x 14 x 14
            img_multi_w_bg_cams_cmm, img_multi_w_bg_cams_smm, _ = cal_cam(img_multi_w_bg_atten, label_multi_w_bg, channel_min_max=True)
            # img_single_cams_cmm, img_single_cams_smm, img_single_cam = cal_cam(img_single_attn.detach(), label_single, channel_min_max=True)
            # ------ 计算 img_single_w 的 fix_match_loss (bg_kl_loss) -------- #
            bg_grid=(F.avg_pool2d(single_bg_mask.unsqueeze(1).expand(img_multi_w_fg.shape).float(), (16, 16)) >=1.0).float()[:, 0]
            bg_score = (torch.ones_like(img_single_cams_cmm[:, 0]) * 0.1).unsqueeze(dim=1)  # B x 1 X H x W
            before_paste_total_score = torch.cat([img_single_attn, bg_score], dim=1)                 # B x (C+1) x H x W
            after_paste_total_score = torch.cat([img_multi_w_bg_atten, bg_score], dim=1)                 # B x (C+1) x H x W
            before_paste_total_score = before_paste_total_score.permute(0, 2, 3, 1)
            after_paste_total_score = after_paste_total_score.permute(0, 2, 3, 1)

            bg_logp_x = F.log_softmax(after_paste_total_score*10, dim=-1)
            bg_p_y = F.softmax(before_paste_total_score, dim=-1)
            FixMatch_BG_loss = F.kl_div(bg_logp_x, bg_p_y, reduction='none')[:, :, :, -1]

            fg_pixel_weight, bg_pixel_weight = PPNet(img_multi_w_fg_cams_smm.detach()*(label_multi_w_fg.unsqueeze(dim=-1).unsqueeze(dim=-1)), fg_grid, \
                img_multi_w_bg_cams_smm.detach()*(label_multi_w_bg.unsqueeze(dim=-1).unsqueeze(dim=-1)), bg_grid)
            
            FixMatch_FG_loss = (FixMatch_FG_loss * fg_pixel_weight).sum()/(FixMatch_FG_loss.shape[0])
            FixMatch_BG_loss = (FixMatch_BG_loss * bg_pixel_weight).sum()/(FixMatch_BG_loss.shape[0])

        bce_loss = F.multilabel_soft_margin_loss(img_cls_logits, label)
        ploss = F.multilabel_soft_margin_loss(img_patch_logits, label)
        metric_logger.update(cls_loss=bce_loss.item())
        metric_logger.update(pat_loss=ploss.item())
        loss = bce_loss + ploss 
        
        metric_logger.update(FixMatch_FG_loss=FixMatch_FG_loss.item())
        metric_logger.update(FixMatch_BG_loss=FixMatch_BG_loss.item())
        if epoch >= args.start_ep:
            loss = loss + FixMatch_FG_loss * args.fixmatch_fg_lambda + FixMatch_BG_loss * args.fixmatch_bg_lambda
        else:
            loss = loss + FixMatch_FG_loss * 0.0 + FixMatch_BG_loss * 0.0

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        # 检查梯度
        # for name, parms in model.named_parameters():	
        #     # print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)
        #     print(parms.sum())
        #     break

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # tensorboard
        if utils.is_main_process():
            writer.add_scalar('train/train_bce_loss', bce_loss.item(), start_iter)
            writer.add_scalar('train/train_patch_loss', ploss.item(), start_iter)
            writer.add_scalar('train/train_FG_loss', FixMatch_FG_loss.item(), start_iter)
            writer.add_scalar('train/train_BG_loss', FixMatch_BG_loss.item(), start_iter)
            writer.add_scalar('train/train_loss', loss.item(), start_iter)

            writer.add_scalar('optimizer/train_lr', optimizer.param_groups[0]["lr"], start_iter)

            wandb.log({
                'iter':start_iter, 
                'train_bce_loss': bce_loss.item(),
                'train_patch_loss': ploss.item(),
                'train_FixMatch_FG_loss': FixMatch_FG_loss.item(),
                'train_FixMatch_BG_loss': FixMatch_BG_loss.item(),
                'train_loss': loss.item(),
                'train_lr':optimizer.param_groups[0]["lr"],
            })


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, start_iter

def self_train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True, args=None, writer=None, start_iter=0):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # header = 'Epoch: [{}]'.format(epoch)
    # print_freq = 10
    # torch.cuda.empty_cache()
    # for samples in metric_logger.log_every(data_loader, print_freq, header):
    for i, ((img, label), 
            (img_multi_w, img_multi_s, label_multi), 
            (img_single_w, img_single_s, label_single, cut_mask)) in enumerate(tqdm(data_loader)):
    
        start_iter += 1
        torch.cuda.empty_cache()
        
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        img_multi_w, img_multi_s, label_multi = img_multi_w.cuda(), img_multi_s.cuda(), label_multi.cuda()
        img_single_w, img_single_s, label_single, cut_mask = img_single_w.cuda(), img_single_s.cuda(), label_single.cuda(), cut_mask.cuda()
        
                
        # patch_outputs = None
        # gcl_loss = None
        # distill_loss = None
        # mKD_loss = None
        
        with torch.cuda.amp.autocast():
            model.train()
            #-------------------------------------------------------------------------------------#
            # 计算主干分支的分类损失
            img_cls_logits, img_patch_logits, loss_dict, _, _ = model(img, label, retun_att_loss=True, n_layers=args.layer_index, attention_type=args.attention_type)
            gcl_loss, distill_loss, mKD_loss = loss_dict['gcl'], loss_dict['distill'], loss_dict['multi_kd']
            
            #-------------------------------------------------------------------------------------#
            # 计算 single-cls image self-training loss
            # cls_attentions : bs x cls x 14 x 14
            # patch_attn     : blk x bs x 196 x 196 
            _, cls_attentions, patch_attn = model(img_single_w, label_single, return_att=True, n_layers=args.layer_index, attention_type=args.attention_type)
            # x_cls_logits, cls_attentions, patch_attn = x_cls_logits.detach(), cls_attentions.detach(), patch_attn.detach()
            patch_attn = torch.sum(patch_attn, dim=0)    # 平均 12 个 blk
            w_featmap, h_featmap = 14, 14
            # img_single_w_attn : bs x cls x 14 x 14
            img_single_w_attn = torch.matmul(patch_attn.unsqueeze(1), 
                                            cls_attentions.view(cls_attentions.shape[0],cls_attentions.shape[1], -1, 1)
                                            ).reshape(cls_attentions.shape[0],cls_attentions.shape[1], w_featmap, h_featmap)
            # img_single_w_cams : bs x cls x 14 x 14
            label_single_idx_list = torch.Tensor([torch.nonzero(label_single[i]).reshape(-1) for i in range(label_single.shape[0])]).long()
            img_single_w_cams, img_single_w_cam = cal_cam(img_single_w_attn.detach(), label_single_idx_list)
            mask_single_w = label_single_idx_list.reshape(-1, 1, 1).expand(label_single.shape[0], 14, 14)                         # BS x H x W
            
            # args.FixMatch_t 温度参数
            FixMatch_loss = F.cross_entropy(img_single_w_cams * args.FixMatch_t, mask_single_w.cuda(), reduction='none')
            # # cut_mask_ = F.interpolate((cut_mask.unsqueeze(1).expand(img_multi_s.shape) == 1).float(), scale_factor=(1/16, 1/16), mode='nearest')[:, 0]
            cut_grid=(F.avg_pool2d(cut_mask.unsqueeze(1).expand(img_multi_s.shape).float(), (16, 16)) >=1.0).float()[:, 0] # bs x h x w
            confi_mask = torch.cat(img_single_w_cam, dim=0) # (conf_u_w >= cfg['conf_thresh']), bs x h x w
            # if torch.isnan(FixMatch_loss).any():
            #     breakpoint()   # torch.isnan(img_multi_s_cams).any()
            # 只计算cut_mix区域且置信度高于阈值的像素分割损失
            FixMatch_loss = (FixMatch_loss * cut_grid * (confi_mask > args.FixMatch_thre)).sum() / (torch.sum(cut_grid * (confi_mask > args.FixMatch_thre)) + 0.001)
            # FixMatch_loss = torch.Tensor([0.]).cuda()
            
            bce_loss = F.multilabel_soft_margin_loss(img_cls_logits, label)
            metric_logger.update(cls_loss=bce_loss.item())
            
            #-------------------------------------------------------------------------------------#
            # if  patch_outputs is not None:
            ploss = F.multilabel_soft_margin_loss(img_patch_logits, label)
            metric_logger.update(pat_loss=ploss.item())
            loss = bce_loss + ploss 
            
            #-------------------------------------------------------------------------------------#
            # if  gcl_loss is not None:
            metric_logger.update(gcl_loss=gcl_loss.item())
            if epoch >= args.start_ep:
                loss = loss + gcl_loss * args.gcl_lambda
            else:
                loss = loss + gcl_loss * 0.0
                
            #-------------------------------------------------------------------------------------#
            # if  distill_loss is not None:
            metric_logger.update(distill_loss=distill_loss.item())
            if epoch >= args.start_ep:
                loss = loss + distill_loss * args.distill_lambda
            else:
                loss = loss + distill_loss * 0.0
                
            #-------------------------------------------------------------------------------------#
            # if  mKD_loss is not None:
            metric_logger.update(mKD_loss=mKD_loss.item())
            if epoch >= args.start_ep:
                loss = loss + mKD_loss * args.multi_kd_lambda
            else:
                loss = loss + mKD_loss * 0.0
                
            # if  FixMatch_loss is not None:
            metric_logger.update(FixMatch_loss=FixMatch_loss.item())
            if epoch >= args.start_ep:
                loss = loss + FixMatch_loss * args.FixMatch_lambda
            else:
                loss = loss + FixMatch_loss * 0.0

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        time.sleep(0.001)
        
        # tensorboard
        if utils.is_main_process():
            writer.add_scalar('loss_iter/train_bce_loss', bce_loss.item(), start_iter)
            writer.add_scalar('loss_iter/train_patch_loss', ploss.item(), start_iter)

            writer.add_scalar('loss_iter/train_gcl_loss', gcl_loss.item(), start_iter)
            writer.add_scalar('loss_iter/train_distill_loss', distill_loss.item(), start_iter)
            writer.add_scalar('loss_iter/train_mKD_loss', mKD_loss.item(), start_iter)
            writer.add_scalar('loss_iter/train_FixMatch_loss', FixMatch_loss.item(), start_iter)
            
            writer.add_scalar('loss_iter/train_loss', loss.item(), start_iter)
            
            writer.add_scalar('optimizer_iter/train_lr', optimizer.param_groups[0]["lr"], start_iter)
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, start_iter

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    mAP = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        # breakpoint()
        images = images.to(device, non_blocking=True)  # bs x c x h x w
        target = target.to(device, non_blocking=True)  # bs x cls_num
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast():
            
            output = model(images, target)                     # bs x cls_num 
            # if not isinstance(output, torch.Tensor):
            #     output, patch_output, _ = output
                
            if not isinstance(output, torch.Tensor):
                if len(output) == 2:
                    output, patch_output = output
                # elif len(output) == 3:
                #     output, patch_output, gcl_loss = output
                # elif len(output) == 4:
                #     output, patch_output, gcl_loss, distill_loss = output
                # elif len(output) == 5:
                #     output, patch_output, gcl_loss, distill_loss, mid_loss = output
                else:
                    pass
                    
            loss = criterion(output, target)
            output = torch.sigmoid(output)

            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list
            metric_logger.meters['mAP'].update(np.mean(mAP_list), n=batch_size)


        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print('* mAP {mAP.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(mAP=metric_logger.mAP, losses=metric_logger.loss))


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
            # print(ap_i)
    return AP


@torch.no_grad()
def generate_attention_maps_ms(data_loader, model, device, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generating attention maps:'
    if args.attention_dir is not None:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True)

    # switch to evaluation mode
    model.eval()
    if 'COCO' in args.data_set:
        img_list = open(os.path.join(args.img_list, 'train_id.txt')).readlines()
    else:
        img_list = open(os.path.join(args.img_list, 'train_id.txt')).readlines()  # 为了快速验证模型，可以只生成train_id的图像
        # img_list = open(os.path.join(args.img_list, 'train_aug_id.txt')).readlines()   # 仅评价单类别图像 sing_cls_id
    index = 0 
    for image_list, target in metric_logger.log_every(data_loader, 10, header):
    # for iter, (image_list, target) in enumerate(data_loader):
        
        images1 = image_list[0].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images1.shape[0]
        img_name = img_list[index].strip()
        index += 1

        img_temp = images1.permute(0, 2, 3, 1).detach().cpu().numpy()
        orig_images = np.zeros_like(img_temp)
        orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.
        orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.
        orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.

        w_orig, h_orig = orig_images.shape[1], orig_images.shape[2]
        # w, h = images1.shape[2] - images1.shape[2] % args.patch_size, images1.shape[3] - images1.shape[3] % args.patch_size
        # w_featmap = w // args.patch_size
        # h_featmap = h // args.patch_size


        with torch.cuda.amp.autocast():
            cam_list = []
            atten_w_list = []
            patch_list = []
            vitattn_list = []
            cam_maps = None
            for s in range(len(image_list)):
                images = image_list[s].to(device, non_blocking=True)
                w, h = images.shape[2] - images.shape[2] % args.patch_size, images.shape[3] - images.shape[3] % args.patch_size
                w_featmap = w // args.patch_size
                h_featmap = h // args.patch_size
                # breakpoint()
                if 'MCTformerV1' in args.model:
                    output, cls_attentions, patch_attn = model(images, return_att=True, n_layers=args.layer_index)
                    cls_attentions = cls_attentions.reshape(batch_size, args.nb_classes, w_featmap, h_featmap)
                    patch_attn = torch.sum(patch_attn, dim=0)

                elif 'MCTformerV2' in args.model:
                    output, cls_attentions, patch_attn = model(images, return_att=True, n_layers=args.layer_index, attention_type=args.attention_type)
                    patch_attn = torch.sum(patch_attn, dim=0)    # 平均 12 个 head
                    
                elif 'MCTformerV3' in args.model:
                    output, cls_attentions, patch_attn = model(images, return_att=True, n_layers=args.layer_index, attention_type=args.attention_type)
                    patch_attn = torch.sum(patch_attn, dim=0)    # 平均 12 个 head

                if args.patch_attn_refine:
                    cls_attentions = torch.matmul(patch_attn.unsqueeze(1), cls_attentions.view(cls_attentions.shape[0],cls_attentions.shape[1], -1, 1)).reshape(cls_attentions.shape[0],cls_attentions.shape[1], w_featmap, h_featmap)
                # breakpoint()
                # continue
                cls_attentions = F.interpolate(cls_attentions, size=(w_orig, h_orig), mode='bilinear', align_corners=False)[0]
                cls_attentions = cls_attentions.cpu().numpy() * target.clone().view(args.nb_classes, 1, 1).cpu().numpy()
                
                if s % 2 == 1:
                    cls_attentions = np.flip(cls_attentions, axis=-1)
                cam_list.append(cls_attentions)
     
                vitattn_list.append(cam_maps)

            sum_cam = np.sum(cam_list, axis=0)
            sum_cam = torch.from_numpy(sum_cam)
            sum_cam = sum_cam.unsqueeze(0).to(device)

            output = torch.sigmoid(output)

        if args.visualize_cls_attn:
            for b in range(images.shape[0]):
                if (target[b].sum()) > 0:
                    cam_dict = {}
                    # feature_dict = {}
                    for cls_ind in range(args.nb_classes):
                        if target[b,cls_ind]>0:
                            cls_score = format(output[b, cls_ind].cpu().numpy(), '.3f')

                            cls_attention = sum_cam[b,cls_ind,:]

                            cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
                            cls_attention = cls_attention.cpu().numpy()

                            cam_dict[cls_ind] = cls_attention
                           
                            if args.attention_dir is not None and len(os.listdir(args.attention_dir))<100:
                                fname = os.path.join(args.attention_dir, img_name + '_' + str(cls_ind) + '_' + str(cls_score) + '.png')
                                show_cam_on_image(orig_images[b], cls_attention, fname)
                            # breakpoint()
                            # tensor = np.zeros((21, h, w),np.float32)
                            # tensor[0,:,:] = 0.4 
                            # predict = np.argmax(tensor, axis=0).astype(np.uint8)
                            # c x h x w
                            # atten_w = F.interpolate(torch.from_numpy(cls_attention).unsqueeze(dim=0).unsqueeze(dim=0), size=patch_list[0].shape[-2:], mode='bilinear', align_corners=False)[0]
                            # atten_w = atten_w*(atten_w > 0.4)
                            # feature_cls = (atten_w * patch_list[0][0].detach().cpu()).sum(dim=-1).sum(dim=-1) / atten_w.squeeze().sum()
                            # feature_dict[cls_ind] = feature_cls



                    # # visualize the salience area
                    # if args.visualize_salience:
                    #     # cls_score = format(output[b, 80].cpu().numpy(), '.3f')

                    #     cls_attention = sum_cam[b, 80,:]

                    #     cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
                    #     cls_attention = cls_attention.cpu().numpy()

                    #     # cam_dict[80] = cls_attention 

                    #     if args.attention_dir is not None and len(os.listdir(args.attention_dir))<100:
                    #         fname = os.path.join(args.attention_dir, img_name + '_' + str("salience") + '.png')
                    #         show_cam_on_image(orig_images[b], cls_attention, fname)
                    
                    # np.save(os.path.join("MCTformer_results/MCTformer_v3/VOC12/Seed-1-Baseline_gpu_1_bs_64/VOC2012_All_Cls/fetaure_dict", img_name + '.npy'), feature_dict)
                    
                    if args.cam_npy_dir is not None:
                        np.save(os.path.join(args.cam_npy_dir, img_name + '.npy'), cam_dict)
                
                    if args.out_crf is not None:
                        for t in [args.low_alpha, args.high_alpha]:
                            orig_image = orig_images[b].astype(np.uint8).copy(order='C')
                            crf = _crf_with_alpha(cam_dict, t, orig_image)
                            folder = args.out_crf + ('_%s' % t)
                            if not os.path.exists(folder):
                                os.makedirs(folder)
                            # print(os.path.join(folder, img_name + '.npy'))
                            np.save(os.path.join(folder, img_name + '.npy'), crf)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return


def _crf_with_alpha(cam_dict, alpha, orig_img):
    from psa.tool.imutils import crf_inference
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = dict()

    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al


def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)