import math
import random
from cProfile import label

import cv2
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms


# disentangle feature
def get_fg_bg(tokens, weights, label=None, t=1.0):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    # tokens :        bs x (n_cls+hw) x c
    # weights :       bs x (n_cls+hw) x (n_cls+hw)
    # label :         bs x n_cls
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    
    # bs x c
    bs, n_cls = label.shape
    cls_token = tokens[:, :n_cls, :]                  # bs x n_cls x c 
    # cls_token = cls_token * label.unsqueeze(dim=-1)   
    # bs x N x c
    patch_token = tokens[:, n_cls:, :]             # bs x N x c

    cls_gnostic_weight = weights[:, :n_cls, n_cls:]            # bs x n_cls x N
    weight_min, weight_max = cls_gnostic_weight.min(dim=-1)[0].unsqueeze(dim=-1), cls_gnostic_weight.max(dim=-1)[0].unsqueeze(dim=-1)
    
    # ------ min_max
    cls_gnostic_weight = (cls_gnostic_weight - weight_min)/(weight_max - weight_min + 0.000001)  
    # ------ pow()
    # fg_weight = torch.pow(input=(fg_weight + 0.000001), exponent=pow)                 
    # ------ sigmoid
    cls_gnostic_weight = torch.sigmoid((cls_gnostic_weight-0.5) * t)                     # 注意 t 这个温度系数, -0.5 这个参数不咋影响结果
    # breakpoint()
    
    # index_list
    idx_list = [torch.nonzero(label[i]).reshape(-1) for i in range(bs)] # (>bs)
    # breakpoint()
    # need to max_pooing along the class_channel to find the background region weight
    cls_gnostic_weight_list = [cls_gnostic_weight[i][idx_list[i]] for i in range(bs)]   # 取出有类别的 cls-weight
    
    # bs x 1 x N
    cls_agnostic_weight = [nn.MaxPool1d(cls_gnostic_weight_list[i].shape[0])(cls_gnostic_weight_list[i].transpose(1, 0)).transpose(1, 0) for i in range(bs)]
    # cls_agnostic_weight = nn.MaxPool1d(n_cls)(cls_gnostic_weight.transpose(2, 1)).transpose(2, 1)  # bs x 1 x N # BUG !
    # breakpoint()
    # print(len(cls_agnostic_weight), bs)
    cls_agnostic_weight = torch.cat(cls_agnostic_weight, dim=0)
    
    # bs x n_cls x N
    # cls_gnostic_weight = cls_gnostic_weight.unsqueeze(dim=1) 
    # bs x n_cls x N  *  bs x N x c = bs x n_cls x c
    fg_feature = torch.matmul(cls_gnostic_weight, patch_token) / patch_token.shape[1]       # divide Patch_num N 
    # bs x 1 x c
    # bg_feature = torch.matmul(1.- cls_agnostic_weight, patch_token) / patch_token.shape[1]   # divide Patch_num N 
    bg_feature = ((1.- cls_agnostic_weight).unsqueeze(dim=-1) * patch_token).mean(dim=1)      # divide Patch_num N
    # bg_feature = torch.matmul(1.- cls_gnostic_weight, patch_token) / patch_token.shape[1]   # divide Patch_num N 
    
    fg_feature = [fg_feature[i][idx_list[i]] for i in range(bs)]  # (>bs) x c
    cls_token = [cls_token[i][idx_list[i]] for i in range(bs)]    # (>bs) x c
    
    fg_feature = torch.cat(fg_feature, dim=0)  # [instance_11, instance_12, instance_21, ....]
    cls_token = torch.cat(cls_token, dim=0)    # [instance_11, instance_12, instance_21, ....]
    
    idx_flat = torch.cat(idx_list, dim=-1)     # [instance_11, instance_12, instance_21, ....]
    
    return fg_feature, bg_feature, cls_token, idx_flat

# cos_similarity
def cos_sim(embedded_fg, embedded_bg, temp):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim =  torch.div(torch.matmul(embedded_fg, embedded_bg.T), temp)
    # if sim.max() > 1.0:
    #         breakpoint()
    return torch.clamp(sim, min=-0.9995, max=0.9995)
    # return sim

def min_max(weights):
    weight_min, weight_max = weights.min(dim=-1)[0].unsqueeze(dim=-1), weights.max(dim=-1)[0].unsqueeze(dim=-1)
    weights_cam = (weights - weight_min)/(weight_max - weight_min + 0.000001)  
    return weights_cam



# 图片短边缩放至x，长宽比保持不变
def resize(img, mask_1=None, mask_2=None, size=None):
    w, h = img.size
    short_side = size

    if h < w:
        oh = short_side
        ow = int(1.0 * w * short_side / h + 0.5)
    else:
        ow = short_side
        oh = int(1.0 * h * short_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    if mask_1 is None:
        return img
    else:
        mask_1 = mask_1.resize((ow, oh), Image.NEAREST)
        mask_2 = mask_2.resize((ow, oh), Image.NEAREST)
        return img, mask_1, mask_2

def crop(img, mask_1=None, mask_2=None, size=None):
    w, h = img.size
    # padw = size - w if w < size else 0
    # padh = size - h if h < size else 0
    # img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    # mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

    w, h = img.size
    # x = 0
    # y = 0
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    
    if mask_1 is None:
        return img
    else:
        mask_1 = mask_1.crop((x, y, x + size, y + size))
        mask_2 = mask_2.crop((x, y, x + size, y + size))
        return img, mask_1, mask_2


def hflip(img, mask_1=None, mask_2=None, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if mask_1 is None:
            return img
        else:
            mask_1 = mask_1.transpose(Image.FLIP_LEFT_RIGHT)
            mask_2 = mask_2.transpose(Image.FLIP_LEFT_RIGHT)
            return img, mask_1, mask_2
    else:
        if mask_1 is None:
            return img
        else:
            return img, mask_1, mask_2

def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

def obtain_cut_box(anno_mask, thre=0.3, p=0.5, size_min=0.2, size_max=0.3, ratio_1=0.5, ratio_2=1/0.5):
    img_size = anno_mask.shape[0]
    
    mask = torch.zeros(img_size, img_size)

    # 2. cut_mask
    mask[anno_mask>thre] = 1
    
    # rand mask
    # mask_one = torch.randint(0, 10, (14,14)).repeat_interleave(16, 0).repeat_interleave(16, 1)
    # mask = mask * (mask_one>=0)
    return mask

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def get_bboxes(cam, cam_thr=0.3):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO
                                    )
    #thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
   
    else:
        estimated_bbox = [0, 0, 1, 1]
    
    return estimated_bbox


class Pixel_Purity(nn.Module):
    def __init__(self, args, padding_mode='zeros'):
        """
        purity_conv: size*size
        entropy_conv: size*size
        """
        super(Pixel_Purity, self).__init__()
        self.args = args
        self.in_channels = args.nb_classes
        size = args.k_size
        # used for foreground copy-and-paste
        self.fg_purity_conv = nn.Conv2d(in_channels=self.in_channels+1, \
                                        out_channels=self.in_channels+1, \
                                        kernel_size=size, \
                                        stride=1, 
                                        padding=int(size / 2), 
                                        bias=False,
                                        padding_mode=padding_mode, 
                                        groups=self.in_channels+1)
        # used for background copy-and-paste
        self.bg_purity_conv = nn.Conv2d(in_channels=self.in_channels+1, \
                                        out_channels=self.in_channels+1, \
                                        kernel_size=size, \
                                        stride=1, 
                                        padding=int(size / 2), 
                                        bias=False,
                                        padding_mode=padding_mode, 
                                        groups=self.in_channels+1)

        weight = torch.ones((size, size), dtype=torch.float32)
        weight = weight.unsqueeze(dim=0).unsqueeze(dim=0)
        
        fg_weight = weight.repeat([self.in_channels+1, 1, 1, 1])
        bg_weight = weight.repeat([self.in_channels+1, 1, 1, 1])
        
        fg_weight = nn.Parameter(fg_weight)
        bg_weight = nn.Parameter(bg_weight)
        
        self.fg_purity_conv.weight = fg_weight
        self.bg_purity_conv.weight = bg_weight
        
        self.fg_purity_conv.requires_grad_(False)
        self.bg_purity_conv.requires_grad_(False)

    def forward(self, fg_cam=None, fg_grid=None, bg_cam=None, bg_grid=None):

        bg_score = (torch.ones_like(fg_cam[:, 0]) * 0.4).unsqueeze(dim=1)  # B x 1 X H x W
        fg_cam = torch.cat([fg_cam, bg_score], dim=1)                 # B x (C+1) x H x W
        bg_cam = torch.cat([bg_cam, bg_score], dim=1)                 # B x (C+1) x H x W

        fg_cam = fg_cam / fg_cam.sum(dim=1).unsqueeze(dim=1)
        bg_cam = bg_cam / bg_cam.sum(dim=1).unsqueeze(dim=1)

        # fg_pixel_weight
        b, cls_num, h, w = fg_cam.shape
        summary = self.fg_purity_conv(fg_cam)  # [b, cls_num, h, w]
        count = torch.sum(summary, dim=1, keepdim=True)  # [b, 1, h, w]
        dist = summary / count  # [b, cls_num, h, w]
        fg_purity = torch.sum(-dist * torch.log(dist + 1e-6), dim=1, keepdim=True) / math.log(cls_num)  # [b, 1, h, w]
        fg_purity = fg_purity.squeeze()
        # breakpoint()
        inf_tensor = torch.ones_like(fg_purity)*(-10000)
        fg_pixel_weight = torch.where(fg_grid==0, inf_tensor, fg_purity) 
        fg_pixel_weight = F.softmax(fg_pixel_weight.reshape(b, -1)*self.args.purity_t, dim=-1).reshape(b, h, w)

        # bg_cam
        b, cls_num, h, w = bg_cam.shape
        summary = self.bg_purity_conv(bg_cam)  # [b, cls_num, h, w]
        count = torch.sum(summary, dim=1, keepdim=True)  # [b, 1, h, w]
        dist = summary / count  # [b, cls_num, h, w]
        bg_purity = torch.sum(-dist * torch.log(dist + 1e-6), dim=1, keepdim=True) / math.log(cls_num)  # [b, 1, h, w]
        bg_purity = bg_purity.squeeze()
        inf_tensor = torch.ones_like(bg_purity)*(-10000)
        bg_pixel_weight = torch.where(bg_grid==0, inf_tensor, bg_purity) 
        bg_pixel_weight = F.softmax(bg_pixel_weight.reshape(b, -1)*self.args.purity_t, dim=-1).reshape(b, h, w)
        
        return fg_pixel_weight, bg_pixel_weight


def cal_cam(cls_attention, label, channel_min_max=True):
    
    bs, c, h, w = cls_attention.shape
    assert c==20, print('the class num error')
    cls_attention = cls_attention.reshape(bs, c, -1)
    # print(torch.min(cls_attention, dim=1, keepdim=True)[0].shape)
    # print(torch.min(cls_attention, dim=-1, keepdim=True)[0].shape)
    
    # 沿着类别通道 Min_Max,  bs x c x hw
    cls_channel_attn = (cls_attention - torch.min(cls_attention, dim=1, keepdim=True)[0]) \
    / (torch.max(cls_attention, dim=1, keepdim=True)[0] - torch.min(cls_attention, dim=1, keepdim=True)[0] + 1e-4)
    
    # 沿着空间通道 Min_Max, bs x c x hw
    space_channel_attn = (cls_attention - torch.min(cls_attention, dim=-1, keepdim=True)[0]) \
    / (torch.max(cls_attention, dim=-1, keepdim=True)[0] - torch.min(cls_attention, dim=-1, keepdim=True)[0] + 1e-4)   # 修改1e-8为1e-4， 否则在混合精度下容易报错
    
    # bs x 14 x 14
    # cls_cam = [space_channel_attn[i][label[i]].reshape(-1, 14, 14) for i in range(bs)]
    # torch.nonzero(label[i]).reshape(-1) 是返回one-hot标签对应的 label_idx
    cls_cam = [torch.max(space_channel_attn[i][torch.nonzero(label[i]).reshape(-1)], dim=0)[0].reshape(-1, 14, 14) for i in range(bs)]
    # bs x 14 x 14
    cls_idx = [torch.nonzero(label[i]).reshape(-1)[torch.max(space_channel_attn[i][torch.nonzero(label[i]).reshape(-1)], dim=0)[1]].reshape(-1, 14, 14)\
         for i in range(bs)]

    return cls_channel_attn.reshape(bs, c, h, w), space_channel_attn.reshape(bs, c, h, w), cls_cam

    # if channel_min_max:
    #     return cls_channel_attn.reshape(bs, c, h, w), cls_cam, cls_idx
    # else:
    #     return space_channel_attn.reshape(bs, c, h, w), cls_cam, cls_idx

class EMA_CAM():
    def __init__(self, class_num=20, init_value=0, thre=0.4, momentum=0.999):
        super(EMA_CAM, self).__init__()
        self.ema_cam = {}
        self.momentum = momentum
        self.thre = thre
        self.class_num = class_num
        self.init_value = init_value

        for cls in range(self.class_num+1):
            self.ema_cam[cls] = self.init_value

    def update(self, cams, labels):
        cams = cams.detach()
        labels = labels.detach()
        bs, _, _, _ = cams.shape
        # breakpoint()
        for i in range(bs):
            tmp_cls_list = torch.nonzero(labels[i]).reshape(-1)
            tmp_cls = tmp_cls_list[0].item()
            tmp_cam = cams[i][tmp_cls]
            mask = tmp_cam>(self.thre)
            tmp_mean = (tmp_cam*mask).sum()/(mask.sum()+0.0001)
            # tmp_mean_list = []
            # for tmp_cls in tmp_cls_list:
            self.ema_cam[tmp_cls+1] = self.ema_cam[tmp_cls+1]*self.momentum + (1-self.momentum)*tmp_mean
            self.ema_cam[0] = self.ema_cam[0]*self.momentum + (1-self.momentum)*tmp_mean
        
    def re_initialize(self):
        for cls in range(self.class_num+1):
            self.ema_cam[cls] = self.init_value
    
def vis_img(img):
    
    img_temp = img.permute(0, 2, 3, 1).detach().cpu().numpy()
    orig_images = np.zeros_like(img_temp)
    orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.
    orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.
    orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.
        
    pass

if __name__ == "__main__":
    tokens, weights = torch.rand(2, 100+196, 384), torch.rand(2, 100+196, 296)
    label = torch.zeros(2, 100).long()
    label[0, 1] = 1
    label[1, 2] = 1
    label[1, 4] = 1
    get_fg_bg(tokens, weights, label)
