import math
from functools import partial
from collections import deque
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from cv2 import log
from numpy import mask_indices
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model

from func import *
from vision_transformer import VisionTransformer, _cfg

__all__ = [
    'deit_small_MCTformerV1_patch16_224', 'deit_small_MCTformerV2_patch16_224', 'deit_small_MCTformerV3_patch16_224'
]

class MCTformerV2(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head.apply(self._init_weights)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_classes, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        print(self.training)

    def interpolate_pos_encoding(self, x, w, h):
        # breakpoint() 
        npatch = x.shape[1] - self.num_classes                 # 196
        N = self.pos_embed.shape[1] - self.num_classes         # 196
        # breakpoint()
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        attn_weights = []

        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights.append(weights_i)

        return x[:, 0:self.num_classes], x[:, self.num_classes:], attn_weights

    def forward(self, x, target=None, return_att=False, n_layers=12, attention_type='fused'):
        w, h = x.shape[2:]
        x_cls, x_patch, attn_weights = self.forward_features(x)
        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)
        x_patch_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

        feature_map = x_patch.detach().clone()  # B * C * 14 * 14
        feature_map = F.relu(feature_map)

        n, c, h, w = feature_map.shape

        mtatt = attn_weights[-n_layers:].sum(0)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w])

        if attention_type == 'fused':
            cams = mtatt * feature_map  # B * C * 14 * 14
        elif attention_type == 'patchcam':
            cams = feature_map
        else:   # 'transcam'
            cams = mtatt
        
        # cams = torch.ones_like(cams).cuda()
        
        patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]

        x_cls_logits = x_cls.mean(-1)

        if return_att:
            return x_cls_logits, cams, patch_attn
        else:
            return x_cls_logits, x_patch_logits

# class MCTformerV2(VisionTransformer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#         self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.head.apply(self._init_weights)
#         num_patches = self.patch_embed.num_patches
#         # class_token + salience token
#         self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes + 1, self.embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_classes + 1, self.embed_dim)) 

#         trunc_normal_(self.cls_token, std=.02)
#         trunc_normal_(self.pos_embed, std=.02)
#         print(self.training)

#     def interpolate_pos_encoding(self, x, w, h):
#         # breakpoint()
#         npatch = x.shape[1] - self.num_classes - 1                 # 196
#         N = self.pos_embed.shape[1] - self.num_classes - 1         # 196
#         # breakpoint()
#         if npatch == N and w == h:
#             return self.pos_embed
#         # else:
#         #     print('check the patch nums.')
#         class_pos_embed = self.pos_embed[:, 0:self.num_classes+1]
#         patch_pos_embed = self.pos_embed[:, self.num_classes+1:]
#         dim = x.shape[-1]

#         w0 = w // self.patch_embed.patch_size[0]
#         h0 = h // self.patch_embed.patch_size[0]
#         # we add a small number to avoid floating point error in the interpolation
#         # see discussion at https://github.com/facebookresearch/dino/issues/8
#         w0, h0 = w0 + 0.1, h0 + 0.1
#         patch_pos_embed = nn.functional.interpolate(
#             patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
#             scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
#             mode='bicubic',
#         )
#         assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
#         patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
#         return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

#     def forward_features(self, x, n=12):
#         B, nc, w, h = x.shape
#         x = self.patch_embed(x)

#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.interpolate_pos_encoding(x, w, h)
#         x = self.pos_drop(x)
#         attn_weights = []

#         for i, blk in enumerate(self.blocks):
#             x, weights_i = blk(x)
#             attn_weights.append(weights_i)

#         return x[:, 0:self.num_classes], x[:, self.num_classes+1:], attn_weights

#     def forward(self, x, return_att=False, n_layers=12, attention_type='fused'):
#         # breakpoint()
#         w, h = x.shape[2:]
#         # x_cls
#         # x_patch : bs x 196 x 384
#         # attn_weights  : bs x n_head x (n_patch + n_cls + 1) x (n_patch + n_cls + 1)
#         x_cls, x_patch, attn_weights = self.forward_features(x)
#         n, p, c = x_patch.shape
        
#         if w != h:
#             w0 = w // self.patch_embed.patch_size[0]
#             h0 = h // self.patch_embed.patch_size[0]
#             x_patch = torch.reshape(x_patch, [n, w0, h0, c])
#         else:
#             x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
            
#         x_patch = x_patch.permute([0, 3, 1, 2])
#         x_patch = x_patch.contiguous()
#         x_patch = self.head(x_patch)
#         x_patch_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

#         attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
#         attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

#         feature_map = x_patch.detach().clone()  # B * C * 14 * 14
#         feature_map = F.relu(feature_map)
#         # breakpoint()
#         n, c, h, w = feature_map.shape

#         # B x C x H x W
#         mtatt = attn_weights[-n_layers:].sum(0)[:, 0:self.num_classes+1, self.num_classes+1:].reshape([n, c+1, h, w])

#         if attention_type == 'fused':
#             cams = mtatt * feature_map  # B * C * 14 * 14
#         elif attention_type == 'patchcam':
#             cams = feature_map
#         else:  # 'transcam'
#             cams = mtatt
#         # 12 x B x C x HW
#         patch_attn = attn_weights[:, :, self.num_classes+1:, self.num_classes+1:]   # N_head x B x HW x HW
#         # 
#         # torch.matmul(patch_attn.unsqueeze(1), cls_attentions.view(cls_attentions.shape[0],cls_attentions.shape[1], -1, 1))
#         x_cls_logits = x_cls.mean(-1)

#         if return_att:
#             return x_cls_logits, cams, patch_attn
#         else:
#             return x_cls_logits, x_patch_logits

class MCTformerV1(VisionTransformer):
    def __init__(self, last_opt='average', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_opt = last_opt
        if last_opt == 'fc':
            self.head = nn.Conv1d(in_channels=self.num_classes, out_channels=self.num_classes, kernel_size=self.embed_dim, groups=self.num_classes)
            self.head.apply(self._init_weights)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_classes, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        print(self.training)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed.shape[1] - self.num_classes
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)

        attn_weights = []

        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            if len(self.blocks) - i <= n:
                attn_weights.append(weights_i)
        return x[:, 0:self.num_classes], attn_weights

    def forward(self, x, n_layers=12, return_att=False):
        x, attn_weights = self.forward_features(x)

        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
        mtatt = attn_weights[-n_layers:].sum(0)[:, 0:self.num_classes, self.num_classes:]
        patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]

        x_cls_logits = x.mean(-1)

        if return_att:
            return x_cls_logits, mtatt, patch_attn
        else:
            return x_cls_logits


# DPTR
class MCTformerV3(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head.apply(self._init_weights)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_classes, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        # print(self.training)
        print('DPTR\n'*10)
        self.gcl = self.global_contrastive_Loss
        self.gcl_t = kwargs['gcl_t']
        self.gcl_blk = kwargs['blk']
        
        self.multi_ins_kd_t = kwargs['multi_kd_t']
        
        print('gcl_blk = {}'.format(self.gcl_blk) * 6)
        
        # self.register_buffer("prototypes", torch.ones(self.num_classes, self.embed_dim))
        # self.class_dict = {}
        # # self.class_prototypes = {}
        # for i in range(self.num_classes):
        #     self.class_dict[i] = {}
        #     self.class_dict[i]['ins_feat'] = deque(maxlen=500)
        #     self.class_dict[i]['ins_area'] = deque(maxlen=500)
        #     # self.class_dict[i]['prototypes'] = []

    def cal_ins_area(self, x, feature_map, mtatt, patch_attn, one_hot_label, epoch):

        B, C, H, W = mtatt.shape
        
        # B x HW x dim 
        x = x.reshape(B, -1, H*W).transpose(2, 1)
        
        label = [torch.nonzero(one_hot_label[i]).reshape(-1) for i in range(B)]
        patch_attn = torch.sum(patch_attn, dim=0)    # B x HW x HW

        # MCTformer三个组件联合得到最终 cam_w
        # B x 1 x HW x HW , B x C x HW x 1  --> B x C x HW x 1
        cam_w = torch.matmul(patch_attn.unsqueeze(1), (mtatt*feature_map).view(B, C, -1, 1))
        # B x C x HW 
        cam_w = cam_w.squeeze()
        cam_w = min_max(cam_w.reshape(B, C, -1))

        # # B x C x HW , B x HW x dim  --> B x C x dim
        # coefficient_matrix = torch.ones((B, self.num_classes)).cuda()

        # instance_feats = torch.matmul(cam_w, x)
        # # # breakpoint()
        # for i in range(B):
        #     instance_feat =instance_feats[i].detach().cpu()  # C x dim
        #     ins_areas = cam_w[i].detach().cpu()              # C x HW
        #     for k in label[i]:
        #         self.class_dict[k.item()]["ins_feat"].append(instance_feat[k])
        #         self.class_dict[k.item()]["ins_area"].append(ins_areas[k].mean())
        #         # mu = torch.mean(torch.Tensor(list(self.class_dict[k.item()]["ins_area"])))
        #         # sigma = torch.std(torch.Tensor(list(self.class_dict[k.item()]["ins_area"])))
        #         # coefficient_matrix[i][k] = torch.exp(-1*(ins_areas[k].mean()-mu)/(sigma*10000))
        # if epoch > 30:
        #     for i in range(self.num_classes):
        #         if (len(self.class_dict[i]["ins_feat"]) == 500):
        #             feat_array = np.array([np.array(i) for i in list(self.class_dict[i]["ins_feat"])])
        #             # ins_feats = torch.Tensor(np.array(array_list))
        #             cluster = KMeans(n_clusters=1).fit(feat_array)
        #             self.prototypes[i] = (torch.from_numpy(cluster.cluster_centers_).cuda()).squeeze()
        # return coefficient_matrix
    
    def global_contrastive_Loss(self, x, weights, label):
        
        # # ======================= 2022-10-22 ， 计算cls-token与背景之间的对比损失
        # # 索引出仅包含单个实例的图像
        single_instance_img_id = ~(label.sum(dim=1)-1.0).bool()  # 只考虑带单个实例的图像 
       
        x = x[single_instance_img_id]
        weights = weights[single_instance_img_id]
        label = label[single_instance_img_id]

        fg_tokens, bg_tokens, cls_token, labels_idx = get_fg_bg(x, weights.mean(dim=1), label, t=self.gcl_t)   # bs x c, bs x c

        sim_pp = cos_sim(fg_tokens, cls_token, temp=1.0)  # Ins_N x Ins_N 
        # sim_pn = cos_sim(bg_tokens, cls_token, temp=1.0)  # Img_N x Ins_N , ------------------------1
        sim_pn = cos_sim(bg_tokens, fg_tokens, temp=1.0)  # Img_N x Ins_N     ------------------------2
        
        # mask = torch.eq(label.view(-1, 1), label.view(-1, 1).t()).float()  # image_label
        mask = torch.eq(labels_idx.view(-1, 1), labels_idx.view(-1, 1).t()).float()   # instance_label
        # sim_pp_mask = (sim_pp * mask)
        # sim_pp_mask = sim_pp_mask.masked_fill(mask == 0.0, 1.0)  # 将不是同一类的前景样本乘积（0.0）填充为1，这样就不参与任何优化
        
        # log_prob = -1 * (torch.log(1. - sim_pn).mean() - (torch.log(1. - sim_pp + 0.00001)*mask).mean())
        log_prob_pn = -1 * (torch.log(1. - sim_pn).mean())   # 0.0001为防止数值计算出错
        
        # # # 最小化cls-token与bg-token的相似度，会导致不同类别的cls-token相似度变弱，尝试加入最小化不同cls-token的相似度
        # # sim_clstoken = cos_sim(cls_token, cls_token, temp=1.0) # Ins_N x Ins_N
        # # log_prob_pp = -1 * ((torch.log(1. - sim_clstoken + 0.00001) * (~mask)).sum() / (~mask).sum()) 

        # ======================= 2022-10-23, 计算cls-token之间的对比损失
        # 索引出仅包含单个实例的图像
        single_instance_img_id = ~(label.sum(dim=1)-1.0).bool()  # 只考虑带单个实例的图像 
       
        x = x[single_instance_img_id]
        weights = weights[single_instance_img_id]
        label = label[single_instance_img_id]
        
        bs, n_cls = label.shape
        cls_token = x[:, :n_cls, :]                  # bs x n_cls x c
        
        # breakpoint()
        real_idx_list = [torch.nonzero(label[i]).reshape(-1) for i in range(bs)] # (>bs)
        fake_idx_list = [torch.nonzero((~label[i].bool()).long()).reshape(-1) for i in range(bs)]
        
        real_cls_token = [cls_token[i][real_idx_list[i]] for i in range(bs)]    
        fake_cls_token = [cls_token[i][fake_idx_list[i]] for i in range(bs)]  
        
        real_fg_feat = torch.cat(real_cls_token, dim=0)       # N x c
        fake_fg_feat = torch.cat(fake_cls_token, dim=0)       # M x c
        sim_rf = cos_sim(real_fg_feat, fake_fg_feat, temp=1.0) 
        log_prob_pn = -1 * (torch.log(1. - sim_rf).mean())
        
        return  log_prob_pn
    
    
    def distillation_loss(self, attn_weights, feat_map, label):
        single_instance_img_id = ~(label.sum(dim=1)-1.0).bool()  # 只考虑带单个实例的图像 
        # breakpoint()
        # x = x[single_instance_img_id]
        attn_weights = attn_weights[-6:].sum(0)
        attn_weights = attn_weights[single_instance_img_id]
        label = label[single_instance_img_id]
        feat_map = feat_map[single_instance_img_id]
        
        n, c, h, w = feat_map.shape
        real_idx_list = [torch.nonzero(label[i]).reshape(-1) for i in range(n)] # (>bs)
        
        attn = attn_weights[:, 0:self.num_classes, self.num_classes:] # B x C x HW
        attn_map = [attn[i][real_idx_list[i]] for i in range(n)]    
        attn_map = torch.cat(attn_map, dim=0)    # Ins_N x HW
        attn_map = min_max(attn_map).detach()           # attn_map 作为伪标签
        
        feat_map = feat_map.reshape(n, c, h*w)
        cam_map = [feat_map[i][real_idx_list[i]] for i in range(n)]    
        cam_map = torch.cat(cam_map, dim=0)      # Ins_N x HW
        cam_map = min_max(cam_map)
        # breakpoint()
        gamma = 0.7                         # gamma 趋向于0时，就是MCTFormer也即baseline
        
        mask = ((cam_map < gamma) & (attn_map < gamma))
        
        distill_loss = torch.nn.MSELoss()(attn_map*mask, cam_map*mask)
        
        return distill_loss
    
    def multi_ins_kd_loss(self, attn_weights, feat_map, label):
        multi_instance_img_id = (label.sum(dim=1)-1.0).bool()  # 只考虑带单个实例的图像 
        # breakpoint()

        attn_weights = attn_weights[-6:].sum(0)
        attn_weights = attn_weights[multi_instance_img_id]
        label = label[multi_instance_img_id]
        feat_map = feat_map[multi_instance_img_id]
        
        n, c, h, w = feat_map.shape
        # real_idx_list = [torch.nonzero(label[i]).reshape(-1) for i in range(n)] # (>bs)
        
        attn_map = attn_weights[:, 0:self.num_classes, self.num_classes:] # B x C x HW
        # attn_map = [attn_map[i][real_idx_list[i]] for i in range(n)]    
        # attn_map = torch.cat(attn_map, dim=0)    # Ins_N x HW
        attn_map = min_max(attn_map)
        # attn_map = F.softmax(attn_map, dim=1)     # B x C x HW
        attn_map = attn_map.transpose(2, 1)       # B x HW x C
        attn_map = attn_map.reshape(-1, c)        # BHW x C
        
        cam_map = feat_map.reshape(n, c, h*w)
        # cam_map = [feat_map[i][real_idx_list[i]] for i in range(n)]    
        # cam_map = torch.cat(cam_map, dim=0)      # Ins_N x HW
        cam_map = min_max(cam_map)
        cam_map = cam_map.detach()
        cam_map = cam_map.transpose(2, 1).reshape(-1, c)  # BHW x C 
        
        mask = (cam_map.max(dim=1)[0] > 0.10).float().unsqueeze(dim=-1) # BHW x 1
        # breakpoint()
    
        cam_map = F.softmax(cam_map*self.multi_ins_kd_t, dim=1)     # B x C x HW
        
        log_prob = nn.functional.log_softmax(attn_map, dim=1)

        mkd_loss = - torch.sum(log_prob * cam_map * mask) / (mask.sum()+0.0001)
        
        # cam_map = (cam_map.detach()>0.4).float()
        
        return mkd_loss
    
    # generate suitable pos_encoding for arbitary shape
    def interpolate_pos_encoding(self, x, w, h):
        # breakpoint() 
        npatch = x.shape[1] - self.num_classes                 # 196
        N = self.pos_embed.shape[1] - self.num_classes         # 196

        if npatch == N and w == h:
            return self.pos_embed
        
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation, see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x, labels, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        attn_weights = []

        gcl_loss = []
        # print(len(self.blocks))
        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights.append(weights_i)

            if (i>=self.gcl_blk) and self.training and labels is not None:
                # gcl_loss.append(self.gcl(x, weights_i, labels))
                gcl_loss.append(torch.Tensor([0.]).cuda())
            if not self.training:
                gcl_loss.append(torch.Tensor([0.]).cuda())
                
        return x[:, 0:self.num_classes], x[:, self.num_classes:], attn_weights, sum(gcl_loss)/len(gcl_loss)

    def forward(self, x, target=None, return_att=False, retun_att_loss=False, n_layers=12, attention_type='fused', epoch=0):
        w, h = x.shape[2:]
        x_cls, x_patch, attn_weights, gcl_loss = self.forward_features(x, target)
        n, p, c = x_patch.shape
        
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
            
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch_p = self.head(x_patch)
        x_patch_logits = self.avgpool(x_patch_p).squeeze(3).squeeze(2)
        
        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

        # feature_map = x_patch.detach().clone()  # B * C * 14 * 14
        feature_map = x_patch_p
        feature_map = F.relu(feature_map)

        n, c, h, w = feature_map.shape

        mtatt = attn_weights[-n_layers:].sum(0)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w])
        patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]

        # calculate the instance_area and feature
        # if self.training:
        #     coefficient_matrix = self.cal_ins_area(x_patch, feature_map, mtatt, patch_attn, target, epoch)
        
        if attention_type == 'fused':
            # B x C x H x W, B x CLS x C
            # breakpoint()
            # feature_map = torch.matmul(F.normalize(self.prototypes.unsqueeze(dim=0).repeat(n, 1, 1), dim=-1), F.normalize(x_patch.reshape(n, 384, h*w), dim=1))
            # feature_map = feature_map.reshape(n, c, h, w)
            # cams = feature_map
            # print(feature_map[:2, 0, 7:8, 3:10])
            cams = mtatt * feature_map  # B * C * 14 * 14
        elif attention_type == 'patchcam':
            cams = feature_map
        else:   # 'transcam'
            cams = mtatt
        
        # cams = torch.ones_like(cams).cuda()

        x_cls_logits = x_cls.mean(-1)

        if return_att:
            return x_cls_logits, cams, patch_attn
        elif retun_att_loss:
            # distill_loss = self.distillation_loss(attn_weights, feature_map.detach(), target)
            # mkd_loss = self.multi_ins_kd_loss(attn_weights, feature_map.detach(), target)
            
            # loss_dict = {}
            # loss_dict['gcl'] = torch.Tensor([0.]).cuda()
            # loss_dict['distill'] = torch.Tensor([0.]).cuda()
            # loss_dict['multi_kd'] = torch.Tensor([0.]).cuda()
            
            return x_cls_logits, x_patch_logits, cams, patch_attn
        else:
            return x_cls_logits, x_patch_logits
            

@register_model
def deit_small_MCTformerV3_patch16_224(pretrained=False, **kwargs):
    model = MCTformerV3(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def deit_small_MCTformerV2_patch16_224(pretrained=False, **kwargs):
    model = MCTformerV2(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def deit_small_MCTformerV1_patch16_224(pretrained=False, **kwargs):
    model = MCTformerV1(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    return model