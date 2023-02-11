import os
import random
from copy import deepcopy

import numpy as np
import PIL.Image
import torch
import torchvision
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision import transforms

from func import *


def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list

def load_image_label_list_from_npy(img_name_list, label_file_path=None):
    if label_file_path is None:
        label_file_path = 'voc12/cls_labels.npy'
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = []
    for id in img_name_list:
        if id not in cls_labels_dict.keys():
            img_name = id + '.jpg'
        else:
            img_name = id
        label_list.append(cls_labels_dict[img_name])
    return label_list
    # return [cls_labels_dict[img_name] for img_name in img_name_list ]

class COCOClsDataset(Dataset):
    def __init__(self, img_name_list_path, coco_root, label_file_path, train=True, transform=None, gen_attn=False):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        self.coco_root = coco_root
        self.transform = transform
        self.train = train
        self.gen_attn = gen_attn

    def __getitem__(self, idx):
        # breakpoint()
        name = self.img_name_list[idx]
        if self.train or self.gen_attn :
            img = PIL.Image.open(os.path.join(self.coco_root, 'train2014', name + '.jpg')).convert("RGB")
        else:
            img = PIL.Image.open(os.path.join(self.coco_root, 'val2014', name + '.jpg')).convert("RGB")

        label = torch.from_numpy(self.label_list[idx])
        if self.transform:
            img = self.transform(img)
        return img, label
        #
        # if self.train:
        #     if self.FixMatch:
        #         # 先按照正常方式读取出图片：既包括多实例图像也包括单实例图像
        #         name = self.img_name_list[idx]
        #         img = PIL.Image.open(os.path.join(self.coco_root, 'train2014', name + '.jpg')).convert("RGB")
        #         label = torch.from_numpy(self.label_list[idx])
        #         if self.transform:
        #             img = self.transform(img)

        #         # --------------------- 读取多实例图像 , 及其 mask---------------------      #
        #         idx_multi = (idx+np.random.randint(0, self.__len__()))%self.__len__()
        #         name_multi = self.img_name_list[idx_multi]
        #         img_multi = PIL.Image.open(os.path.join(self.coco_root, 'train2014', name_multi + '.jpg')).convert("RGB")
        #         label_multi = torch.from_numpy(self.label_list[idx_multi])
                
        #         mlti_anno_mask_dict = np.array(Image.open(os.path.join('MCTformer_results/MCTformer_v3/COCO', \
        #             '%s.png'%name_multi)))
        #         multi_bg_mask = mlti_anno_mask_dict==0
        #         multi_bg_mask = Image.fromarray(multi_bg_mask)

        #         #  ------------------------------- 读取出单实例图像, 及其 mask -------------------------------   #
        #         label_single_, name_single = self.generate_cls_sample(torch.nonzero(label_multi).reshape(-1))
        #         img_single = PIL.Image.open(os.path.join(self.coco_root, 'train2014', name_single + '.jpg')).convert("RGB")
        #         label_single = torch.nn.functional.one_hot(torch.Tensor([label_single_]).long(), 20).float().squeeze()

        #         # ---------------------              pseudo mask                  --------------------- # 
        #         single_anno_mask_dict = np.array(Image.open(os.path.join('MCTformer_results/MCTformer_v3/COCO', \
        #             '%s.png'%name_single)))
        #         single_fg_mask = single_anno_mask_dict==(label_single_+1)
        #         # breakpoint()
        #         single_bg_mask = ~single_fg_mask
        #         single_fg_mask = Image.fromarray(single_fg_mask)
        #         single_bg_mask = Image.fromarray(single_bg_mask)

        #         # # 添加旋转增强操作， 指定逆时针旋转的角度
        #         # img_single = img_single.rotate(self.degrees[idx % len(self.degrees)]) 
        #         # single_anno_mask = single_anno_mask.rotate(self.degrees[idx % len(self.degrees)]) 
        #         # im_rotate.show()

        #         # 对 single_img 和 multi_img 做 weak_transform
        #         assert single_fg_mask is not None
        #         img_single, single_fg_mask, single_bg_mask = resize(img_single, single_fg_mask, single_bg_mask, size=256)
        #         img_single, single_fg_mask, single_bg_mask = crop(img_single, single_fg_mask,  single_bg_mask, size=224)
        #         img_single, single_fg_mask, single_bg_mask = hflip(img_single, single_fg_mask, single_bg_mask, p=0.5)

        #         img_multi, multi_bg_mask, multi_bg_mask = resize(img_multi, multi_bg_mask, multi_bg_mask, size=256)
        #         img_multi, multi_bg_mask, multi_bg_mask = crop(img_multi, multi_bg_mask, multi_bg_mask, size=224)
        #         img_multi, multi_bg_mask, multi_bg_mask = hflip(img_multi, multi_bg_mask, multi_bg_mask, p=0.5)
                
        #         # single_img 和 multi_img 拷贝出来用于 strong_transform
        #         img_single_w, img_single_s= deepcopy(img_single), deepcopy(img_single)
        #         img_multi_w_fg, img_multi_w_bg, img_multi_s= deepcopy(img_multi), deepcopy(img_multi), deepcopy(img_multi)
                
        #         if self.scale_aug:
        #             scale_size = np.random.uniform(0.6, 1.0)
        #             img_single_s, scale_single_anno_mask=resize(img_single_s, single_fg_mask, size=int(224*scale_size))
        #             single_fg_mask = obtain_cut_box(np.array(scale_single_anno_mask), thre=0.4)
        #         else:
        #             scale_size = 1.0 
        #             single_fg_mask = obtain_cut_box(np.array(single_fg_mask), thre=0.4)
        #             # breakpoint()
        #             multi_bg_mask = obtain_cut_box(np.array(multi_bg_mask), thre=0.4)
        #             single_bg_mask = (1 - single_fg_mask) * multi_bg_mask

        #         img_single = normalize(img_single)
        #         img_single_w = normalize(img_single_w)
        #         img_single_s = normalize(img_single_s)
                
        #         img_multi_w_fg = normalize(img_multi_w_fg)
        #         img_multi_w_bg = normalize(img_multi_w_bg)
        #         img_multi_s = normalize(img_multi_s)

        #         # ------------------------ 将 single_instance_fg 随机 paste 到 multi_img 上 -------------------------- #
        #         fg_paste_w, fg_paste_h = int(scale_size*224), int(scale_size*224)
        #         fg_paste_position_x1, fg_paste_position_y1 = int((random.randint(0, int(224 - fg_paste_w))//16)*16), \
        #             int((random.randint(0, int(224 - fg_paste_h))//16)*16)
        #         fg_paste_position_x2, fg_paste_position_y2 = fg_paste_position_x1 + fg_paste_w, \
        #             fg_paste_position_y1 + fg_paste_h

        #         img_multi_w_fg[:, fg_paste_position_x1 : fg_paste_position_x2, fg_paste_position_y1 : fg_paste_position_y2]\
        #             [single_fg_mask.unsqueeze(0).expand((3, fg_paste_w, fg_paste_h)) == 1] = \
        #             img_single_s[single_fg_mask.unsqueeze(0).expand(3, fg_paste_w, fg_paste_h) == 1]

        #         label_multi_w_fg= (label_multi.bool() | label_single.bool()).long()
                
        #         fg_paste_position = np.array([fg_paste_position_x1, fg_paste_position_y1, \
        #             fg_paste_position_x2, fg_paste_position_y2, fg_paste_w, fg_paste_h, scale_size])
        #         single_fg_mask_ = torch.zeros(224, 224)
        #         single_fg_mask_[:fg_paste_w, :fg_paste_h] = single_fg_mask

        #         # ------------------------ 将 single_instance_bg 随机 paste 到 single_img 上 -------------------------- #
        #         bg_paste_w, bg_paste_h = int(scale_size*224), int(scale_size*224)
        #         bg_paste_position_x1, bg_paste_position_y1 = int((random.randint(0, int(224 - bg_paste_w))//16)*16), \
        #             int((random.randint(0, int(224 - bg_paste_h))//16)*16)
        #         bg_paste_position_x2, bg_paste_position_y2 = bg_paste_position_x1 + bg_paste_w, bg_paste_position_y1 + bg_paste_h
                
        #         img_multi_w_bg[:, bg_paste_position_x1 : bg_paste_position_x2, bg_paste_position_y1 : bg_paste_position_y2]\
        #             [single_bg_mask.unsqueeze(0).expand((3, bg_paste_w, bg_paste_h)) == 1] = \
        #             img_single_s[single_bg_mask.unsqueeze(0).expand(3, bg_paste_w, bg_paste_h) == 1]
                
        #         label_multi_w_bg = label_multi
                
        #         bg_paste_position = np.array([bg_paste_position_x1, bg_paste_position_y1, bg_paste_position_x2, bg_paste_position_y2, \
        #             bg_paste_w, bg_paste_h, scale_size])
        #         single_bg_mask_ = torch.zeros(224, 224)
        #         single_bg_mask_[:bg_paste_w, :bg_paste_h] = single_bg_mask

                
        #         return img, label, \
        #             img_single, label_single, \
        #                 img_multi_w_fg, label_multi_w_fg, single_fg_mask_, fg_paste_position, \
        #                     img_multi_w_bg, label_multi_w_bg, single_bg_mask_, bg_paste_position
        #     else:
        #         name = self.img_name_list[idx]
        #         img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        #         label = torch.from_numpy(self.label_list[idx])
                
        #         if self.transform:
        #             img = self.transform(img)

                # return img, label
        # else:
        #     name = self.img_name_list[idx]
        #     img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        #     label = torch.from_numpy(self.label_list[idx])
            
        #     if self.transform:
        #         img = self.transform(img)
                
            # return img, label

    def __len__(self):
        return len(self.img_name_list)

class COCOClsDatasetMS(Dataset):
    def __init__(self, img_name_list_path, coco_root, label_file_path, part, scales, train=True, transform=None, gen_attn=False, unit=1):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val"}_id.txt')
        # breakpoint()
        self.img_name_list = load_img_name_list(img_name_list_path)[part*10000 : (part+1)*10000]
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        self.coco_root = coco_root
        self.transform = transform
        self.train = train
        self.unit = unit
        self.scales = scales
        self.gen_attn = gen_attn
        
        self.existing_files = []
        for file in os.listdir('MCTformer_results/MCTformer_v2/COCO2014/attn-patchrefine-npy'):
            self.existing_files.append(file[:27])
    def __getitem__(self, idx):
        breakpoint()
        # print(idx)
        while self.img_name_list[idx] in self.existing_files:
            print('{} has been done'.format(self.img_name_list[idx]))
            idx += 1
        name = self.img_name_list[idx]
        if self.train or self.gen_attn:
            img = PIL.Image.open(os.path.join(self.coco_root, 'train2014', name + '.jpg')).convert("RGB")
        else:
            img = PIL.Image.open(os.path.join(self.coco_root, 'val2014', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        rounded_size = (int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])

            # msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))
            
        return msf_img_list, label

    def __len__(self):
        return len(self.img_name_list)


class VOC12Dataset(Dataset):
    def __init__(self, img_name_list_path, voc12_root, train=True, transform=None, gen_attn=False, FixMatch=False, strong_transform=None):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.voc12_root = voc12_root
        self.transform = transform
        # self.gt_anno = "/data/code/WSSS/VOC2012/SegmentationClassAug"
        self.single_cls_name_list, self.single_cls_label_list = self.filter_single_cls_image_list(self.img_name_list, self.label_list)
        self.scale_aug = False
        self.strong_transform = strong_transform   # 如果要使用FixMatch实验，务必设置为True
        self.train = train
        self.FixMatch = FixMatch
        self.degrees = [0, 30, 60, 90, 0]
        self.class_id_dict=self.get_class_dict()
        self.class_sample_p = self.coupling_degree_estimation()
    
    def __getitem__(self, idx):
        if self.train:
            if self.FixMatch:
                # 先按照正常方式读取出图片：既包括多实例图像也包括单实例图像
                name = self.img_name_list[idx]
                img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
                label = torch.from_numpy(self.label_list[idx])
                if self.transform:
                    img = self.transform(img)

                # --------------------- 读取多实例图像 , 及其 mask---------------------      #
                idx_multi = (idx+np.random.randint(0, self.__len__()))%self.__len__()
                name_multi = self.img_name_list[idx_multi]
                img_multi = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name_multi + '.jpg')).convert("RGB")
                label_multi = torch.from_numpy(self.label_list[idx_multi])
                
                mlti_anno_mask_dict = np.array(Image.open(os.path.join('MCTformer_results/MCTformer_v3/VOC12/Seed-1-Baseline_gpu_1_bs_64/pgt-psa-rw-0.36-label', \
                    '%s.png'%name_multi)))
                multi_bg_mask = mlti_anno_mask_dict==0
                multi_bg_mask = Image.fromarray(multi_bg_mask)

                #  ------------------------------- 读取出单实例图像, 及其 mask -------------------------------   #
                label_single_, name_single = self.generate_cls_sample(torch.nonzero(label_multi).reshape(-1))
                img_single = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name_single + '.jpg')).convert("RGB")
                # label_single = torch.from_numpy(self.single_cls_label_list[idx % len(self.single_cls_name_list)])
                label_single = torch.nn.functional.one_hot(torch.Tensor([label_single_]).long(), 20).float().squeeze()

                # ---------------------              pseudo mask                  --------------------- # 
                single_anno_mask_dict = np.array(Image.open(os.path.join('MCTformer_results/MCTformer_v3/VOC12/Seed-1-Baseline_gpu_1_bs_64/pgt-psa-rw-0.36-label', \
                    '%s.png'%name_single)))
                single_fg_mask = single_anno_mask_dict==(label_single_+1)
                # breakpoint()
                single_bg_mask = ~single_fg_mask
                single_fg_mask = Image.fromarray(single_fg_mask)
                single_bg_mask = Image.fromarray(single_bg_mask)

                # # 添加旋转增强操作， 指定逆时针旋转的角度
                # img_single = img_single.rotate(self.degrees[idx % len(self.degrees)]) 
                # single_anno_mask = single_anno_mask.rotate(self.degrees[idx % len(self.degrees)]) 
                # im_rotate.show()

                # 对 single_img 和 multi_img 做 weak_transform
                assert single_fg_mask is not None
                img_single, single_fg_mask, single_bg_mask = resize(img_single, single_fg_mask, single_bg_mask, size=256)
                img_single, single_fg_mask, single_bg_mask = crop(img_single, single_fg_mask,  single_bg_mask, size=224)
                img_single, single_fg_mask, single_bg_mask = hflip(img_single, single_fg_mask, single_bg_mask, p=0.5)

                img_multi, multi_bg_mask, multi_bg_mask = resize(img_multi, multi_bg_mask, multi_bg_mask, size=256)
                img_multi, multi_bg_mask, multi_bg_mask = crop(img_multi, multi_bg_mask, multi_bg_mask, size=224)
                img_multi, multi_bg_mask, multi_bg_mask = hflip(img_multi, multi_bg_mask, multi_bg_mask, p=0.5)
                
                # single_img 和 multi_img 拷贝出来用于 strong_transform
                img_single_w, img_single_s= deepcopy(img_single), deepcopy(img_single)
                img_multi_w_fg, img_multi_w_bg, img_multi_s= deepcopy(img_multi), deepcopy(img_multi), deepcopy(img_multi)
                
                # if self.strong_transform is not None:
                #     img_single_s = self.strong_transform(img_single_s)
                #     img_single_s = blur(img_single_s, p=0.5)
                    
                #     img_multi_s = self.strong_transform(img_multi_s)
                #     img_multi_s = blur(img_multi_s, p=0.5)
                # else:
                #     pass
                
                if self.scale_aug:
                    scale_size = np.random.uniform(0.6, 1.0)
                    img_single_s, scale_single_anno_mask=resize(img_single_s, single_fg_mask, size=int(224*scale_size))
                    single_fg_mask = obtain_cut_box(np.array(scale_single_anno_mask), thre=0.4)
                else:
                    scale_size = 1.0 
                    single_fg_mask = obtain_cut_box(np.array(single_fg_mask), thre=0.4)
                    # breakpoint()
                    multi_bg_mask = obtain_cut_box(np.array(multi_bg_mask), thre=0.4)
                    single_bg_mask = (1 - single_fg_mask) * multi_bg_mask

                img_single = normalize(img_single)
                img_single_w = normalize(img_single_w)
                img_single_s = normalize(img_single_s)
                
                img_multi_w_fg = normalize(img_multi_w_fg)
                img_multi_w_bg = normalize(img_multi_w_bg)
                img_multi_s = normalize(img_multi_s)
                # breakpoint()
                # ------------------------ 将 single_instance_fg 随机 paste 到 multi_img 上 -------------------------- #
                fg_paste_w, fg_paste_h = int(scale_size*224), int(scale_size*224)
                fg_paste_position_x1, fg_paste_position_y1 = int((random.randint(0, int(224 - fg_paste_w))//16)*16), \
                    int((random.randint(0, int(224 - fg_paste_h))//16)*16)
                fg_paste_position_x2, fg_paste_position_y2 = fg_paste_position_x1 + fg_paste_w, \
                    fg_paste_position_y1 + fg_paste_h

                img_multi_w_fg[:, fg_paste_position_x1 : fg_paste_position_x2, fg_paste_position_y1 : fg_paste_position_y2]\
                    [single_fg_mask.unsqueeze(0).expand((3, fg_paste_w, fg_paste_h)) == 1] = \
                    img_single_s[single_fg_mask.unsqueeze(0).expand(3, fg_paste_w, fg_paste_h) == 1]

                label_multi_w_fg= (label_multi.bool() | label_single.bool()).long()
                
                fg_paste_position = np.array([fg_paste_position_x1, fg_paste_position_y1, \
                    fg_paste_position_x2, fg_paste_position_y2, fg_paste_w, fg_paste_h, scale_size])
                single_fg_mask_ = torch.zeros(224, 224)
                single_fg_mask_[:fg_paste_w, :fg_paste_h] = single_fg_mask

                # ------------------------ 将 single_instance_bg 随机 paste 到 single_img 上 -------------------------- #
                bg_paste_w, bg_paste_h = int(scale_size*224), int(scale_size*224)
                bg_paste_position_x1, bg_paste_position_y1 = int((random.randint(0, int(224 - bg_paste_w))//16)*16), \
                    int((random.randint(0, int(224 - bg_paste_h))//16)*16)
                bg_paste_position_x2, bg_paste_position_y2 = bg_paste_position_x1 + bg_paste_w, bg_paste_position_y1 + bg_paste_h
                
                img_multi_w_bg[:, bg_paste_position_x1 : bg_paste_position_x2, bg_paste_position_y1 : bg_paste_position_y2]\
                    [single_bg_mask.unsqueeze(0).expand((3, bg_paste_w, bg_paste_h)) == 1] = \
                    img_single_s[single_bg_mask.unsqueeze(0).expand(3, bg_paste_w, bg_paste_h) == 1]
                
                label_multi_w_bg = label_multi
                
                bg_paste_position = np.array([bg_paste_position_x1, bg_paste_position_y1, bg_paste_position_x2, bg_paste_position_y2, \
                    bg_paste_w, bg_paste_h, scale_size])
                single_bg_mask_ = torch.zeros(224, 224)
                single_bg_mask_[:bg_paste_w, :bg_paste_h] = single_bg_mask

                
                return img, label, \
                    img_single, label_single, \
                        img_multi_w_fg, label_multi_w_fg, single_fg_mask_, fg_paste_position, \
                            img_multi_w_bg, label_multi_w_bg, single_bg_mask_, bg_paste_position
            else:
                name = self.img_name_list[idx]
                img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
                label = torch.from_numpy(self.label_list[idx])
                
                if self.transform:
                    img = self.transform(img)

                return img, label
        else:
            name = self.img_name_list[idx]
            img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
            label = torch.from_numpy(self.label_list[idx])
            
            if self.transform:
                img = self.transform(img)
                
            return img, label

    def __len__(self):
        return len(self.img_name_list)

        # filter the image with only single class
    def filter_single_cls_image_list(self, img_name_list, label_list):
        new_image_list = []
        new_label_list = []
        for i in range(len(img_name_list)):
            if(label_list[i].sum()==1.0):  # 单类别图像
                new_image_list.append(img_name_list[i])
                new_label_list.append(label_list[i])
        return new_image_list, new_label_list
    
    def generate_cls_sample(self, label_multi):

        class_candidates = np.arange(20)

        cls_id_list = np.array(label_multi)
        init_p = np.zeros((1, 20))
        for i in cls_id_list:
            init_p = init_p + self.class_sample_p[i]
        prob = init_p / init_p.sum()

        # 按照指定概率 prob进行采样
        cls_id = np.random.choice(a=class_candidates, size=1, replace=True, p=prob.squeeze()).item()
        img_name = np.random.choice(self.class_id_dict[cls_id])

        return cls_id, img_name
    
    def get_class_dict(self):
        class_dict = {}
        for idx in range(20):
            class_dict[idx] = []
        
        # 仅收集单类别图像数据集
        for i, item in enumerate(self.label_list):
            if item.sum() == 1.0:
                idxs = torch.nonzero(torch.Tensor(item)).reshape(-1)
                for idx in idxs:
                    class_dict[idx.item()].append(self.img_name_list[i])
        for idx in range(20):
            print(idx, len(class_dict[idx]))
        return class_dict

    def coupling_degree_estimation(self):
        # 统计各个类别拥有的单类别图像数目
        mat = np.ones((20, 20))
        # 
        for i , group in enumerate(self.label_list):
            cls_idx_list = torch.nonzero(torch.Tensor(group)).reshape(-1)
            length = len(cls_idx_list)
            if length==1:
                mat[cls_idx_list[0]][cls_idx_list[0]] += 1
            else:
                for i in range(length):
                    for j in range(i+1, length):
                        mat[cls_idx_list[i]][cls_idx_list[j]] +=1
                        mat[cls_idx_list[j]][cls_idx_list[i]] +=1

        # class_prior_p = mat / mat.sum(axis=0)
        class_prior_p = np.array(F.softmax(torch.Tensor(mat)*0.1, dim=1))

        return class_prior_p


class VOC12DatasetMS(Dataset):
    def __init__(self, img_name_list_path, voc12_root, scales, train=True, transform=None, gen_attn=False, unit=1):
        # img_name_list_path = os.path.join(img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

        # # 筛选出只有单类别的图像
        # print('Is filtering the image with only single class')
        # self.img_name_list, self.label_list = self.filter_single_cls_image_list(self.img_name_list, self.label_list)
        # print('The image with only single class is {}'.format(len(self.img_name_list)))
        
        # file_write_obj = open("voc12/sing_cls_id.txt", 'w')
        # for var in self.img_name_list:
        #     file_write_obj.writelines(var)
        #     file_write_obj.write('\n')
        # file_write_obj.close()
        # breakpoint()
        
        # 筛选出2个类别的图像
        # print('Is filtering the image with multi classes')
        # self.img_name_list, self.label_list = self.filter_two_cls_image_list(self.img_name_list, self.label_list)
        # print('The image with two classes is {}'.format(len(self.img_name_list)))
        
        # file_write_obj = open("voc12/two_cls_id.txt", 'w')
        # for var in self.img_name_list:
        #     file_write_obj.writelines(var)
        #     file_write_obj.write('\n')
        # file_write_obj.close()
        # breakpoint()
        
        
        self.voc12_root = voc12_root
        self.transform = transform
        self.unit = unit
        self.scales = scales
        
    
    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        rounded_size = (int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])

            # msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))
        return msf_img_list, label

    # filter the image with only single class
    def filter_single_cls_image_list(self, img_name_list, label_list):
        new_image_list = []
        new_label_list = []
        for i in range(len(img_name_list)):
            if(label_list[i].sum()==1.0):  # 单类别图像
                new_image_list.append(img_name_list[i])
                new_label_list.append(label_list[i])
        return new_image_list, new_label_list
    
    # filter the image with two classes
    def filter_two_cls_image_list(self, img_name_list, label_list):
        new_image_list = []
        new_label_list = []
        for i in range(len(img_name_list)):
            if(label_list[i].sum()==2.0):  # 多类别图像
                new_image_list.append(img_name_list[i])
                new_label_list.append(label_list[i])
        return new_image_list, new_label_list
    
    # filter the image with three classes
    def filter_three_cls_image_list(self, img_name_list, label_list):
        new_image_list = []
        new_label_list = []
        for i in range(len(img_name_list)):
            if(label_list[i].sum()==3.0):  # 多类别图像
                new_image_list.append(img_name_list[i])
                new_label_list.append(label_list[i])
        return new_image_list, new_label_list
    
    # filter the image with four classes
    def filter_four_cls_image_list(self, img_name_list, label_list):
        new_image_list = []
        new_label_list = []
        for i in range(len(img_name_list)):
            if(label_list[i].sum()==4.0):  # 多类别图像
                new_image_list.append(img_name_list[i])
                new_label_list.append(label_list[i])
        return new_image_list, new_label_list
    
    def __len__(self):
        return len(self.img_name_list)


def build_dataset(is_train, args, gen_attn=False, FixMatch=False):
    transform = build_transform(is_train, args)
    if FixMatch:
        strong_transform = build_strong_transform(is_train, args)
        FixMatch = True
    else:
        strong_transform = None
        FixMatch = False
        
    dataset = None
    nb_classes = None

    if args.data_set == 'VOC12':
        # dataset = VOC12Dataset(img_name_list_path=args.img_list, voc12_root=args.data_path,
        #                        train=is_train, gen_attn=gen_attn, transform=transform)
        # dataset_multi = VOC12Dataset(img_name_list_path=args.img_list, voc12_root=args.data_path,
        #                        train=is_train, gen_attn=gen_attn, transform=transform, weak_strong_transform=weak_strong_transform)
        dataset = VOC12Dataset(img_name_list_path=args.img_list, voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform, FixMatch=FixMatch, strong_transform=strong_transform)
        nb_classes = 20
        
    elif args.data_set == 'VOC12MS':
        dataset = VOC12DatasetMS(img_name_list_path=args.img_list, voc12_root=args.data_path, scales=tuple(args.scales),
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 20
    elif args.data_set == 'COCO':
        dataset = COCOClsDataset(img_name_list_path=args.img_list, coco_root=args.data_path, label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 80
    elif args.data_set == 'COCOMS':
        dataset = COCOClsDatasetMS(img_name_list_path=args.img_list, coco_root=args.data_path, scales=tuple(args.scales), label_file_path=args.label_file_path, part=args.part,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 80

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im and not args.gen_attention_maps:
        size = int((256/ args.input_size) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def build_strong_transform(is_train, args):
    size = int((256 / 224) * args.input_size)   # 256, 320
    
    weak_strong_transform = [transforms.Resize(size, interpolation=3), 
                             transforms.CenterCrop(args.input_size), 
                             transforms.RandomHorizontalFlip(p=0.5)
                             ]
    
    strong_transform =[transforms.ColorJitter(0.5, 0.5, 0.5, 0.25), transforms.RandomGrayscale(p=0.2)]
    
    return transforms.Compose(strong_transform)

