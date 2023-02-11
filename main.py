from torch.utils.tensorboard import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler

import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path
import pickle
import shutil
import numpy as np
import wandb

import models
import utils
from datasets import build_dataset
from engine import (evaluate, generate_attention_maps_ms, self_train_one_epoch,
                    train_one_epoch, vanilla_train_one_epoch)
from func import Pixel_Purity, EMA_CAM

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=60, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')


    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='', type=str, help='dataset path')
    parser.add_argument('--img-list', default='', type=str, help='image list path')
    parser.add_argument('--data-set', default='', type=str, help='dataset')


    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)


    # generating attention maps
    parser.add_argument('--gen_attention_maps', action='store_true')
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--attention-dir', type=str, default=None)
    parser.add_argument('--layer-index', type=int, default=12, help='extract attention maps from the last layers')

    parser.add_argument('--patch-attn-refine', action='store_true')
    parser.add_argument('--visualize-cls-attn', action='store_true')

    parser.add_argument('--gt-dir', type=str, default=None)
    parser.add_argument('--cam-npy-dir', type=str, default=None)
    parser.add_argument("--scales", nargs='+', type=float)
    parser.add_argument('--label-file-path', type=str, default=None)
    parser.add_argument('--attention-type', type=str, default='fused')

    parser.add_argument('--out-crf', type=str, default=None)
    parser.add_argument("--low_alpha", default=1, type=int)
    parser.add_argument("--high_alpha", default=12, type=int)
    
    parser.add_argument("--part", default=0, type=int)
    # salience 
    parser.add_argument('--salience', action='store_true')
    parser.add_argument('--visualize_salience', action='store_true')
    
    # 
    parser.add_argument('--gcl_t', type=float, default=1.0,
                        help='global constrative loss sigmoid temperature(1.0)')
    parser.add_argument('--gcl_lambda', type=float, default=0.0, help='coefficient of global constrative loss')
    parser.add_argument('--blk', type=int, default=0, help='from blk block use gcl loss')
    parser.add_argument('--distill_lambda', type=float, default=0.0, help='coefficient of distill loss')
    parser.add_argument('--multi_kd_lambda', type=float, default=0.0, help='coefficient of multi-instance distill loss')
    parser.add_argument('--multi_kd_t', type=float, default=1.0, help='kd loss temperature(1.0)')
    
    # for fixmatch
    parser.add_argument('--fixmatch_fg_thre', type=float, default=0.5, help='choose foreground pusedo label for fixmatch')
    parser.add_argument('--fixmatch_bg_thre', type=float, default=0.5, help='choose background pusedo label for fixmatch')
    parser.add_argument('--fixmatch_temp', type=float, default=1.0, help='temperature for fixmatch_fg_loss')
    parser.add_argument('--fixmatch_fg_lambda', type=float, default=1.0, help='coefficient of fixmatch_fg_loss')
    parser.add_argument('--fixmatch_bg_lambda', type=float, default=1.0, help='coefficient of fixmatch_bg_loss')
    
    parser.add_argument('--start_ep', type=int, default=0, help='from which epoch to use gcl loss')
    
    parser.add_argument('--comment', type=str, default='base', help='help distinguish experiment')
    parser.add_argument('--seed', type=int, default=0, help='seed for reproduce')

    # purity-based reweighting
    parser.add_argument('--purity_t', type=float, default=1.0, help='temperature for purity')
    parser.add_argument('--k_size', type=int, default=1, help='kernel_size for computing purity')

    return parser


def main(args):

    print(args)


    SEED = args.seed
    def worker_init_fn(worker_id):
        random.seed(SEED + worker_id)
        
    utils.fix_seed_all(seed=SEED)   # for reproducing the results !!!!!!!!!!!!!!!!!
    g = torch.Generator()
    g.manual_seed(SEED)
    
    
    device = torch.device(args.device)
    # cudnn.benchmark = True  # https://zhuanlan.zhihu.com/p/370743229

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args, FixMatch=True)
    # dataset_train, args.nb_classes = build_dataset(is_train=True, args=args, FixMatch=False)
    # dataset_train_multi, args.nb_classes = build_dataset(is_train=True, args=args, FixMatch=True, Cut=False)
    # dataset_train_single, args.nb_classes = build_dataset(is_train=True, args=args, FixMatch=True, Cut=True)
    
    dataset_train_, args.nb_classes = build_dataset(is_train=False, gen_attn=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=g,
        persistent_workers=args.num_workers>0
    )
    # data_loader_train_multi = torch.utils.data.DataLoader(
    #     dataset_train_multi, 
    #     sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    #     worker_init_fn=worker_init_fn,
    #     generator=g,
    #     persistent_workers=args.num_workers>0
    # )
    # data_loader_train_single = torch.utils.data.DataLoader(
    #     dataset_train_single, 
    #     sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    #     worker_init_fn=worker_init_fn,
    #     generator=g,
    #     persistent_workers=args.num_workers>0
    # )

    data_loader_train_ = torch.utils.data.DataLoader(
        dataset_train_,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val,
        batch_size=int(2 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"Creating model: {args.model}")


    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        gcl_t=args.gcl_t,
        blk=args.blk,
        multi_kd_t=args.multi_kd_t
    )

    # breakpoint()
    if args.finetune:
        # 读取预训练好的 checkpoint, 这里是 deit
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        try:
            checkpoint_model = checkpoint['model']
        except:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed'] # [1, 197, 384]
        embedding_size = pos_embed_checkpoint.shape[-1]      
        num_patches = model.patch_embed.num_patches          # 196
        if args.finetune.startswith('https'):                # True
            num_extra_tokens = 1
        else:
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)  # 14

        new_size = int(num_patches ** 0.5)
        # MCTFormer把原先的CLs-token的位置编码复制多遍
        if args.finetune.startswith('https') and 'MCTformer' in args.model:
            if args.salience:
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens].repeat(1,args.nb_classes+1,1)  # +1表示对应添加salience的pos embedding
            else:
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens].repeat(1,args.nb_classes,1)
        else:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)  # [1, 384, 14, 14]
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed
        # MCTFormer把原先的CLs-token的参数也复制多遍
        if args.finetune.startswith('https') and 'MCTformer' in args.model:
            cls_token_checkpoint = checkpoint_model['cls_token']
            if args.salience:
                new_cls_token = cls_token_checkpoint.repeat(1,args.nb_classes+1,1)           # +1表示对应添加salience的token embedding
            else:
                new_cls_token = cls_token_checkpoint.repeat(1,args.nb_classes,1) 
                
            checkpoint_model['cls_token'] = new_cls_token

        model.load_state_dict(checkpoint_model, strict=False)
    # breakpoint()
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    # # 注意这里学习率的设置 # # 
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    output_dir = Path(os.path.join(args.output_dir, args.data_set, args.comment+'_gpu_{}'.format(utils.get_world_size())+'_bs_{}'.format(args.batch_size)))


    # 如果是评测模式，就直接评测结果
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"mAP of the network on the {len(dataset_val)} test images: {test_stats['mAP']*100:.1f}%")
        return
    
    # 生成弱监督分割 map
    if args.gen_attention_maps:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        generate_attention_maps_ms(data_loader_train_, model, device, args)
        return

    # exit if having carry out the same experiment
    if os.path.exists(output_dir):
        if 'Debug' in str(output_dir) or 'checkpoint_resume.pth' in os.listdir(output_dir):
            pass
        else:
            print('WARNING: This experiment {} has existed.\n\n'.format(output_dir) * 10)
            sys.exit(-1) 
    else:
        print('mkdir output_dir {}.'.format(output_dir))
        os.makedirs(output_dir)

    # 保存部分代码在当前实验文件夹路径下
    output_code_dir = os.path.join(output_dir, 'backup')
    if not os.path.exists(output_code_dir):
        os.mkdir(output_code_dir)
    for name in ['engine.py', 'dataset.py', 'func.py', 'evaluation.py', 'main.py', 'models.py']:
        shutil.copy('/data/code/WSSS/MCTformer/engine.py', os.path.join(output_code_dir, name))

    #################################################### tensorboard ####################################################
    if args.output_dir and utils.is_main_process():
        writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    else:
        writer = None
    #################################################### tensorboard ####################################################
    
    exp_name = os.path.join(args.data_set, args.comment+'_gpu_{}'.format(utils.get_world_size())+'_bs_{}'.format(args.batch_size))
    config = wandb.config
    wandb.init(project='MCTformer',  name=exp_name, dir="/data/code/WSSS/MCTformer", config=config)
    wandb.config.update(args)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    start_iter = 0
    
    # 从断点续训
    if os.path.exists(os.path.join(output_dir, 'checkpoint_resume.pth')):
        resume_ckpt = torch.load(os.path.join(output_dir, 'checkpoint_resume.pth'))
        if 'epoch' in resume_ckpt.keys():
            args.start_epoch = resume_ckpt['epoch'] + 1
        else:
            resume_ckpt['epoch'] = 20
            args.start_epoch = resume_ckpt['epoch'] + 1
        start_iter += len(data_loader_train)*resume_ckpt['epoch']
        model.load_state_dict(resume_ckpt['model'])
        optimizer.load_state_dict(resume_ckpt['optimizer'])
        print("=> loaded checkpoint (epoch {})".format(resume_ckpt['epoch']))
    else:
        print('{} not exists'.format(str(os.path.join(output_dir, 'checkpoint_resume.pth'))))
    
    PPNet = Pixel_Purity(args=args).cuda()
    EMA_C = EMA_CAM(class_num=args.nb_classes)
    for epoch in range(args.start_epoch, args.epochs):
        # loader = zip(data_loader_train, data_loader_train_multi, data_loader_train_single)
        # model.class_dict = {}
        # self.class_prototypes = {}
        # for i in range(model.num_classes):
        #     model.class_dict[i] = {}
        #     model.class_dict[i]['ins_feat'] = [] 
        #     model.class_dict[i]['ins_area'] = []
        #     model.class_dict[i]['prototypes'] = []
            
        train_stats, start_iter = train_one_epoch(
            model, 
            data_loader_train,
            optimizer, 
            device, 
            epoch, 
            loss_scaler,
            args.clip_grad,
            args=args,
            writer=writer,
            start_iter=start_iter,
            PPNet=PPNet,
            EMA_CAM=EMA_C
        )
        lr_scheduler.step(epoch)
        
        # 断点保存, 以便之后断点续训
        checkpoint_resume = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint_resume, output_dir / 'checkpoint_resume.pth')
        
        # 保存warm_up的模型
        if epoch == (args.start_ep-1):
            torch.save({'model': model.state_dict()}, os.path.join(output_dir, 'checkpoint_warmup.pth'))

        # if (epoch % 5)==0:
        #     #     torch.save({'model': model.state_dict()}, os.path.join(output_dir, 'checkpoint_{}.pth'.format(epoch)))
        #     # 保存ema_cam
        #     np.save(os.path.join(output_dir, 'ema_cam_{}.npy'.format(epoch)), EMA_C.ema_cam)
        #     EMA_C.re_initialize()

        # 保存最优的模型
        test_stats = evaluate(data_loader_val, model, device)
        print(f"mAP of the network on the {len(dataset_val)} test images: {test_stats['mAP']*100:.1f}%")
        if test_stats["mAP"] > max_accuracy and args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_best.pth']
            for checkpoint_path in checkpoint_paths:
                torch.save({'model': model.state_dict()}, checkpoint_path)
            
        max_accuracy = max(max_accuracy, test_stats["mAP"])
        print(f'Max mAP: {max_accuracy * 100:.2f}%')

        #################################################### tensorboard ####################################################
        if utils.is_main_process():
            # writer.add_scalar('Test/epoch', epoch, epoch)
            writer.add_scalar('Test/cls_top1_acc', test_stats["mAP"], epoch)
            writer.add_scalar('Test/cls_top1_acc_max', max_accuracy, epoch)
            wandb.log({
                'epoch':epoch, 
                'cls_top1_acc': test_stats["mAP"],
                'cls_top1_acc_max': max_accuracy
            })
        #################################################### tensorboard ####################################################
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    

    # # 保存每个类别对应的实例特征以及区域估计
    # with open(output_dir / 'instance_dict.pkl', 'wb') as f:
    #     pickle.dump(model.class_dict, f)

    os.remove(output_dir / 'checkpoint_resume.pth') 
    torch.save({'model': model.state_dict()}, output_dir / 'checkpoint.pth')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    import sys
    print(sys.path)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)