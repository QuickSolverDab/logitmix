#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import random
import sys
import math
import shutil
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models_tv

from torch.utils.data import DataLoader
from torchsummary import summary
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

"""my functions"""
from datasetload import *
from utils import *
from submain import *
from tensorboardX import SummaryWriter

""" networks """
from models.networks import *

""" argparse
        python -W ignore main.py --gpu 0
"""
dims   = {'cifar10': (3,32,32), 'cifar100':(3,32,32), 'tiny_imagenet': (3,64,64), 'imagenet':(3,224,224)}
classes = {'cifar10': 10, 'cifar100':100, 'tiny_imagenet_100':100, 'tiny_imagenet':200, 'imagenet':1000}

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--network', default='resnet50',
                    type=str, help='name of dataset')
parser.add_argument('--dataset', default='cifar100',
                    choices=['cifar10', 'cifar100', 'tiny_imagenet'],
                    type=str, help='name of dataset')
parser.add_argument('--dataset_dir', default='/home/Databases/cifar/cifar-100-python',
                    type=str, metavar='PATH',
                    help='path to the dataset')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size (default: 128), this is the total'
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--gamma', default=0.2, type=float,
                    help='learning rate decaying')
parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 2e-4)',
                    dest='wd')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epoch', default=200, type=int, metavar='N',
                    help='terminate epoch')
parser.add_argument('--print_freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 2)')
parser.add_argument('--resume', default='./best_ckpt/model_best.pth.tar',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world_size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--port', default='8888', type=str)
parser.add_argument('--gpu', default=None, type=str, help='GPU id to use.')
parser.add_argument('--time', default=None, type=str, help='start time of training')
parser.add_argument('--mp', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--pretrained', action='store_true', help='using pre-trained Imagenet Model')

""" Mixup """
parser.add_argument('--mixmethod', default='org', type=str,
                    choices=['org', 'mixup', 'cutmix', 'logitmix_M', 'logitmix_C'])
parser.add_argument('--weights', default=[1, 1e-0, 1e-0], type=float, nargs=3,
                    help='ratio of similarity loss')
parser.add_argument('--dist', default='beta', type=str, choices=['beta', 'normal'],
                    help='distribution for mixup')
parser.add_argument('--alpha', default=3.0, type=float,
                    help='percentage of mixing images'
                         '[0,0.1): U shape, 1: Uniform, bigger than 2: Gaussian')
parser.add_argument('--warmup_mode', default=None, type=str, choices=[None, 'grad', 'step'],
                    help='warmup mode for logitmix')
parser.add_argument('--repeat', default= '0', type=str, help='num of repeating')
parser.add_argument('--LS', action='store_true', help='Label Smoothing')
parser.add_argument('--is_summary', action='store_true',
                    help='Measure confidence and adversarial')
parser.add_argument('--info', default=None, type=str)

# best test accuracy
best_acc1 = 0

def main(args):
    # check the number and id of gpu for training
    ngpus_per_node = torch.cuda.device_count()
    args.distributed = args.world_size > 1 or args.mp
    if args.mp:
        args.world_size = ngpus_per_node * args.world_size
        print(mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:'+args.port,
                                world_size=args.world_size, rank=args.rank)
    """ Create models
        define models / check both the arcitecture and parameters
        torchvision: model = models.__dict__[args.arch]()
        print("=> creating model '{}'".format(args.arch))
    """
    print("=> creating model")
    args.dataset = args.dataset.lower()
    args.network = args.network.lower()
    if 'imagenet' in args.dataset:
        if 'resnet' in args.network:
            network = models_tv.resnet50(pretrained=args.pretrained)
            if 'tiny' in args.dataset:
                network.maxpool = nn.Identity()
                num_ftrs = network.fc.in_features
                network.fc = nn.Linear(num_ftrs, classes[args.dataset])

        elif 'resnext' in args.network:
            network = models_tv.resnext50_32x4d(pretrained=args.pretrained)
            if 'tiny' in args.dataset:
                network.maxpool = nn.Identity()
                num_ftrs = network.fc.in_features
                network.fc = nn.Linear(num_ftrs, classes[args.dataset])

        elif 'shuffle' in args.network:
            network = models_tv.shufflenet_v2_x1_0(pretrained=args.pretrained)
            if 'tiny' in args.dataset:
                network.maxpool = nn.Identity()
                num_ftrs = network.fc.in_features
                network.fc = nn.Linear(num_ftrs, classes[args.dataset])

        elif 'mobilenet' in args.network:
            network = models_tv.mobilenet_v2(pretrained=args.pretrained)
            if 'tiny' in args.dataset:
                layers = list(network.features.children())
                layers[0] = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
                                          nn.BatchNorm2d(32),
                                          nn.ReLU6(inplace=True))
                network.features = nn.Sequential(*layers)
                num_ftrs = network.last_channel
                network.classifier = nn.Linear(num_ftrs, classes[args.dataset])
    else:
        network = networks(args.network)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            network.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            network = torch.nn.parallel.DistributedDataParallel(network,
                                                            device_ids=[args.gpu])
        else:
            network.cuda()
            network = torch.nn.parallel.DistributedDataParallel(network)

    else: #args.gpu is not None: args.gpu = 0
        args.gpu = int(args.gpu)
        torch.cuda.set_device(args.gpu)
        network.cuda(args.gpu)

    print()
    network.eval()
    summary(network, dims[args.dataset])
    print()


    """ Training scheme
        select optimizer
    """
    network_lr  = lr_scheduler(args.network, args.lr, args)
    network_optm= torch.optim.SGD(network.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.wd,
                                  nesterov=True)

    """ Optionally resume from a checkpoint
        resume: dir of check point
        load_state_dict: load pre-trained parameters
    """
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpu))
            start_epoch = checkpoint['start_epoch']
            if args.gpu is not None:
                best_acc1 = checkpoint['best_acc1']
            if args.distributed:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = 'module.' + k # remove `module.`
                    new_state_dict[name] = v
                network.load_state_dict(new_state_dict)
            else:
                network.load_state_dict(checkpoint['state_dict'])#,strict=False)
            network_optm.load_state_dict(checkpoint['network_optm'])
            print("=> loaded checkpoint '{}' (start_epoch {})"
                  .format(args.resume, checkpoint['start_epoch']))
            if 'best' in args.resume:
                load_networks = True
            else:
                load_networks = False
                if 'logit' in args.mixmethod:
                    num = checkpoint['num']
                    _schedule = list(args.schedule)
                    min_epoch, max_epoch = _schedule[num-1], _schedule[num]
            network_save_dir = args.resume[:-18]
            writer = SummaryWriter(log_dir=network_save_dir)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            start_epoch = 0
            load_networks = False

    """ Data loading code
        train/valication dataset load
        padding_mode = [constant, reflect, edge, symmetric]
    """
    train_dataset, val_dataset = datasetload(args.dataset, padding_mode='reflect')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=(train_sampler is None), num_workers=args.workers,
                        pin_memory=True, sampler=train_sampler)
    val_loader   = DataLoader(val_dataset, batch_size=100,
                        shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.evaluate or args.resume:
        acc1, acc5 = validate(val_loader, network, args=args)
        print(' *** Acc@1 {0:7.4f} Acc@5 {1:7.4f}'.format(acc1, acc5))

    """ network training
    """
    num = 0
    if not load_networks:
        best_acc1 = 0
        for epoch in range(start_epoch, args.epoch):
            if epoch == 0:
                network_save_dir = make_save_dir(args, network=args.network)
                writer = SummaryWriter(log_dir=network_save_dir)
                backup_dir = './ckpt/backup/gpu{}_{}'.format(args.gpu, args.time)
                make_dir(backup_dir)
                shutil.copy('./main.py', backup_dir)
                shutil.copy('./submain.py', backup_dir)
                shutil.copy('./utils.py', backup_dir)
            if not 'mobilenet' in args.network and epoch in args.schedule:
                for param_group in network_optm.param_groups:
                    param_group['lr'] = args.schedule[epoch]
                _schedule = list(args.schedule)
                min_epoch, max_epoch = _schedule[num], _schedule[num+1]
                print(num, param_group['lr'])
                num += 1

            # train for one epoch
            if 'logit' in args.mixmethod:
                if args.warmup_mode is None:
                    args.logit_prob = 1
                elif args.warmup_mode == 'step':
                    args.logit_prob = 0 if (epoch < args.epoch // 10) else 1
                elif args.warmup_mode == 'grad':
                    if num == 1:
                        args.logit_prob = warmup(epoch, min_epoch, max_epoch, 0.2, 0.3, first=num)
                    else:
                        args.logit_prob = warmup(epoch, min_epoch, max_epoch, 0.0, 0.3, first=num)
                tr_acc1, m_loss, c_loss, s_loss = train(train_loader, network, network_optm, epoch, args)
            else:
                tr_acc1  = train(train_loader, network, network_optm, epoch, args)

            if tr_acc1 is None:
                break
            # evaluate on validation set
            if epoch % args.print_freq == 0 or epoch + 1 == args.epoch:
                acc1, acc5 = validate(val_loader, network, args=args)
                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
                if (args.mp and args.gpu == 0) or args.mp == False:
                    print(' *** Acc@1 {0:7.4f} Acc@5 {1:7.4f} Acc@Best {2:7.4f}'.format(acc1, acc5, best_acc1))
                    save_checkpoint(network_save_dir, args, is_best,
                                    {'start_epoch': epoch + 1,
                                    'state_dict': network.state_dict(),
                                    'best_acc1': best_acc1,
                                    'network_optm' : network_optm.state_dict(),
                                    'num': num,
                                    })

                    if 'logit' in args.mixmethod:
                        record(network_save_dir, args,
                               {'epoch': epoch + 1, 'train_acc': tr_acc1, 'test_acc': acc1, 'best_acc': best_acc1,
                                'mixup': m_loss, 'cls': c_loss, 'sim': s_loss})
                    else:
                        record(network_save_dir, args,
                               {'epoch': epoch + 1, 'train_acc': tr_acc1, 'test_acc': acc1, 'best_acc': best_acc1})

                    writer.add_scalar('{}_{}/accuracy/train'.format(args.mixmethod, args.dataset), torch.tensor(tr_acc1), epoch)
                    writer.add_scalar('{}_{}/accuracy/test'.format(args.mixmethod, args.dataset), torch.tensor(acc1), epoch)
                    if 'logit' in args.mixmethod:
                        writer.add_scalar('{}_{}/losses/mixup'.format(args.mixmethod, args.dataset), torch.tensor(m_loss), epoch)
                        writer.add_scalar('{}_{}/losses/cls'.format(args.mixmethod, args.dataset), torch.tensor(c_loss), epoch)
                        writer.add_scalar('{}_{}/losses/sim'.format(args.mixmethod, args.dataset), torch.tensor(s_loss), epoch)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    args = parser.parse_args()
    # For multi-processing
    if len(args.gpu) != 1:
        gpu_devices = args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        args.mp = True
        print('Multi_process mode is {} using {}'.format(args.mp, gpu_devices))
    else:
        print('Multi_process mode is {} using {}'.format(args.mp, args.gpu))

    torch.set_num_threads(4)
    # Pre-setting
    if args.mixmethod == 'logitmix_M':
        args.weights = [1,1e-0,1e-0] # Mixup, CE, logitmix
        args.dist, args.alpha = 'beta', 3.0
    elif args.mixmethod == 'logitmix_C':
        args.weights = [1,1e-0,1e-1] # Mixup, CE, logitmix
        args.dist, args.alpha = 'beta', 1.0
    else:
        args.weights = [0,0,0]

    args.time  = datetime.now().strftime("%y%m%d_%H-%M-%S")
    main(args)
