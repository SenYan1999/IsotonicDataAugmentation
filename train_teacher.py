from comet_ml import Experiment, OfflineExperiment

import argparse
import os
import sys
import time
import utils

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.distributed as dist
import numpy as np

from tqdm import tqdm
import utils
import random
from utils.metric import *
from utils.data import get_dataloader
from utils.models import select_model

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    import apex
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print("Please install apex from https://www.github.com/nvidia/apex to run this example.")

'''model_names = sorted([name for name in resnet_for_cifar.__dict__
                      if name.islower() and not name.startswith("__")
                      and name.startswith("resnet")
                      and callable(resnet_for_cifar.__dict__[name])] + ["wideres_28_10",'resnet18'])

print(model_names)'''

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--dataset', default='CIFAR10', help='dataset')
parser.add_argument('--random', action='store_true')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    help='model architecture:  (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--imagenet_dir', default='data/tiny-imagenet-200', type=str, metavar='PATH',
                    help='path to imagenet')
parser.add_argument('--cinic_dir', default='data/cinic', type=str, metavar='PATH',
                    help='path to cinic')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=20)
parser.add_argument('--alpha', default=1., type=float, help='interpolation strength (uniform=1., ERM=0.)')
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--num_classes", default=10, type=int)
parser.add_argument('--distributed')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:4319', type=str)
parser.add_argument('--dist-backend', default='nccl', type=str)
parser.add_argument('--gpu', default=None, type=int)
parser.add_argument('--rank', default=0)
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument('--ngpus-per-node', default=1, type=int)
parser.add_argument('--multiprocessing-distributed', action='store_true')
parser.add_argument('--random_seed', default=230, type=int)

args = parser.parse_args()

def train(train_loader, model, criterion, optimizer, epoch, experiment, args):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    pbar = tqdm(train_loader)
    for i, (input, target) in enumerate(pbar):
        pbar.set_description('Epoch {}'.format(epoch))
        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        if args.gpu is not None:
            input_var = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        else:
            target = target.cuda()
            input_var = input.cuda()

        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        if args.half:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        # lr_scheduler.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_postfix(epoch=epoch, loss=losses.avg, top1_val=top1.val, top1_avg=top1.avg)
        step = i + epoch * len(train_loader)
        experiment.log_metric('loss', losses.val, step=step)
        experiment.log_metric('top1', top1.val, step=step)
        experiment.log_metric('lr', optimizer.param_groups[0]['lr'], step=step)

def validate(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        pbar = tqdm(val_loader)
        for i, (input, target) in enumerate(pbar):
            if args.gpu is not None:
                input_var = input.cuda(args.gpu, non_blocking=True)
                target_var = target.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            else:
                target = target.cuda()
                input_var = input.cuda()
                target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_postfix(loss=losses.avg, top1_val=top1.val, top1_avg=top1.avg)

    return top1.avg

if __name__ == '__main__':
    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    # set comet ml parameters
    if args.local_rank == 0:
        experiment = Experiment(
            api_key="vmCRwu7hHUSAp05U6DDME06pR",
            project_name="kd-mixup",
            workspace="senyan1999",
        )
        # experiment = OfflineExperiment(
        #     workspace='senyan1999',
        #     project_name='kd-mixup',
        #     offline_directory='comet_offline'
        # )

        experiment.add_tag('Train %s in %s' % (args.arch, args.dataset))
        experiment.log_parameters({
            'teacher-arch': args.arch,
            'batch_size': args.batch_size,
            'num_epoch': args.epochs,
            'lr': args.lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'fp16': args.half,
        })
    else:
        # experiment = Experiment(api_key="vmCRwu7hHUSAp05U6DDME06pR", disabled=True)
        experiment = OfflineExperiment(offline_directory='comet_offline', disabled=True)

    # distributed training setting
    if args.half:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        print(f'Now local rank is {args.local_rank}')
    
    # prepare data
    print('Loading the datasets...')
    train_dl, dev_dl, train_sampler = get_dataloader(args.dataset, args.half, args.batch_size, args.workers)
    print('- done.')

    # select model and optimizer
    model = select_model(args.arch, args).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.half:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        model = DDP(model)
    else:
        model = torch.nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.dataset == 'CIFAR10':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[100, 150, 250], last_epoch=args.start_epoch - 1)
    elif args.dataset == 'STL10':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[30, 60, 100, 150], last_epoch=args.start_epoch - 1)
    elif args.dataset == 'CIFAR100':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    elif args.dataset == 'IMAGENET' or args.dataset == 'mini-imagenet':
        print('mini')
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                            milestones=[30, 60, 90, 120, 150, 180], last_epoch=args.start_epoch - 1)

    best_prec1 = 0.0
    for epoch in range(args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        with experiment.train():
            train(train_dl, model, criterion, optimizer, epoch, experiment, args)

        with experiment.test():
            # evaluate on validation set
            prec1 = validate(dev_dl, model, criterion, args)
            experiment.log_metric('top1', prec1, step=epoch)

        if args.half:
            train_sampler.set_epoch(epoch)

        try:
            lr_scheduler.step()
        except:
            pass

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        print('Epoch: [%d/%d] Acc: %.2f' % (epoch, args.epochs, prec1))
        if is_best:
            model_file = os.path.join(args.save_dir, '{}_{}_best_model.th'.format(args.dataset, args.arch))
            print('new best: %.4f' % best_prec1)
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=model_file)
    print('dataset:{} arch:{} best acc:{}'.format(args.dataset, args.arch,best_prec1))
