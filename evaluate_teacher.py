"""Main entrance for train/eval with/without KD on CIFAR-10"""

# import comet_ml at the top of your file
from comet_ml import Experiment, OfflineExperiment

import argparse
import logging
import os
import sys
import shutil
import time
import math
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from tqdm import tqdm

import utils
from utils.metric import AverageMeter, accuracy
from utils.evaluate import evaluate_kd, evaluate
from utils.models import select_model
from utils.data import get_dataloader
from distill.loss_fn import *
from distill.kd import *
from distill.methods import *

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    import apex
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print("Please install apex from https://www.github.com/nvidia/apex to run this example.")

parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory for the dataset")
parser.add_argument('--model_dir', default='experiments',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('--dataset', default='CIFAR10', help='dataset')
parser.add_argument('--arch', metavar='ARCH', default='resnet20')
parser.add_argument('--mixup_teacher', action='store_true')
parser.add_argument('--teacher-arch', metavar='TEA_ARCH', default='resnet110')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--alpha', default=0.95, type=float, metavar='ALPHA',
                    help='alpha')
parser.add_argument('--temperature', default=0.6, type=float, metavar='T',
                    help='temperature')
parser.add_argument('--imagenet-dir', default='/home/cuiwanyun/imagenet/ILSVRC2012', type=str, metavar='PATH',
                    help='path to imagenet')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                    help='not use cuda')
parser.add_argument('--no-kd', dest='no_kd', action='store_true',
                    help='do not kd')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--mixup', dest='mixup', action='store_true',
                    help='use mixup ')
parser.add_argument('--mixup-method', default='train_kd_mixup', type=str)
parser.add_argument('--loss-func', default=None, type=str)
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument("--num_classes", default=10, type=int)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--random_seed', default=230, type=int)
parser.add_argument('--log_dir', default='logs/', type=str)
parser.add_argument('--tensorboard_dir', default=None, type=str)
parser.add_argument('--kd_method', default='vanilla', \
    choices=['vanilla', 'mixup'], type=str)

args = parser.parse_args()

def get_lr_scheduler(dataset, optimizer):
    if args.dataset == 'CIFAR10':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[100, 150], last_epoch=args.start_epoch - 1)
    elif args.dataset == 'CIFAR100':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    elif args.dataset == 'IMAGENET':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[30, 60, 90, 120, 150, 180], last_epoch=args.start_epoch - 1)
    return lr_scheduler

def train_and_evaluate(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                          experiment, params):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
    """

    best_val_acc = 0.0
    metrics = {'accuracy': accuracy}

    '''
    # Tensorboard logger setup
    tensorboard_dir = params.tensorboard_dir if params.tensorboard_dir else os.path.join(params.log_dir, f'{params.dataset}_{args.teacher_arch}_{params.arch}_{args.kd_method}_{args.loss_func}')
    # if os.path.exists(tensorboard_dir):
    #     # need_delete = input('The tensorboard dir has existed, do you want to delete it and continue? [y / n]')
    #     need_delete = 'y'
    #     if need_delete[0].lower() == 'y':
    #         shutil.rmtree(tensorboard_dir)
    #     else:
    #         sys.exit(0)
    summary_writer = tensorboard.SummaryWriter(log_dir=tensorboard_dir)
    '''

    # fetch teacher outputs using teacher_model under eval() mode
    loading_start = time.time()
    teacher_model.eval()
    # teacher_outputs = fetch_teacher_outputs(teacher_model, train_dataloader, params)
    elapsed_time = math.ceil(time.time() - loading_start)
    logging.info("- Finished computing teacher outputs after {} secs..".format(elapsed_time))

    lr_scheduler = get_lr_scheduler(args.dataset, optimizer)

    for epoch in range(params.epochs):

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.epochs))

        with experiment.train():
            # compute number of batches in one epoch (one full pass over the training set)
            if params.kd_method == 'vanilla':
                kd_vanilla(epoch, model, teacher_model, optimizer, train_dataloader, experiment, params)
            elif params.kd_method == 'mixup':
                kd_mixup(epoch, model, teacher_model, optimizer, train_dataloader, experiment, params)
            else:
                print('Not implamentation for kd method %s' % params.kd_method)

        with experiment.test():
            # Evaluate for one epoch on validation set
            val_metrics = evaluate_kd(model, val_dataloader, metrics, params)

            # write tensorboard
            experiment.log_metric('top_1', val_metrics['accuracy'], step=epoch)
            experiment.log_metric('loss', val_metrics['loss'], step=epoch)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc
        best_val_acc = best_val_acc if best_val_acc > val_acc else val_acc

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print('dataset:{} teacher-arch:{} student-arch:{} current lr:{} best acc:{}'.format(args.dataset, args.teacher_arch, args.arch, current_lr, best_val_acc))

if __name__ == '__main__':
    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    # # set comet ml parameters
    # if args.local_rank == 0:
    #     # experiment = Experiment(
    #     #     api_key="vmCRwu7hHUSAp05U6DDME06pR",
    #     #     project_name="kd-mixup",
    #     #     workspace="senyan1999",
    #     # )
    #     experiment = OfflineExperiment(
    #         workspace='senyan1999',
    #         project_name='kd-mixup',
    #         offline_directory='comet_offline'
    #     )

    #     experiment.add_tag(args.dataset)
    #     experiment.log_parameters({
    #         'teacher-arch': args.teacher_arch,
    #         'student-arch': args.arch,
    #         'batch_size': args.batch_size,
    #         'num_epoch': args.epochs,
    #         'lr': args.lr,
    #         'momentum': args.momentum,
    #         'weight_decay': args.weight_decay,
    #         'kd-method': args.kd_method,
    #         'loss-func': args.loss_func,
    #         'temperature': args.temperature,
    #         'alpha': args.alpha,
    #         'fp16': args.half,
    #     })
    # else:
    #     # experiment = Experiment(api_key="vmCRwu7hHUSAp05U6DDME06pR", disabled=True)
    #     experiment = OfflineExperiment(offline_directory='comet_offline', disabled=True)

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
    teacher_model = select_model(args.arch, args).cuda()
    # teacher_model = torchvision.models.resnet34(pretrained=True).cuda()
    if args.mixup_teacher:
        teacher_checkpoint = os.path.join(args.save_dir, '{}_{}_mixup2_best_model.th'.format(args.dataset, args.arch))
    else:
        teacher_checkpoint = os.path.join(args.save_dir, '{}_{}_best_model.th'.format(args.dataset, args.arch))
    if args.half:
        teacher_model = amp.initialize(teacher_model, opt_level='O1')
        teacher_model = DDP(teacher_model)

    else:
        teacher_model = torch.nn.DataParallel(teacher_model)
    checkpoint = torch.load(teacher_checkpoint, map_location=lambda storage, loc: storage.cuda(args.local_rank))
    teacher_model.load_state_dict(checkpoint['state_dict'])


    evaluate(teacher_model, dev_dl, {'accuracy': accuracy}, args)
