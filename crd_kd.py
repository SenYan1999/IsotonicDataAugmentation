"""Main entrance for train/eval with/without KD on CIFAR-10"""

# import comet_ml at the top of your file
import logging
import os
import sys
import time
import math
import random
import numpy as np
import torch
import torchvision
import torch.distributed as dist

import distill
import distill.crd
from utils.metric import AverageMeter, accuracy, accuracy_5
from utils.evaluate import evaluate_kd, evaluate
from utils.models import select_model
from utils.crd_data import get_dataloader
from utils.utils import save_checkpoint
from args import args

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    import apex
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print("Please install apex from https://www.github.com/nvidia/apex to run this example.")

# redirect std out
log_name = 'crd_%s-%s-%s-%s-%s-%s-%s.log' % (args.dataset, args.teacher_arch, args.arch, args.mixup_method, args.calibration_method, str(args.soft_constraint_ratio), str(args.temperature))
# log_name = '%s-%s-%s-%s-%s-%s.log' % (args.dataset, args.teacher_arch, args.arch, args.mixup_method, args.calibration_method, str(args.soft_constraint_ratio))
stdout_file = open(os.path.join(args.log_dir, log_name), 'w')
sys.stdout = stdout_file

def get_lr_scheduler(optimizer):
    if args.dataset == 'CIFAR10' or args.dataset == 'STL10':
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        #                                                     milestones=[150, 180, 210], last_epoch=args.start_epoch - 1)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[100, 150], last_epoch=args.start_epoch - 1)
    elif args.dataset == 'CIFAR100':
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        #                                                     milestones=[150, 180, 210], last_epoch=args.start_epoch - 1)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    elif args.dataset == 'mini-imagenet':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    elif args.dataset == 'IMAGENET':
        lr_scheduler = None
    return lr_scheduler

def train_and_evaluate(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                          experiment, params, criterion_crd):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
    """

    best_val_acc = 0.0
    metrics = {'accuracy': accuracy, 'top5': accuracy_5}

    # fetch teacher outputs using teacher_model under eval() mode
    loading_start = time.time()
    teacher_model.eval()
    # teacher_outputs = fetch_teacher_outputs(teacher_model, train_dataloader, params)
    elapsed_time = math.ceil(time.time() - loading_start)
    logging.info("- Finished computing teacher outputs after {} secs..".format(elapsed_time))

    lr_scheduler = get_lr_scheduler(optimizer)

    start_time = time.time()
    for epoch in range(params.epochs):

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.epochs))

        if args.mixup_method == 'none':
            train_metric = distill.crd.crd_vanilla(epoch, model, teacher_model, optimizer, train_dataloader, criterion_crd, args)
        else:
            train_metric = distill.crd.crd_mixup(epoch, model, teacher_model, optimizer, train_dataloader, criterion_crd, args)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate_kd(model, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        val_top5 = val_metrics['top5']

        if best_val_acc < val_acc:
            model_file = os.path.join('experiments', '%s-%s-%s-%s-%s-%s-%s.pt' % (params.dataset, params.teacher_arch, params.arch, params.mixup_method, params.calibration_method, str(params.soft_constraint_ratio), str(params.temperature)))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': val_acc,
            }, True, filename=model_file)
        best_val_acc = best_val_acc if best_val_acc > val_acc else val_acc

        if args.dataset != 'IMAGENET':
            lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if params.local_rank == 0:
            print('epoch: {} dataset:{} teacher-arch:{} student-arch:{} current lr:{} current train loss:{} current acc:{} top5: {} best acc:{}'.format(epoch, args.dataset, args.teacher_arch, args.arch, current_lr, train_metric[1], val_acc, val_top5, best_val_acc))

if __name__ == '__main__':
    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    experiment = None

    # distributed training setting
    if args.half:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        print(f'Now local rank is {args.local_rank}')
    
    # prepare data
    print('Loading the datasets...')
    train_dl, dev_dl, train_sampler = get_dataloader(args.dataset, args.half, args.batch_size, args.workers, args)
    print('- done.')

    # select model and optimizer
    if args.dataset != 'IMAGENET':
        teacher_model = select_model(args.teacher_arch, args).cuda()
    else:
        teacher_model = torchvision.models.resnet34(pretrained=True).cuda()
    student_model = select_model(args.arch, args).cuda()

    all_parameters = student_model.parameters()
    criterion_crd = None
    args, all_parameters, criterion_crd = distill.crd.prepare_crd_distill(student_model, teacher_model, args)
    criterion_crd = criterion_crd.cuda()
    optimizer = torch.optim.SGD(all_parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # get teacher checkpoint file
    teacher_checkpoint = os.path.join(args.save_dir, '{}_{}_best_model.th'.format(args.dataset, args.teacher_arch))

    if args.half:
        teacher_model = amp.initialize(teacher_model, opt_level='O1')
        student_model, optimizer = amp.initialize(student_model, optimizer, opt_level='O1')
        teacher_model = DDP(teacher_model)
        student_model = DDP(student_model)

    else:
        teacher_model = torch.nn.DataParallel(teacher_model)
        student_model = torch.nn.DataParallel(student_model)

    # load teacher checkpoint
    if args.dataset != 'IMAGENET':
        print('load teacher')
        checkpoint = torch.load(teacher_checkpoint, map_location=lambda storage, loc: storage.cuda(args.local_rank))
        teacher_model.load_state_dict(checkpoint['state_dict'])

    # before start, you may want to evaluate the teacher model's performance 
    # evaluate(teacher_model, dev_dl, {'accuracy': accuracy}, args)
    
    # train
    # TODO: train_sampler.set_epoch()
    train_and_evaluate(student_model, teacher_model, train_dl, dev_dl, optimizer, experiment, args, criterion_crd=criterion_crd)
