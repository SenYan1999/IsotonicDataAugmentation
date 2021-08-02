import torch
import numpy as np

from torch.autograd import Variable
from utils import AverageMeter, RunningAverage, accuracy, imagenet_adjust_lr
from tqdm import tqdm
from apex import amp
from .loss_fn import *
from .utils import *

def kd(epoch, model, teacher_model, optimizer, dataloader, params):
    model.train()
    teacher_model.eval()

    # summary for current training loop and a running average object for loss
    loss_avg = RunningAverage()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Use tqdm for progress bar
    pbar = tqdm(dataloader)
    for i, (train_batch, labels_batch) in enumerate(pbar):
        # adjust lr
        if params.dataset == 'IMAGENET':
            imagenet_adjust_lr(optimizer, epoch, i, len(dataloader), params)

        pbar.set_description('Epoch {}'.format(epoch))
        # move to GPU if available
        train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
        # convert to torch Variables
        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

        if params.mixup_method == 'none':
            mixed_x, mixed_y, lam, index = train_batch, labels_batch, 0, None
        elif params.mixup_method == 'mixup':
            mixed_x, mixed_y, lam, index = mixup(train_batch, labels_batch, params)
        elif params.mixup_method == 'cutmix':
            mixed_x, mixed_y, lam, index = cutmix(train_batch, labels_batch, params)
        else:
            raise Exception('Not support mixup method %s' % params.mixup_method)

        output_teacher_batch = teacher_model(mixed_x).detach()

        output_batch_mixed = model(mixed_x)
        output_batch = model(train_batch)

        # get loss
        loss = loss_fn(output_batch_mixed, output_teacher_batch, mixed_y, labels_batch,
                                labels_batch[index], lam, params)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        if params.half:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # update the average loss
        loss_avg.update(loss.item())
        prec1 = accuracy(output_batch.data, labels_batch)[0]
        losses.update(loss.item(), train_batch.size(0))
        top1.update(prec1.item(), train_batch.size(0))

        if params.local_rank == 0:
            pbar.set_postfix(epoch=epoch, loss=losses.avg, top1_val=top1.val, top1_avg=top1.avg)

        # write to tensorboard
        step = i + epoch * len(dataloader)
        lr = optimizer.param_groups[0]['lr']

    return (top1.avg, losses.avg)

