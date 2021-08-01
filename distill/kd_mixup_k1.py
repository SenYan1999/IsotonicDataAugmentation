import torch
import numpy as np

from torch.autograd import Variable
from utils import AverageMeter, RunningAverage, accuracy, update_experiment, imagenet_adjust_lr
from tqdm import tqdm
from apex import amp
from .loss_fn import *


def loss_fn_kd(outputs, labels, teacher_outputs, params):
    loss_s = soft_loss(outputs, teacher_outputs, params)
    loss_h = hard_loss(outputs, labels)

    # correction loss
    topk_ground_truth = (labels.unsqueeze(1) == torch.arange(params.num_classes).cuda().reshape(1, params.num_classes)).float()
    loss_soft_constraint = soft_loss_mixup_correction_min_gap(outputs, topk_ground_truth)

    loss = loss_s * params.alpha + loss_h * params.gamma + loss_soft_constraint * 0.5

    return loss

def kd_mixup_k1(epoch, model, teacher_model, optimizer, dataloader, experiment, params):
    model.train()
    teacher_model.eval()

    # summary for current training loop and a running average object for loss
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

        output_batch = model(train_batch)
        output_teacher_batch = teacher_model(train_batch).detach()

        loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        if params.half:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # update the average loss
        prec1 = accuracy(output_batch.data, labels_batch)[0]
        losses.update(loss.item(), train_batch.size(0))
        top1.update(prec1.item(), train_batch.size(0))

        if params.local_rank == 0:
            pbar.set_postfix(epoch=epoch, loss=losses.avg, top1_val=top1.val, top1_avg=top1.avg)
            # write in tensorboard
            step = i + epoch * len(dataloader)
            lr = optimizer.param_groups[0]['lr']
            update_experiment(experiment, step, losses, top1, lr)

    return (top1.avg, losses.avg)
