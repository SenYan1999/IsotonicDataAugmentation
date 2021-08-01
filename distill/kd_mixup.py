import torch
import numpy as np

from torch.autograd import Variable
from utils import AverageMeter, RunningAverage, accuracy, update_experiment, imagenet_adjust_lr
from tqdm import tqdm
from apex import amp
from .loss_fn import *
from .calibration import *

def get_calibrated_logit(pred_y, mixed_y, param):
    if param.calibration_method == 'isotonic':
        return pred_y
    elif param.calibration_method == 'none':
        return pred_y
    else:
        raise Exception('Not Support calibration_method %s' % param.calibration_method)

def mixup(x, y, params):
    lam = np.random.beta(1.0, 1.0)
    index = torch.randperm(x.size(0)).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y_a = (y.unsqueeze(1) == torch.arange(params.num_classes).cuda().reshape(1, params.num_classes)).float()
    mixed_y_b = (y[index].unsqueeze(1) == torch.arange(params.num_classes).cuda().reshape(1, params.num_classes)).float()
    mixed_y = lam * mixed_y_a + (1 - lam) * mixed_y_b

    return mixed_x, mixed_y, lam, index

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(x, y, params):
    lam = np.random.beta(1.0, 1.0)
    rand_index = torch.randperm(x.size()[0]).cuda()
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), 1 - lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lam
    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-1])

    mixed_y_a = (y.unsqueeze(1) == torch.arange(params.num_classes).cuda().reshape(1, params.num_classes)).float()
    mixed_y_b = (y[rand_index].unsqueeze(1) == torch.arange(params.num_classes).cuda().reshape(1, params.num_classes)).float()

    mixed_y = lam * mixed_y_a + (1 - lam) * mixed_y_b

    return x, mixed_y, lam, rand_index


def loss_fn(outputs_mixed, teacher_outputs_mixed, mixed_y, target_a, target_b, lam, params):
    # get hard loss
    if params.mixup_method == 'none':
        loss_hard = hard_loss(outputs_mixed, mixed_y)
        loss_soft = soft_loss(outputs_mixed, teacher_outputs_mixed, params)
        loss = params.alpha * loss_soft + (1 - params.alpha) * loss_hard
        return loss

    else:
        loss_hard = hard_loss_mixup(outputs_mixed, target_a, target_b, lam)

        # calibration and get soft loss
        teacher_outputs_mixed = get_calibrated_logit(teacher_outputs_mixed, mixed_y)
        loss_soft = soft_loss(outputs_mixed, teacher_outputs_mixed, params)

        # combine soft loss and hard loss
        if params.soft_constraint:
            topk_ground_truth = mixed_y.gt(0).float()
            loss_constraint = loss_soft_constraint(outputs_mixed, topk_ground_truth)
            loss = params.alpha * loss_soft + (1 - params.alpha) * loss_hard + params.soft_constraint_ratio * loss_constraint
        else:
            loss = params.alpha * loss_soft + (1 - params.alpha) * loss_hard
        return loss

def kd(epoch, model, teacher_model, optimizer, dataloader, experiment, params):
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
            mixed_x, mixed_y, lam, index = mixup(train_batch, labels_batch)
        elif params.mixup_method == 'cutmix':
            mixed_x, mixed_y, lam, index = cutmix(train_batch, labels_batch)
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
        if params.local_rank == 0:
            update_experiment(experiment, step, losses, top1, lr)

    return (top1.avg, losses.avg)

