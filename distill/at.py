import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from utils import AverageMeter, RunningAverage, accuracy, update_experiment
from tqdm import tqdm
from apex import amp
from .loss_fn import *

class AT(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""

    def __init__(self, p = 2):
        super(AT, self).__init__()
        self.p = p 

    def forward(self, g_s, g_t):
        return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))

def loss_fn_at(s_logit, t_logit, s_feature, t_feature, ground_truth, at_criterion, params):
    # hard loss
    loss_hard = hard_loss(s_logit, ground_truth)

    # soft loss
    loss_soft = soft_loss(s_logit, t_logit, params)

    # at loss
    loss_at = sum(at_criterion(s_feature, t_feature))

    # total loss
    loss = params.gamma * loss_hard + params.alpha * loss_soft + params.beta * loss_at
    print('hard loss:%.4f' % loss_hard)
    print('soft loss:%.4f' % loss_soft)
    print('at loss:%.4f' % loss_at)


    return loss

def loss_fn_at_mixup(outputs_mixed, teacher_outputs_mixed, s_feature, t_feature, topk_ground_truth, target_a, target_b, lam, at_criterion, params):
    # hard loss
    loss_h = hard_loss_mixup(outputs_mixed, target_a, target_b, lam)

    # soft loss
    if params.mixup_method == 'no_correction':
        loss_s = soft_loss_mixup_no_correction(outputs_mixed, teacher_outputs_mixed, params)
    elif params.mixup_method == 'correction':
        loss_s = soft_loss_mixup_correction(outputs_mixed, teacher_outputs_mixed, topk_ground_truth, params)
    else:
        raise(Exception('Not support mixup method: %s' % (params.mixup_method)))

    if params.mixup_method == 'correction':
        if params.correction_method == 'naive_plus':
            loss_s = soft_loss_mixup_correction(outputs_mixed, teacher_outputs_mixed, topk_ground_truth, params)
        elif params.correction_method == 'plus_remain_sum':
            loss_s = soft_loss_mixup_correction_plus_remain_sum(outputs_mixed, teacher_outputs_mixed, topk_ground_truth, params)
        elif params.correction_method == 'min_gap':
            loss_s = soft_loss_mixup_no_correction(outputs_mixed, teacher_outputs_mixed, params)
            loss_soft_constraint = soft_loss_mixup_correction_min_gap(torch.softmax(outputs_mixed, dim=-1), topk_ground_truth)
            loss_s = loss_soft_constraint * params.soft_constraint_ratio / params.alpha + loss_s
        else:
            raise(Exception('Not support correction method: %s' % (params.correction_method)))

    # at loss
    loss_at = sum(at_criterion(s_feature, t_feature))

    # total loss
    loss = params.gamma * loss_h + params.alpha * loss_s + params.beta * loss_at

    return loss

def at_vanilla(epoch, model, teacher_model, optimizer, dataloader, experiment, params):
    model.train()
    teacher_model.eval()

    # summary for current training loop and a running average object for loss
    losses = AverageMeter()
    top1 = AverageMeter()

    # criterion
    at_criterion = AT(p=params.at_p)

    # Use tqdm for progress bar
    pbar = tqdm(dataloader)
    for i, (train_batch, labels_batch) in enumerate(pbar):
        pbar.set_description('Epoch {}'.format(epoch))
        # move to GPU if available
        train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
        # convert to torch Variables
        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

        student_features, output_batch = model(train_batch, is_feat=True)
        teacher_features, output_teacher_batch = teacher_model(train_batch, is_feat=True)
        output_teacher_batch = output_teacher_batch.detach()

        t_feature, s_feature = teacher_features[1:-1], student_features[1:-1]
        loss = loss_fn_at(output_batch, output_teacher_batch, s_feature, t_feature, labels_batch, at_criterion, params)

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

        pbar.set_postfix(epoch=epoch, loss=losses.avg, top1_val=top1.val, top1_avg=top1.avg)
        # write in tensorboard
        step = i + epoch * len(dataloader)
        lr = optimizer.param_groups[0]['lr']
        update_experiment(experiment, step, losses, top1, lr)
    return (top1.avg, losses.avg)

def at_mixup(epoch, model, teacher_model, optimizer, dataloader, experiment, params):
    # top k的class乘以alpha，使得P(c)是满足topk ground truth的
    model.train()
    teacher_model.eval()

    # define at criterion
    at_criterion = AT(params.at_p)

    # summary for current training loop and a running average object for loss
    loss_avg = RunningAverage()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Use tqdm for progress bar
    pbar = tqdm(dataloader)
    for i, (train_batch, labels_batch) in enumerate(pbar):
        pbar.set_description('Epoch {}'.format(epoch))
        # move to GPU if available
        train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
        # convert to torch Variables
        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

        lam = np.random.beta(1.0, 1.0)
        index = torch.randperm(train_batch.size(0)).cuda()
        mixed_x = lam * train_batch + (1 - lam) * train_batch[index, :]
        mixed_y_a = (labels_batch.unsqueeze(1) == torch.arange(params.num_classes).cuda().reshape(1,
                                                                                                params.num_classes)).float()
        mixed_y_b = (labels_batch[index].unsqueeze(1) == torch.arange(params.num_classes).cuda().reshape(1,
                                                                                                        params.num_classes)).float()
        mixed_y = lam * mixed_y_a + (1 - lam) * mixed_y_b
        topk_ground_truth = (mixed_y_a + mixed_y_b).gt(0).float()
        t_feature, output_teacher_batch = teacher_model(mixed_x, is_feat=True)
        output_teacher_batch = output_teacher_batch.detach()

        s_feature, output_batch_mixed = model(mixed_x, is_feat=True)
        output_batch = model(train_batch)

        # get loss
        t_feature, s_feature = t_feature[1:-1], s_feature[1:-1]
        loss = loss_fn_at_mixup(output_batch_mixed, output_teacher_batch, s_feature, t_feature, topk_ground_truth, labels_batch, labels_batch[index], lam, at_criterion, params)

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
