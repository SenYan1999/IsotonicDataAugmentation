import torch
import numpy as np

from torch.autograd import Variable
from utils import AverageMeter, RunningAverage, accuracy
from tqdm import tqdm
from apex import amp
from .loss_fn import *

def update_summary_writer(summary_writer, step, loss, top1, lr):
    try:
        summary_writer.add_scalar('train/top_1', top1.val, step)
        summary_writer.add_scalar('train/loss', loss.val, step)
        summary_writer.add_scalar('train/lr', lr, step)
    except:
        pass

def kd_vanilla(epoch, model, teacher_model, optimizer, dataloader, summary_writer, params):
    model.train()
    teacher_model.eval()

    # summary for current training loop and a running average object for loss
    summ = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
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

        pbar.set_postfix(epoch=epoch, loss=losses.avg, top1_val=top1.val, top1_avg=top1.avg)
        # write in tensorboard
        step = i + epoch * len(dataloader)
        lr = optimizer.param_groups[0]['lr']
        if params.local_rank == 0:
            update_summary_writer(summary_writer, step, losses, top1, lr)

def kd_mixup(epoch, model, teacher_model, optimizer, dataloader, summary_writer, params):
    # top k的class乘以alpha，使得P(c)是满足topk ground truth的
    model.train()
    teacher_model.eval()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = RunningAverage()
    batch_time = AverageMeter()
    data_time = AverageMeter()
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
        output_teacher_batch = teacher_model(mixed_x).detach()

        output_batch_mixed = model(mixed_x)
        output_batch = model(train_batch)

        if params.loss_func in ['loss_fn_kd_mixup_origin', 'loss_fn_kd_mixup_correction', 'loss_fn_kd_mixup_correction_plus_respective']:
            loss =globals()[params.loss_func](output_batch, output_batch_mixed, labels_batch, output_teacher_batch,
                                            topk_ground_truth, params, lam, labels_batch, labels_batch[index])
        elif params.loss_func is not None:
            loss = globals()[params.loss_func](output_batch, output_batch_mixed, labels_batch, output_teacher_batch,
                                            topk_ground_truth, params, mixed_y)
        else:
            loss = loss_fn_kd_mixup_prod_alpha(output_batch, output_batch_mixed, labels_batch, output_teacher_batch,
                                            topk_ground_truth, params)

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

        pbar.set_postfix(epoch=epoch, loss=losses.avg, top1_val=top1.val, top1_avg=top1.avg)
        
        # write to tensorboard
        step = i + epoch * len(dataloader)
        lr = optimizer.param_groups[0]['lr']
        if params.local_rank == 0:
            update_summary_writer(summary_writer, step, losses, top1, lr)
