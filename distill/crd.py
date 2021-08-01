import torch
import numpy as np
import math

from torch.autograd import Variable
from utils import AverageMeter, RunningAverage, accuracy, imagenet_adjust_lr
from tqdm import tqdm
from apex import amp
from .loss_fn import *
from .utils import *

eps = 1e-7

class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_n: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_n, opt.nce_t, opt.nce_m)
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_n]
        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss

class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2


class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj

def prepare_crd_distill(s_model, t_model, params):
    data = torch.randn(2, 3, 32, 32).cuda()

    s_model.eval()
    t_model.eval()

    s_feature, _ = s_model(data, is_feat=True)
    t_feature, _ = t_model(data, is_feat=True)

    params.s_dim = s_feature[-1].shape[1]
    params.t_dim = t_feature[-1].shape[1]
    assert params.dataset in ['CIFAR10', 'CIFAR100']
    params.n_data = 50000 # only for cifar
    criterion_kd = CRDLoss(params)

    embed_s, embed_t = criterion_kd.embed_s, criterion_kd.embed_t

    # prepare trainable parameters
    trainable_module = torch.nn.ModuleList([])
    trainable_module.append(s_model)
    trainable_module.append(embed_s)
    trainable_module.append(embed_t)

    parameters = trainable_module.parameters()
    
    return params, parameters, criterion_kd

def loss_fn_crd(outputs, labels, teacher_outputs, features_s, features_t, index, contrastive_idx, criterion_crd, params):
    loss_s = soft_loss(outputs, teacher_outputs, params)
    loss_h = hard_loss(outputs, labels)
    loss_crd = criterion_crd(features_s, features_t, index, contrastive_idx)

    loss = loss_s * params.alpha + loss_h * (1-params.alpha) + loss_crd * params.beta

    return loss

def loss_fn_crd_mixup(outputs_mixed, teacher_outputs_mixed, topk_ground_truth, mixed_y, target_a, target_b, lam, features_s, features_t, index, contrastive_idx, criterion_crd, params):
    # hard loss
    loss_h = hard_loss_mixup(outputs_mixed, target_a, target_b, lam)

    # soft loss
    loss_s = soft_loss(outputs_mixed, teacher_outputs_mixed, params)

    # soft loss
    if params.calibration_method == 'isotonic_appr':
        loss_constraint = loss_isotonic_appr(torch.softmax(outputs_mixed, dim=-1), topk_ground_truth, mixed_y)
    elif params.calibration_method == 'isotonic':
        loss_constraint = loss_isotonic(torch.softmax(outputs_mixed, dim=-1), teacher_outputs_mixed, lam, target_a, target_b, params)
    else:
        loss_constraint = 0

    # crd loss
    loss_crd = criterion_crd(features_s, features_t, index, contrastive_idx)

    loss = (1-params.alpha) * loss_h + params.alpha * loss_s + params.beta * loss_crd + params.soft_constraint_ratio * loss_constraint

    return loss

def crd_vanilla(epoch, model, teacher_model, optimizer, dataloader, criterion_crd, params):
    model.train()
    teacher_model.eval()

    # summary for current training loop and a running average object for loss
    losses = AverageMeter()
    top1 = AverageMeter()

    # Use tqdm for progress bar
    pbar = tqdm(dataloader)
    for i, (train_batch, labels_batch, index, contrastive_idx) in enumerate(pbar):
        # adjust lr
        if params.dataset == 'IMAGENET':
            imagenet_adjust_lr(optimizer, epoch, i, len(dataloader), params)

        pbar.set_description('Epoch {}'.format(epoch))
        # move to GPU if available
        train_batch, labels_batch, index, contrastive_idx = train_batch.cuda(), labels_batch.cuda(), index.long().cuda(), contrastive_idx.cuda()

        features_s, output_batch = model(train_batch, is_feat=True)
        features_t, output_teacher_batch = teacher_model(train_batch, is_feat=True)
        output_teacher_batch = output_teacher_batch.detach()

        features_t = features_t[-1]
        features_s = features_s[-1]
        loss = loss_fn_crd(output_batch, labels_batch, output_teacher_batch, features_s, features_t, \
            index, contrastive_idx, criterion_crd, params)

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

    return (top1.avg, losses.avg)

def crd_mixup(epoch, model, teacher_model, optimizer, dataloader, criterion_crd, params):
    # top k的class乘以alpha，使得P(c)是满足topk ground truth的
    model.train()
    teacher_model.eval()

    # summary for current training loop and a running average object for loss
    loss_avg = RunningAverage()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Use tqdm for progress bar
    pbar = tqdm(dataloader)
    for i, (train_batch, labels_batch, index, contrastive_idx) in enumerate(pbar):
        # adjust lr
        if params.dataset == 'IMAGENET':
            imagenet_adjust_lr(optimizer, epoch, i, len(dataloader), params)

        pbar.set_description('Epoch {}'.format(epoch))
        # move to GPU if available
        train_batch, labels_batch, index, contrastive_idx = train_batch.cuda(), labels_batch.cuda(), index.long().cuda(), contrastive_idx.cuda()

        '''
        lam = np.random.beta(1.0, 1.0)
        index = torch.randperm(train_batch.size(0)).cuda()
        mixed_x = lam * train_batch + (1 - lam) * train_batch[index, :]
        mixed_y_a = (labels_batch.unsqueeze(1) == torch.arange(params.num_classes).cuda().reshape(1,
                                                                                                params.num_classes)).float()
        mixed_y_b = (labels_batch[index].unsqueeze(1) == torch.arange(params.num_classes).cuda().reshape(1,
                                                                                                        params.num_classes)).float()
        mixed_y = lam * mixed_y_a + (1 - lam) * mixed_y_b
        topk_ground_truth = (mixed_y_a + mixed_y_b).gt(0).float()
        '''

        if params.mixup_method == 'mixup':
            mixed_x, mixed_y, lam, index = mixup(train_batch, labels_batch, params)
            topk_ground_truth = mixed_y.gt(0).float()
        elif params.mixup_method == 'cutmix':
            mixed_x, mixed_y, lam, index = cutmix(train_batch, labels_batch, params)
            topk_ground_truth = mixed_y.gt(0).float()
        else:
            raise Exception('Not support mixup method %s' % params.mixup_method)

        features_t, output_teacher_batch = teacher_model(mixed_x, is_feat=True)
        output_teacher_batch = output_teacher_batch.detach()

        features_s, output_batch_mixed = model(mixed_x, is_feat=True)
        output_batch = model(train_batch)

        # get loss
        features_t = features_t[-1]
        features_s = features_s[-1]
        loss = loss_fn_crd_mixup(output_batch_mixed, output_teacher_batch, topk_ground_truth, mixed_y, labels_batch, labels_batch[index], lam, \
            features_s, features_t, index, contrastive_idx, criterion_crd, params)

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
