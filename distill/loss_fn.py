import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *

def hard_loss(outputs, truth):
    return F.cross_entropy(outputs, truth)

def soft_loss(s_logit, t_logit, params):
    T = params.temperature
    loss = F.kl_div(F.log_softmax(s_logit / T, dim=1), F.softmax(t_logit / T, dim=1), reduction=params.reduction_method) * T * T
    return loss

def hard_loss_mixup(outputs_mixed, target_a, target_b, lam):
    hard_loss = lam * F.cross_entropy(outputs_mixed, target_a) + (1 - lam) * F.cross_entropy(outputs_mixed, target_b)
    return hard_loss

def soft_loss_mixup_no_correction(outputs_mixed, teacher_outputs_mixed, params):
    T = params.temperature
    return F.kl_div(F.log_softmax(outputs_mixed / T, dim=1), F.softmax(teacher_outputs_mixed / T, dim=1), reduction=params.reduction_method) * T * T

def soft_loss_mixup_correction(outputs_mixed, teacher_outputs_mixed, topk_ground_truth, params):
    T = params.temperature

    # correction
    min_in_ground_truth = torch.min(teacher_outputs_mixed + (1 - topk_ground_truth) * 1e8, dim=-1)[0]
    max_in_false = torch.max(teacher_outputs_mixed - (topk_ground_truth) * 1e8, dim=-1)[0]
    increment = torch.clamp(max_in_false - min_in_ground_truth, min=0).unsqueeze(-1)
    teacher_outputs_mixed_correction = teacher_outputs_mixed + increment * topk_ground_truth

    # soft label (there may be some errors! havn't multiply T^2, see the correction label)
    loss = nn.KLDivLoss(reduction=params.reduction_method)(F.log_softmax(outputs_mixed / T, dim=-1), F.softmax(teacher_outputs_mixed_correction / T, dim=-1)) * T * T

    return loss

def soft_loss_mixup_correction_plus_remain_sum(outputs_mixed, teacher_outputs_mixed, topk_ground_truth, params):
    T = params.temperature

    # correction
    min_in_ground_truth = torch.min(teacher_outputs_mixed + (1 - topk_ground_truth) * 1e8, dim=-1)[0]
    max_in_false = torch.max(teacher_outputs_mixed - (topk_ground_truth) * 1e8, dim=-1)[0]
    value = torch.clamp(max_in_false - min_in_ground_truth, min=0).unsqueeze(-1)
    increment = value / 2 - value / params.num_classes
    decrement = value / params.num_classes
    teacher_outputs_mixed_correction = teacher_outputs_mixed + increment * topk_ground_truth - decrement * (1 - topk_ground_truth)

    # soft label (there may be some errors! havn't multiply T^2, see the correction label)
    loss = nn.KLDivLoss(reduction=params.reduction_method)(F.log_softmax(outputs_mixed / T, dim=-1), F.softmax(teacher_outputs_mixed_correction / T, dim=-1)) * T * T

    return loss

def loss_isotonic_appr(outputs_mixed, topk_ground_truth, mixed_y):
    min_in_ground_truth = torch.min(outputs_mixed + (1 - topk_ground_truth) * 1e8, dim=-1)[0]
    max_in_false = torch.max(outputs_mixed - (topk_ground_truth) * 1e8, dim=-1)[0]

    loss1 = torch.mean(torch.clamp(max_in_false - min_in_ground_truth, min=0))

    # constraint that top1 > top2
    topk = torch.topk(mixed_y, k=2)[1]
    loss2 = torch.mean(torch.clamp(outputs_mixed.gather(1, topk[:, 1].reshape(-1, 1)) -
                                   outputs_mixed.gather(1, topk[:, 0].reshape(-1, 1)), min=0))

    soft_constraint_loss = loss1 + loss2

    return soft_constraint_loss

'''
def loss_isotonic_appr(outputs_mixed, teacher_outputs_mixed, topk_ground_truth, labels_mixed, params):
    alpha = params.alpha
    T = params.temperature

    loss3 = nn.KLDivLoss()(F.log_softmax(outputs_mixed,dim=1),labels_mixed)* (1. - alpha)
    #F.cross_entropy(outputs, labels) * (1. - alpha)

    loss1 = nn.KLDivLoss(reduction='none')(F.log_softmax(outputs_mixed / T, dim=1),
                                           F.softmax(teacher_outputs_mixed / T, dim=1)) * (alpha * T * T)/(teacher_outputs_mixed.size(0)*teacher_outputs_mixed.size(1))

    teacher_outputs_mixed_constraint = teacher_outputs_mixed * topk_ground_truth - (1-topk_ground_truth) * 1e8
    loss2 = nn.KLDivLoss(reduction='none')(F.log_softmax(outputs_mixed / T, dim=1),
                                           F.softmax(teacher_outputs_mixed_constraint / T, dim=1)) * (alpha * T * T)/(teacher_outputs_mixed.size(0)*teacher_outputs_mixed.size(1))

    ratio=0.1

    ret_loss = torch.sum(loss1*ratio + loss2*(1-ratio))+loss3
    return ret_loss
'''

def loss_isotonic(outputs_mixed, teacher_outputs_mixed, lam, target_a, target_b, params):
    alpha = params.alpha
    T = params.temperature
    soft_constraint_ratio = params.soft_constraint_ratio

    teacher_corrected = isotonic_regression(teacher_outputs_mixed,
                                            (target_a.unsqueeze(1) == torch.arange(params.num_classes).cuda().view(1,-1)) * lam
                                            + (target_b.unsqueeze(1) == torch.arange(params.num_classes).cuda().view(1,-1)) *(1 - lam))

    loss = nn.KLDivLoss(reduction=params.reduction_method)(F.log_softmax(outputs_mixed / T, dim=-1), F.softmax(teacher_corrected / T, dim=-1)) * T * T

    return loss
