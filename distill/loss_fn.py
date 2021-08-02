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

def loss_isotonic(outputs_mixed, teacher_outputs_mixed, lam, target_a, target_b, params):
    alpha = params.alpha
    T = params.temperature
    soft_constraint_ratio = params.soft_constraint_ratio

    teacher_corrected = isotonic_regression(teacher_outputs_mixed,
                                            (target_a.unsqueeze(1) == torch.arange(params.num_classes).cuda().view(1,-1)) * lam
                                            + (target_b.unsqueeze(1) == torch.arange(params.num_classes).cuda().view(1,-1)) *(1 - lam))

    loss = nn.KLDivLoss(reduction=params.reduction_method)(F.log_softmax(outputs_mixed / T, dim=-1), F.softmax(teacher_corrected / T, dim=-1)) * T * T

    return loss

def loss_fn(outputs_mixed, teacher_outputs_mixed, mixed_y, target_a, target_b, lam, params):
    # get hard loss
    if params.mixup_method == 'none':
        loss_hard = hard_loss(outputs_mixed, mixed_y)
        loss_soft = soft_loss(outputs_mixed, teacher_outputs_mixed, params)
        loss = params.alpha * loss_soft + (1 - params.alpha) * loss_hard
        return loss

    else:
        loss_hard = hard_loss_mixup(outputs_mixed, target_a, target_b, lam)

        # get soft loss
        loss_soft = soft_loss(outputs_mixed, teacher_outputs_mixed, params)

        # combine hard loss, soft loss and calibrated loss
        if params.calibration_method == 'isotonic_appr':
            topk_ground_truth = mixed_y.gt(0).float()
            loss_constraint = loss_isotonic_appr(torch.softmax(outputs_mixed, dim=-1), topk_ground_truth, mixed_y)
            loss = params.alpha * loss_soft + (1 - params.alpha) * loss_hard + params.soft_constraint_ratio * loss_constraint
        elif params.calibration_method == 'isotonic':
            loss_constraint = loss_isotonic(torch.softmax(outputs_mixed, dim=-1), teacher_outputs_mixed, lam, target_a, target_b, params)
            loss = params.alpha * loss_soft + (1 - params.alpha) * loss_hard + params.soft_constraint_ratio * loss_constraint
        else:
            loss = params.alpha * loss_soft + (1 - params.alpha) * loss_hard

        return loss