import torch
import numpy as np

def isotonic_regression(soft,hard):
    soft=soft.detach().clone()
    _, indices = torch.topk(hard,k=2,dim=-1)
    soft_t =soft.detach().clone()
    for i in range(soft_t.size(0)):
        soft_t[i][indices[i][0]] += 1e8
        soft_t[i][indices[i][1]] += 1e4
    _, indices_i = torch.sort(soft_t,-1,descending=True)
    for ii in range(soft.size(0)):
        i=2
        size_2 = 1.0
        size_1 = 1.0
        set2 = []
        set1 = []
        while i<soft.size(1) and soft[ii][indices_i[ii][i]]>soft[ii][indices[ii][1]]:
            index=indices_i[ii][i]
            soft[ii][indices[ii][1]] =(soft[ii][indices[ii][1]]*size_2+soft[ii][index]) /(size_2+1)
            size_2+=1
            set2.append(index)
            i+=1
        if soft[ii][indices[ii][1]]>soft[ii][indices[ii][0]]:
            soft[ii][indices[ii][0]] = (soft[ii][indices[ii][1]] * size_2 + soft[ii][indices[ii][0]]) / (size_2 + 1)
            size_1 = size_2+1
            set1.extend(set2)
            set1.append(indices[ii][1])
            while i < soft.size(1) and soft[ii][indices_i[ii][i]] > soft[ii][indices[ii][0]]:
                index = indices_i[ii][i]
                soft[ii][indices[ii][0]] = (soft[ii][indices[ii][0]] * size_1 + soft[ii][index]) / (size_1 + 1)
                size_1 += 1
                set1.append(index)
                i += 1
            for t in set1:
                soft[ii][t]=soft[ii][indices[ii][0]]
        else:
            for t in set2:
                soft[ii][t]=soft[ii][indices[ii][1]]
    return soft.detach()

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

