import torch
import numpy as np

def isotonic_regression(soft_label, hard_label):
    soft_label = soft_label.detach().clone()
    _, indices_h = torch.topk(hard_label, k=2, dim=-1)
    soft_t = soft_label.detach().clone()

    for i in range(soft_t.size(0)):
        soft_t[i][indices_h[i][0]] += 1e8 # top1 label
        soft_t[i][indices_h[i][1]] += 1e4 # top2 label
    
    _, indices_s = torch.sort(soft_t, -1, descending=True)

    for i in range(soft_label.size(0)):
        j = 2
        top1_idx, top2_idx = indices_s[i][0], indices_s[i][1]
        bin1, bin2 = [top1_idx], [top2_idx]

        while j < soft_label.size(1) and soft_label[i][indices_s[i][j]] > soft_label[i][top2_idx]:
            index = indices_s[i][j] # find order violation and record the index
            bin_avg = (soft_label[i][top2_idx] * len(bin2) + soft_label[i][index]) / (len(bin2) + 1)
            soft_label[i][top2_idx] = bin_avg
            bin2.append(index) # append order violation node to bin 2
            j += 1
        
        if soft_label[i][top2_idx] > soft_label[i][top1_idx]:
            bin_avg = (soft_label[i][top2_idx] * len(bin2) + soft_label[i][top1_idx]) / (len(bin2) + 1)
            soft_label[i][top1_idx] = bin_avg
            bin1 += bin2
            
            while j < soft_label.size(1) and soft_label[i][indices_s[i][j]] > soft_label[i][top1_idx]:
                index = indices_s[i][j]
                bin_avg = (soft_label[i][top1_idx] * len(bin1) + soft_label[i][index]) / (len(bin1) + 1)
                soft_label[i][top1_idx] = bin_avg
                bin1.append(index)
                j += 1

            for t in bin1:
                soft_label[i][t] = soft_label[i][top1_idx]
        else:
            for t in bin2:
                soft_label[i][t] = soft_label[i][top2_idx]
    return soft_label.detach()

