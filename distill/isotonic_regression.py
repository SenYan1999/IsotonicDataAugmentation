import torch

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

