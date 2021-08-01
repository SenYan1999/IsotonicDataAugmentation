from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
from PIL import Image
import torchvision.datasets as dst

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
Modified from https://github.com/HobbitLong/RepDistiller/blob/master/dataset/cifar100.py
'''

class CIFAR10IdxSample(dst.CIFAR10):
    def __init__(self, root, train=True, 
                 transform=None, target_transform=None,
                 download=False, n=4096, mode='exact', percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.n = n
        self.mode = mode

        num_classes = 10
        num_samples = len(self.data)
        labels = self.targets

        self.cls_positive = [[] for _ in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[labels[i]].append(i)

        self.cls_negative = [[] for _ in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            num = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:num]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.mode == 'exact':
            pos_idx = index
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positive[target], 1)[0]
        else:
            raise NotImplementedError(self.mode)
        replace = True if self.n > len(self.cls_negative[target]) else False
        neg_idx = np.random.choice(self.cls_negative[target], self.n, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

        return img, target, index, sample_idx


class CIFAR100IdxSample(dst.CIFAR100):
    def __init__(self, root, train=True, 
                 transform=None, target_transform=None,
                 download=False, n=4096, mode='exact', percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.n = n
        self.mode = mode

        num_classes = 100
        num_samples = len(self.data)
        labels = self.targets

        self.cls_positive = [[] for _ in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[labels[i]].append(i)

        self.cls_negative = [[] for _ in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            num = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:num]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.mode == 'exact':
            pos_idx = index
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positive[target], 1)[0]
        else:
            raise NotImplementedError(self.mode)
        replace = True if self.n > len(self.cls_negative[target]) else False
        neg_idx = np.random.choice(self.cls_negative[target], self.n, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

        return img, target, index, sample_idx

def get_dataloader(dataset, half, batch_size, workers, args):
    mean = {
        'CIFAR10': (0.4914, 0.4822, 0.4465),
        'CIFAR100': (0.5071, 0.4867, 0.4408),
        'STL10': (0.5, 0.5, 0.5),
        'mini-imagenet': (0.485, 0.456, 0.406),
        'IMAGENET': (0.485, 0.456, 0.406)
    }

    std = {
        'CIFAR10': (0.2023, 0.1994, 0.2010),
        'CIFAR100': (0.2675, 0.2565, 0.2761),
        'STL10': (0.5, 0.5, 0.5),
        'IMAGENET': (0.229, 0.224, 0.225),
        'mini-imagenet': (0.229, 0.224, 0.225)
    }
    normalize = transforms.Normalize(mean=mean[dataset],
                                     std=std[dataset])

    if dataset.startswith('CIFAR'):
        Dataset = CIFAR100IdxSample if dataset == 'CIFAR100' else CIFAR10IdxSample
        train_dataset = Dataset(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), download=True, n=args.nce_n, mode=args.crd_data_mode)
        if half:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       num_workers=workers, pin_memory=True, sampler=train_sampler)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       num_workers=workers, pin_memory=True, shuffle=True)

        val_dataset = datasets.__dict__[dataset](root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=128, shuffle=False,
                                                 num_workers=workers, pin_memory=True)

    elif dataset == 'STL10':
        train_dataset = datasets.STL10(
                root='./data', split='train', download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(96),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))

        if half:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       num_workers=workers, pin_memory=True, sampler=train_sampler)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       num_workers=workers, pin_memory=True, shuffle=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.STL10(
                root='./data', split='test', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False)

    elif dataset == 'IMAGENET':
        train_dataset = datasets.ImageFolder(
            root='/nfsshare/home/cuiwanyun/imagenet/ILSVRC2012/train',
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )

        if half:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=workers, pin_memory=True, sampler=train_sampler
        )

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root='/nfsshare/home/cuiwanyun/imagenet/ILSVRC2012/val', transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True
        )

    if not half:
        train_sampler = None

    return train_loader, val_loader, train_sampler
