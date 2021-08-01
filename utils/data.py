import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .crd_data import CIFAR100IdxSample, CIFAR10IdxSample

def get_dataloader(dataset, half, batch_size, workers):
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
        train_dataset = datasets.__dict__[dataset](root='./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
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
    
    elif dataset == 'mini-imagenet':
        train_dataset = datasets.ImageFolder(
            root='~/mini-imagenet/train',
            transform=transforms.Compose([
                transforms.RandomResizedCrop(size=[224, 224]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize
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
            datasets.ImageFolder(root='~/mini-imagenet/val', transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(size=[224, 224]),
                transforms.ToTensor(),
                normalize
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True
        )
    
    elif dataset == 'IMAGENET':
        train_dataset = datasets.ImageFolder(
            root='~/imagenet/train',
            transform=transforms.Compose([
                transforms.RandomResizedCrop(size=[224, 224]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize
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
            datasets.ImageFolder(root='~/imagenet/val', transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(size=[224, 224]),
                transforms.ToTensor(),
                normalize
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True
        )
    
    if not half:
        train_sampler = None

    return train_loader, val_loader, train_sampler
