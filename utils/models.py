import torchvision.models as tvmodels
from models import models_cifar10, models_cifar100

def select_model(arch_name,args):
    if args.dataset == 'CIFAR10':
        model = models_cifar10.__dict__[arch_name]()
        print('Now is CIFAR10 model')
    elif args.dataset == 'CIFAR100':
        model = models_cifar100.__dict__[arch_name]()
        print('Now is CIFAR100 model')
    elif args.dataset == 'STL10':
        model = tvmodels.__dict__[arch_name](pretrained=False, num_classes=args.num_classes)
        print('Now is TORCHVISION model')
    elif args.dataset == 'IMAGENET':
        pretrained = True if arch_name == 'resnet34' else False
        print(arch_name)
        print(pretrained)
        model = tvmodels.__dict__[arch_name](pretrained=pretrained, num_classes=args.num_classes)
        print('Now is TORCHVISION model')
    elif args.dataset == 'mini-imagenet':
        if arch_name == 'resnet34':
            pretrained=True
        else:
            pretrained=False
        pretrained=False
        print(f"model {arch_name} pretrained {pretrained}")
        model = tvmodels.__dict__[arch_name](pretrained=pretrained, num_classes=args.num_classes)
        print('Now is TORCHVISION model')
    else:
        raise Exception('Please input the supported datasets: [CIAFR10, CIFAR100, STL10]')

    return model

