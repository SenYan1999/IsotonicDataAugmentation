import argparse

parser = argparse.ArgumentParser()

# global setting
parser.add_argument('--random_seed', default=230, type=int)


# data details
parser.add_argument('--dataset', default='CIFAR10', help='dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')


# distributed and fp16 details
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')


# model details
parser.add_argument('--arch', metavar='ARCH', default='resnet20')
parser.add_argument('--teacher_arch', metavar='TEA_ARCH', default='resnet110')
parser.add_argument("--num_classes", default=10, type=int)
parser.add_argument("--local_rank", default=0, type=int)


# optim details
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# distill details
parser.add_argument('--temperature', default=6, type=float, metavar='T', help='temperature')
parser.add_argument('--mixup_method', default='none', choices=['none', 'mixup', 'cutmix'], type=str)
parser.add_argument('--calibration_method', type=str, default='none', choices=['none', 'isotonic', 'isotonic_appr'])
parser.add_argument('--soft_constraint_ratio', type=float, default=2.)
parser.add_argument('--alpha', type=float, default=0.95)
parser.add_argument('--beta', type=float, default=0.8)

# log and save details
parser.add_argument('--model_dir', default='experiments',
                    help="Directory containing params.json")
parser.add_argument('--save_dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--log_dir', default='logs/', type=str)
parser.add_argument('--tensorboard_dir', default=None, type=str)

# something
parser.add_argument('--reduction_method', default='batchmean', type=str)
parser.add_argument('--prob', default=1.0, type=float)

# for crd
parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
parser.add_argument('--nce_n', default=16384, type=int, help='number of negative samples for NCE')
parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
parser.add_argument('--crd_data_mode', type=str, default='exact')
parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

args = parser.parse_args()
