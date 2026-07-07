import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample

from helper.util import adjust_learning_rate



from helper.loops import train_distill as train, validate
from helper.pretrain import init

def parse_option():


    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100','cifar10'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--arch', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--distill', type=str, default='None', choices=['kd', 'mlkd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst']) 

    opt = parser.parse_args()
    
    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-1].split('_')
    print(segments)
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

def load_teacher(model_path, n_cls,opt):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[opt.arch](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    #model.load_state_dict(torch.load(model_path)['state_dict'])
    print('==> done')
    return model



def main():
    criterion_cls = nn.CrossEntropyLoss()
    opt = parse_option()
   
    if opt.dataset == 'cifar100':
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                       num_workers=opt.num_workers,
                                                                       is_instance=True)

            n_cls = 100
    elif opt.dataset == 'cifar10':
            train_loader, val_loader, n_data = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                                       num_workers=opt.num_workers,
                                                                       is_instance=True)
            n_cls = 10

    model_t = load_teacher(opt.path_t, n_cls,opt).cuda()
    teacher_acc, teacher_acctop5, _ = validate(val_loader, model_t, criterion_cls, opt)

    print("top1: ",teacher_acc)
    print("top5: ",teacher_acctop5)

if __name__ == '__main__':
    main()