import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample

from helper.util import adjust_learning_rate,AverageMeter, accuracy

from matplotlib import pyplot as plt
from matplotlib import cm

from helper.loops import train_distill as train
from helper.pretrain import init
import os
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
    parser.add_argument('--distill', type=str, default='None') 

    opt = parser.parse_args()
    
    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    print(segments)
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

def load_teacher(model_path, n_cls,opt):
    print('==> loading teacher model')
    #model_t = get_teacher_name(model_path)
    model = model_dict[opt.arch](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    #model.load_state_dict(torch.load(model_path)['state_dict'])
    print('==> done')
    return model


def validate_SNE(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    try:
        from sklearn.manifold import TSNE

        HAS_SK = True
    except:
        HAS_SK = False
        print('Please install sklearn for layer visualization')
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        feature_list = []
        target_list = []
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            if opt.distill in ['clkd','pckd']:
                input = input[:, 0, :, :, :]

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            feat, logit,_= model(input,is_feat=True)
            feature=feat[-1]
            #print('2:')
            #print(feature.shape)
            loss = criterion(logit, target)

            feature_list.append(feature)
            target_list.append(target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logit, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        feature = torch.cat(feature_list, dim=0)
        feature = torch.nn.functional.normalize(feature, dim=-1)
        target = torch.cat(target_list, dim=0)
        print(feature.size())
        print(target.size())
       
        tsne = TSNE(perplexity=30, n_components=3, n_iter=5000)

        # plot_only = 5000

        low_dim_embs = tsne.fit_transform(feature.cpu().data.numpy()[:, :])
        print(len(low_dim_embs))
        labels = target.cpu().numpy()[:]
        print(len(labels))
        plot_with_labels(low_dim_embs, labels,opt)

    return top1.avg, top5.avg, losses.avg

def plot_with_labels(lowDWeights, labels,opt):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    fig = plt.figure()
   # temp = [0,9,18,27,36,45,54,63,72,81]
    # temp = [0,10,20,30,40,50,60,70,80,90]
    #temp = [0,20,40,60,80,99]
    # temp = [0,1,2,3,4,95,96,97,98,99]
    #temp = [0,1,2,4,50,95,96,97,98,99]
    
    
    
    #temp = [0,9,22,30,36,54,63,75,95]
    #temp = [0,5,15,30,45,60,75,90,99]   
    #temp = [0,10,20,40,60,70,80,90,99]
    
    #temp = [0,5,15,35,45,60,75,90,99]
    #temp = [0,5,15,20,45,60,75,90,99]
    temp = [0,5,15,20,50,60,75,90,99]
    
    colors = ['orange', 'pink', 'yellowgreen', 'grey', 'plum', 'r','lightgreen','blue','tan'] # ,''
    for index in range(9):
        x = lowDWeights[np.where(labels == temp[index]),0]
        y = lowDWeights[np.where(labels == temp[index]),1]
        plt.scatter(x, y, c=colors[index], s=10)

    #plt.scatter(X, Y, c=plt.cm.Set1(labels / 10.), s=10)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualization of CRD')
    #plt.show()
    print('1')
    img = os.path.join("/data/dingmuhe/CLCKD_NEW/photo/"+ opt.distill +".png")
    fig.savefig(img)
    #plt.pause(1)
    print('2')


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
    teacher_acc, teacher_acctop5, _ = validate_SNE(val_loader, model_t, criterion_cls, opt)

    print("top1: ",teacher_acc)
    print("top5: ",teacher_acctop5)

if __name__ == '__main__':
    main()