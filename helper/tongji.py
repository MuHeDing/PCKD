from __future__ import print_function, division

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import AverageMeter, accuracy
from helper.losses import focal_loss,crossentropyloss
import math

def train_distill_tongji(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]

    criterion_corr = criterion_list[1]
    
    criterion_kd = criterion_list[2]

    

    if opt.distill == 'pckd':
        
        criterion_classifier = criterion_list[3]
        criterion_align= criterion_list[4]
        criterion_cls_id = nn.CrossEntropyLoss(reduction='none')
        
        
    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    pt = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):

        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        elif opt.distill in ['pckd']:
            input, target = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if opt.distill not in ['pckd']:
                index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True

        if opt.distill == 'pckd':
            c, h, w = input.size()[-3:]
            input = input.view(-1, c, h, w)
            batch_size = int(input.size(0) / 4)
            nor_index = (torch.arange(4 * batch_size) % 4 == 0).cuda()
            aug_index = (torch.arange(4 * batch_size) % 4 != 0).cuda()

        feat_s, logit_s,fc_s = model_s(input, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_t, logit_t,fc_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        if opt.distill == 'pckd':
            loss_cls = criterion_cls(logit_s[nor_index], target)
        else:
            loss_cls = criterion_cls(logit_s, target)

        if opt.distill == 'pckd':
            loss_corr=0
        else :
            loss_corr = criterion_corr(logit_s, logit_t)
        
        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
            
        elif opt.distill == 'focal_kd':
            loss_kd = criterion_kd(logit_s, target, logit_t)
            
            
            
        elif opt.distill == 'pckd':

            f_s = feat_s[-1]
            f_t = feat_t[-1]
            
            l=[]
            for a in fc_s.parameters():
                if a.ndim==1:
                    a=torch.unsqueeze(a,1)
                else :
                    fcl_s=a
                l.append(a)
                
            fc_sm=torch.cat(l,1)
            
            l=[]
            for a in fc_t.parameters():
                if a.ndim==1:
                    a=torch.unsqueeze(a,1)
                else :
                    fcl_t=a
                l.append(a)
                
            fc_tm=torch.cat(l,1)  

            if fc_sm.shape!=fc_tm.shape:
                loss_corr=0
                # print('no_corr')
            else:
                loss_corr = criterion_corr(fc_sm,fc_tm)
                # print('corr')
            
            fcl_s=fcl_s.transpose(0,1)
            fcl_t=fcl_t.transpose(0,1)     
            
            aug_target = target.unsqueeze(1).expand(-1, 4).contiguous().view(-1).long().cuda()

            loss_kd = criterion_kd(f_s, f_t, fcl_s, fcl_t, aug_target)
            
            
            f_s_nor = f_s[nor_index]
            f_t_nor = f_t[nor_index]
            f_t_list = []
            for i in range(4):
                aug_index = (torch.arange(4 * batch_size) % 4 == i).cuda()
                f_t_aug = f_t[aug_index]
                f_t_list.append(f_t_aug)
            
            loss_align=criterion_align(f_s, f_t)      
            
                        
            loss_tem=criterion_classifier(logit_s[nor_index], logit_t[nor_index],'none') # 64  100
            loss_tem=loss_tem.sum(1).cuda()
            
            loss_cls_id = criterion_cls_id(logit_s[nor_index],target)

            loss_cls_mean = criterion_cls(logit_s[nor_index],target)
            
            alpha=torch.div(loss_cls_id,loss_cls_mean)
           
            # t = math.pow(1.015, epoch)
            # t = math.pow(1.005, epoch)
            t = math.pow(1.01, epoch) # 为了画图
            one = torch.ones(alpha.shape).cuda()
            #zero= torch.zeros(alpha.shape).cuda()
            hard = torch.pow(alpha, 2)
            hard = -hard
            hard = torch.exp(hard)

            percent = torch.where(alpha <= t, one, hard)
            #p1= torch.where(alpha <= t, one,zero)
            #p2= torch.where(alpha <= t, zero,hard)
            
            l = []
            #l1 = []
            #l2 = []
            
            for i in range(batch_size):
              for j in range(4):
                l.append(percent[i])

            percent2=torch.Tensor(l).cuda()


                                  
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)
        
        if opt.distill =='pckd':

            loss1=torch.dot(percent,loss_tem).cuda()           
            loss1=opt.delta*loss1
            
            
            
                                 
            loss2 =torch.dot(percent2,opt.beta * loss_kd).cuda()
            loss2=torch.div(loss2,4*batch_size)
            
                        
            loss=loss_cls+loss1+loss2+opt.alpha * loss_corr +opt.ff*loss_align
      
            
        else :
            
            loss= opt.r * loss_cls + opt.alpha * loss_corr + opt.beta * loss_kd 
           
            
            
        if opt.distill == 'pckd':
            acc1, acc5 = accuracy(logit_s[nor_index], target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        

        pt.update(percent.mean().item(), input.size(0)/4)
        
        print(pt.val)
        #print(pt.avg)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush() 

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg, pt.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            if opt.distill in ['clkd','pckd']:
                input = input[:, 0, :, :, :]

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
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

    return top1.avg, top5.avg, losses.avg
