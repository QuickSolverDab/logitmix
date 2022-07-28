import time
import torch
import math
import random
import numpy as np
import torch.nn.functional as F
from tqdm  import trange
from utils import *

def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if 'logit' in args.mixmethod:
        m_loss, c_loss, s_loss = AverageMeter(), AverageMeter(), AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    total_step = len(train_loader)
    tbar = trange(0, total_step, total=total_step, initial=0)
    train_flow = iter(train_loader)
    vis_step = total_step // 50
    for steps in tbar:
        try:
            input, target = next(train_flow)
        except StopIteration:
            continue

        if args.gpu is not None:
            input = input.cuda(args.gpu)
            target = target.cuda(args.gpu)

        if 'mobilenet' in args.network:
            adjust_learning_rate(optimizer, epoch, args.epoch, steps, total_step,
                                 init_lr=args.lr, gamma=args.gamma)

        mixed_loss = 0

        if args.mixmethod == 'mixup':
            # default alpha = 0.4
            mixed_input, prob, index = mixup_data(input, alpha=0.4)
            output = model(mixed_input)
            loss = mixup_criterion(output, target, prob, index, is_LS=args.LS)

        elif args.mixmethod == 'cutmix':
            if 'cifar' in args.dataset:
                cutmix_prob = 0.5
            else:
                cutmix_prob = 1
            r = np.random.rand(1)
            if r < cutmix_prob:
                mixed_input, prob, index = cutmix_data(input, dist='beta', alpha=1.0)
                output = model(mixed_input)
                loss = mixup_criterion(output, target, prob, index, is_LS=args.LS)
            else:
                output = model(input)
                loss = cross_entropy(output, target, is_LS=args.LS)

        elif args.mixmethod == 'logitmix_M':
            r = np.random.rand(1)
            if args.logit_prob == 0:
                output = model(input)
                loss = cross_entropy(output, target, is_LS=args.LS)
            elif r < args.logit_prob or args.logit_prob == 1:
                batch_size = input.size()[0]
                mixed_input, prob, index = logitmix_data(input, alpha=args.alpha, dist=args.dist)
                output = midftr = model(torch.cat((input, mixed_input),0))
                mixed_loss, ce_loss, similarity = logitmix_criterion(output[:batch_size], output[batch_size:],
                                                                     midftr[:batch_size], midftr[batch_size:],
                                                                     target, prob, index, is_LS=args.LS)
                loss = args.weights[0] * mixed_loss + args.weights[1] * ce_loss + args.weights[2] * similarity
                output = output[batch_size:]
            else:
                mixed_input, prob, index = logitmix_data(input, alpha=args.alpha, dist=args.dist)
                output = model(mixed_input)
                loss = mixup_criterion(output, target, prob, index, is_LS=args.LS)

        elif args.mixmethod == 'logitmix_C':
            if 'cifar' in args.dataset:
                cutmix_prob = 0.5
            else:
                cutmix_prob = 1
            r = np.random.rand(1)
            if r < cutmix_prob:
                r = np.random.rand(1)
                if args.logit_prob == 0:
                    output = model(input)
                    loss = cross_entropy(output, target, is_LS=args.LS)
                elif r < args.logit_prob or args.logit_prob == 1:
                    batch_size = input.size(0)
                    mixed_input, prob, index = cutmix_data(input, dist=args.dist, alpha=args.alpha)
                    output = midftr = model(torch.cat((input, mixed_input),0))
                    mixed_loss, ce_loss, similarity = logitmix_criterion(output[:batch_size], output[batch_size:],
                                                                         midftr[:batch_size], midftr[batch_size:],
                                                                         target, prob, index, is_LS=args.LS)
                    loss = args.weights[0] * mixed_loss + args.weights[1] * ce_loss + args.weights[2] * similarity
                    output = output[batch_size:]
                else:
                    mixed_input, prob, index = cutmix_data(input, dist=args.dist, alpha=args.alpha)
                    output = model(mixed_input)
                    loss = mixup_criterion(output, target, prob, index, is_LS=args.LS)
            else:
                output = model(input)
                loss = cross_entropy(output, target, is_LS=args.LS)

        else:
            output = model(input)
            loss = cross_entropy(output, target, is_LS=args.LS)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        if 'logit' in args.mixmethod and mixed_loss != 0:
            m_loss.update(mixed_loss.item(), input.size(0))
            c_loss.update(ce_loss.item(), input.size(0))
            s_loss.update(similarity.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (steps + 1) % vis_step == 0 or steps == 0:
            if 'logit' in args.mixmethod and mixed_loss != 0:
                tbar.set_description('Epoch: [{:3d}] '
                        'Loss  {loss.avg:7.4f} '
                        'M/C/S  [{m.avg:7.4f}, {c.avg:7.4f}, {s.avg:7.4f}] '
                        'Acc@1 {top1.avg:7.3f} '
                        'Acc@5 {top5.avg:7.3f}'.format(
                        epoch, loss=losses, m=m_loss, c=c_loss, s=s_loss ,top1=top1, top5=top5))
            else:
                tbar.set_description('Epoch: [{:3d}] '
                        'Loss  {loss.val:7.4f} ({loss.avg:7.4f}) '
                        'Acc@1 {top1.val:7.3f} ({top1.avg:7.3f}) '
                        'Acc@5 {top5.val:7.3f} ({top5.avg:7.3f})'.format(
                        epoch, loss=losses, top1=top1, top5=top5))
    tbar.set_description('\n')

    if 'logit' in args.mixmethod:
        return top1.avg, m_loss.avg, c_loss.avg, s_loss.avg
    else:
        return top1.avg

def validate(val_loader, model, args=None, ece=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logit_output = []
    logit_label = []

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        total_step = len(val_loader)
        val_flow = iter(val_loader)
        tbar = trange(0, total_step, total=total_step, initial=0)
        for steps in tbar:
            try:
                input, target = next(val_flow)
            except StopIteration:
                continue

            # if args.gpu is not None:
            input = input.cuda(args.gpu) #args.gpu, non_blocking=True)
            target = target.cuda(args.gpu)#args.gpu, non_blocking=True)
            batch_size = input.size(0)
            # compute output
            output = model(input)
            if ece:
                if len(logit_output) == 0:
                    logit_output = output.detach().cpu().numpy()
                    logit_label  = target.detach().cpu().numpy()
                else:
                    logit_output = np.append(logit_output, output.detach().cpu().numpy(), axis=0)
                    logit_label  = np.append(logit_label, target.detach().cpu().numpy(), axis=0)

            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if steps == (total_step // 4) or steps == 0:
                tbar.set_description('Test'
                            'Loss  {loss.val:7.4f} ({loss.avg:7.4f}) '
                            'Acc@1 {top1.val:7.3f} ({top1.avg:7.3f}) '
                            'Acc@5 {top5.val:7.3f} ({top5.avg:7.3f}) '.format(
                            loss=losses, top1=top1, top5=top5))

        tbar.set_description('\n')
    if ece:
        return top1.avg, top5.avg, logit_output, logit_label
    else:
        return top1.avg, top5.avg
