import os
import shutil
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import torchvision


""" Training scheduler for cifar, tiny_imagenet, and imagenet training
"""
def lr_scheduler(model, init_lr, args):
    if 'cifar' in args.dataset:
        if 'pyramid' in model:
            schedule = [0, 150, 225]
            init_lr = 0.25
            gamma = 0.1
            epoch = 300
            weight_decay = 1e-4
        else:
            epoch = args.epoch
            gamma = args.gamma
            weight_decay = args.wd
            if args.gamma == 0.1:
                schedule = [0, 100, 150]
            elif args.gamma == 0.2:
                schedule = [0, 60, 120, 160]
            else:
                raise UserError('Need to other setting')

    elif 'imagenet' in (args.dataset).lower():
        ## tiny_imagenet or imagenet
        if ('mobile' in args.network):
            print('mobilenetv2 training scheme setting!')
            epoch = args.epoch = 150
            weight_decay = args.wd == 4e-5
            args.lr == 0.05
            args.schedule = args.gamma = 'Cyclic'
            return None
        else:
            epoch = args.epoch
            gamma = args.gamma
            weight_decay = args.wd
            if epoch == 300:
                schedule = [0, 75, 150, 225]
            else: #200
                schedule = [0, 60, 120, 160]

    ## training scheduler
    _lr_schedule = dict({'total_epoch': epoch,
                        'weight_decay': weight_decay,
                        'gamma': gamma,
                        'schedule': schedule})
    _lr_steps = OrderedDict()
    schedule += [epoch]
    for i, step in enumerate(schedule):
        _lr_schedule.update({step: round(init_lr*(gamma**i), 8)})
        _lr_steps.update({step: round(init_lr*(gamma**i), 8)})
    args.wd, args.gamma = weight_decay, gamma
    args.epoch, args.schedule = epoch, _lr_steps
    return _lr_schedule

def warmup(epoch, min_epoch, max_epoch, min_ratio, max_ratio, first=None):
    eps = 1e-12
    if first == 1:
        bound = 0
    else:
        bound = eps
    range= max_epoch - min_epoch
    init = min_epoch + int(range * min_ratio)
    fin  = max_epoch - int(range * max_ratio)
    inc  = (1 - bound) / (fin - init + eps)
    if epoch < init:
        return bound
    elif (init <= epoch) and (epoch < fin):
        return  min(max(0, bound + (epoch - init) * inc), 1)
    else:
        return 1

from math import cos, pi
def adjust_learning_rate(optimizer, epoch, max_epoch, iteration, num_iter, init_lr, gamma=None, warmup=None, lr_decay='cos'):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = max_epoch * num_iter

    if lr_decay == 'step':
        lr = init_lr * (gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif lr_decay == 'cos':
        lr = init_lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif lr_decay == 'linear':
        lr = init_lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    else:
        raise ValueError('Unknown lr mode {}'.format(lr_decay))

    if epoch < warmup_epoch:
        lr = init_lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


""" Class and function for recording
"""
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def make_dir(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError:
            pass
    return dir

def make_save_dir(args, save_dir = './ckpt', network=None):
    assert network is not None
    if args.mp:
        save_dir = os.path.join(save_dir, network + '_' + args.dataset, args.time)
    else:
        save_dir = os.path.join(save_dir, network + '_' + args.dataset,
                                'gpu'+ str(args.gpu) + '_' + args.mixmethod + '_' + args.repeat + '_' + args.time)
    make_dir(save_dir)
    return save_dir

def save_checkpoint(save_dir, args, is_best, state, filename='checkpoint.pth.tar'):
    ckpt_dir = os.path.join(save_dir, filename)
    torch.save(state, ckpt_dir)
    if is_best:
        best_dir = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(ckpt_dir, best_dir)

def record(save_dir, args, state):
    record_dir = 'GPU{}_{}_Record_{}.txt'.format(args.gpu, args.repeat, args.time)
    if not os.path.exists(os.path.join(save_dir, record_dir)):
        file = open(os.path.join(save_dir, record_dir), 'w')
        file.write('gpu : {} \n'.format(args.gpu))
        file.write('pretrained : {} \n'.format(args.pretrained))
        file.write('Network / Dataset: {} / {} \n'.format(args.network, args.dataset))
        file.write('Mix method : {} \n'.format(args.mixmethod))

        file.write('\n')
        file.write('Epoch / Batch_size / weight_decay: {} / {} / {} \n'.format(args.epoch, args.batch_size, args.wd))
        file.write('init learning {} with gamma {} \n'.format(args.lr, args.gamma))
        file.write('learning rate schedule : {} \n'.format(args.schedule))
        file.write('\n')

        if 'logitmix' in args.mixmethod:
            file.write('Weights : {} \n'.format(args.weights))
            if args.dist == 'normal':
                file.write('Distribution: {}, std: {} \n'.format(args.dist, args.alpha))
            elif args.dist == 'beta':
                file.write('Distribution: {}, alpha: {} \n'.format(args.dist, args.alpha))
            file.write('Label smoothing : {} \n'.format(args.LS))
            if args.info is not None:
                file.write('information : {} \n'.format(args.info))
        else:
            file.write('alpha : {} \n'.format(args.alpha))
        file.write('\n')
        if 'logitmix' in args.mixmethod:
            file.write('Epoch: {:3d}, M/C/S: [{:.4f} / {:.4f} / {:.4f}], Train acc {:.2f}, Test acc {:.2f}, Best_acc: {:2f} \n'
                       .format(state['epoch'], state['mixup'], state['cls'], state['sim'],
                               state['train_acc'], state['test_acc'], state['best_acc']))
        else:
            file.write('Epoch: {:3d}, Train acc {:.4f}, Test acc {:.4f}, Best_acc: {:4f} \n'
                       .format(state['epoch'], state['train_acc'], state['test_acc'], state['best_acc']))
        file.close()
    else:
        file = open(os.path.join(save_dir, record_dir), 'a')
        if 'logitmix' in args.mixmethod:
            file.write('Epoch: {:3d}, M/C/S: [{:.4f} / {:.4f} / {:.4f}], Train acc {:.2f}, Test acc {:.2f}, Best_acc: {:2f} \n'
                       .format(state['epoch'], state['mixup'], state['cls'], state['sim'],
                               state['train_acc'], state['test_acc'], state['best_acc']))
        else:
            file.write('Epoch: {:3d}, Train acc {:.4f}, Test acc {:.4f}, Best_acc: {:4f} \n'
                       .format(state['epoch'], state['train_acc'], state['test_acc'], state['best_acc']))
        file.close()
    shutil.copy(os.path.join(save_dir, record_dir),
                os.path.join(save_dir, '../'))

""" Classes and functions
"""
class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        oe = torch.zeros(1, device=logits.device)
        ece_per_bin = []
        prob_CE = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].float().mean()
                CE = avg_confidence_in_bin - accuracy_in_bin
                ece += torch.abs(CE) * prop_in_bin
                ece_per_bin.append(accuracy_in_bin.cpu().numpy())
                oe += avg_confidence_in_bin * torch.max(CE, torch.zeros_like(CE, device=logits.device)) * prop_in_bin
                prob_CE.append((prop_in_bin.cpu().numpy(), torch.abs(CE).cpu().numpy()))
            else:
                ece_per_bin.append(0)
                prob_CE.append((0,0))

        return ece.cpu().numpy()[0], ece_per_bin, oe.cpu().numpy()[0], prob_CE


""" Mixup
"""
def mixup_data(x, alpha=0.4, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = round(lam, 6)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda(x.device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    return mixed_x, lam, index

def mixup_criterion(pred, target, prob, index, is_LS=False):
    if is_LS:
        target = label_smoothing(pred, target)
    y_a, y_b = target, target[index]
    return prob * cross_entropy(pred, y_a, is_LS=is_LS) + \
           (1 - prob) * cross_entropy(pred, y_b, is_LS=is_LS)

""" Manifold Mixup
"""
def mixing_manifold(x, alpha='1.0', dist='beta', is_half=False):
    if is_half:
        batch_size = x.size(0)
        x1, x2 = torch.split(x, batch_size//2, dim=0)
        mixed_x, prob, index = logitmix_data(x1, alpha=alpha, dist=dist)
        x = torch.cat((mixed_x, x2))
    else:
        x, prob, index = logitmix_data(x, alpha=alpha, dist=dist)
    return x, prob, index

"""Cutmix
"""
def cutmix_data(x, alpha=1.0, dist='beta', use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if 'beta' in dist:
        lam = np.random.beta(alpha, alpha)
    elif 'normal' in dist:
        lam = np.clip(np.random.normal(0.5, alpha), 0, 1.)
    else:
        raise NotImplementedError

    lam = round(lam, 6)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda(x.device)
    else:
        index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    mixed_x = x
    return mixed_x, lam, index

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

""" LogitMix
"""
def logitmix_data(x, alpha=.5, dist='beta', use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if 'beta' in dist:
        lam = np.random.beta(alpha, alpha)
    elif 'normal' in dist:
        lam = np.clip(np.random.normal(0.5, alpha), 0, 1.)
    else:
        raise NotImplementedError
    lam = round(lam, 6)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda(x.device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, lam, index

def logitmix_criterion(org_pred, mixed_pred, org_mid, mixed_mid,
                       target, prob, index, is_LS=False):
    batch_size = target.size(0)
    if is_LS:
        target = label_smoothing(org_pred, target)
    # labels
    y_a, y_b = target, target[index]
    # mixup loss
    mixed_loss = prob * cross_entropy(mixed_pred, y_a, is_LS=is_LS) + \
                (1 - prob) * cross_entropy(mixed_pred, y_b, is_LS=is_LS)
    # cross entropy loss
    ce_loss = cross_entropy(org_pred, y_a, is_LS=is_LS)
    # similarity loss
    similarity =  similarity_loss(org_mid, mixed_mid, index, prob)
    return mixed_loss, ce_loss, similarity

def similarity_loss(original, mixed, index, prob):
    """ similartiy losses
        by using mse, kl_div...
    """
    assert mixed.size() == original.size()
    mixed_org = prob * original + (1 - prob) * original[index]
    loss = sym_mse_loss(mixed, mixed_org)
    return loss


""" Losses
"""
def sym_mse_loss(input_logits, target_logits):
    """ mse_loss for update input and target simultaneously
    """
    return torch.mean(torch.sum((input_logits - target_logits)**2, dim=1) / input_logits.size(1))

def cross_entropy(input, target, is_LS=False):
    """ Cross entropy for one-hot labels
    """
    if is_LS:
        target = label_smoothing(input, target)
        return -torch.mean(torch.sum(target * F.log_softmax(input, dim=1), dim=1))
    else:
        return F.cross_entropy(input, target)

def softmax_kl_loss(input_logits, target_logits):
    """ Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax)

""" etc
"""
def idx2onehot(idx, n):
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).cuda(idx.gpu)
    onehot.scatter_(1, idx, 1)
    return onehot

def label_smoothing(pred, target, alpha=0.1):
    target_ = target.contiguous()
    one_hot = torch.zeros_like(pred).scatter(1, target_.view(-1, 1), 1)
    n_class = pred.size(1)
    smoothed_one_hot = one_hot * (1-alpha) + (1-one_hot) * alpha / (n_class - 1)
    return smoothed_one_hot
