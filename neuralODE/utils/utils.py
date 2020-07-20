import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import logging
import torch
import os


FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'
FORMAT_MINIMAL = '%(message)s'

logger = logging.getLogger('neuralODE')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def learning_rate_with_decay(
        lr,
        batch_size,
        batch_denom,
        batches_per_epoch,
        boundary_epochs,
        decay_rates):
    initial_learning_rate = lr*1. #* batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader, args):
    total_correct = 0
    for x, y in dataset_loader:
        if args.gpu:
            x = x.cuda()
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def flatten(grad_calculator):
    def wrapper():
        return torch.cat([grad.view(-1) for grad in grad_calculator()])

    return wrapper


def exp_lr_scheduler(epoch,
                     optimizer,
                     strategy=True,
                     decay_eff=0.1,
                     decayEpoch=[]):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""

    if strategy == 'normal':
        if epoch in decayEpoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_eff
    else:
        print('wrong strategy')
        raise ValueError('A very specific bad thing happened.')

    return optimizer


def lr_calculator(epoch,
                  current_lr,
                  strategy=True,
                  decay_eff=0.1,
                  decayEpoch=[]):

    if strategy == 'normal':
        if epoch in decayEpoch:
            return current_lr * decay_eff
        else: 
            return current_lr
    else:
        print('wrong strategy')
        raise ValueError('A very specific bad thing happened.')


def fb_warmup(optimizer, epoch, baselr, large_ratio):
    for param_group in optimizer.param_groups:
        param_group['lr'] = epoch * \
            (baselr * large_ratio - baselr) / 5. + baselr
    return optimizer


# useful
def group_add(params, update, lmbd=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].add_(update[i] * lmbd + 0.)
    return params


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def get_p_g_m(opt, layers):
    i = 0
    paramlst = []
    grad = []
    mum = []

    for group in opt.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']

        for p in group['params']:
            if p.grad is None:
                continue
            p.grad.data = p.grad.data
            d_p = p.grad.data
            if momentum != 0:
                param_state = opt.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(
                        p.data)
                else:
                    buf = param_state['momentum_buffer']
            if i in layers:
                paramlst.append(p.data)
                grad.append(d_p + 0.)  # get grad
                mum.append(buf * momentum + 0.)
            i += 1
    return paramlst, grad, mum


def manually_update(opt, grad):
    for group in opt.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for i, p in enumerate(group['params']):
            d_p = grad[i]
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            if momentum != 0:
                param_state = opt.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(
                        p.data)
                    buf.mul_(momentum).add_(d_p)
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf
            p.data.add_(-group['lr'], d_p)


def change_lr_single(optimizer, best_lr):
    """change learning rate"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = best_lr
    return optimizer


def bad_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal(m.weight, std=0.5)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=0.5)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)