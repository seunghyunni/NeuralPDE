import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import grad, Variable
import time
import copy
from model.PDE_solver import EntireNet, PDE_solver
from utils.utils import RunningAverageMeter, Flatten, learning_rate_with_decay, one_hot, count_parameters, norm, norm_
from data_loader import get_mnist_loaders, inf_generator

parser = argparse.ArgumentParser()
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    # logger.info(args)

    # Model
    downsampling_layers = [nn.Conv2d(1, 64, 3, 1), norm(64),
                           nn.ReLU(inplace=True), nn.Conv2d(64, 64, 4, 2, 1), norm(64),
                           nn.ReLU(inplace=True), nn.Conv2d(64, 64, 4, 2, 1), norm(64), nn.ReLU(inplace=True)]

    PDE_layers = [nn.Conv2d(67, 67, 3, 1, 1), norm_(67), nn.ReLU(inplace=True), nn.Conv2d(67, 64, 3, 1, 1),
                  norm(64), nn.ReLU(inplace=True)]

    fc_layers =  [nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]
    #in adaptiveaveragepooling target output size is (1, 1) and stride, kernelsize will be choose automatically

    down_model = nn.Sequential(*downsampling_layers)
    feature_model = nn.Sequential(*PDE_layers)
    fc_model = nn.Sequential(*fc_layers)
    feature_fc_model = nn.Sequential(feature_model, fc_model)

    model = EntireNet(down_model, feature_model, fc_model).to(device)
    model_fe = PDE_Solver(feature_model).to(device)

    print(model.parameters)
    print("=================================================")
    print(model_fe.parameters)

    # Loss Fn
    criterion = nn.CrossEntropyLoss().to(device)

    # Data Loader
    train_loader, test_loader, train_eval_loader = get_mnist_loaders(args.data_aug, args.batch_size,
                                                                     args.test_batch_size)

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    print("Train Data: {}, Test Data: {}, Train_Eval Data: {}".format(len(train_loader), len(test_loader),
                                                                      len(train_eval_loader)))

    # Evaluation - Accuracy
    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    # Learning Rate Scheduler
    lr_fn = learning_rate_with_decay(args, batch_denom=128, batches_per_epoch=batches_per_epoch,
                                     boundary_epochs=[60, 100, 140],
                                     decay_rates=[1, 0.1, 0.01, 0.001]
                                     )
    # parameter
    down_pixel_size = 6

    # Initialize X1, X2, T
    x_index = torch.tensor(np.array([[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6],
                                      [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]]) / 6.0,
                           dtype=torch.float32).to(device)
    x_index2 = torch.tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3],
                                       [4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]]]) / 6.0,
                            dtype=torch.float32).to(device)
    t_index = torch.zeros((1, down_pixel_size, down_pixel_size), dtype=torch.float32).to(device)

    init_x_t_pairs = torch.cat([x_index, x_index2, t_index], dim=0).to(device)

    # Randomized X1, X2, T (Grad True)
    rand_x_index = torch.tensor(np.array([[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6],
                                           [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]]) / 6.0,
                                dtype=torch.float32, requires_grad=True)
    rand_x_index2 = torch.tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3],
                                            [4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]]]) / 6.0,
                                 dtype=torch.float32, requires_grad=True)
    rand_t_index = torch.ones((1, down_pixel_size, down_pixel_size), dtype=torch.float32, requires_grad=True)

    for num in ['00', '10', '20', '30', '01', '11', '21', '31', '02', '12', '22', '32', 'b01', 'b11', 'b21', 'b31', 'b02',
                'b12', 'b22', 'b32']:
        exec("a_%s = torch.tensor(1, dtype=torch.float32, requires_grad=True)" % num)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_fe = torch.optim.Adam(model_fe.parameters(), lr=args.lr)
    optimizer_para = torch.optim.Adam([a_00, a_10, a_20, a_30, a_01, a_11, a_21, a_31, a_02, a_22, a_12, a_32, a_b01,
                                       a_b11, a_b21, a_b31, a_b02, a_b22, a_b12, a_b32], lr=args.lr)
    optimizer_x_t = torch.optim.Adam([rand_x_index, rand_x_index2, rand_t_index], lr=args.lr)

    print(('Number of parameters: {}'.format(count_parameters(model))))

    start_time = time.time()

    for epoch in range(args.nepochs):
        # -------------------------------- Train Dataset --------------------------------
        for itr, (data) in enumerate(train_loader):
            # update x_t pairs
            rand_x_t_pairs = torch.cat([rand_x_index, rand_x_index2, rand_t_index], dim=0).to(device)

            # learning rate scheduling
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_fn(itr + epoch * batches_per_epoch)

            x, y = data
            x = x.to(device)
            y = y.to(device)

            down_logits = down_model(x)  # Original Image To 64 x 6 x 6 feature map
            added_down_logit_init = torch.empty(
                (len(down_logits), len(down_logits[0]) + 3, len(down_logits[0][0]), len(down_logits[0][0]))).to(device)

            # feature map + initial x_t pairs
            for i in range(len(down_logits)):
                added_down_logit_init[i] = torch.cat([down_logits[i], init_x_t_pairs]).to(device)

            # convolution on new expanded featuremap, 67x6x6--> 64x6x6
            u_init = feature_model(added_down_logit_init)

            # Lu - MSE Loss
            init_criterion = nn.MSELoss()
            loss_init = init_criterion(down_logits, u_init)

            added_down_logit_rand = torch.empty(
                (len(down_logits), len(down_logits[0]) + 3, len(down_logits[0][0]), len(down_logits[0][0]))).to(device)

            for i in range(len(down_logits)):
                added_down_logit_rand[i] = torch.cat([down_logits[i], rand_x_t_pairs]).to(device)

            # convolution on new expanded featuremap, 67x6x6--> 64x6x6
            u = feature_model(added_down_logit_rand)

            u_2 = torch.pow(u, 2)
            u_3 = torch.pow(u, 3)

            u_x_sum = a_01.to(device) + torch.mul(a_11.to(device), u) + torch.mul(a_21.to(device), u_2) + torch.mul(
                a_31.to(device), u_3)
            u_x2_sum = a_b01.to(device) + torch.mul(a_b11.to(device), u) + torch.mul(a_b21.to(device), u_2) + torch.mul(
                a_b31.to(device), u_3)
            basic_sum = a_00.to(device) + torch.mul(a_10.to(device), u) + torch.mul(a_20.to(device), u_2) + torch.mul(
                a_30.to(device), u_3)

            u_t = grad(u, rand_t_index, grad_outputs=u.data.new(u.shape).fill_(1), create_graph=True)[0].to(device)
            u_x = grad(u, rand_x_index, grad_outputs=u_x_sum, create_graph=True)[0].to(device)

            u_x2 = grad(u, rand_x_index2, grad_outputs=u_x2_sum, create_graph=True)[0].to(device)
            ux = basic_sum.sum(0).sum(0) + u_x + u_x2

            # Lf - MSE Loss --> finding governing equation
            dyn_criterion = nn.MSELoss()
            loss_dyn = dyn_criterion(u_t, ux)*(1.0/(128*64))

            # classification 
            logits = fc_model(u)

            # Task Loss
            loss_task = criterion(logits, y)

            loss_init.backward(retain_graph=True)  # downsampled featuremap <--> u_init
            loss_dyn.backward(retain_graph=True)  # u_t <--> governing eq
            optimizer_fe.step()  # update solver(feature_model)
            optimizer_para.step()  # update governing eq

            # print(a_00, a_10, a_20, a_30, a_01, a_11, a_21, a_31)
            optimizer.zero_grad()
            optimizer_fe.zero_grad()
            optimizer_x_t.zero_grad()
            optimizer_para.zero_grad()

            loss_task.backward(retain_graph=True)  # classification loss
            optimizer.step()
            optimizer_x_t.step()
            optimizer.zero_grad()
            optimizer_fe.zero_grad()
            optimizer_x_t.zero_grad()
            optimizer_para.zero_grad()

            if itr % 100 == 0: print(
                "loss_init: {}, loss_dyn: {}, loss_task(CE): {}".format(loss_init, loss_dyn, loss_task))

        # -------------------------------- Validation Dataset --------------------------------
        total_correct = 0
        with torch.no_grad():
            train_acc = accuracy(down_model, feature_fc_model, train_eval_loader, rand_x_t_pairs)
            val_acc = accuracy(down_model, feature_fc_model, test_loader, rand_x_t_pairs)

            if val_acc > best_acc:
                torch.save({'state_dict': model.state_dict(), 'args': args, 'a_00': a_00, 'a_10': a_10, 'a_20': a_20,
                            'a_30': a_30, 'a_01': a_01, 'a_11': a_11, 'a_21': a_21, 'a_31': a_31, 'a_02': a_02,
                            'a_12': a_12,
                            'a_22': a_22, 'a_32': a_32, 'b01': a_b01, 'b11': a_b11, 'b21': a_b21, 'b31': a_b31,
                            'b02': a_b02,
                            'b12': a_b12, 'b22': a_b22, 'b32': a_b32, 'rand_x_index': rand_x_index,
                            'rand_x_index2': rand_x_index2, 'rand_t_index': rand_t_index},
                           os.path.join(args.save, 'model_pde_%s_normalization.pth' % args.rectifier))  #
                torch.save({'state_dict': model.state_dict(), 'args': args},
                           os.path.join(args.save, 'model_pde_only_dict_%s_normalization.pth' % args.rectifier))
                best_acc = val_acc

            print("Epoch {:04d} | Train Acc {:.4f} | Test Acc {:.4f}".format(epoch + 1, train_acc, val_acc))
            print(rand_x_index, rand_t_index, down_logits.sum(0).sum(0).sum(0).sum(0), torch.abs(u-u_init).sum(0).sum(0).sum(0).sum(0))
