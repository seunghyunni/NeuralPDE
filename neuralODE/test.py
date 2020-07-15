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
import torchvision
import pandas as pd


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

    func = ConvODEF(64)
    ode = NeuralODE(func)
    model = ContinuousNeuralMNISTClassifier(ode).to(device)

    print(model.parameters)
    print("=================================================")
    print(model_fe.parameters)

    # Data Loader
    img_std = 0.3081
    img_mean = 0.1307

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("data/mnist", train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((img_mean,), (img_std,))
                                   ])
                                   ),
        batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("data/mnist", train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((img_mean,), (img_std,))
                                   ])
                                   ),
        batch_size=128, shuffle=True
    )

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

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(('Number of parameters: {}'.format(count_parameters(model))))

    start_time = time.time()

    num_items = 0
    train_losses = []

    for epoch in range(args.nepochs):
        # -------------------------------- Train Dataset --------------------------------
        model.train()
        criterion = nn.CrossEntropyLoss()
        print(f"Training Epoch {epoch}...")
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_losses += [loss.item()]
            num_items += data.shape[0]
        print('Train loss: {:.5f}'.format(np.mean(train_losses)))

        # -------------------------------- Validation Dataset --------------------------------
        accuracy = 0.0
        num_items = 0

        model.eval()
        criterion = nn.CrossEntropyLoss()
        print(f"Testing...")
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                if use_cuda:
                    data = data.cuda()
                    target = target.cuda()
                output = model(data)
                accuracy += torch.sum(torch.argmax(output, dim=1) == target).item()
                num_items += data.shape[0]
            accuracy = accuracy * 100 / num_items
            print("Test Accuracy: {:.3f}%".format(accuracy))

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

        n_epochs = 5
        test()
        train_losses = []
        for epoch in range(1, n_epochs + 1):
            train_losses += train(epoch)
            test()



plt.figure(figsize=(9, 5))
history = pd.DataFrame({"loss": train_losses})
history["cum_data"] = history.index * batch_size
history["smooth_loss"] = history.loss.ewm(halflife=10).mean()
history.plot(x="cum_data", y="smooth_loss", figsize=(12, 5), title="train error")