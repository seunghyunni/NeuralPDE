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
from utils.utils import RunningAverageMeter, flatten, learning_rate_with_decay, one_hot, count_parameters, Identity
import torchvision
from model.ode import NeuralODE
from model.model import ContinuousMobileNetV3
from model.mobilenetv3 import MobileNetV3
from thop import profile
from data_loader import get_dataset, inf_generator
from tqdm import tqdm 
from torch.optim.lr_scheduler import StepLR
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument('--nepochs', type=int, default=300)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--save', type=str, default='./ODE_experiment1')
parser.add_argument('--dataset', type=str, default='tinyimagenet')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])
parser.add_argument('--is_ode', type=eval, default=True, choices=[True, False])
parser.add_argument('--double', type=eval, default=False, choices=[True, False])
args = parser.parse_args()

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("cuda")
else:
    print("CPU")

if __name__ == '__main__':
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # Data Loader
    transformer = ToDouble if args.double else Identity
    
    train_dataset, test_dataset, num_classes = get_dataset(
        args.dataset, tensor_type_transformer=transformer)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    print("Train Data: {}, Test Data: {}".format(len(train_dataset), len(test_dataset)))
    print("Total number of classes: {}".format(num_classes))

    # Model
    if args.is_ode:
        model = ContinuousMobileNetV3(n_class=num_classes, input_size=64, dropout=0.8, mode='small', width_mult=1.).to(device)
    else:
        model = MobileNetV3(n_class=num_classes, input_size=64, dropout=0.8, mode='small', width_mult=1.).to(device)

    #model= nn.DataParallel(model)

    #print('mobilenetv3:\n', model)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print("=================================================")

    # Loss Fn
    criterion = nn.CrossEntropyLoss()

    # Evaluation - Accuracy
    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    # parameter
    down_pixel_size = 6

    # Optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.99, weight_decay= 5e-4)

    # Learning Rate Scheduler
    lr_fn = StepLR(optimizer, step_size = 3, gamma =0.02, last_epoch = -1)

    start_time = time.time()

    num_items = 0
    train_losses = []
    correct = 0

    for epoch in range(args.nepochs):
        epoch_start_time = time.time()
        # -------------------------------- Train Dataset --------------------------------
        criterion = nn.CrossEntropyLoss()
        correct = 0
        print('\n===> Epoch [%d/%d]' % (epoch+1, args.nepochs))
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.train()
            data, target = Variable(data).to(device), Variable(target).to(device)
            
            output = model(data)
            
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses += [loss.item()]
            num_items += data.shape[0]
            
            _, predicted = torch.max(output.data, 1) 
            correct += (predicted == target).sum().item()    
            #accuracy = torch.sum(torch.argmax(output, dim=1) == target).item()
        train_acc = correct * 100 / num_items
        print('Train loss: {:.4f}, Train Accuracy: {:.4f}%'.format(np.mean(train_losses), train_acc))

        # -------------------------------- Validation Dataset --------------------------------
        accuracy = 0.0
        num_items = 0

        model.eval()
        criterion = nn.CrossEntropyLoss()
        print(f"Testing...")
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                data = data.cuda()
                target = target.cuda()
                output = model(data)
                accuracy += torch.sum(torch.argmax(output, dim=1) == target).item()
                num_items += data.shape[0]
            val_acc = accuracy * 100 / num_items
            print("Test Accuracy: {:.3f}%".format(val_acc))

        if val_acc > best_acc:
            torch.save({'state_dict': model.state_dict(), 'args': args},
                       os.path.join(args.save, 'model_dict_%.4f.pth' % val_acc))
            best_acc = val_acc

        print("Epoch {:04d}: Train Acc {:.4f}%, Test Acc {:.4f}%".format(epoch + 1, train_acc, val_acc))
        
        lr_fn.step()
        print("epoch time: %.4f min" %((time.time() - epoch_start_time)/60))
    
    end_time = time.time()
    print("total time: %.1f h" %((end_time - start_time)/3600))