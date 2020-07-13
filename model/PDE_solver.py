import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import grad, Variable


class EntireNet(nn.Module):
    def __init__(self, down_model, feature_model, fc_model):
        super(EntireNet, self).__init__()
        
        self.output = nn.Sequential(down_model, feature_model, fc_model)
    
    def forward(self,x):
        x = self.output(x)
        return x


class PDE_Solver(nn.Module):
    def __init__(self, feature_model):
        super(PDE_Solver, self).__init__()
        
        self.output = feature_model
    
    def forward(self,x):
        x = self.output(x)
        return x