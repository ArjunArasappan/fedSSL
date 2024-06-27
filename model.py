import flwr as fl
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from flwr.common.logger import log
from logging import INFO, DEBUG

from dataset import global_batch



class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.batch_size = None
        self.criterion = nn.CrossEntropyLoss(reduction="mean",)

    def forward(self, z_i, z_j):

        self.batch_size = z_i.size(0)
        feature_dim = z_i.size(1)
        
        z_i = F.normalize(z_i)
        z_j = F.normalize(z_j)
        
        z = torch.cat((z_i, z_j), dim=0)  

        sim_matrix = torch.matmul(z, z.T) / self.temperature
        sim_matrix.fill_diagonal_(-float('inf'))

        labels = torch.arange(self.batch_size, 2 * self.batch_size, device=self.device)
        labels = torch.cat((labels, labels - self.batch_size))  

        loss = self.criterion(sim_matrix, labels)
        return loss


        loss = self.criterion(sim_matrix, labels) #ignores zero index which 
        
        return loss

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096, num_layer=2):
        super().__init__()
        self.in_features = dim
        self.out_features = projection_size
        
        
        if num_layer == 1:
            self.net = nn.Sequential(
                nn.Linear(dim, projection_size),
            )
        elif num_layer == 2:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, projection_size),
            )
        else:
            raise NotImplementedError(f"Not defined MLP: {num_layer}")

    def forward(self, x):
        return self.net(x)

class SimCLR(nn.Module):
    def __init__(self, device, useResnet18, image_size=32, projection_size=2048, projection_hidden_size=4096, num_layer = 2) -> None:
        super(SimCLR, self).__init__()
        
        if useResnet18:
            self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        else:
            self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)

        self.encoded_size = self.encoder.fc.in_features

        self.encoder.fc = nn.Identity()
                
        self.proj_head = None
        
        if num_layer == 1:
            self.proj_head = nn.Sequential(
                nn.Linear(self.encoded_size, projection_size),
            )
        elif num_layer == 2:
            self.proj_head = nn.Sequential(
                nn.Linear(self.encoded_size, projection_hidden_size),
                nn.BatchNorm1d(projection_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(projection_hidden_size, projection_size),
            )
        
        
        self.isInference = False
        
    def setInference(self, isInference):
        self.isInference = isInference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder(x)
        if not self.isInference:
            return self.proj_head(e1)
        return e1

   
    
class SimCLRPredictor(nn.Module):
    def __init__(self, num_classes, device, tune_encoder = False):
        super(SimCLRPredictor, self).__init__()
        
        self.simclr = SimCLR(device, useResnet18 = True).to(device)
        self.linear_predictor = nn.Linear(self.simclr.proj_head.out_features, num_classes)
        
        if not tune_encoder:
            for param in self.simclr.parameters():
                param.requires_grad = False
                
    def set_encoder_parameters(weights):
        self.simclr.set_weights(weights)

    def forward(self, x):
        self.simclr.setInference(True)
        features = self.simclr(x)
        output = self.linear_predictor(features)
        return output
    
