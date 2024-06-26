import flwr as fl
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, ResNet18_Weights

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positives = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        negatives = sim[self.mask].view(N, -1)
        labels = torch.zeros(N).to(self.device).long()
        logits = torch.cat((positives, negatives), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
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
    def __init__(self, encoder=resnet18, image_size=32, projection_size=2048, projection_hidden_size=4096) -> None:
        super(SimCLR, self).__init__()
        super(SimCLR, self).__init__()
        self.encoder = encoder(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoded_size = self.encoder.fc.in_features

        self.encoder.fc = nn.Identity()
        self.proj_head = MLP(self.encoded_size, projection_size, projection_hidden_size)
        self.isInference = False
        
    def setInference(self, isInference):
        self.isInference = isInference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder(x)
        if self.isInference:
            return self.proj_head(e1)
        return e1
    
class SimCLRPredictor(nn.Module):
    def __init__(self, simclr_model, num_classes, tune_encoder = False):
        super(SimCLRPredictor, self).__init__()
        self.simclr = simclr_model
        self.linear_predictor = nn.Linear(self.simclr.proj_head.out_features, num_classes)
        
        if not tune_encoder:
            for param in self.simclr.parameters():
                param.requires_grad = False

    def forward(self, x):
        self.simclr.setInference(True)
        features = self.simclr(x)
        output = self.linear_predictor(features)
        return output
    
