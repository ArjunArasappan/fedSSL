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
from flwr.common import NDArrays, Scalar
from typing import Dict, Optional, Tuple

from utils import NUM_CLASSES, NUM_CLIENTS, NUM_ROUNDS, DEVICE, BATCH_SIZE






class NTXentLoss(nn.Module):
    def __init__(self, device, temperature=0.2, ):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.batch_size = None
        self.criterion = nn.CrossEntropyLoss(reduction="mean",)
        
    def testBatch(self):
        batchSize = 2
        random1Tensor = torch.randn((batchSize, 5))
        random2Tensor = torch.randn((batchSize, 5))
        
        return random1Tensor, random2Tensor
        

    def forward(self, z_i, z_j):

        self.batch_size = z_i.size(0)
        feature_dim = z_i.size(1)
        
        z_i = F.normalize(z_i).to(self.device)
        z_j = F.normalize(z_j).to(self.device)
        
        z = torch.cat((z_i, z_j), dim=0).to(self.device)
        
        # print(z)

        sim_matrix = torch.matmul(z, z.T).to(self.device) / self.temperature
        sim_matrix.fill_diagonal_(-float('inf'))
        
        # print(sim_matrix)

        labels = torch.arange(self.batch_size, 2 * self.batch_size, device=self.device)
        labels = torch.cat((labels, labels - self.batch_size))  
        
        # print(labels)

        loss = self.criterion(sim_matrix, labels)
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
            print("Using ResNet 18 Encoder")
        else:
            self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
            print("Using ResNet 50 Encoder")


        self.encoded_size = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        
        self.projected_size = projection_size


                
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
        if self.isInference:
            return e1
        return self.proj_head(e1)

   
    
class SimCLRPredictor(nn.Module):
    def __init__(self, num_classes, device, useResnet18 = True, tune_encoder = False):
        super(SimCLRPredictor, self).__init__()
        
        print("New Predictor Created!")
        
        self.simclr = SimCLR(device, useResnet18 = useResnet18).to(device)
        self.linear_predictor = nn.Linear(self.simclr.encoded_size, num_classes)
        
        if not tune_encoder:
            for param in self.simclr.parameters():
                param.requires_grad = False
                
    def set_encoder_parameters(self, weights):
        
        params_dict = zip(self.simclr.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        self.simclr.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        self.simclr.setInference(True)
        features = self.simclr(x)
        output = self.linear_predictor(features)
        return output
    
    
class GlobalPredictor:
    
    def __init__(self, tune_encoder, trainloader, testloader, device, useResnet18 = True):
        self.round = 0
        
        self.simclr_predictor = SimCLRPredictor(NUM_CLASSES, device, useResnet18 = useResnet18, tune_encoder = tune_encoder).to(device)
        
        self.trainloader = trainloader
        self.testloader = testloader
        
        self.epochs = 30
        self.optimizer = torch.optim.Adam(self.simclr_predictor.parameters(), lr=3e-4)
        self.criterion = nn.CrossEntropyLoss()
        
    def get_evaluate_fn(self):
        
        def evaluate(server_round: int, parameters, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            if self.round != NUM_ROUNDS:
                self.round = self.round + 1
                return -1, {"accuracy": -1}
            
            self.update_encoder(parameters)
            
            self.fine_tune_predictor()
            loss, accuracy = self.evaluate()
            print("Global Model Accuracy: ", accuracy)
            
            return loss, {"accuracy": accuracy}

        return evaluate

    def fine_tune_predictor(self):
        self.simclr_predictor.train()
    
        for epoch in range(self.epochs):
            batch = 0
            num_batches = len(self.trainloader)
        

            for item in self.trainloader:
                (x, x_i, x_j), labels = item['img'], item['label']
                x, labels = x.to(DEVICE), labels.to(DEVICE)
                
                self.optimizer.zero_grad()
                
                outputs = self.simclr_predictor(x)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                # if batch % (print_interval * num_batches) == 0:
                print(f"Epoch: {epoch} Predictor Train Batch: {batch} / {num_batches}")

                
                batch += 1
                
    def evaluate(self):
        self.simclr_predictor.eval()
        
        total = 0
        count = 0
        correct = 0
        loss = 0
    
        for epoch in range(self.epochs):
            batch = 0
            num_batches = len(self.testloader)
            
            with torch.no_grad():
                
                for item in self.testloader:
                    (x, x_i, x_j), labels = item['img'], item['label']

                    x, labels = x.to(DEVICE), labels.to(DEVICE)
                    
                    logits = self.simclr_predictor(x)
                    values, predicted = torch.max(logits, 1)  
                    
                    total += labels.size(0)
                    
                    loss += self.criterion(logits, labels)
                    
                    
                    correct += (predicted == labels).sum().item()
                    

                    # if batch % (print_interval * num_batches) == 0:
                    print(f"Epoch: {epoch} Predictor Test Batch: {batch} / {num_batches}")
                    
                    batch += 1
                    count += 1
                    
                break
            
        return loss / count, correct / total
            
            


        
    def update_encoder(self, weights):
        self.simclr_predictor.set_encoder_parameters(weights)

    
