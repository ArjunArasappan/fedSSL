import flwr as fl
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from transform import SimCLRTransform
from model import SimCLR, NTXentLoss, SimCLRPredictor
from utils import load_partition, load_centralized_data, BATCH_SIZE, NUM_CLASSES, NUM_CLIENTS, NUM_ROUNDS, useResnet18, DEVICE






def train(net, trainloader, optimizer, criterion, epochs):
    net.train()
    
    num_batches = len(trainloader)
    batch = 0
    total_loss = 0
    
    for epoch in range(epochs):

        
        for item in trainloader:
            x, x_i, x_j = item['img']
   
            x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
            optimizer.zero_grad()
            z_i = net(x_i).to(DEVICE)
            z_j = net(x_j).to(DEVICE)
            loss = criterion(z_i, z_j)
            total_loss += loss
            # print("training loss: ", loss)

            loss.backward()
            optimizer.step()


            # if(batch % (print_interval * num_batches) == 0):
            print("Client Train Batch:", batch, "/", num_batches)
            
            
            batch += 1
            
    return {'Avg Train Loss' : float(total_loss / batch)}
                 

def test(net, testloader, criterion):
    net.eval()
    loss_epoch = 0
    batch = 0
    num_batches = len(testloader)
    
    with torch.no_grad():
        for item in testloader:
            x, x_i, x_j = item['img']
            
            
            x, x_i, x_j = x.to(DEVICE), x_i.to(DEVICE), x_j.to(DEVICE)
            
            z_i = net(x_i).to(DEVICE)
            z_j = net(x_j).to(DEVICE)
            loss = criterion(z_i, z_j)
            
            loss_epoch += loss.item()
            
            # if(batch % (print_interval * num_batches) == 0):
            print("Client Train Batch:", batch, "/", num_batches)
            
            batch += 1
    return loss_epoch / (batch), -1



ntxent = NTXentLoss( device=DEVICE).to(DEVICE)


class CifarClient(fl.client.NumPyClient):
    def __init__(self, cid, simclr):
        self.cid = cid
        self.simclr = simclr
        self.optimizer = torch.optim.Adam(self.simclr.parameters(), lr=3e-4)
                
        self.trainloader, self.testloader = load_partition(self.cid)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.simclr.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.simclr.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        self.simclr.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        results = train(self.simclr, self.trainloader, self.optimizer, ntxent, epochs=1)
                
        return self.get_parameters(config={}), len(self.trainloader), results

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        loss, accuracy = test(self.simclr, self.testloader, ntxent)
        
        print("Loss: ", float(loss), ', ', self.cid)

        return float(loss), len(self.testloader), {"accuracy": accuracy}

def client_fn(cid):
    clientID = int(cid)
    simclr = SimCLR(DEVICE, useResnet18=useResnet18).to(DEVICE)
    return CifarClient(clientID, simclr).to_client()


