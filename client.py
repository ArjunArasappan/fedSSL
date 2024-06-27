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
from dataset import load_data, global_batch, num_iters, NUM_CLASSES

NUM_CLIENTS = 2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


progress_interval = 0.1


def train(net, trainloader, optimizer, criterion, epochs):
    net.train()
    
    for epoch in range(epochs):
        num_batches = len(trainloader)
        batch = 0
        for (x, x_i, x_j), _ in trainloader:
            x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
            optimizer.zero_grad()
            z_i = net(x_i)
            z_j = net(x_j)
            loss = criterion(z_i, z_j)
            # print("training loss: ", loss)

            loss.backward()
            optimizer.step()


            # if(batch % (progress_interval * num_batches) == 0):
            print("Client Train Batch:", batch, "/", num_batches)
                
            batch += 1
                 

def test(net, predictor, testloader, criterion):
    net.eval()
    loss_epoch = 0
    count = 0
    
    with torch.no_grad():
        for (x, x_i, x_j), label in testloader:
            x, x_i, x_j = x.to(DEVICE), x_i.to(DEVICE), x_j.to(DEVICE)
            
            z_i = net(x_i)
            z_j = net(x_j)
            loss = criterion(z_i, z_j)
            
            

            
            loss_epoch += loss.item()
            count += 1
            if count == num_iters:
                print("Exited Test Loop")
                break 
    return loss_epoch / (count), -1



trainloaders, valloaders, testloader, predictorloader, num_examples = load_data(NUM_CLIENTS)

#batch size usually at 16
ntxent = NTXentLoss(batch_size=global_batch, temperature=0.5, device=DEVICE)


class CifarClient(fl.client.NumPyClient):
    def __init__(self, cid, simclr, trainloader, valloader):
        self.cid = cid
        self.simclr = simclr
        self.optimizer = torch.optim.Adam(self.simclr.parameters(), lr=3e-4)
        self.simclr_predictor = None
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.simclr.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.simclr.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        self.simclr.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        train(self.simclr, self.trainloader, self.optimizer, ntxent, epochs=1)
        
        # print("Type: ", type(num_examples))
        
        return self.get_parameters(config={}), num_examples[self.cid], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # self.simclr_predictor = train_predictor(self.simclr, self.trainloader, self.optimizer, nn.CrossEntropyLoss(reduction='mean'), epochs=1)
        
        loss, accuracy = test(self.simclr, self.simclr_predictor, testloader, ntxent)
        
        print("Loss: ", float(loss), ', ', self.cid)
    
        # print("Accuracy: ", float(accuracy))
        return float(loss), num_examples[self.cid], {"accuracy": accuracy}

def client_fn(cid):
    clientID = int(cid)
    simclr = SimCLR(DEVICE, useResnet18=True).to(DEVICE)
    trainloader = trainloaders[clientID]
    valloader = valloaders[clientID]
    return CifarClient(clientID, simclr, trainloader, valloader).to_client()


