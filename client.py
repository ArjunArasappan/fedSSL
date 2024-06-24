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

NUM_CLIENTS = 3

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple




from transform import SimCLRTransform
from model import SimCLR, NTXentLoss, SimCLRPredictor
from dataset import load_data, global_batch, global_epoch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, trainloader, optimizer, criterion, epochs):
    net.train()
    
    for epoch in range(epochs):
        num_batches = len(trainloader)
        batch = 0
        for (x_i, x_j), _ in trainloader:
            x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
            optimizer.zero_grad()
            z_i = net(x_i)
            z_j = net(x_j)
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()
            batch += 1

            if(batch == global_epoch):
                print("Exited at batch 10")
                break
            
def train_predictor(base_encoder, trainloader, optimizer, criterion, epochs):
    simclr_predictor = SimCLRPredictor(base_encoder, tune_encoder = False, num_classes=10).to(DEVICE)
    
    simclr_predictor.train()
    
    for epoch in range(epochs):
        iter = 0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = simclr_predictor(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if iter == 10:
                break
                     

def test(net, predictor, testloader, criterion):
    net.eval()
    loss_epoch = 0
    count = 0
    with torch.no_grad():
        for (x_i, x_j), label in testloader:
            print(label)
            x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
            z_i = predictor(x_i)
            z_j = predictor(x_j)
            
            loss = criterion(z_i, z_j)
            loss_epoch += loss.item()
            count += 1
            if count == global_epoch:
                print("Exited Test Loop")
                break 
    return loss_epoch / (count)



trainloaders, valloaders, testloader, num_examples = load_data(NUM_CLIENTS)

#batch size usually at 16
criterion = NTXentLoss(batch_size=global_batch, temperature=0.5, device=DEVICE)


class CifarClient(fl.client.NumPyClient):
    def __init__(self, cid, simclr, trainloader, valloader):
        self.cid = cid
        self.simclr = simclr
        self.optimizer = optimizer = torch.optim.Adam(self.simclr.parameters(), lr=3e-4)
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
        
        train(self.simclr, self.trainloader, self.optimizer, criterion, epochs=1)
        
        print("NUMEXAMPLES: ", num_examples)
        return self.get_parameters(config={}), num_examples[self.cid], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.simclr_predictor = train_predictor(self.simclr, self.trainloader, self.optimizer, nn.CrossEntropyLoss, epochs=1)

        # train_predictor(net)
        loss, accuracy = test(self.simclr, self.simclr_predictor, testloader, self.optimizer, nn.CrossEntropyLoss)
        
        print("Loss: ", float(loss))
        print("Accuracy: ", float(accuracy))
        return float(loss), num_examples["testset"], {"accuracy": accuracy}

def client_fn(cid):
    clientID = int(cid)
    print("CID: ", clientID)
    print(clientID)
    simclr = SimCLR().to(DEVICE)
    trainloader = trainloaders[clientID]
    valloader = valloaders[clientID]
    return CifarClient(clientID, simclr, trainloader, valloader).to_client()


