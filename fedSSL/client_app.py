import flwr as fl
from flwr.common import Context
from flwr.client import ClientApp

import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm
import os


from fedSSL.transform import SimCLRTransform
from fedSSL.model import SimCLR, NTXentLoss

import fedSSL.utils as utils

DEVICE = utils.DEVICE

def train(net, trainloader, optimizer, criterion, cid, epochs = 1):
    net.train()
    num_batches = len(trainloader)
    batch = 0
    total_loss = 0
    
    print(f'Client {cid} Training...')
    for epoch in range(epochs):

        for item in tqdm(trainloader):
            x_i, x_j = item['img']
   
            x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
            optimizer.zero_grad()
            
            z_i = net(x_i).to(DEVICE)
            z_j = net(x_j).to(DEVICE)
            
            loss = criterion(z_i, z_j)
            total_loss += loss

            loss.backward()
            optimizer.step()

            # print("Client Train Batch:", batch, "/", num_batches)
            
            batch += 1
            
    return {'Loss' : float(total_loss / batch)}
                 

def test(net, testloader, criterion, cid):
    
    if testloader == None:
        return -1, -1
    
    net.eval()
    loss_epoch = 0
    batch = 0
    num_batches = len(testloader)
    
    with torch.no_grad():
        print(f'Client {cid} Local Evaluation...')
        for item in tqdm(testloader):
            x_i, x_j = item['img']
            
            x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
            
            z_i = net(x_i).to(DEVICE)
            z_j = net(x_j).to(DEVICE)
            loss = criterion(z_i, z_j)
            
            loss_epoch += loss.item()
            
            # print("Client Train Batch:", batch, "/", num_batches)
            
            batch += 1
    return loss_epoch / (batch), -1


class CifarClient(fl.client.NumPyClient):
    def __init__(self, cid, simclr, trainset, testset, useResnet18, num_clients, loss):
        self.cid = cid
        self.simclr = simclr
        self.optimizer = torch.optim.Adam(self.simclr.parameters(), lr=3e-4)
        self.loss = loss
        
        self.useResnet18 = useResnet18
        self.num_clients = num_clients
        

        self.trainloader = DataLoader(trainset, batch_size = utils.BATCH_SIZE)
        
        if testset == None:
            self.testloader = None
        else:
            self.testloader = DataLoader(testset, batch_size = utils.BATCH_SIZE)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.simclr.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.simclr.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        self.simclr.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):

        self.set_parameters(parameters)
        self.simclr.setInference(False)
        results = train(self.simclr, self.trainloader, self.optimizer, self.loss, self.cid)
        
        return self.get_parameters(config={}), len(self.trainloader), results

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.simclr.setInference(False)
        
        loss, accuracy = test(self.simclr, self.testloader, self.loss, self.cid)

        return float(loss), 1, {"accuracy": accuracy}



def client_fn(context):
    clientID = context.node_config["partition-id"]
    num_clients = context.node_config["num-partitions"]
    useResnet18 = True if context.run_config['use-resnet18'] == 1 else False
    
    ntxent = NTXentLoss( device=DEVICE).to(DEVICE)
    
    fds = utils.get_fds(num_clients)
    
    train, test = utils.load_partition(fds, clientID, client_test_split = context.run_config['local-eval-fraction'])
    simclr = SimCLR(DEVICE, useResnet18=useResnet18).to(DEVICE)
    
    return CifarClient(clientID, simclr, train, test, useResnet18, num_clients, ntxent).to_client()





app = ClientApp(client_fn)