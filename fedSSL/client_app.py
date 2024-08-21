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

def train(net, cid, trainloader, optimizer, criterion, epochs = 1):
    net.train()
    num_batches = len(trainloader)
    total_loss = 0
    
    with tqdm(total=num_batches * epochs, desc=f'Client {cid} Local Train', position=0, leave=True) as pbar:
        for epoch in range(epochs):

            for item in trainloader:
                x_i, x_j = item['img']
    
                x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
                optimizer.zero_grad()
                
                z_i = net(x_i).to(DEVICE)
                z_j = net(x_j).to(DEVICE)
                
                loss = criterion(z_i, z_j)
                total_loss += loss

                loss.backward()
                optimizer.step()
                
                pbar.update(1)
            
            
    return {'loss' : float(total_loss / num_batches)}
                 

def test(net, cid, testloader, criterion):
    
    if testloader == None:
        return 0
    
    net.eval()
    loss_epoch = 0
    num_batches = len(testloader)
    
    with tqdm(total=num_batches, desc=f'Client {cid} Local Eval', position=0, leave=True) as pbar:

        with torch.no_grad():
            for item in testloader:
                x_i, x_j = item['img']
                
                x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
                
                z_i = net(x_i).to(DEVICE)
                z_j = net(x_j).to(DEVICE)
                loss = criterion(z_i, z_j)
                
                loss_epoch += loss.item()
                
                pbar.update(1)
                        
    return loss_epoch / (num_batches)


class CifarClient(fl.client.NumPyClient):
    def __init__(self, cid, simclr, trainloader, testloader, criterion, train_epochs):
        self.cid = cid
        self.simclr = simclr
        self.optimizer = torch.optim.Adam(self.simclr.parameters(), lr=3e-4)
        self.criterion = criterion
        
        self.train_epochs = train_epochs
        
        self.trainloader = trainloader
        self.testloader = testloader
        

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.simclr.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.simclr.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        self.simclr.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.simclr.setInference(False)
        
        results = train(self.simclr, self.cid, self.trainloader, self.optimizer, self.criterion, self.train_epochs)
        
        return self.get_parameters(config={}), len(self.trainloader), results

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.simclr.setInference(False)
        
        loss = test(self.simclr, self.cid, self.testloader, self.criterion)
 
        return float(loss), -1, {'loss': loss}



def client_fn(context):
    clientID = context.node_config["partition-id"]
    num_clients = context.node_config["num-partitions"]
    useResnet18 = True if context.run_config['use-resnet18'] == 1 else False
    
    ntxent = NTXentLoss(device=DEVICE)
    fds = utils.get_fds(num_clients)
    
    simclr = SimCLR(DEVICE, useResnet18=useResnet18).to(DEVICE)

    trainset, testset = utils.load_partition(fds, clientID, client_test_split = context.run_config['local-eval-fraction'])
        
    trainloader = DataLoader(trainset, batch_size = context.run_config['batch-size'])
    testloader = None
    
    if testset != None:
        testloader = DataLoader(testset, batch_size = context.run_config['batch-size'])
    
    return CifarClient(clientID, simclr, trainloader, testloader, ntxent, context.run_config['local-train-epochs']).to_client()





app = ClientApp(client_fn)