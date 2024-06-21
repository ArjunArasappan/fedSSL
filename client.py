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
from transform import SimCLRTransform


from model import SimCLR, NTXentLoss
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
            # if batch % int(0.05*(num_batches)) == 0:
            #     print("Epoch: ", epoch)
            #     print("Batch: ", batch)
            if(batch == global_epoch):
                print("Exited at batch 10")
                break
            
def train_predictor(base_encoder, trainloader, optimizer, criterion, epochs):
    simclr_predictor = simclr_predictor(base_encoder, fine_tune_base = False).to(DEVICE)
    
    simclr_predictor.train()
    
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    

def test(net, testloader, criterion):
    net.eval()
    loss_epoch = 0
    count = 0
    with torch.no_grad():
        for (x_i, x_j), _ in testloader:
            x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
            z_i = net(x_i)
            z_j = net(x_j)
            
            loss = criterion(z_i, z_j)
            loss_epoch += loss.item()
            count += 1
            if count == global_epoch:
                print("Exited Test Loop")
                break 
    return loss_epoch / (count)


net = SimCLR().to(DEVICE)


trainloader, testloader, num_examples = load_data()
#batch size usually at 16
criterion = NTXentLoss(batch_size=global_batch, temperature=0.5, device=DEVICE)
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, optimizer, criterion, epochs=1)
        return self.get_parameters(config={}), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # train_predictor(net)
        loss = test(net, testloader, criterion)
        print("Loss: ", float(loss))
        return float(loss), num_examples["testset"], {"accuracy": 0}

# Define Flower client
def client_fn(cid: str):
    return CifarClient().to_client()


