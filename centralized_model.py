import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_centralized_data
from model import SimCLR, SimCLRPredictor
import utils
import flwr as fl

import csv

simclr_predictor = None

DEVICE = utils.DEVICE

EPOCHS = 10

def centralized_train(useResnet18):
    global simclr_predictor
    simclr_predictor = SimCLRPredictor(utils.NUM_CLASSES, DEVICE, useResnet18=utils.useResnet18, tune_encoder=utils.fineTuneEncoder).to(DEVICE)
    

    
    train, test = utils.load_centralized_data()
    
    trainloader = DataLoader(train, batch_size = utils.BATCH_SIZE)
    testloader = DataLoader(test, batch_size = utils.BATCH_SIZE)   

    optimizer = torch.optim.Adam(simclr_predictor.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_predictor(trainloader, optimizer, criterion)
    
    loss, accuracy = evaluate(testloader, criterion)
    


    return loss, accuracy


def load_model(useResnet18):
    global simclr_predictor
    
    simclr = SimCLR(DEVICE, useResnet18=useResnet18).to(DEVICE)

    
    list_of_files = [fname for fname in glob.glob("./centralized_weoghts/model_round_*")]
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from:", latest_round_file)
    state_dict = torch.load(latest_round_file)
    
    simclr.load_state_dict(state_dict)
    
    weights = [v.cpu().numpy() for v in simclr.state_dict().values()]
    
    simclr_predictor.set_encoder_parameters(weights)


def train_predictor(trainloader, testloader, optimizer, criterion):
    accuracy = None
    
    while accuracy is None or accuracy <= 95:
        simclr_predictor.train()
    

        for epoch in range(EPOCHS):
            batch = 0
            num_batches = len(trainloader)

            for item in trainloader:
                (x, x_i, x_j), labels = item['img'], item['label']
                x, labels = x.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                outputs = simclr_predictor(x)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                print(f"Epoch: {epoch} Predictor Train Batch: {batch} / {num_batches}")
                batch += 1
                
        _, accuracy = evaluate(testloader, criterion)
        print("Accuracy:", accuracy)

def evaluate(testloader, criterion):
    simclr_predictor.eval()
    
    total = 0
    correct = 0
    loss = 0

    batch = 0
    num_batches = len(testloader)

    with torch.no_grad():
        for item in testloader:
            (x, x_i, x_j), labels = item['img'], item['label']
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            
            logits = simclr_predictor(x)
            values, predicted = torch.max(logits, 1)  
            
            total += labels.size(0)
            loss += criterion(logits, labels).item()
            correct += (predicted == labels).sum().item()

            print(f"Test Batch: {batch} / {num_batches}")
            batch += 1
  
    return loss / batch, correct / total


count = 0
def save_model():
    global count
    aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

    params_dict = zip(simclr_predictor.state_dict().keys(), aggregated_ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    simclr_predictor.load_state_dict(state_dict, strict=True)

    torch.save(simclr_predictor.state_dict(), f"./centralized_weights/centralized_model_{count}.pth")
    count += 1

if __name__ == "__main__":
    centralized_train(False)

