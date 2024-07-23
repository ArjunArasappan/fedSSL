import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_centralized_data
from model import SimCLR, SimCLRPredictor
from flwr_datasets import FederatedDataset

import utils
import flwr as fl

import csv

simclr_predictor = None

DEVICE = utils.DEVICE

EPOCHS = 1
count = 0

logpath = "./log_files/central_log.csv"




def load_data():
    fds = FederatedDataset(dataset="cifar10", partitioners = {'train' : 1, 'test' : 1})
        
    train_data = fds.load_split("train")
    train_data = train_data.with_transform(utils.apply_transforms)

    
    test_data = fds.load_split("test")
    test_data = test_data.with_transform(utils.apply_transforms)
    
    test_data = test_data.train_test_split(test_size=0.1, shuffle = True, seed=42)
    
    val_data = test_data['test']
    test_data = test_data['train']
    
    
    return train_data, val_data, test_data


def centralized_train(useResnet18):
    global simclr_predictor
    simclr_predictor = SimCLRPredictor(utils.NUM_CLASSES, DEVICE, useResnet18=utils.useResnet18, tune_encoder=utils.fineTuneEncoder).to(DEVICE)
    
    train, val, test = load_data()
    
    trainloader = DataLoader(train, batch_size = utils.BATCH_SIZE)
    testloader = DataLoader(test, batch_size = utils.BATCH_SIZE)   

    optimizer = torch.optim.Adam(simclr_predictor.parameters(), lr=3e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    train_predictor(trainloader, testloader, optimizer, criterion)
    


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
    total_epochs = 0
    while accuracy is None or accuracy < 95 or total_epochs < 200:
        simclr_predictor.train()


        for epoch in range(EPOCHS):
            batch = 0
            num_batches = len(trainloader)
            
            total = 0
            loss = 0
            correct = 0
            
            for item in trainloader:
                (x, x_i, x_j), labels = item['img'], item['label']
                x, labels = x_i.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                outputs = simclr_predictor(x)
                
                values, predicted = torch.max(outputs, 1)  
                
                loss = criterion(outputs, labels)
            
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                
                
                loss.backward()
                optimizer.step()
                
                print(f"Epoch: {total_epochs} Predictor Train Batch: {batch} / {num_batches}")
                batch += 1
                
        total_epochs += 1
        
        print(f"Train Accuracy: {correct/total}")
        
        data = [total_epochs, "train accuracy", int(correct/total * 10000) / 10000]

        with open(log_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
                
        _, accuracy = evaluate(testloader, criterion)
        print("Test Accuracy:", accuracy)
        
        data = [total_epochs, "test accuracy", int(correct/total * 10000) / 10000]

        with open(log_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        
        save_model(int(accuracy * 10000)/10000)

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



def save_model(acc):
    global count
    torch.save(simclr_predictor.state_dict(), f"./centralized_weights/centralized_model_{count}_{acc}.pth")
    count += 1

if __name__ == "__main__":
    centralized_train(False)

