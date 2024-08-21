import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SimCLR, SimCLRPredictor
import os, glob

from tqdm import tqdm

import fedSSL.utils as utils

DEVICE = utils.DEVICE



def evaluate_gb_model(useResnet18):
    simclr_predictor = SimCLRPredictor(utils.NUM_CLASSES, DEVICE, useResnet18=useResnet18, tune_encoder=utils.fineTuneEncoder).to(DEVICE)
    
    load_model(useResnet18, simclr_predictor)
    
    train, test = utils.load_centralized_data()
    
    trainloader = DataLoader(train, batch_size = utils.BATCH_SIZE)
    testloader = DataLoader(test, batch_size = utils.BATCH_SIZE)   

    optimizer = torch.optim.Adam(simclr_predictor.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    fine_tune_predictor(simclr_predictor, trainloader, optimizer, criterion)
    
    loss, accuracy = evaluate(simclr_predictor, testloader, criterion)
    


    return loss, accuracy


def load_model(useResnet18, simclr_predictor):
    simclr = SimCLR(DEVICE, useResnet18=useResnet18).to(DEVICE)

    
    list_of_files = [fname for fname in glob.glob("./fedSSL/model_weights/model_round_*")]
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from:", latest_round_file)
    state_dict = torch.load(latest_round_file)
    
    simclr.load_state_dict(state_dict)
    
    weights = [v.cpu().numpy() for v in simclr.state_dict().values()]
    
    simclr_predictor.set_encoder_parameters(weights)


def fine_tune_predictor(simclr_predictor, trainloader, optimizer, criterion):
    simclr_predictor.train()

    with tqdm(total=utils.FINETUNE_EPOCHS * len(trainloader), desc=f'Global Model Finetune', position=0, leave=True) as pbar:

        for epoch in range(utils.FINETUNE_EPOCHS):
            batch = 0
            num_batches = len(trainloader)

            for item in trainloader:
                x , labels = item['img'], item['label']
                x, labels = x.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                outputs = simclr_predictor(x)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                pbar.update(1)
                
                # print(f"Epoch: {epoch} Predictor Train Batch: {batch} / {num_batches}")
                batch += 1

def evaluate(simclr_predictor, testloader, criterion):
    simclr_predictor.eval()
    
    total = 0
    correct = 0
    loss = 0

    batch = 0
    num_batches = len(testloader)
    
    with tqdm(total=num_batches, desc=f'Global Model Test', position=0, leave=True) as pbar:

        with torch.no_grad():
            for item in testloader:
                x , labels = item['img'], item['label']
                x, labels = x.to(DEVICE), labels.to(DEVICE)
                
                logits = simclr_predictor(x)
                values, predicted = torch.max(logits, 1)  
                
                total += labels.size(0)
                loss += criterion(logits, labels).item()
                correct += (predicted == labels).sum().item()

                pbar.update(1)

                batch += 1
  
    return loss / batch, correct / total

if __name__ == "__main__":
    loss, accuracy = evaluate_gb_model(utils.useResnet18)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
