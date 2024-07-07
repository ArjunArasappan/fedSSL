import flwr as fl
from client import client_fn, NUM_CLIENTS
from flwr.server.strategy import FedAvg
import torch
import argparse

from client import predictorloader, testloader, DEVICE
from dataset import NUM_CLASSES
from model import SimCLRPredictor, NTXentLoss
import torch.nn as nn
import numpy as np
import torch

from flwr.common import NDArrays, Scalar

from typing import Dict, Optional, Tuple

NUM_ROUNDS = 7


fl.common.logger.configure(identifier="debug", filename="log.txt")

batch_break = -1
print_interval = 0.1

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument(
    "--num_cpus",
    type=int,
    default=1,
    help="Number of CPUs to assign to a virtual client",
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default=2.0,
    help="Ratio of GPU memory to assign to a virtual client",
)



    
class global_predictor:
    
    def __init__(self, tune_encoder, trainloader, testloader, useResnet18 = True):
        self.round = 0
        
        self.simclr_predictor = SimCLRPredictor(NUM_CLASSES, DEVICE, useResnet18 = useResnet18, tune_encoder = tune_encoder).to(DEVICE)
        
        self.trainloader = trainloader
        self.testloader = testloader
        
        self.epochs = 20
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
            percent_trained = .1

            for (x, x_i, x_j), labels in self.trainloader:

                x, labels = x.to(DEVICE), labels.to(DEVICE)
                
                self.optimizer.zero_grad()
                
                outputs = self.simclr_predictor(x)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                

                if batch >= int(percent_trained * num_batches):
                    break        

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
                
                for (x, x_i, x_j), labels in self.testloader:
                    x, labels = x.to(DEVICE), labels.to(DEVICE)
                    
                    logits = self.simclr_predictor(x)
                    values, predicted = torch.max(logits, 1)  
                    
                    total += labels.size(0)
                    
                    loss += self.criterion(logits, labels)
                    
                    
                    correct += (predicted == labels).sum().item()
                    
                    if batch == batch_break:
                        break

                    # if batch % (print_interval * num_batches) == 0:
                    print(f"Epoch: {epoch} Predictor Train Batch: {batch} / {num_batches}")
                    
                    batch += 1
                    count += 1
                    
                break
            
        return loss / count, correct / total
            
            


        
    def update_encoder(self, weights):
        self.simclr_predictor.set_encoder_parameters(weights)


gb_pred = global_predictor(True, predictorloader, testloader, useResnet18 = False)

strategy = fl.server.strategy.FedAvg(
    evaluate_fn = gb_pred.get_evaluate_fn(),
)


if __name__ == "__main__":
    print("Cuda?:", torch.cuda.is_available())
    print("name:", torch.cuda.get_device_name(0))
    
    print(torch.__version__)
    print(torch.version.cuda)
    
    args = parser.parse_args()

    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        client_resources=client_resources,
        strategy=strategy
    )