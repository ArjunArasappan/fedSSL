import flwr as fl
import torch
import numpy as np
from typing import List
from collections import OrderedDict
import argparse

from client import get_client_fn
from model import SimCLR, SimCLRPredictor, NTXentLoss
import utils
from test import evaluate_gb_model 
import os

DEVICE = utils.DEVICE


fl.common.logger.configure(identifier="debug", filename="./log.txt")

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument(
    "--num_cpus",
    type=int,
    default= 12,
    help="Number of CPUs to use during simulation",
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default= 1,
    help="Number of GPUs to use during simulation",
)

parser.add_argument(
    "--num_clients",
    type=int,
    default=5,
    help="Number of clients",
)

parser.add_argument(
    "--use_resnet18",
    type=bool,
    default=False,
    help="Use Resnet18 over Resnet50",
)

parser.add_argument(
    "--num_rounds",
    type=int,
    default=5,
    help="Number of FL training rounds",
)


centralized_finetune, centralized_test = utils.load_centralized_data()

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            
            gb_simclr = SimCLR(DEVICE, useResnet18=useResnet18).to(DEVICE)

            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(gb_simclr.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            
            dir = './model_weights/'
            if not os.path.exists(dir):
                os.makedirs(dir)

            torch.save(state_dict, f"./model_weights/model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics

strategy = SaveModelStrategy()

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
    
    NUM_CLIENTS = args.num_clients
    NUM_ROUNDS = args.num_rounds
    useResnet18 = args.use_resnet18
    
    print("Num Clients: ", NUM_CLIENTS)
    print("Num Rounds: ", NUM_ROUNDS)
    print("Resnet18", useResnet18)
        
    fds = utils.get_fds(NUM_CLIENTS)
    
    fl.simulation.start_simulation(
        client_fn=get_client_fn(fds, useResnet18, NUM_CLIENTS),
        num_clients= NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds= NUM_ROUNDS),
        client_resources=client_resources,
        strategy=strategy
    )
    
    loss, accuracy = evaluate_gb_model(utils.useResnet18)
    
    print("FINAL GLOBAL MODEL RESULTS:")
    print("Loss:", loss)
    print("Accuracy:", accuracy)
    
    
    
    