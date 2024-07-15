import flwr as fl

from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes
import torch
import argparse
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from collections import OrderedDict

import os



import client
from model import SimCLR, SimCLRPredictor, NTXentLoss, GlobalPredictor
import utils
from test import evaluate_gb_model 

DEVICE = utils.DEVICE

global gb_pred, gb_simclr




fl.common.logger.configure(identifier="debug", filename="./log_files/log.txt")

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument(
    "--num_cpus",
    type=int,
    default= 12,
    help="Number of CPUs to assign to a virtual client",
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default= 1,
    help="Ratio of GPU memory to assign to a virtual client",
)

parser.add_argument(
    "--num_clients",
    type=int,
    default=utils.NUM_CLIENTS,
    required= True,
    help="Ratio of GPU memory to assign to a virtual client",
)

parser.add_argument(
    "--use_resnet18",
    type=bool,
    default=utils.useResnet18,
    help="Ratio of GPU memory to assign to a virtual client",
)

parser.add_argument(
    "--num_rounds",
    type=int,
    default=utils.NUM_ROUNDS,
    help="Ratio of GPU memory to assign to a virtual client",
)


centralized_finetune, centralized_test = utils.load_centralized_data()





class SaveModelStrategy(fl.server.strategy.FedAvg):
    global gb_simclr
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model weights using weighted average and store checkpoint"""


        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            params_dict = zip(gb_simclr.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            gb_simclr.load_state_dict(state_dict, strict=True)

            torch.save(gb_simclr.state_dict(), f"./model_weights/model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics

strategy = SaveModelStrategy(
    # evaluate_fn = gb_pred.get_evaluate_fn()
)

if __name__ == "__main__":
    global gb_pred, gb_simclr

    print("Cuda?:", torch.cuda.is_available())
    print("name:", torch.cuda.get_device_name(0))
    
    print(torch.__version__)
    print(torch.version.cuda)
    
    args = parser.parse_args()

    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }
    
    utils.NUM_CLIENTS = args.num_clients
    utils.NUM_ROUNDS = args.num_rounds
    utils.useResnet18 = args.use_resnet18
    
    print("Num Clients: ", utils.NUM_CLIENTS)
    print("Num Rounds: ", utils.NUM_ROUNDS)
    print("Resnet18", utils.useResnet18)
    
    utils.printClients()
    
    fds = utils.get_fds(utils.NUM_CLIENTS)
    
    gb_pred = GlobalPredictor(utils.fineTuneEncoder, centralized_finetune, centralized_test, DEVICE, useResnet18 = utils.useResnet18)
    gb_simclr = SimCLR(DEVICE, useResnet18=utils.useResnet18).to(DEVICE)

    fl.simulation.start_simulation(
        client_fn=client.get_client_fn(fds, utils.useResnet18, utils.NUM_CLIENTS),
        num_clients= utils.NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds= utils.NUM_ROUNDS),
        client_resources=client_resources,
        strategy=strategy
    )
    
    # loss, accuracy = evaluate_gb_model(utils.useResnet18)
    
    # print("FINAL GLOBAL MODEL RESULTS:")
    # print("Loss:", loss)
    # print("Accuracy:", accuracy)
    
    
    
    
    