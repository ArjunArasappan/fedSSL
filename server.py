import flwr as fl
from client import client_fn, NUM_CLIENTS
from flwr.server.strategy import FedAvg
import torch
import argparse
import torch.nn as nn
import numpy as np

from model import SimCLRPredictor, NTXentLoss, GlobalPredictor
from utils import NUM_CLIENTS, NUM_CLASSES, NUM_ROUNDS, DEVICE, useResnet18, fineTuneEncoder, load_centralized_data



fl.common.logger.configure(identifier="debug", filename="log.txt")

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
    default=1.0,
    help="Ratio of GPU memory to assign to a virtual client",
)


centralized_finetune, centralized_test = load_centralized_data()

gb_pred = GlobalPredictor(fineTuneEncoder, centralized_finetune, centralized_test, DEVICE, useResnet18 = useResnet18)

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