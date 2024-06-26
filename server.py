import flwr as fl
from client import client_fn, NUM_CLIENTS
from flwr.server.strategy import FedAvg
import torch

strategy = fl.server.strategy.FedAvg()

fl.common.logger.configure(identifier="debug", filename="log.txt")

# Start Flower simulation
if __name__ == "__main__":
    print("Cuda?:", torch.cuda.is_available())
    print("name:", torch.cuda.get_device_name(0))
    
    print(torch.__version__)
    print(torch.version.cuda)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )