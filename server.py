import flwr as fl
from client import client_fn, NUM_CLIENTS
from flwr.server.strategy import FedAvg




# Start Flower simulation
if __name__ == "__main__":
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
    )