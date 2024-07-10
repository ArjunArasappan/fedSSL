import os
import warnings
from hydra import compose, initialize

import flwr as fl
from flwr_datasets import FederatedDataset

from client import client_fn

from dataset import get_tokenizer_and_data_collator_and_propt_formatting
from client import gen_client_fn
from utils import get_on_fit_config, fit_weighted_average
from server import SaveModelStrategy


warnings.filterwarnings("ignore", category=UserWarning)

NUM_ROUNDS = 7
save_path = "./results/"

with initialize(config_path="conf"):
    cfg = compose(config_name="config")

# Reset the number of number
cfg.num_rounds = NUM_ROUNDS
cfg.train.num_rounds = NUM_ROUNDS

# Create output directory
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Partition dataset and get dataloaders
# We set the number of partitions to 20 for fast processing.
fds = FederatedDataset(dataset=cfg.dataset.name, partitioners={"train": cfg.num_clients})


# ClientApp for client #1 (Flower Next)
client1 = fl.client.ClientApp(
    client_fn = client_fn(1)
)


# ClientApp for client #2 (Flower Next)
client2 = fl.client.ClientApp(
    client_fn = client_fn(2)
)


# Instantiate strategy.
strategy = SaveModelStrategy()

# ServerApp for Flower-Next
server = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)