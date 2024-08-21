"""pytorchexample: A Flower / PyTorch app."""

from typing import List, Tuple
import flwr as fl
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

import fedSSL.utils as utils
from fedSSL.model import SimCLR
from typing import List
from collections import OrderedDict
import torch
import os


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m['loss'] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

class SaveModelStrategy(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            
            gb_simclr = SimCLR(utils.DEVICE, useResnet18=utils.useResnet18).to(utils.DEVICE)

            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(gb_simclr.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            gb_simclr.load_state_dict(state_dict)
            
            dir = '.fedSSL/model_weights/'
            if not os.path.exists(dir):
                os.makedirs(dir)

            torch.save(gb_simclr.state_dict(), dir + f"model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics
    
def server_fn(context: Context):

    num_rounds = context.run_config["num-server-rounds"]

    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=2,
        fit_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)