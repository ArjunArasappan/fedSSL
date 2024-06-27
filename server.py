import flwr as fl
from client import client_fn, NUM_CLIENTS
from flwr.server.strategy import FedAvg
import torch

from client import predictorloader, testloader, DEVICE
from dataset import num_classes

strategy = fl.server.strategy.FedAvg()

fl.common.logger.configure(identifier="debug", filename="log.txt")

def train_predictor(model, trainloaders, optimizer, criterion, epochs):    

    
class global_predictor():
    
    def __init__(self, tune_encoder, trainloader, testloader):
        
        self.simclr_predictor = SimCLRPredictor(DEVICE, useResnet18 = True, tune_encoder = tune_encoder, num_classes=NUM_CLASSES).to(DEVICE)
        
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = torch.optim.Adam(self.simclr.parameters(), lr=3e-4)
        self.epochs = 1
        self.criterion = nn.CrossEntropyLoss()
        
        def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            self.update_encoder(parameters)
            self.fine_tune_predictor()
            loss, accuracy = model.evaluate(x_val, y_val)
            return loss, {"accuracy": accuracy}

        return evaluate

    self.fine_tune_predictor():
        self.simclr_predictor.train()
    
        for epoch in range(self.epochs):
            batch = 0
            num_batches = len(self.trainloader)


            for (x, x_i, x_j), labels in self.trainloader:

                x, labels = x.to(DEVICE), labels.to(DEVICE)
                
                self.optimizer.zero_grad()
                
                outputs = self.simclr_predictor(x)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                print("Predictor Train Batch:", batch, "/", num_batches)


        
    def update_encoder(self, weights):
        self.simclr_predictor.set_encoder_parameters(weights)


gb_pred = global_predictor(tune_encoder = False, predictorloader, testloader)

if __name__ == "__main__":
    print("Cuda?:", torch.cuda.is_available())
    print("name:", torch.cuda.get_device_name(0))
    
    print(torch.__version__)
    print(torch.version.cuda)
    
    hist = client_resources = {
        "num_cpus": 8,
        "num_gpus": 1.0,
    }


    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
        client_resources=client_resources,
        evaluate_fn = 
        strategy=strategy
    )