import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100
from transform import SimCLRTransform
from torch.utils.data import TensorDataset, Dataset
from torch import Generator
import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


DRY_RUN = True

NUM_CLIENTS = 5
NUM_CLASSES = 10
NUM_ROUNDS = 8
useResnet18 = False
fineTuneEncoder = True
addGausainBlur = True
evaluateEveryRound = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

centralized_finetune_split = 0.1
centralized_test_split = 0.1

client_train_split = 0.05

FINETUNE_EPOCHS = 5

BATCH_SIZE = 512
transform = SimCLRTransform(size=32, gaussian=addGausainBlur)

client_fds = None
server_fds = None

datalog_path = './datalog.csv'

def apply_transforms(batch):
    batch["img"] = [transform(img) for img in batch["img"]]
    return batch

def load_partition(partition_id, image_size=32):
    global client_fds

    if client_fds is None:
        client_fds = FederatedDataset(dataset="cifar10", partitioners={'train': IidPartitioner(NUM_CLIENTS)})

    partition = client_fds.load_partition(partition_id)
    partition = partition.with_transform(apply_transforms)
    partition = partition.train_test_split(test_size=client_train_split, shuffle = True)
    
    # if DRY_RUN:
    #     partition["train"]['img'] = partition["train"]['img'][:BATCH_SIZE * NUM_CLIENTS]
    #     partition["test"]['img'] = partition["test"]['img'][:BATCH_SIZE * NUM_CLIENTS]
        
    #     partition["train"]['label'] = partition["train"]['label'][:BATCH_SIZE * NUM_CLIENTS]
    #     partition["test"]['label'] = partition["test"]['label'][:BATCH_SIZE * NUM_CLIENTS]

    trainloader = DataLoader(partition["train"], batch_size=BATCH_SIZE, shuffle = True)
    testloader = DataLoader(partition["test"], batch_size=BATCH_SIZE, shuffle = True)
    

        
    
    return trainloader, testloader

def load_centralized_data(image_size=32, batch_size=BATCH_SIZE):
    global server_fds
    
    if server_fds is None:
        server_fds = FederatedDataset(dataset="cifar10", partitioners={'train': 1, 'test': 1})

    centralized_train_data = server_fds.load_split("test")
    centralized_train_data = centralized_train_data.with_transform(apply_transforms)
    
    centralized_train_data = centralized_train_data.train_test_split(test_size=centralized_finetune_split, shuffle = True)['test']

    centralized_test_data = server_fds.load_split("train")
    centralized_test_data = centralized_test_data.with_transform(apply_transforms)
    
    centralized_test_data = centralized_test_data.train_test_split(test_size=centralized_test_split, shuffle = True)['test']

    

    trainloader = DataLoader(centralized_train_data, batch_size=batch_size, shuffle = True)
    testloader = DataLoader(centralized_test_data, batch_size=batch_size, shuffle = True)

    return trainloader, testloader

