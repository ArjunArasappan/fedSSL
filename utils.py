import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100
from transform import SimCLRTransform
from torch.utils.data import TensorDataset, Dataset
from torch import Generator
import torch
from flwr_datasets import FederatedDataset

NUM_CLIENTS = 2
NUM_CLASSES = 10
NUM_ROUNDS = 4
useResnet18 = False
fineTuneEncoder = True
addGausainBlur = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
centralized_fine_tune = 0.9

BATCH_SIZE = 512
transform = SimCLRTransform(size=32, gaussian=addGausainBlur)
federated_dataset = None

def apply_transforms(batch):
    batch["img"] = [transform(img) for img in batch["img"]]
    return batch

def load_partition(partition_id, image_size=32):
    global federated_dataset

    if federated_dataset is None:
        federated_dataset = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})

    partition = federated_dataset.load_partition(partition_id)
    partition = partition.with_transform(apply_transforms)
    partition = partition.train_test_split(test_size=0.2, seed=42)

    trainloader = DataLoader(partition["train"], batch_size=BATCH_SIZE)
    testloader = DataLoader(partition["test"], batch_size=BATCH_SIZE)
    
    return trainloader, testloader

def load_centralized_data(image_size=32, batch_size=BATCH_SIZE):
    fds = FederatedDataset(dataset="cifar10", partitioners={"test": 1})
    centralized_data = fds.load_split("test")
    centralized_data = centralized_data.with_transform(apply_transforms)
    
    centralized_data = centralized_data.train_test_split(test_size=centralized_fine_tune)
    
    trainloader = DataLoader(centralized_data["train"], batch_size=batch_size)
    testloader = DataLoader(centralized_data["test"], batch_size=batch_size)

    return trainloader, testloader

