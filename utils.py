import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100
from transform import SimCLRTransform
from torch.utils.data import TensorDataset, Dataset
from torch import Generator
import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

import csv
NUM_CLASSES = 10

useResnet18 = False
fineTuneEncoder = False
addGausainBlur = True
evaluateEveryRound = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

centralized_finetune_split = 1
centralized_test_split = 0.25

FINETUNE_EPOCHS = 1
BATCH_SIZE = 512

transform = SimCLRTransform(size=32)

def sim_log(data, path = './log_files/datalog.csv'):
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

    
def apply_transforms(batch):
    batch["img"] = [transform(img) for img in batch["img"]]
    return batch

def get_fds(partitions):
    client_fds = FederatedDataset(dataset="cifar10", partitioners={'train': IidPartitioner(partitions)})
    return client_fds

def load_partition(fds, partition_id, client_test_split = 0):
    
    partition = fds.load_partition(partition_id, "train")
    partition = partition.with_transform(apply_transforms)
    
    if client_test_split != 0:
        partition = partition.train_test_split(test_size=client_test_split)
        return partition["train"], partition["test"]
    
    return partition, None


def load_centralized_data(image_size=32, batch_size=BATCH_SIZE):
    fds = FederatedDataset(dataset="cifar10", partitioners = {'train' : 1, 'test' : 1})
        
    centralized_train_data = fds.load_split("train")
    centralized_train_data = centralized_train_data.with_transform(apply_transforms)
    
    if centralized_finetune_split != 1:
        centralized_train_data = centralized_train_data.train_test_split(test_size=centralized_finetune_split, shuffle = True, seed=42)['test']

    centralized_test_data = fds.load_split("test")
    centralized_test_data = centralized_test_data.with_transform(apply_transforms)

    if centralized_test_split != 1:
        centralized_test_data = centralized_test_data.train_test_split(test_size=centralized_test_split, shuffle = True, seed=42)['test']
    
    return centralized_train_data, centralized_test_data