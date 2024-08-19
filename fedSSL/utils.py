import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

from transform import SimCLRTransform
import csv


NUM_CLASSES = 10

useResnet18 = False
fineTuneEncoder = True
evaluateEveryRound = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

centralized_finetune_split = 0.1
centralized_test_split = 0.2

FINETUNE_EPOCHS = 10
BATCH_SIZE = 512



datalog_path = './log_files/datalog.csv'

def transform_fn(augmentData):
    simclrTransform = SimCLRTransform(size=32)  
    def apply_transforms(batch):
        batch["img"] = [simclrTransform(img, augmentData) for img in batch["img"]]
        return batch

    return apply_transforms

def get_fds(partitions):
    client_fds = FederatedDataset(dataset="cifar10", partitioners={'train': IidPartitioner(partitions)})
    return client_fds

def load_partition(fds, partition_id, client_test_split = 0):
    
    partition = fds.load_partition(partition_id, "train")
    partition = partition.with_transform(transform_fn(True))
    
    if client_test_split == 0:
        return partition, None
    
    partition = partition.train_test_split(test_size=client_test_split)
    return partition["train"], partition["test"]
    

def load_centralized_data(image_size=32, batch_size=BATCH_SIZE):
    fds = FederatedDataset(dataset="cifar10", partitioners = {'train' : 1, 'test' : 1})
        
    centralized_train_data = fds.load_split("train")
    centralized_train_data = centralized_train_data.with_transform(transform_fn(False))
    
    centralized_train_data = centralized_train_data.train_test_split(test_size=centralized_finetune_split, shuffle = True, seed=42)['test']

    centralized_test_data = fds.load_split("test")
    centralized_test_data = centralized_test_data.with_transform(transform_fn(False))
    
    centralized_test_data = centralized_test_data.train_test_split(test_size=centralized_test_split, shuffle = True, seed=42)['test']
    
    return centralized_train_data, centralized_test_data

def sim_log(data, path = './sim_log.txt'):
    with open('path', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
