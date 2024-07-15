import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100
from transform import SimCLRTransform
from torch.utils.data import TensorDataset, Dataset
from torch import Generator
import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


global NUM_CLIENTS

NUM_CLIENTS = 2

NUM_CLASSES = 10
NUM_ROUNDS = 7
useResnet18 = False
fineTuneEncoder = True
addGausainBlur = True
evaluateEveryRound = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

centralized_finetune_split = 0.1
centralized_test_split = 0.2

client_train_split = 0.05

FINETUNE_EPOCHS = 10

BATCH_SIZE = 512
transform = SimCLRTransform(size=32, gaussian=addGausainBlur)

client_fds, server_fds = None, None

client_dict, server_dict = {}, {}


datalog_path = './log_files/datalog.csv'


def printClients():
    global NUM_CLIENTS
    print(NUM_CLIENTS)
    
def apply_transforms(batch):
    batch["img"] = [transform(img) for img in batch["img"]]
    # batch['label'] = [transform(img) for label in batch["img"]]
    return batch


def load_partition(partition_id, partitions, image_size=32):
    global client_fds, client_dict, calls, NUM_CLIENTS
    

    if partition_id not in client_dict.keys():
        client_dict = {}
        if client_fds is None:
            print('NUM CLIENTS WHEN NONE', partitions)
            printClients()
            client_fds = FederatedDataset(dataset="cifar10", partitioners={'train': IidPartitioner(partitions)})

        partition = client_fds.load_partition(partition_id, "train")
        print("Partition Size",partition)
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(test_size=client_train_split)
        
        client_dict[partition_id] = partition["train"], partition["test"]
            
    return client_dict[partition_id]

fds = None

def load_centralized_data(image_size=32, batch_size=BATCH_SIZE):
    global fds
    if fds is None:
        fds = FederatedDataset(dataset="cifar10", partitioners = {'train' : 1, 'test' : 1})
        
    centralized_train_data = fds.load_split("train")
    centralized_train_data = centralized_train_data.with_transform(apply_transforms)
    
    centralized_train_data = centralized_train_data.train_test_split(test_size=centralized_finetune_split, shuffle = True, seed=42)['test']

    print('Train Len', len(centralized_train_data))

    centralized_test_data = fds.load_split("test")
    centralized_test_data = centralized_test_data.with_transform(apply_transforms)
    
    centralized_test_data = centralized_test_data.train_test_split(test_size=centralized_test_split, shuffle = True, seed=42)['test']

    print('Test Len', len(centralized_test_data))
    
    return centralized_train_data, centralized_test_data

