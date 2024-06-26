import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from transform import SimCLRTransform
from torch.utils.data import Dataset
from torch import Generator


    
global_batch = 512
num_iters = -1
validation_split = 0.1
#batch size usually at 512, num workers at 8

def load_data(num_clients, image_size=32, batch_size = global_batch, num_workers = 0):
    
    transformation = SimCLRTransform(size = image_size, gaussian = False)
    
    train_dataset = CIFAR10(".", train=True, download=True, transform = transformation)
    test_dataset = CIFAR10(".", train=False, download=True, transform = transformation)
    
    partition_size = len(train_dataset) // num_clients

    partition_lengths = [partition_size] * num_clients
    print("LENGTHS: ", partition_lengths)
    
    for i in range(0, len(train_dataset) % num_clients):
        partition_lengths[i] = partition_lengths[i] + 1

    datasets = random_split(train_dataset, partition_lengths, Generator().manual_seed(42))
    
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = int(len(ds) * validation_split)  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        # print(type(Generator().manual_seed(42)))
        # print(len(ds))
        
        ds_train, ds_val = random_split(ds, lengths)
        
        trainloaders.append(
            DataLoader(ds_train, batch_size = batch_size, shuffle = True))
        valloaders.append(
            DataLoader(ds_val, batch_size = batch_size))
        
    testloader = DataLoader(test_dataset, batch_size = batch_size)
    return trainloaders, valloaders, testloader, partition_lengths
