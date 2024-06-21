import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from transform import SimCLRTransform
from torch.utils.data import Dataset

    
global_batch = 16
global_epoch = 10
#batch size usually at 512, num workers at 8

def load_data(image_size=32, batch_size=global_batch, num_workers=0):
    transformation = SimCLRTransform(size=image_size, gaussian=False)
    train_dataset = CIFAR10(".", train=True, download=True, transform=transformation)
    test_dataset = CIFAR10(".", train=False, download=True, transform=transformation)
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
    
    num_examples = {"trainset": len(train_dataset), "testset": len(test_dataset)}
    return train_loader, test_loader, num_examples
